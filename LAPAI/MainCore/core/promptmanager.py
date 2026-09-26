from .state import *


@dataclass
class PromptMeta:
    created_at: str
    relation: float = 1.0


class PromptManager:
    conl = cache.conf.get('CONTEXT_LIMIT')
    limrat = cache.conf.get('SOFT_LIMIT_RATIO')
    resout = cache.conf.get('RESERVED_OUTPUT')
    tbat = cache.conf.get('TRIM_BATCH')
    das = cache.conf.get('DECAY_AFTER_SECONDS')
    dis = cache.conf.get('DECAY_INTERVAL_SECONDS')
    ds = cache.conf.get('DECAY_STEP')

    def __init__(self):
        self.meta: dict[int, PromptMeta] = {}
        self.last_stats: dict[str, Any] = {}

    def _conf(self, key: str, default: Any) -> Any:
        value = cache.conf.get(key, default)
        return default if value is None else value

    @property
    def context_limit(self) -> int:
        return max(1, int(self._conf(
            "promptContextLimit",
            self.conl,
        )))

    @property
    def soft_limit_ratio(self) -> float:
        value = float(self._conf(
            "promptSoftLimitRatio",
            self.limrat,
        ))
        return min(max(value, 0.50), 0.99)

    @property
    def reserved_output(self) -> int:
        return max(0, int(self._conf(
            "promptReservedOutputTokens",
            self.resout,
        )))

    @property
    def trim_batch(self) -> int:
        return max(1, int(self._conf(
            "promptTrimBatchMessages",
            self.tbat,
        )))

    @property
    def decay_after_seconds(self) -> float:
        return max(0.0, float(self._conf(
            "promptDecayAfterSeconds",
            self.das,
        )))

    @property
    def decay_interval_seconds(self) -> float:
        return max(1.0, float(self._conf(
            "promptDecayIntervalSeconds",
            self.dis,
        )))

    @property
    def decay_step(self) -> float:
        value = float(self._conf(
            "promptDecayStep",
            self.ds,
        ))
        return min(max(value, 0.0), 1.0)

    @property
    def prompt_soft_limit(self) -> int:
        effective_capacity = max(
            1,
            self.context_limit - self.reserved_output,
        )
        return max(
            1,
            int(effective_capacity * self.soft_limit_ratio),
        )


    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _parse_timestamp(value: str | None) -> datetime:
        if not value:
            return datetime.now(timezone.utc)

        try:
            dt = datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return datetime.now(timezone.utc)

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        return dt.astimezone(timezone.utc)

    def _relation(self, created_at: str) -> float:      #<--Decay Relation
        elapsed = max(
            0.0,
            (self._now() - self._parse_timestamp(created_at)).total_seconds(),
        )

        if elapsed <= self.decay_after_seconds:
            return 1.0

        extra = elapsed - self.decay_after_seconds

        # First 0.5% decay happens immediately after the first 3-minute
        # threshold; subsequent decay happens every configured interval.
        steps = int(extra // self.decay_interval_seconds) + 1

        return max(
            0.0,
            1.0 - (steps * self.decay_step),
        )

    def _is_protected(self, message: dict[str, Any]) -> bool:
        return str(message.get("role", "")).lower() == "system"

    # ---------- metadata ----------

    def sync(self, messages: list[dict[str, Any]]) -> None:
        live_ids = set()

        for message in messages:
            if not isinstance(message, dict):
                continue

            message_id = id(message)
            live_ids.add(message_id)

            if message_id not in self.meta:
                created_at = message.get("_lapai_created_at")
                if created_at is None:
                    created_at = datetime.now(timezone.utc).isoformat()

                self.meta[message_id] = PromptMeta(
                    created_at=str(created_at),
                    relation=1.0,
                )

        stale_ids = set(self.meta) - live_ids
        for message_id in stale_ids:
            self.meta.pop(message_id, None)

    def relation_for(self, message: dict[str, Any]) -> float:
        if self._is_protected(message):
            return 1.0

        meta = self.meta.get(id(message))
        if meta is None:
            self.sync([message])
            meta = self.meta[id(message)]

        meta.relation = self._relation(meta.created_at)
        return meta.relation


    def _groups(self,messages: list[dict[str, Any]],) -> list[list[dict[str, Any]]]:
        groups: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []

        for message in messages:
            role = str(message.get("role", "")).lower()

            if role == "system":
                if current:
                    groups.append(current)
                    current = []
                groups.append([message])
                continue

            if role == "user" and current:
                groups.append(current)
                current = []

            current.append(message)

        if current:
            groups.append(current)

        return groups

    def _group_created_at(self, group: list[dict[str, Any]]) -> str:
        timestamps = []

        for message in group:
            meta = self.meta.get(id(message))
            if meta is not None:
                timestamps.append(meta.created_at)

        if not timestamps:
            return datetime.now(timezone.utc).isoformat()

        return min(
            timestamps,
            key=lambda value: self._parse_timestamp(value).timestamp(),
        )

    def _group_relation(self, group: list[dict[str, Any]]) -> float:
        if any(self._is_protected(message) for message in group):
            non_system = [
                message
                for message in group
                if not self._is_protected(message)
            ]
            if not non_system:
                return 1.0
            return min(self.relation_for(message) for message in non_system)

        return min(self.relation_for(message) for message in group)

    def remove_expired(self, messages: list[dict[str, Any]]) -> int:
        removed = 0

        for group in self._groups(messages):
            if not group:
                continue

            if all(self._is_protected(message) for message in group):
                continue

            if self._group_relation(group) <= 0.0:
                group_ids = {id(message) for message in group}
                messages[:] = [
                    message
                    for message in messages
                    if id(message) not in group_ids
                ]
                removed += len(group)

        self.sync(messages)
        return removed


    def estimate_tokens(self, messages: list[dict[str, Any]]) -> int:
        total = 0

        tokenizer = getattr(cache, "tokenizer", None)
        if tokenizer is None:
            for message in messages:
                total += len(str(message.get("content", "")).split()) + 4
            return total

        for message in messages:
            if not isinstance(message, dict):
                continue

            total += 4

            content = message.get("content")
            if content:
                total += len(
                    tokenizer.encode(
                        str(content),
                        add_special_tokens=False,
                    )
                )

            if message.get("name"):
                total += len(
                    tokenizer.encode(
                        str(message["name"]),
                        add_special_tokens=False,
                    )
                )

            if message.get("tool_call_id"):
                total += len(
                    tokenizer.encode(
                        str(message["tool_call_id"]),
                        add_special_tokens=False,
                    )
                )

            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                total += len(
                    tokenizer.encode(
                        str(tool_calls),
                        add_special_tokens=False,
                    )
                )

        return total + 2

    def trim_oldest_batch(self,messages: list[dict[str, Any]],target_count: int | None = None,) -> int:
        target_count = self.trim_batch if target_count is None else max(
            1,
            int(target_count),
        )

        groups = self._groups(messages)

        removable = []
        for index, group in enumerate(groups):
            if not group:
                continue

            if all(self._is_protected(message) for message in group):
                continue
            removable.append({
                "index": index,
                "group": group,
                "created_at": self._group_created_at(group),
                "relation": self._group_relation(group),
            })

        if len(groups) > 0:
            newest_index = max(
                item["index"]
                for item in removable
            ) if removable else None

            if newest_index is not None:
                removable = [
                    item
                    for item in removable
                    if item["index"] != newest_index
                ]

        removable.sort(
            key=lambda item: (
                self._parse_timestamp(
                    item["created_at"]
                ).timestamp(),
                item["relation"],
            )
        )

        remove_ids: set[int] = set()
        removed_count = 0

        for item in removable:
            group = item["group"]
            for message in group:
                remove_ids.add(id(message))
                removed_count += 1

            if removed_count >= target_count:
                break

        if not remove_ids:
            return 0

        messages[:] = [
            message
            for message in messages
            if id(message) not in remove_ids
        ]

        self.sync(messages)
        return removed_count

    def prepare_for_model(self,messages: list[dict[str, Any]],) -> list[dict[str, Any]]:
        self.sync(messages)

        expired_removed = self.remove_expired(messages)

        trimmed = 0
        token_count = self.estimate_tokens(messages)

        while token_count >= self.prompt_soft_limit:
            removed = self.trim_oldest_batch(messages)
            if removed <= 0:
                break

            trimmed += removed
            token_count = self.estimate_tokens(messages)

        self.last_stats = {
            "tokens": token_count,
            "context_limit": self.context_limit,
            "reserved_output": self.reserved_output,
            "soft_limit": self.prompt_soft_limit,
            "expired_removed": expired_removed,
            "trimmed_messages": trimmed,
        }

        return [
            {
                key: value
                for key, value in message.items()
                if not key.startswith("_lapai_")
            }
            for message in messages
        ]


def get_prompt_manager() -> PromptManager:
    manager = getattr(cache, "prompt_manager", None)

    if manager is None:
        manager = PromptManager()
        cache.prompt_manager = manager

    return manager
