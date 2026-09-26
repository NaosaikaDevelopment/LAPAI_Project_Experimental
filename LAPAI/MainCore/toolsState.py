from .core import *

from typing import Any, Union, get_args, get_origin, get_type_hints

class ToolRegistry:

    def __init__(self):
        self.tools: dict[str, dict[str, Any]] = {}
        self.schemas: list[dict[str, Any]] = []
        self.source_package: str | None = None

    def clear(self):
        self.tools.clear()
        self.schemas.clear()

    def discover(self, package_name: str):

        root = importlib.import_module(package_name)
        self.clear()
        self.source_package = package_name
        modules = [(root.__name__, root)]
        if hasattr(root, "__path__"):

            for _, module_name, _ in pkgutil.walk_packages(root.__path__,root.__name__ + ".",):
                print(f"[TOOL DISCOVERY] FOUND: {module_name}")
                module = importlib.import_module(module_name)

                modules.append((module_name, module))
        else:

            print(
                f"[TOOL DISCOVERY] FOUND: {root.__name__}"
            )

        for _, module in modules:
            self._discover_module(module)

        self.schemas = [
            build_tool_schema(
                name,
                info
            )
            for name, info in self.tools.items()
        ]

        return self

    def _discover_module(self, module):

        for name, function in inspect.getmembers(module,inspect.isfunction):
            if function.__module__ != module.__name__:
                continue
            if name.startswith("_"):
                continue

            if name in self.tools:

                existing = self.tools[name]["module"]

                raise ValueError(
                    f"Duplicate tool name '{name}' found in "
                    f"{existing} and {module.__name__}. "
                    "Rename one of the functions."
                )

            description = (
                inspect.getdoc(function)
                or f"Tool: {name}"
            )

            metadata = _parse_tool_metadata(description)

            self.tools[name] = {
                "function": function,
                "description": description,
                "module": module.__name__,
                **metadata,
            }
_TOOL_META_PREFIXES = {
    "requires_state": "@requires_state:",
    "provides_state": "@provides_state:",
    "required_when_state": "@required_when_state:",
}


def _parse_tool_metadata(description: str) -> dict[str, Any]:
    """Parse generic tool lifecycle/dependency metadata from a docstring."""
    text = description or ""
    metadata = {
        "required_each_turn": False,
        "requires_state": [],
        "provides_state": [],
        "required_when_state": [],
    }

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if line == "@required_each_turn":
            metadata["required_each_turn"] = True
            continue

        for key, prefix in _TOOL_META_PREFIXES.items():
            if line.startswith(prefix):
                state_name = line[len(prefix):].strip()
                if state_name:
                    metadata[key].append(state_name)
                break

    return metadata


def _state_is_active(context_state: dict[str, Any], state_name: str) -> bool:
    return bool(context_state.get(state_name))


def get_required_tool_names(
    registry: "ToolRegistry",
    context_state: dict[str, Any] | None = None,
) -> set[str]:
    """Return currently mandatory tools using only generic tool metadata."""
    context_state = context_state or {}
    required = set()

    for name, info in registry.tools.items():
        if info.get("required_each_turn", False):
            required.add(name)
            continue

        conditions = info.get("required_when_state", [])
        if conditions:
            all_active = True
            for state_name in conditions:
                if not _state_is_active(context_state, state_name):
                    all_active = False
                    break
            if all_active:
                required.add(name)

    return required


def get_eligible_tool_names(
    registry: "ToolRegistry",
    context_state: dict[str, Any] | None = None,
) -> set[str]:
    """Return tools whose state requirements are currently satisfied."""
    context_state = context_state or {}
    eligible = set()

    for name, info in registry.tools.items():
        requirements = info.get("requires_state", [])

        all_active = True
        for state_name in requirements:
            if not _state_is_active(context_state, state_name):
                all_active = False
                break

        if all_active:
            eligible.add(name)

    return eligible

def get_tool_state_requirements(registry: "ToolRegistry", tool_name: str) -> list[str]:
    """Return state names that must be active before a tool may execute."""
    info = registry.tools.get(tool_name)
    if info is None:
        raise KeyError(tool_name)
    return list(info.get("requires_state", []))


def _annotation_to_schema(annotation: Any) -> dict[str, Any]:

    if (annotation is inspect.Parameter.empty or annotation is Any):
        return {}

    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Union:

        non_none = [
            arg
            for arg in args
            if arg is not type(None)
        ]

        if len(non_none) == 1:

            schema = _annotation_to_schema(
                non_none[0]
            )

            schema["nullable"] = True

            return schema
    if origin is list:

        item_type = args[0] if args else Any

        return {
            "type": "array",
            "items": _annotation_to_schema(
                item_type
            ),
        }

    if origin is dict:

        return {
            "type": "object"
        }

    primitive_types = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }

    json_type = primitive_types.get(
        annotation
    )

    if json_type:

        return {
            "type": json_type
        }

    return {}


def build_tool_schema(name: str,info: dict[str, Any]) -> dict[str, Any]:

    function = info["function"]

    signature = inspect.signature(function)

    try:
        hints = get_type_hints(function)
    except Exception:
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for parameter_name, parameter in signature.parameters.items():

        if parameter_name in ("self","cls"):
            continue

        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL,inspect.Parameter.VAR_KEYWORD,):
            continue

        annotation = hints.get(parameter_name,parameter.annotation,)

        properties[parameter_name] = (_annotation_to_schema(annotation))

        if (parameter.default is inspect.Parameter.empty):
            required.append(
                parameter_name
            )

    description = info["description"]
    contract_lines = []

    if info.get("required_each_turn"):
        contract_lines.append("Execution contract: this tool is required on every user turn.")

    if info.get("requires_state"):
        contract_lines.append(
            "Execution contract: requires active state(s): "
            + ", ".join(info["requires_state"])
            + "."
        )

    if info.get("required_when_state"):
        contract_lines.append(
            "Execution contract: required when state(s) are active: "
            + ", ".join(info["required_when_state"])
            + "."
        )

    if contract_lines:
        description = description + "\n\n" + "\n".join(contract_lines)

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def build_all_schemas(registry: ToolRegistry):

    if registry.schemas:
        return registry.schemas

    registry.schemas = [
        build_tool_schema(
            name,
            info
        )
        for name, info
        in registry.tools.items()
    ]

    return registry.schemas


class ToolRanker:

    def __init__(self):
        self.index = None
        self.tool_names = []

    def build(self, registry):

        vectors = []
        self.tool_names = []

        for name, info in registry.tools.items():
            description = (info["description"]or name)
            tool_text = (f"Tool Name:\n"f"{name}\n\n"f"Description:\n"f"{description}")
            vector = generate_embedding(tool_text,embedding_type="passage")
            vector = np.asarray(vector,dtype=np.float32).reshape(1, -1)
            vectors.append(vector)

            self.tool_names.append(name)

        if not vectors:
            self.index = None

            cache.tool_faiss = None
            cache.tool_names = []

            return

        matrix = np.vstack(vectors).astype(np.float32)

        dimension = matrix.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(matrix)

        cache.tool_faiss = self.index
        cache.tool_names = self.tool_names.copy()

        print( "[TOOL RANKER] Indexed " f"{len(self.tool_names)} tools.")

    def search(self,query: str,top_k: int = 5,threshold: float | None = None,):

        if self.index is None:
            return []

        if not query:
            return []

        query_vector = generate_embedding(query,embedding_type="query")

        query_vector = np.asarray(query_vector,dtype=np.float32).reshape(1, -1)

        if query_vector.shape[1] != self.index.d:
            raise ValueError( "Tool embedding dimension mismatch: " f"query={query_vector.shape[1]}, " f"index={self.index.d}")

        k = min(max(int(top_k), 1),self.index.ntotal)

        scores, indices = self.index.search(query_vector,k)

        results = []

        for score, index_id in zip(scores[0],indices[0]):

            if index_id < 0:
                continue

            if index_id >= len(self.tool_names):
                continue

            score = float(score)

            if (threshold is not None and score < threshold):
                continue

            results.append({"tool": self.tool_names[index_id],"score": score,})

        return results

registry = ToolRegistry()


def initialize_tools(package_name: str = "MainCore.core.tools",force: bool = False,):
    if (not force and getattr(cache,"tools_initialized",False ) and getattr(cache,"tool_registry",None ) is not None and getattr(cache,"tool_ranker",None ) is not None):
        return cache.tool_registry
    print("[TOOLS] Initializing tool system...")

    registry.discover(package_name)

    schemas = build_all_schemas(
        registry
    )

    ranker = ToolRanker()

    ranker.build(
        registry
    )

    cache.tool_registry = registry
    cache.tool_schemas = list(schemas)

    cache.tool_package = (package_name)

    cache.tool_ranker = (ranker)

    cache.tools_initialized = True

    print(
        "[TOOLS] Registered "
        f"{len(registry.tools)} tools."
    )

    return registry