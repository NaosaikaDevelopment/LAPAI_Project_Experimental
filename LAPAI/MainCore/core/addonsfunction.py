#this can be experimental feature and adding another function
from .state import *


def parse_yesno(user_it: str):
    match = re.search(r"\b(yes|no)\b", user_it, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    return "No"

def generate_question(client, model, text):
    prompt = f"Create a brief question about something odd or needing investigation from the following text: {text}\n\n"
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=cache.conf.get('gqMaxToken')
    )
    return resp.choices[0].message.content.strip()

def add_question(question):
    questions = load_json(cache.QUESTIONS_FILE, [])
    if any(q["question"] == question for q in questions):
       return
    questions.append({
        "question": question,
        "answered": False,
        "timestamp": datetime.now().isoformat(),
        "answer": []
    })
    save_json(cache.QUESTIONS_FILE, questions)

def load_json(path, default=None):
    if not os.path.exists(path):
        return list(default) if default is not None else []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


#Guard Scan
def estimate_tokens(messages):
    total = 0

    for message in messages:
        if isinstance(message, dict) and "content" in message:
            content = str(message["content"])
        elif isinstance(message, str):
            content = message
        else:
            continue

        total += len(
            cache.tokenizer.encode(
                content,
                add_special_tokens=False
            )
        )

    return total

def compute_importance(text):
    words = len(text.split())

    if words > 80:
        return 1.3
    if words < 10:
        return 0.7
    return 1.0

def normalize_fts(bm25_score):
    score = float(bm25_score)

    if score <= 0:
        return 1.0 / (1.0 + abs(score))

    return 1.0 / (1.0 + score)

def normalize_faiss(dist):
    return 1 / (1 + dist)

def compute_recency(timestamp):
    age = time.time() - timestamp
    return 1 / (1 + age / 86400) 

def calculate_context_budget(model_context_limit, reserved_output_tokens, used_tokens):
    return max(
        0,
        model_context_limit
        - reserved_output_tokens
        - used_tokens
    )
def trim_prompt(prompt,memory_prompt=None,chat_history=None,max_tokens=None):
    if max_tokens is None:
        return prompt

    while estimate_tokens(prompt) >= max_tokens:
        if memory_prompt and len(memory_prompt) > 0:
            memory_prompt.pop(0)
            continue

        if chat_history and len(chat_history) > 0:
            chat_history.pop(0)
            continue

        removed = False

        for i, msg in enumerate(prompt):
            if msg.get("role") != "system":
                del prompt[i]
                removed = True
                break

        if not removed:
            break

    return prompt

def load_persona():
    persona_paths = [cache.pepath]

    for path in persona_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                value = f.read().strip()

            if value:
                return value

    return None


def build_memory_prompt(user_msg,summary,active_memory,recalled_memory,knowledge):
    prompt_items = []

    if summary:
        prompt_items.append({
            "role": "system",
            "content": (
                "Persistent conversation memory:\n"
                + summary
            )
        })

    if knowledge:
        for item in knowledge:
            prompt_items.append({
                "role": item.get("role", "system"),
                "content": str(item.get("content", ""))
            })

    if recalled_memory:
        prompt_items.append({
            "role": "system",
            "content": (
                "Relevant previous conversation:\n"
                + format_items(recalled_memory)
            )
        })

    if active_memory:
        for item in active_memory:
            prompt_items.append({
                "role": item["role"],
                "content": str(item["content"])
            })

    prompt_items.append({
        "role": "user",
        "content": user_msg
    })

    return prompt_items

def format_items(items):
    formatted = []

    for m in items:
        content = str(m.get("content", ""))
        if "relation_score" in m:
            formatted.append(
                "- [Memory] "
                f"kind={m.get('kind', 'unknown')} "
                f"category={m.get('category', 'general')} "
                f"date={m.get('date', '')} "
                f"relation={float(m.get('relation_score', 0.0)):.4f} "
                f"score={float(m.get('score', 0.0)):.4f} "
                f"confidence={float(m.get('confidence', 0.0)):.4f} "
                f"support={m.get('support_count', 1)}\n"
                f"  Evidence: {content}"
            )
        else:
            formatted.append(f"- {content}")

    return "\n".join(formatted)
def get_current_time_context():
    now = datetime.now()
    return (
        f"Current time:\n"
        f"- Date: {now.strftime('%Y-%m-%d')}\n"
        f"- Time: {now.strftime('%H:%M:%S')}\n"
        f"- Day: {now.strftime('%A')}"
    )

def thoughtm(user_msg):
    completion = cache.client.chat.completions.create(
        model=cache.Sum_model,
        messages=[{"role":"system", "content":f"Just answer what is actually meant by this user input: '{user_msg}'"}],
        max_tokens = 10
    )
    return completion.choices[0].message.content


import inspect


def python_type_to_json(annotation):

    if annotation is str:
        return {
            "type": "string"
        }

    if annotation is int:
        return {
            "type": "integer"
        }

    if annotation is float:
        return {
            "type": "number"
        }

    if annotation is bool:
        return {
            "type": "boolean"
        }

    return {
        "type": "string"
    }


def build_tool_schema(name, info):

    function = info["function"]

    signature = inspect.signature(
        function
    )

    properties = {}
    required = []

    for param_name, parameter in signature.parameters.items():

        annotation = parameter.annotation

        if annotation is inspect.Parameter.empty:
            annotation = str

        properties[param_name] = \
            python_type_to_json(annotation)

        if parameter.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": info["description"],
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }