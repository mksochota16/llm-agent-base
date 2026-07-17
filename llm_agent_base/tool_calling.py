import inspect
import json
from typing import Callable, Union, get_args, get_origin, get_type_hints

_JSON_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _is_optional(tp) -> bool:
    return get_origin(tp) is Union and type(None) in get_args(tp)


def _unwrap_optional(tp):
    args = [a for a in get_args(tp) if a is not type(None)]
    return args[0] if args else str


def _to_json_type(tp) -> str:
    if _is_optional(tp):
        tp = _unwrap_optional(tp)
    return _JSON_TYPE_MAP.get(tp, "string")


def build_tool_schema(fn: Callable) -> dict:
    hints = get_type_hints(fn)
    hints.pop("return", None)
    sig = inspect.signature(fn)

    properties = {}
    required = []
    for name, param in sig.parameters.items():
        tp = hints.get(name, str)
        properties[name] = {"type": _to_json_type(tp)}
        if not _is_optional(tp) and param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": inspect.getdoc(fn) or "",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def execute_tool_loop(
    client,
    model: str,
    messages: list,
    tools: dict[str, tuple[Callable, dict]],
    debug: bool = False,
    temperature: float | None = None,
    response_format: dict | None = None,
) -> str:
    """Run the agentic tool-calling loop and return the final text response."""
    tool_schemas = [schema for _, schema in tools.values()]

    while True:
        kwargs = {"model": model, "messages": messages}
        if tool_schemas:
            kwargs["tools"] = tool_schemas
        if temperature is not None:
            kwargs["temperature"] = temperature
        if response_format is not None:
            kwargs["response_format"] = response_format

        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message

        if choice.finish_reason != "tool_calls" or not msg.tool_calls:
            return msg.content

        messages.append(msg)

        for tc in msg.tool_calls:
            fn, _ = tools.get(tc.function.name, (None, None))
            if fn is None:
                result = f"Error: unknown tool '{tc.function.name}'"
                if debug:
                    print(f"[debug] LLM called unknown tool '{tc.function.name}'")
            else:
                try:
                    args = json.loads(tc.function.arguments)
                    result = str(fn(**args))
                    if debug:
                        print(f"[debug] tool '{tc.function.name}' args={args} result={result}")
                except Exception as e:
                    result = f"Error: {e}"

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
