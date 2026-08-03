import inspect
import json
import re
from typing import Callable, Union, get_args, get_origin, get_type_hints

_MAX_TEXT_TOOL_CALLS = 5

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


def _extract_text_tool_call(text: str) -> tuple[str, dict] | None:
    """Find an inline tool-call JSON object in ``text``.

    Recognizes both the ``{"name": <tool>, "arguments": {<args>}}`` shape and
    the single-key ``{<tool>: {<args>}}`` shape emitted by some models.

    Returns ``(tool_name, arguments_dict)`` or ``None``.
    """
    if not text or "{" not in text:
        return None
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        args = obj.get("arguments")
        if isinstance(name, str) and isinstance(args, dict):
            return name, args
        if len(obj) == 1:
            key, value = next(iter(obj.items()))
            if isinstance(key, str) and isinstance(value, dict):
                return key, value
    return None


def _strip_tool_call(text: str, tool_name: str) -> str:
    """Remove a leading/standalone tool-call JSON object from ``text``."""
    pattern = re.compile(
        r"^\s*```(?:json)?\s*\{[^`]*?\"name\"\s*:\s*\""
        + re.escape(tool_name)
        + r"\"[^`]*?}\s*```\s*",
        re.DOTALL,
    )
    stripped = pattern.sub("", text, count=1)
    if stripped != text:
        return stripped
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, end = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if obj.get("name") == tool_name:
            return text[:i] + text[i + end:]
        if len(obj) == 1 and isinstance(obj.get(tool_name), dict):
            return text[:i] + text[i + end:]
    return text


def execute_tool_loop(
    client,
    model: str,
    messages: list,
    tools: dict[str, tuple[Callable, dict]],
    debug: bool = False,
    temperature: float | None = None,
    response_format: dict | None = None,
) -> str:
    """Run the agentic tool-calling loop and return the final text response.

    Handles both structural tool calls (via ``message.tool_calls``) and tool
    calls emitted as inline JSON text in ``message.content`` (a behavior seen
    with some OpenAI-compatible endpoints, e.g. GLM models).
    """
    tool_schemas = [schema for _, schema in tools.values()]
    text_tool_calls = 0

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

        # Structural tool calls: execute and feed back (existing behavior).
        if choice.finish_reason == "tool_calls" and msg.tool_calls:
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
            continue

        # No structural tool calls: check for a tool call emitted as inline
        # JSON text before returning the content as the final answer.
        content = msg.content
        found = _extract_text_tool_call(content) if content else None
        if (
            found is not None
            and found[0] in tools
            and text_tool_calls < _MAX_TEXT_TOOL_CALLS
        ):
            tool_name, args = found
            fn, _ = tools[tool_name]
            try:
                result = str(fn(**args))
                if debug:
                    print(f"[debug] text tool call '{tool_name}' args={args} result={result}")
            except Exception as e:
                result = f"Error: {e}"
                if debug:
                    print(f"[debug] text tool call '{tool_name}' raised: {e}")
            assistant_text = _strip_tool_call(content, tool_name).strip()
            if assistant_text:
                messages.append({"role": "assistant", "content": assistant_text})
            messages.append({
                "role": "tool",
                "tool_call_id": f"text_{tool_name}",
                "content": result,
            })
            text_tool_calls += 1
            continue

        return content
