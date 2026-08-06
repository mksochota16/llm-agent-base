import inspect
import json
import re
from enum import Enum
from typing import Callable, Literal, Union, get_args, get_origin, get_type_hints

_MAX_TEXT_TOOL_CALLS = 5
_MAX_ITERATIONS = 10

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


def _json_schema_for(tp) -> dict:
    """Build a JSON Schema fragment for a single type annotation.

    Resolves container item types and enum values, which strict providers
    require and which measurably improve tool-calling accuracy.
    """
    if _is_optional(tp):
        tp = _unwrap_optional(tp)

    origin = get_origin(tp)

    if origin is Literal:
        values = list(get_args(tp))
        item_type = _to_json_type(type(values[0])) if values else "string"
        return {"type": item_type, "enum": values}

    if isinstance(tp, type) and issubclass(tp, Enum):
        values = [m.value for m in tp]
        item_type = _to_json_type(type(values[0])) if values else "string"
        return {"type": item_type, "enum": values}

    if origin in (list, set, frozenset, tuple) or tp in (list, set, frozenset, tuple):
        schema: dict = {"type": "array"}
        args = [a for a in get_args(tp) if a is not Ellipsis]
        if args:
            schema["items"] = _json_schema_for(args[0])
        return schema

    if origin is dict or tp is dict:
        return {"type": "object"}

    return {"type": _JSON_TYPE_MAP.get(tp, "string")}


_ARGS_HEADER_RE = re.compile(r"^(args|arguments|parameters)\s*:\s*$", re.IGNORECASE)
_SECTION_HEADER_RE = re.compile(
    r"^(returns?|raises?|yields?|examples?|notes?|attributes?)\s*:\s*$", re.IGNORECASE
)
_GOOGLE_PARAM_RE = re.compile(r"^(\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$")
_SPHINX_PARAM_RE = re.compile(r"^:param\s+(?:[^:]+\s+)?(\w+)\s*:\s*(.*)$")


def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Split a docstring into its summary and per-parameter descriptions.

    Understands Google style (an ``Args:`` section) and Sphinx style
    (``:param name:`` lines). Returns ``(summary, {param_name: description})``.
    """
    if not doc:
        return "", {}

    summary_lines: list[str] = []
    params: dict[str, str] = {}
    in_args = False
    # Once any section starts, the summary is over — trailing sections such as
    # Returns:/Raises: must not leak into the tool description.
    seen_section = False
    current: str | None = None

    for raw in doc.splitlines():
        line = raw.strip()

        if _ARGS_HEADER_RE.match(line):
            in_args, current, seen_section = True, None, True
            continue
        if _SECTION_HEADER_RE.match(line):
            in_args, current, seen_section = False, None, True
            continue

        sphinx = _SPHINX_PARAM_RE.match(line)
        if sphinx:
            current = sphinx.group(1)
            params[current] = sphinx.group(2).strip()
            seen_section = True
            continue

        if line.startswith(":"):
            # Some other Sphinx field (:returns:, :rtype:) — ends the parameter.
            current, seen_section = None, True
            continue

        if in_args:
            if not line:
                current = None
            elif (google := _GOOGLE_PARAM_RE.match(line)):
                current = google.group(1)
                params[current] = google.group(2).strip()
            elif current:
                # Continuation of the previous parameter's description.
                params[current] = f"{params[current]} {line}".strip()
            continue

        if current is not None:
            # Continuation of a Sphinx :param: description.
            if line:
                params[current] = f"{params[current]} {line}".strip()
            else:
                current = None
            continue

        if not seen_section:
            summary_lines.append(line)

    summary = "\n".join(summary_lines).strip()
    return summary, {k: v for k, v in params.items() if v}


def build_tool_schema(fn: Callable) -> dict:
    hints = get_type_hints(fn)
    hints.pop("return", None)
    sig = inspect.signature(fn)
    summary, param_docs = _parse_docstring(inspect.getdoc(fn) or "")

    properties = {}
    required = []
    for name, param in sig.parameters.items():
        tp = hints.get(name, str)
        schema = _json_schema_for(tp)
        if name in param_docs:
            schema["description"] = param_docs[name]
        properties[name] = schema
        if not _is_optional(tp) and param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": summary,
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
    max_iterations: int = _MAX_ITERATIONS,
) -> str:
    """Run the agentic tool-calling loop and return the final text response.

    Handles both structural tool calls (via ``message.tool_calls``) and tool
    calls emitted as inline JSON text in ``message.content`` (a behavior seen
    with some OpenAI-compatible endpoints, e.g. GLM models).

    ``max_iterations`` bounds how many rounds of tool calling are allowed. Once
    the budget is spent the tools are withheld from the request, forcing the
    model to answer in text, so the loop always terminates and always returns a
    string.
    """
    tool_schemas = [schema for _, schema in tools.values()]
    text_tool_calls = 0
    iterations = 0

    while True:
        offer_tools = bool(tool_schemas) and iterations < max_iterations
        if tool_schemas and not offer_tools and debug:
            print(f"[debug] max_iterations ({max_iterations}) reached; requesting final answer without tools")

        kwargs = {"model": model, "messages": messages}
        if offer_tools:
            kwargs["tools"] = tool_schemas
        if temperature is not None:
            kwargs["temperature"] = temperature
        if response_format is not None:
            kwargs["response_format"] = response_format

        response = client.chat.completions.create(**kwargs)
        iterations += 1
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
            and offer_tools
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

        return content or ""
