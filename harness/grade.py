"""Tool-call repair rules for the lenient score. Deterministic; no model in the loop. Written before any model was run.

Adapted from tests/bonsai2/harness/grade.py: the reasoning and coding graders are gone (tools only). The strict score
uses none of this. The lenient score may (a) repair malformed argument JSON, (b) recover calls that the server left in
the visible reply as text. (b) recognises three fixed shapes, tried in this order for both arms alike:
  1. JSON objects in <tool_call>...</tool_call> tags or in a ```json fence   (unchanged from bonsai2)
  2. Qwen3-Coder-style XML: [<tool_call>] <function=NAME> <parameter=K>V</parameter> ... </function>
  3. LFM-style Pythonic list: [<|tool_call_start|>] [NAME(k=v, ...), ...] [<|tool_call_end|>]
Shapes 2 and 3 were added because they are the documented native formats of the two arms and the open llama.cpp
issues #30118 and #30490 show each arm's calls leaking into the reply as text in exactly these shapes.
"""
import ast
import json
import re

# ---------- argument repair ----------


def parse_arguments(raw):
    """Returns (args_dict_or_None, repaired: bool). Strict path is plain json.loads to a dict."""
    if isinstance(raw, dict):
        return raw, False
    try:
        val = json.loads(raw)
        if isinstance(val, dict):
            return val, False
        if isinstance(val, str):  # double-encoded
            val2 = json.loads(val)
            if isinstance(val2, dict):
                return val2, True
    except (TypeError, ValueError):
        pass
    if not isinstance(raw, str):
        return None, False
    s = raw.strip()
    s = re.sub(r"^```(?:json)?|```$", "", s).strip()
    start = s.find("{")
    if start == -1:
        return None, False
    s = s[start:]
    candidates = [s, re.sub(r",\s*([}\]])", r"\1", s)]
    candidates.append(candidates[-1].replace("'", '"'))
    for c in list(candidates):
        opened = c.count("{") - c.count("}")
        sq = c.count("[") - c.count("]")
        if opened > 0 or sq > 0:
            candidates.append(c + "]" * max(sq, 0) + "}" * max(opened, 0))
        end = c.rfind("}")
        if end != -1:
            candidates.append(c[: end + 1])
    for c in candidates:
        try:
            val = json.loads(c)
            if isinstance(val, dict):
                return val, True
        except ValueError:
            continue
    return None, False


# ---------- calls printed as text (lenient only) ----------

JSON_CALL_RES = [
    re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL),
    re.compile(r"```(?:json|tool_call)?\s*(\{\s*\"name\".*?\})\s*```", re.DOTALL),
]
XML_FUNCTION_RE = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.DOTALL)
XML_PARAM_RE = re.compile(r"<parameter=([^>\s]+)>(.*?)</parameter>", re.DOTALL)
PY_TOKENS_RE = re.compile(r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>", re.DOTALL)
_PY_NAMES = {"true": True, "false": False, "null": None, "True": True, "False": False, "None": None}


def _schemas(tools):
    return {t["function"]["name"]: t["function"].get("parameters", {}) for t in tools}


def _from_json(content, schemas):
    out = []
    for rx in JSON_CALL_RES:
        for blob in rx.findall(content):
            obj, _ = parse_arguments(blob)
            if not obj or obj.get("name") not in schemas:
                continue
            args = obj.get("arguments", obj.get("parameters", {}))
            if isinstance(args, str):
                args, _ = parse_arguments(args)
            if isinstance(args, dict):
                out.append({"name": obj["name"], "arguments": args})
        if out:
            break
    return out


def _xml_value(raw, prop_schema):
    """One leading and one trailing newline belong to the format. String-typed parameters stay strings; anything
    else is read as JSON if it parses (the same schema-guided conversion llama.cpp's own parser does)."""
    v = raw[1:] if raw.startswith("\n") else raw
    v = v[:-1] if v.endswith("\n") else v
    t = prop_schema.get("type")
    types = t if isinstance(t, list) else [t]
    if "string" in types and len(types) == 1:
        return v
    try:
        return json.loads(v)
    except ValueError:
        return v


def _from_xml(content, schemas):
    out = []
    for name, body in XML_FUNCTION_RE.findall(content):
        if name not in schemas:
            continue
        props = schemas[name].get("properties", {})
        args = {k: _xml_value(v, props.get(k, {})) for k, v in XML_PARAM_RE.findall(body)}
        out.append({"name": name, "arguments": args})
    return out


def _py_literal(node):
    """Literal values only (no names other than true/false/null, no calls): nothing is ever evaluated."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in _PY_NAMES:
        return _PY_NAMES[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_py_literal(e) for e in node.elts]
    if isinstance(node, ast.Dict):
        return {_py_literal(k): _py_literal(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)) and isinstance(node.operand, ast.Constant):
        return -node.operand.value if isinstance(node.op, ast.USub) else node.operand.value
    raise ValueError("not a literal")


def _parse_py_list(text, schemas):
    try:
        tree = ast.parse(text.strip(), mode="eval").body
    except (SyntaxError, ValueError):
        return []
    calls = tree.elts if isinstance(tree, (ast.List, ast.Tuple)) else [tree]
    out = []
    for c in calls:
        if not (isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id in schemas) or c.args:
            return []
        try:
            args = {kw.arg: _py_literal(kw.value) for kw in c.keywords if kw.arg is not None}
        except (ValueError, TypeError):
            return []
        out.append({"name": c.func.id, "arguments": args})
    return out


def _from_pythonic(content, schemas):
    out = []
    for blob in PY_TOKENS_RE.findall(content):
        out += _parse_py_list(blob, schemas)
    if out:
        return out
    # Without the special tokens: only a reply that is nothing but a call list, so prose and markdown links never count.
    stripped = content.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        return _parse_py_list(stripped, schemas)
    return []


TEXT_FORMATS = (("json", _from_json), ("xml", _from_xml), ("pythonic", _from_pythonic))


def calls_from_text(content, tools):
    """Lenient only: recover tool calls the server failed to parse out of the visible reply (never the reasoning).
    Returns (calls, format_name); calls is a list of {"name", "arguments"}; ([], None) if nothing was found."""
    if not content:
        return [], None
    schemas = _schemas(tools)
    for name, fn in TEXT_FORMATS:
        found = fn(content, schemas)
        if found:
            return found, name
    return [], None
