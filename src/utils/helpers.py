import json
import os
import ast
import re
from typing import Any, Dict, List, Optional

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def write_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            # ensure 'id' is the first key when present
            if isinstance(it, dict) and "id" in it:
                ordered = {"id": it["id"]}
                for k, v in it.items():
                    if k == "id":
                        continue
                    ordered[k] = v
                f.write(json.dumps(ordered, ensure_ascii=False) + "\n")
            else:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")


def append_jsonl(path: str, item: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        if isinstance(item, dict) and "id" in item:
            ordered = {"id": item["id"]}
            for k, v in item.items():
                if k == "id":
                    continue
                ordered[k] = v
            payload = ordered
        else:
            payload = item

        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=4)

def robust_json_extract(text: str) -> Optional[str]:
    if not text:
        return None
    s = text.find("{")
    e = text.rfind("}")
    if s >= 0 and e >= s:
        return text[s:e + 1]
    return None


def parse_llm_json(text: str) -> Optional[Any]:
    """Extract and parse JSON-like object from LLM output.

    Attempts multiple strategies:
    1. Use robust_json_extract to get substring between first '{' and last '}'.
    2. Try json.loads on that substring.
    3. Fallback to ast.literal_eval (accepts Python-style dicts with single quotes).
    4. Try simple fixes (single->double quotes, remove trailing commas, None/True/False replacement) and json.loads again.

    Returns the parsed object on success, or None on failure.
    """
    # if not text:
    #     return None

    # jstr = robust_json_extract(text)
    # if not jstr:
    #     return None

    # # 1) try strict json
    try:
        return json.loads(text)
    except Exception:
        pass

    # # 2) try python literal eval (handles single quotes, python True/None)
    # try:
    #     return ast.literal_eval(jstr)
    # except Exception:
    #     pass

    # # 3) try simple normalization heuristics and json.loads
    # s = jstr.strip()
    # # replace single quotes with double quotes (simple heuristic)
    # s2 = s.replace("'", '"')
    # # remove trailing commas before } or ]
    # s2 = re.sub(r",\s*}\s*", "}", s2)
    # s2 = re.sub(r",\s*\]\s*", "]", s2)
    # # replace python literals
    # s2 = s2.replace("None", "null").replace("True", "true").replace("False", "false")
    # try:
    #     return json.loads(s2)
    # except Exception:
    #     return None
