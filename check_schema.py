"""Offline schema checks. Optional --payload uses a real run as the valid baseline."""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

from agent import GeminiVideoAgent
from schemas import LearningOutput

parser = argparse.ArgumentParser()
parser.add_argument("--payload")
parser.add_argument("--out", required=True)
args = parser.parse_args()
valid = (json.loads(Path(args.payload).read_text(encoding="utf-8"))["result"]
         if args.payload else {
             "chu_de": "Dữ liệu giả để kiểm tra schema",
             "tom_tat": ["Ý 1", "Ý 2", "Ý 3"], "y_chinh": ["Ý 1", "Ý 2", "Ý 3"],
             "cau_hoi_trac_nghiem": [{"cau_hoi": "Câu hỏi mẫu", "lua_chon":
                 {"A": "A", "B": "B", "C": "C", "D": "D"}, "dap_an_dung": "A"}]})
results = []

def check(name, response, expected):
    error = None
    try:
        GeminiVideoAgent._parse_learning_output_response(response)
        accepted = True
    except (ValueError, TypeError) as exc:
        accepted = False
        error = str(exc)
    results.append({"case": name, "expected_accept": expected, "accepted": accepted,
                    "passed": accepted == expected, "error": error})

check("valid_res_parsed", SimpleNamespace(parsed=LearningOutput.model_validate(valid)), True)
check("valid_raw_json", SimpleNamespace(parsed=None, text=json.dumps(valid)), True)
for name, mutate in [
    ("missing_chu_de", lambda d: d.pop("chu_de")),
    ("only_2_summaries", lambda d: d.update(tom_tat=d["tom_tat"][:2])),
    ("invalid_answer_E", lambda d: d["cau_hoi_trac_nghiem"][0].update(dap_an_dung="E")),
    ("empty_quiz", lambda d: d.update(cau_hoi_trac_nghiem=[])),
]:
    altered = copy.deepcopy(valid)
    mutate(altered)
    check(name, SimpleNamespace(parsed=None, text=json.dumps(altered)), False)
check("malformed_json", SimpleNamespace(parsed=None, text='{broken'), False)
Path(args.out).write_text(json.dumps({
    "scope": "Offline local validation; invalid inputs are deliberately mutated, not Gemini failures.",
    "baseline": args.payload or "synthetic fixture", "cases": results,
    "passed": sum(r["passed"] for r in results), "total": len(results),
}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Schema checks: {sum(r['passed'] for r in results)}/{len(results)} passed")
assert all(r["passed"] for r in results)
