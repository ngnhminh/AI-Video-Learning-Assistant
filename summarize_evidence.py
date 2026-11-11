"""Measured usage and auditable Standard token costs, never an invoice."""
import argparse
import copy
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

PRICE_SOURCE = "https://ai.google.dev/gemini-api/docs/pricing"
PRICE_CHECKED = "2026-10-24"
PRICES = {
    "gemini-2.5-flash-lite": {"input": .10, "audio": .30, "output": .40, "cache": .01, "cache_audio": .03},
    "gemini-2.5-flash": {"input": .30, "audio": 1.00, "output": 2.50, "cache": .03, "cache_audio": .10},
}
NOTE = ("Ước tính tiền token theo giá Standard trả phí, không phải hóa đơn hoặc số tiền đã trừ. "
        "Chưa gồm phí lưu cache, công cụ dịch vụ và thuế. Request thiếu usage/đơn giá được ghi chưa biết; "
        "không suy rộng từ một video.")

def counts(details):
    result = Counter()
    for item in details or []:
        modality=getattr(item["modality"],"value",item["modality"])
        result[modality] += item["token_count"]
    return result

def calculate_call(call):
    usage = call.get("usage") or {}
    model = call.get("model_version") or call["model_requested"]
    price = PRICES.get(model)
    unknown = {"estimated_standard_usd": None, "cost_terms": [], "cost_formula": "Chưa đủ dữ liệu"}
    if call["status"] != "ok" or not usage or not price:
        return {**unknown, "cost_note": "Request lỗi, thiếu usage hoặc model chưa có đơn giá đã kiểm chứng."}
    prompt = usage.get("prompt_token_count")
    output = usage.get("candidates_token_count")
    thinking = usage.get("thoughts_token_count") or 0
    basis = "usage_metadata"
    if output is None and prompt is not None and usage.get("total_token_count") == prompt and thinking == 0:
        output = 0
        basis = "Suy ra output=0: total_token_count bằng prompt_token_count, không có thinking."
    inputs, cached = counts(usage.get("prompt_tokens_details")), counts(usage.get("cache_tokens_details"))
    cached_total = usage.get("cached_content_token_count") or 0
    if not inputs or sum(inputs.values()) != prompt or output is None:
        return {**unknown, "cost_note": "Thiếu hoặc không khớp số token theo loại input/output."}
    if sum(cached.values()) != cached_total or any(v > inputs.get(k, 0) for k,v in cached.items()):
        return {**unknown, "cost_note": "Thiếu hoặc không khớp phân loại token cache."}
    if set(inputs) - {"TEXT", "IMAGE", "VIDEO", "AUDIO"}:
        return {**unknown, "cost_note": "Có loại input chưa có đơn giá."}
    terms = []
    for modality, n in inputs.items():
        audio = modality == "AUDIO"
        terms.append({"kind": modality, "tokens": n-cached.get(modality,0), "usd_per_million": price["audio" if audio else "input"]})
        if cached.get(modality):
            terms.append({"kind": "CACHE_"+modality, "tokens": cached[modality], "usd_per_million": price["cache_audio" if audio else "cache"]})
    terms.extend([{"kind":"OUTPUT", "tokens":output, "usd_per_million":price["output"]},
                  {"kind":"THINKING", "tokens":thinking, "usd_per_million":price["output"]}])
    cost = sum(Decimal(t["tokens"])*Decimal(str(t["usd_per_million"])) for t in terms)/Decimal(1_000_000)
    formula = "(" + " + ".join(f"{t['tokens']} × {t['usd_per_million']:g} [{t['kind']}]" for t in terms) + f") / 1.000.000 = {cost:.8f} USD"
    return {"estimated_standard_usd":float(cost), "cost_terms":terms, "cost_formula":formula,
            "output_count_basis":basis, "output_tokens_for_cost":output, "thinking_tokens_for_cost":thinking,
            "input_counts":dict(inputs), "cost_note":None}

def estimate(call):
    return calculate_call(call)["estimated_standard_usd"]

def summarize_calls(calls, run_id=None, elapsed_s=None):
    calls=copy.deepcopy(calls)
    for call in calls:call.update(calculate_call(call))
    known=[c for c in calls if c["estimated_standard_usd"] is not None]
    complete=bool(calls) and len(known)==len(calls)
    totals={"total_prompt_tokens":sum((c.get("usage") or {}).get("prompt_token_count",0) for c in known),
            "total_output_tokens":sum(c["output_tokens_for_cost"] for c in known),
            "total_thinking_tokens":sum(c["thinking_tokens_for_cost"] for c in known),
            "total_cached_tokens":sum((c.get("usage") or {}).get("cached_content_token_count",0) for c in known)}
    subtotal=sum(Decimal(str(c["estimated_standard_usd"])) for c in known)
    return {"run_id":run_id, "calls":calls, "requests":len(calls),
            "successful_requests":sum(c["status"]=="ok" for c in calls),
            "api_latency_sum_s":round(sum(c.get("latency_s",0) for c in calls),4),
            "elapsed_s":elapsed_s, "usage_complete":complete,
            **{k:v if complete else None for k,v in totals.items()},
            "observed_token_subtotals":totals, "priced_requests":len(known),
            "estimated_standard_usd":float(subtotal) if complete else None,
            "known_cost_subtotal_usd":float(subtotal) if known else None,
            "price_source":PRICE_SOURCE, "price_checked":PRICE_CHECKED,
            "prices_per_million_tokens":PRICES, "note":NOTE}

def summarize(directory):
    calls=json.loads((directory/"requests.json").read_text(encoding="utf-8"))
    run=json.loads((directory/"run.json").read_text(encoding="utf-8")) if (directory/"run.json").exists() else {}
    result=summarize_calls(calls,directory.name,run.get("elapsed_s"))
    (directory/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    return result

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("run")
    args=parser.parse_args()
    result=summarize(Path(args.run))
    print(json.dumps({k:v for k,v in result.items() if k not in ["calls","prices_per_million_tokens"]},ensure_ascii=False))
