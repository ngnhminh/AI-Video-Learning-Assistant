"""Local, credential-free evidence for actual Gemini calls; no simulated results."""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def serializable(value):
    if isinstance(value, bytes):
        return {"binary_bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, type) and hasattr(value, "model_json_schema"):
        return value.model_json_schema()
    if hasattr(value, "model_dump"):
        return serializable(value.model_dump(mode="python", exclude_none=True))
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()
                if str(k).lower() not in {"api_key", "authorization", "x-goog-api-key"}}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if callable(value):
        return {"function_name": value.__name__, "description": value.__doc__}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class EvidenceRecorder:
    def __init__(self):
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        root = Path(os.getenv("EVIDENCE_DIR", Path(__file__).parent / "evidence"))
        self.directory = root / self.run_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.calls = []
        self.started = time.perf_counter()

    def write(self, name, value):
        (self.directory / name).write_text(
            json.dumps(serializable(value), ensure_ascii=False, indent=2), encoding="utf-8")

    def generate(self, client, *, model, contents, config):
        call_id = len(self.calls) + 1
        record = {"call_id": call_id, "model_requested": model,
                  "started_at": datetime.now(timezone.utc).isoformat(), "status": "started"}
        config_data=serializable(config) or {}
        record["generation_config"]={k:config_data[k] for k in ["temperature","max_output_tokens","response_mime_type"] if k in config_data}
        self.calls.append(record)
        self.write(f"request_{call_id:02}.json", {"model": model, "contents": contents, "config": config})
        self.write("requests.json", self.calls)
        started = time.perf_counter()
        try:
            response = client.models.generate_content(model=model, contents=contents, config=config)
            record.update(status="ok", model_version=getattr(response, "model_version", None),
                          response_id=getattr(response, "response_id", None),
                          usage=serializable(getattr(response, "usage_metadata", None)))
            self.write(f"response_{call_id:02}.json", response)
            return response
        except Exception as exc:
            # Persist exception type only; service error text may contain request credentials.
            record.update(status="error", error_type=type(exc).__name__,
                          status_code=getattr(exc, "code", None))
            raise
        finally:
            record["latency_s"] = round(time.perf_counter() - started, 4)
            self.write("requests.json", self.calls)

    def finish(self, payload, transcript, video_name, video_bytes):
        from summarize_evidence import summarize_calls
        from run_report import video_illustrations, build_html
        run = {
            "run_id": self.run_id, "video_name": video_name, "video": video_bytes,
            "elapsed_s": round(time.perf_counter() - self.started, 4),
            "sdk_calls": len(self.calls), "schema_valid": True,
            "trace_scope": "Every generate_content attempt; SDK internal retries disabled. Binary input represented by length and SHA256. Credentials omitted. Tool arguments and results are in workflow_logs in payload.json.",
        }
        payload["run_info"]=serializable(run)
        payload["transcript"]=transcript
        payload["measurement_summary"]=summarize_calls(self.calls,self.run_id,run["elapsed_s"])
        payload["video_preview"]=video_illustrations(self.directory,video_bytes)
        self.write("run.json",run)
        self.write("payload.json", payload)
        self.write("transcript.json", {"transcript": transcript})
        self.write("summary.json",payload["measurement_summary"])
        build_html(payload,self.directory)

    def abort(self, state):
        self.write("workflow.json", state.workflow_logs)
        self.write("run.json", {
            "run_id": self.run_id, "status": "failed", "video_name": state.video_name,
            "video": state.video_bytes, "elapsed_s": round(time.perf_counter() - self.started, 4),
            "sdk_calls": len(self.calls), "schema_valid": None,
            "last_status_code": self.calls[-1].get("status_code") if self.calls else None,
        })
