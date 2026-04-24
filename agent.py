from __future__ import annotations

from dotenv import load_dotenv
import os
import json
import time
import random
import tempfile
import requests

from dataclasses import dataclass, field
from typing import Optional, Literal, Any, Callable

from google import genai
from google.genai import types

from pymongo import MongoClient
from datetime import datetime, UTC
from pydantic import BaseModel, Field

from schemas import LearningOutput

load_dotenv()


# =========================
# ACTION SCHEMA
# =========================
class AgentAction(BaseModel):
    thought: str = Field(description="Lý do ngắn gọn cho bước tiếp theo")
    action: Literal[
        "extract_transcript",
        "detect_topic",
        "search_topic_in_transcript",
        "lookup_wikipedia",
        "save_to_database",
        "finalize",
    ]
    action_input: Optional[str] = Field(default=None)
    reason: Optional[str] = Field(default=None)
    needs_external_context: Optional[bool] = Field(
        default=None,
        description="User có cần thông tin nền / kiến thức mở rộng ngoài transcript hay không",
    )


# =========================
# STATE
# =========================
@dataclass
class AgentState:
    transcript: Optional[str] = None
    detected_topic: Optional[str] = None
    topic_result: Optional[dict] = None
    external_context: list[dict] = field(default_factory=list)

    video_bytes: Optional[bytes] = None
    video_mime: Optional[str] = None
    video_name: Optional[str] = None
    uploaded_file_name: str = ""

    user_language: str = "vi"

    observations: list[str] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    workflow_logs: list[dict] = field(default_factory=list)


# =========================
# AGENT
# =========================
class GeminiVideoAgent:
    def __init__(self, api_key: str, model: str):
        self.client = genai.Client(api_key=api_key)
        self.model = model

        self.primary_model = model
        self.fallback_model = "gemini-2.5-flash-lite"

        self.task_models = {
            "default": self.primary_model,
            "planner": self.fallback_model,
            "detect": self.fallback_model,
            "translate": self.fallback_model,
            "language": self.fallback_model,
            "semantic_search": self.primary_model,
            "final": self.primary_model,
        }

        self.state = AgentState()

        self.inline_limit = 20 * 1024 * 1024
        self.file_limit = 2 * 1024 * 1024 * 1024
        self.max_steps = 6

        self.tools: dict[str, Callable[..., Any]] = {
            "extract_transcript": self.tool_extract_transcript,
            "detect_topic": self.tool_detect_topic,
            "search_topic_in_transcript": self.tool_search_topic,
            "lookup_wikipedia": self.tool_lookup_wikipedia,
            "save_to_database": self.tool_save,
        }
        
    def _log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        self.state.logs.append(log_line)
        print(log_line, flush=True)

    def _json_safe(self, value: Any):
        try:
            json.dumps(value, ensure_ascii=False)
            return value
        except Exception:
            if isinstance(value, dict):
                return {str(k): self._json_safe(v) for k, v in value.items()}
            if isinstance(value, (list, tuple, set)):
                return [self._json_safe(v) for v in value]
            return str(value)

    def _workflow_event(self, event: str, **payload):
        entry = {
            "ts": datetime.now(UTC).isoformat(),
            "event": event,
            **{k: self._json_safe(v) for k, v in payload.items()},
        }
        self.state.workflow_logs.append(entry)
        try:
            preview = json.dumps(entry, ensure_ascii=False)
        except Exception:
            preview = str(entry)
        self._log(f"workflow_event={preview[:800]}")
        return entry

    def _state_snapshot(self) -> dict:
        return {
            "has_transcript": bool(self.state.transcript),
            "transcript_len": len(self.state.transcript or ""),
            "detected_topic": self.state.detected_topic,
            "has_topic_result": self.state.topic_result is not None,
            "has_external_context": bool(self.state.external_context),
            "observations_count": len(self.state.observations),
        }

    def _extract_function_calls_from_response(self, response) -> list[Any]:
        calls = []
        if response is None:
            return calls

        direct_calls = getattr(response, "function_calls", None)
        if direct_calls:
            return list(direct_calls)

        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", None) or []:
                function_call = getattr(part, "function_call", None)
                if function_call:
                    calls.append(function_call)
        return calls

    def _execute_function_call(self, function_call: Any, user_request: str) -> dict:
        fn_name = getattr(function_call, "name", None)
        fn_args = dict(getattr(function_call, "args", None) or {})
        call_id = getattr(function_call, "id", None)

        self._workflow_event(
            "tool_call_requested",
            tool_name=fn_name,
            call_id=call_id,
            args=fn_args,
            state=self._state_snapshot(),
        )

        if fn_name == "fc_extract_transcript":
            result = self.fc_extract_transcript()
        elif fn_name == "fc_detect_topic":
            if not fn_args.get("user_request"):
                fn_args["user_request"] = user_request
            result = self.fc_detect_topic(**fn_args)
        elif fn_name == "fc_search_topic_in_transcript":
            if not fn_args.get("topic"):
                fn_args["topic"] = self.state.detected_topic or ""
            result = self.fc_search_topic_in_transcript(**fn_args)
        elif fn_name == "fc_lookup_wikipedia":
            if not fn_args.get("query"):
                fn_args["query"] = self.state.detected_topic or ""
            result = self.fc_lookup_wikipedia(**fn_args)
        elif fn_name == "fc_finalize_research":
            result = self.fc_finalize_research()
        else:
            raise RuntimeError(f"Unknown function call: {fn_name}")

        self._workflow_event(
            "tool_call_result",
            tool_name=fn_name,
            call_id=call_id,
            result=result,
            state=self._state_snapshot(),
        )
        return {
            "id": call_id,
            "name": fn_name,
            "response": result,
        }

    # =========================
    # RETRY
    # =========================
    def _is_retryable_error(self, exc: Exception) -> bool:
        msg = str(exc).upper()
        return any(
            s in msg
            for s in [
                "502",
                "503",
                "BAD_GATEWAY",
                "UNAVAILABLE",
                "RATE",
                "TIMEOUT",
                "RESOURCE_EXHAUSTED",
                "INTERNAL",
            ]
        )

    def _generate_with_retry(self, *, contents, config=None, model_name: Optional[str] = None):
        last_exc = None
        max_retries_per_model = 2

        preferred = model_name or self.primary_model
        fallback_chain = list(dict.fromkeys([preferred, self.fallback_model]))

        for chosen_model in fallback_chain:
            for attempt in range(max_retries_per_model):
                try:
                    self._log(
                        f"model_call model={chosen_model} attempt={attempt+1}/{max_retries_per_model}"
                    )
                    return self.client.models.generate_content(
                        model=chosen_model,
                        contents=contents,
                        config=config,
                    )
                except Exception as e:
                    last_exc = e
                    err_text = str(e).upper()
                    self._log(f"model_error model={chosen_model} err={e}")

                    if "404" in err_text or "NOT_FOUND" in err_text:
                        self._log(f"skip_unavailable_model={chosen_model}")
                        break

                    if not self._is_retryable_error(e):
                        raise RuntimeError(f"Model failed: {e}") from e

                    if attempt < max_retries_per_model - 1:
                        delay = min(10, 1.5 * (2 ** attempt)) + random.uniform(0, 0.5)
                        self._log(f"retry_sleep={delay:.2f}s")
                        time.sleep(delay)

            self._log(f"switch_model_from={chosen_model}")

        raise RuntimeError(
            f"Model failed after retries across fallbacks: {last_exc}"
        ) from last_exc

    # =========================
    # HELPERS
    # =========================
    def _chunk_transcript(self, text: str, chunk_size: int = 4000, overlap: int = 500) -> list[str]:
        """
        Chia transcript dài thành nhiều đoạn có overlap để tránh mất ngữ cảnh.
        chunk_size và overlap đang tính theo số ký tự để đơn giản và ổn định.
        """
        text = (text or "").strip()
        if not text:
            return []

        chunks: list[str] = []
        start = 0
        n = len(text)

        while start < n:
            end = min(start + chunk_size, n)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= n:
                break

            start = max(0, end - overlap)

        self._log(f"chunk_transcript total_chars={n} chunks={len(chunks)}")
        return chunks

    def _summarize_chunk(self, chunk_text: str, chunk_index: int) -> dict:
        """
        Tóm tắt một chunk transcript.
        Trả về JSON gồm tom_tat và y_chinh.
        """
        prompt = f"""
[ROLE]
Bạn là AI hỗ trợ tóm tắt một đoạn transcript của video bài giảng.

[GOAL]
Tạo bản tóm tắt ngắn gọn và trích xuất ý chính từ đoạn transcript này.

[INSTRUCTIONS]
- Chỉ dựa trên nội dung transcript được cung cấp.
- Không bịa thêm thông tin ngoài transcript.
- Tóm tắt ngắn gọn, rõ ràng, dễ hiểu.
- Rút ra các ý chính quan trọng nhất của đoạn này.
- Nếu transcript chỉ là một phần của video dài, chỉ tóm tắt đúng phần này.

[CONSTRAINTS]
- Không trả lời ngoài phạm vi transcript.
- Không thêm markdown.
- Chỉ trả JSON hợp lệ.

[OUTPUT FORMAT]
{{
  "tom_tat": ["ý tóm tắt 1", "ý tóm tắt 2", "ý tóm tắt 3"],
  "y_chinh": ["ý chính 1", "ý chính 2", "ý chính 3"]
}}

[CHUNK INDEX]
{chunk_index}

[TRANSCRIPT CHUNK]
{chunk_text}
"""
        res = self._generate_with_retry(
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            ),
            model_name=self.task_models["final"],
        )

        data = self._safe_json_load(
            getattr(res, "text", "") or "",
            {"tom_tat": [], "y_chinh": []},
        )

        if not isinstance(data, dict):
            data = {"tom_tat": [], "y_chinh": []}

        if not isinstance(data.get("tom_tat"), list):
            data["tom_tat"] = []
        if not isinstance(data.get("y_chinh"), list):
            data["y_chinh"] = []

        self._log(
            f"summarize_chunk index={chunk_index} summary_items={len(data['tom_tat'])} keypoints={len(data['y_chinh'])}"
        )
        return data

    def _summarize_long_transcript(self) -> dict:
        """
        Xử lý transcript dài theo kiểu:
        1) chia đoạn
        2) tóm tắt từng đoạn
        3) gộp và tổng hợp thành summary cuối
        """
        transcript = self.state.transcript or ""
        if not transcript.strip():
            return {"tom_tat": [], "y_chinh": []}

        chunks = self._chunk_transcript(transcript)
        if not chunks:
            return {"tom_tat": [], "y_chinh": []}

        # Nếu transcript ngắn thì không cần map-reduce quá nhiều
        if len(chunks) == 1:
            return self._summarize_chunk(chunks[0], 1)

        chunk_summaries: list[dict] = []
        for idx, chunk in enumerate(chunks, start=1):
            chunk_summaries.append(self._summarize_chunk(chunk, idx))

        combined_summaries: list[str] = []
        combined_keypoints: list[str] = []

        for item in chunk_summaries:
            combined_summaries.extend(
                [x for x in item.get("tom_tat", []) if isinstance(x, str) and x.strip()]
            )
            combined_keypoints.extend(
                [x for x in item.get("y_chinh", []) if isinstance(x, str) and x.strip()]
            )

        # Khử trùng lặp giữ nguyên thứ tự
        combined_summaries = list(dict.fromkeys(combined_summaries))
        combined_keypoints = list(dict.fromkeys(combined_keypoints))

        reduce_prompt = f"""
[ROLE]
Bạn là AI tổng hợp nội dung từ nhiều bản tóm tắt trung gian của transcript video bài giảng.

[GOAL]
Tạo ra bản tóm tắt cuối cùng phản ánh đầy đủ toàn bộ video, không chỉ phần đầu transcript.

[INSTRUCTIONS]
- Dựa trên các tóm tắt trung gian và các ý chính trung gian.
- Phải bao phủ các giai đoạn/nội dung quan trọng trên toàn bộ video.
- Không chỉ tập trung vào phần đầu.
- Loại bỏ các ý trùng lặp.
- Giữ lại các ý có giá trị thông tin cao nhất.
- Không bịa thêm thông tin ngoài dữ liệu đã cho.

[CONSTRAINTS]
- Chỉ trả JSON hợp lệ.
- Không thêm markdown hay giải thích.

[OUTPUT FORMAT]
{{
  "tom_tat": ["ý tóm tắt 1", "ý tóm tắt 2", "ý tóm tắt 3", "ý tóm tắt 4", "ý tóm tắt 5"],
  "y_chinh": ["ý chính 1", "ý chính 2", "ý chính 3", "ý chính 4", "ý chính 5"]
}}

[TÓM TẮT TRUNG GIAN]
{json.dumps(combined_summaries, ensure_ascii=False)}

[Ý CHÍNH TRUNG GIAN]
{json.dumps(combined_keypoints, ensure_ascii=False)}
"""
        res = self._generate_with_retry(
            contents=reduce_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            ),
            model_name=self.task_models["final"],
        )

        final_data = self._safe_json_load(
            getattr(res, "text", "") or "",
            {"tom_tat": [], "y_chinh": []},
        )

        if not isinstance(final_data, dict):
            final_data = {"tom_tat": [], "y_chinh": []}

        if not isinstance(final_data.get("tom_tat"), list):
            final_data["tom_tat"] = combined_summaries[:5]
        if not isinstance(final_data.get("y_chinh"), list):
            final_data["y_chinh"] = combined_keypoints[:5]

        self._log(
            f"summarize_long_transcript final_summary_items={len(final_data['tom_tat'])} final_keypoints={len(final_data['y_chinh'])}"
        )
        return final_data

    def _build_transcript_context_for_final(self, transcript: str, chunk_size: int = 12000) -> str:
        """
        Tạo context transcript cho prompt cuối mà hạn chế thất thoát thông tin.

        - Transcript ngắn/vừa: đưa toàn bộ vào prompt cuối.
        - Transcript rất dài: giữ nhiều cửa sổ (đầu/giữa/cuối) thay vì chỉ lấy 20k ký tự đầu.
        """
        transcript = (transcript or "").strip()
        if not transcript:
            return ""

        if len(transcript) <= 120000:
            self._log(f"final_context_mode=full chars={len(transcript)}")
            return transcript

        n = len(transcript)
        windows: list[str] = []
        anchors = [0, max(0, n // 3 - chunk_size // 2), max(0, (2 * n) // 3 - chunk_size // 2), max(0, n - chunk_size)]

        seen = set()
        for start in anchors:
            end = min(start + chunk_size, n)
            key = (start, end)
            if key in seen:
                continue
            seen.add(key)
            section = transcript[start:end].strip()
            if section:
                windows.append(f"[TRANSCRIPT WINDOW {start}:{end}]\n{section}")

        self._log(
            f"final_context_mode=windowed chars={len(transcript)} windows={len(windows)} window_size={chunk_size}"
        )
        return "\n\n".join(windows)
    
    def _normalize_user_request(self, user_request: str) -> str:
        return " ".join(user_request.strip().split())

    def _extract_db_request(self, user_request: str) -> bool:
        lower = user_request.lower()
        return any(k in lower for k in ["lưu", "save", "store", "database", "db"])

    def _infer_external_need_fallback(self, user_request: str, topic: Optional[str]) -> bool:
        if not topic:
            return False

        lower = user_request.lower()
        soft_signals = [
            "tìm thêm",
            "biết thêm",
            "giải thích thêm",
            "nói rõ hơn",
            "mở rộng",
            "bổ sung",
            "tra cứu",
            "wikipedia",
            "kiến thức nền",
            "thông tin nền",
            "bối cảnh",
            "context",
            "background",
        ]
        return any(s in lower for s in soft_signals)

    def _safe_json_load(self, text: str, default):
        try:
            return json.loads(text)
        except Exception:
            return default

    def _observe(self, message: str):
        clipped = message[:500]
        self.state.observations.append(clipped)
        if len(self.state.observations) > 10:
            self.state.observations = self.state.observations[-10:]
        self._workflow_event("observation", message=clipped, state=self._state_snapshot())

    def _clean_wikipedia_snippet(self, text: str) -> str:
        if not text:
            return ""
        return (
            text.replace('<span class="searchmatch">', "")
            .replace("</span>", "")
            .replace("&quot;", '"')
            .replace("&#39;", "'")
            .replace("&amp;", "&")
            .strip()
        )

    def _detect_user_language(self, user_request: str) -> str:
        prompt = f"""
Xác định ngôn ngữ chính của yêu cầu người dùng dưới đây.
Chỉ trả về đúng một mã ISO 639-1, ví dụ: vi, en, fr, ja.

Yêu cầu:
{user_request}
"""
        try:
            res = self._generate_with_retry(
                contents=prompt,
                model_name=self.task_models["language"],
            )
            lang = (res.text or "").strip().lower()
            if len(lang) <= 5:
                return lang
            return "vi"
        except Exception:
            return "vi"

    def _translate_to_user_language(self, text: str) -> str:
        if not text:
            return text

        target_lang = getattr(self.state, "user_language", "vi")

        prompt = f"""
Dịch đoạn sau sang ngôn ngữ có mã ISO là: {target_lang}.
Chỉ trả về nội dung đã dịch, không giải thích, không thêm tiêu đề.

{text}
"""

        try:
            res = self._generate_with_retry(
                contents=prompt,
                model_name=self.task_models["translate"],
            )
            return (res.text or "").strip()
        except Exception:
            return text

    def _get_wikipedia_summary(self, title: str) -> str:
        if not title:
            return ""

        try:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}"
            resp = requests.get(
                url,
                timeout=10,
                headers={"User-Agent": "AI-Video-Learning-Assistant/1.0"},
            )
            resp.raise_for_status()
            data = resp.json()
            summary = data.get("extract", "") or ""
            return summary.strip()
        except Exception as e:
            self._observe(f"Lấy summary Wikipedia lỗi với title={title}: {e}")
            return ""

    def _generate_wikipedia_queries(self, topic: str) -> list[str]:
        topic = (topic or "").strip()
        if not topic:
            return []

        prompt = f"""
[ROLE]
Bạn là hệ thống tạo truy vấn tìm kiếm Wikipedia tiếng Anh.

[GOAL]
Sinh ra các truy vấn tìm kiếm tối ưu để tìm đúng bài Wikipedia liên quan đến chủ đề.

[INPUT]
Chủ đề:
{topic}

[QUERY GENERATION STRATEGY]
Bạn cần suy luận và tạo các biến thể truy vấn dựa trên:

- tên chính xác của chủ đề
- tên tiếng Anh (nếu có)
- tên gọi phổ biến hơn
- tên học thuật hoặc chính thức
- tên sự kiện hoặc nhân vật liên quan trực tiếp
- cách viết không dấu hoặc biến thể đơn giản hơn (nếu hữu ích)

[QUALITY RULES]
- Mỗi query phải:
  - ngắn gọn (2-6 từ)
  - rõ nghĩa
  - có khả năng khớp với tiêu đề Wikipedia

- Ưu tiên:
  - danh từ riêng
  - tên sự kiện
  - tên nhân vật
  - khái niệm chính xác

[CONSTRAINTS]
- KHÔNG tạo query quá chung chung (ví dụ: "history", "lesson")
- KHÔNG lệch chủ đề
- KHÔNG giải thích
- KHÔNG thêm văn bản ngoài JSON
- KHÔNG tạo query vô nghĩa hoặc trùng lặp

[OUTPUT RULES]
- Trả từ 3 đến 6 queries
- Mỗi query là một chuỗi ngắn
- Không lặp lại ý nghĩa

[OUTPUT FORMAT]
Chỉ trả JSON hợp lệ:

{{
  "queries": ["query 1", "query 2", "query 3"]
}}
"""

        fallback_queries = [topic, topic.lower()]
        try:
            res = self._generate_with_retry(
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
                model_name=self.task_models["detect"],
            )
            data = self._safe_json_load(res.text, {"queries": []})
            raw_queries = data.get("queries", [])
            queries: list[str] = []
            for q in raw_queries:
                if isinstance(q, str):
                    q = " ".join(q.strip().split())
                    if q:
                        queries.append(q)
            queries.extend(fallback_queries)
            deduped = list(dict.fromkeys(queries))
            return deduped[:6]
        except Exception as e:
            self._observe(f"Sinh truy vấn Wikipedia fallback vì lỗi: {e}")
            return list(dict.fromkeys(fallback_queries))

    # =========================
    # VIDEO
    # =========================
    def ingest_video(self):
        if not self.state.video_bytes:
            return {"error": "No video"}

        size = len(self.state.video_bytes)

        if size > self.file_limit:
            return {"error": f"Video too large: {size} bytes"}

        if size <= self.inline_limit:
            return {"strategy": "inline"}

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(self.state.video_bytes)
            path = tmp.name

        uploaded = self.client.files.upload(file=path)

        while True:
            f = self.client.files.get(name=uploaded.name)
            state_name = getattr(f.state, "name", "")
            if state_name == "ACTIVE":
                return {"strategy": "file", "file": f}
            if state_name == "FAILED":
                return {"error": "File processing failed"}
            time.sleep(3)

    def tool_extract_transcript(self):
        self._log("tool=extract_transcript")

        if self.state.transcript:
            return self.state.transcript

        ingest = self.ingest_video()
        if "error" in ingest:
            raise RuntimeError(ingest["error"])

        prompt = """
[ROLE]
Bạn là hệ thống trích xuất lời nói từ video bài giảng.

[GOAL]
Trích xuất transcript từ video một cách trung thực, đầy đủ và bám sát nội dung audio.

[INSTRUCTIONS]

- Chuyển toàn bộ lời nói trong video thành văn bản.
- Giữ nguyên ngôn ngữ gốc được nói trong video.
- Nếu có thể xác định được mốc thời gian, hãy đặt timestamp gần đúng theo định dạng [MM:SS] ở đầu câu hoặc đoạn.
- Nếu một đoạn không nghe rõ, ghi [không rõ].
- Giữ nội dung theo đúng thứ tự xuất hiện trong video.

[TRANSCRIPT RULES]

- Transcript phải bám sát lời nói thực tế trong video.
- Không được rút gọn ý.
- Không được diễn giải lại theo cách khác.
- Không được thêm thông tin không xuất hiện trong audio.
- Không được sửa nội dung theo suy đoán.

[TIMESTAMP RULES]

- Chỉ thêm timestamp khi có đủ căn cứ từ video.
- Timestamp phải ở dạng [MM:SS].
- Nếu không xác định được timestamp ở một đoạn, bỏ qua timestamp cho đoạn đó.
- Không được bịa timestamp.

[CONSTRAINTS]

- Không tóm tắt.
- Không phân tích.
- Không nhận xét.
- Không trả thêm giải thích.

[OUTPUT FORMAT]
Chỉ trả về transcript thuần.
Ví dụ:
[00:15] Hai Bà Trưng nổi dậy chống nhà Hán
[00:22] [không rõ]
Nội dung tiếp theo...

"""

        if ingest["strategy"] == "inline":
            res = self._generate_with_retry(
                contents=types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            inline_data=types.Blob(
                                data=self.state.video_bytes,
                                mime_type=self.state.video_mime or "video/mp4",
                            )
                        ),
                        types.Part(text=prompt),
                    ],
                ),
                model_name=self.task_models["default"],
            )
        else:
            res = self._generate_with_retry(
                contents=[ingest["file"], prompt],
                model_name=self.task_models["default"],
            )

        full_text = (res.text or "").strip()

        # lưu bản full
        self.state.raw_transcript = full_text

        # dùng transcript làm việc (có thể chunk sau)
        self.state.transcript = full_text
        self._observe(f"Đã có transcript, độ dài {len(self.state.transcript)} ký tự.")
        return self.state.transcript

    # =========================
    # CORE TOOLS
    # =========================
    def tool_detect_topic(self, user_request: str) -> Optional[str]:
        self._log("tool=detect_topic")

        prompt = f"""
[ROLE]
Bạn là hệ thống phân tích yêu cầu người dùng.

[GOAL]
Trích xuất chủ đề cụ thể mà người dùng muốn kiểm tra trong video bài giảng.

[INPUT]
Yêu cầu người dùng:
{user_request}

[INSTRUCTIONS]
- Phân tích yêu cầu người dùng.
- Nếu người dùng hỏi về một chủ đề cụ thể → trích xuất chủ đề đó.
- Chủ đề có thể là:
  - một khái niệm
  - một sự kiện
  - một nhân vật
  - một nội dung cụ thể trong bài giảng

[DECISION RULE]
- Nếu yêu cầu rõ ràng → trả chủ đề.
- Nếu yêu cầu chung chung (ví dụ: "tóm tắt video", "nội dung chính là gì") → trả null.
- Nếu không chắc chắn → ưu tiên trả null.

[CONSTRAINTS]
- KHÔNG giải thích.
- KHÔNG thêm văn bản ngoài JSON.
- KHÔNG suy diễn quá mức từ yêu cầu người dùng.

[OUTPUT FORMAT]
Chỉ trả JSON hợp lệ:

{{
  "topic": "..."
}}

Hoặc:

{{
  "topic": null
}}
"""

        res = self._generate_with_retry(
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            ),
            model_name=self.task_models["detect"],
        )

        data = self._safe_json_load(res.text, {"topic": None})
        topic = data.get("topic")
        if isinstance(topic, str):
            topic = topic.strip()

        self.state.detected_topic = topic or None
        self._observe(f"Topic phát hiện: {self.state.detected_topic}")
        return self.state.detected_topic

    def tool_search_topic(self, topic: str) -> dict:
        self._log(f"tool=search_topic topic={topic}")

        transcript_source = getattr(self.state, "raw_transcript", None) or self.state.transcript or ""
        self._log(
            f"search_topic_source={'raw_transcript' if getattr(self.state, 'raw_transcript', None) else 'transcript'} chars={len(transcript_source)}"
        )

        chunks = self._chunk_transcript(transcript_source)

        matches: list[dict] = []

        for index, chunk in enumerate(chunks, start=1):
            prompt = f"""
    [GOAL]
    Tìm xem đoạn transcript này có chứa topic không

    [TOPIC]
    {topic}

    [TRANSCRIPT]
    {chunk}

    [OUTPUT JSON]
    {{
    "co_tim_thay": true hoặc false,
    "doan_lien_quan": "...",
    "moc_thoi_gian": null hoặc "MM:SS"
    }}
    """
            res = self._generate_with_retry(
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                ),
                model_name=self.task_models["semantic_search"],
            )

            data = self._safe_json_load(res.text, {})

            if data.get("co_tim_thay"):
                data["chunk_index"] = index
                matches.append(data)

        relevant_passages = []
        timestamps = []
        for item in matches:
            passage = item.get("doan_lien_quan")
            timestamp = item.get("moc_thoi_gian")
            if isinstance(passage, str) and passage.strip():
                relevant_passages.append(passage.strip())
            if isinstance(timestamp, str) and timestamp.strip():
                timestamps.append(timestamp.strip())

        relevant_passages = list(dict.fromkeys(relevant_passages))
        timestamps = list(dict.fromkeys(timestamps))

        result = {
            "chu_de": topic,
            "co_tim_thay": bool(matches),
            "moc_thoi_gian": timestamps[0] if timestamps else None,
            "cac_moc_thoi_gian": timestamps,
            "doan_lien_quan": "\n\n".join(relevant_passages[:5]) if relevant_passages else None,
            "so_lan_xuat_hien": len(matches),
            "thong_tin_mo_rong": [],
            "nguon_wikipedia": [],
        }

        self._log(
            f"search_topic matches={len(matches)} timestamps={len(timestamps)} passages={len(relevant_passages)}"
        )

        self.state.topic_result = result
        return result

    def tool_lookup_wikipedia(self, query: str) -> list[dict]:
        self._log(f"tool=lookup_wikipedia query={query}")

        search_url = "https://en.wikipedia.org/w/api.php"
        cleaned: list[dict] = []
        seen_titles: set[str] = set()
        search_queries = self._generate_wikipedia_queries(query)
        self._log(f"wiki_queries={search_queries}")

        for search_query in search_queries:
            params = {
                "action": "query",
                "list": "search",
                "srsearch": search_query,
                "srlimit": 5,
                "srprop": "snippet|titlesnippet",
                "utf8": 1,
                "format": "json",
            }

            try:
                resp = requests.get(
                    search_url,
                    params=params,
                    timeout=10,
                    headers={"User-Agent": "AI-Video-Learning-Assistant/1.0"},
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                self._observe(f"Wikipedia lookup lỗi với query={search_query}: {e}")
                continue

            raw_results = data.get("query", {}).get("search", [])
            for item in raw_results:
                title = (item.get("title") or "").strip()
                if not title or title in seen_titles:
                    continue

                seen_titles.add(title)
                page_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                short_snippet = self._clean_wikipedia_snippet(item.get("snippet", ""))
                full_summary = self._get_wikipedia_summary(title)
                source_text = full_summary if full_summary else short_snippet
                translated_text = self._translate_to_user_language(source_text)

                cleaned.append(
                    {
                        "title": title,
                        "url": page_url,
                        "snippet": translated_text,
                    }
                )

            # dừng sớm nếu đã có vài kết quả tốt
            if len(cleaned) >= 3:
                break

        self.state.external_context = cleaned
        self._observe(f"Tra cứu Wikipedia được {len(cleaned)} kết quả.")
        return cleaned

    def tool_save(self, data):
        self._log("tool=save_db")

        mongo_uri = os.getenv("MONGO_URI")
        db_name = os.getenv("MONGO_DB")
        collection_name = os.getenv("MONGO_COLLECTION")

        if not mongo_uri or not db_name or not collection_name:
            self._observe("Bỏ qua lưu DB vì thiếu cấu hình.")
            return {
                "success": False,
                "message": "Thiếu cấu hình MongoDB",
            }

        try:
            with MongoClient(mongo_uri) as client:
                db = client[db_name]
                col = db[collection_name]
                inserted = col.insert_one(
                    {
                        "data": data,
                        "video_name": self.state.video_name,
                        "created_at": datetime.now(UTC),
                    }
                )
            self._observe("Đã lưu database thành công.")
            return {
                "success": True,
                "inserted_id": str(inserted.inserted_id),
            }
        except Exception as e:
            self._observe(f"Lưu DB thất bại: {e}")
            return {
                "success": False,
                "message": str(e),
            }


    # =========================
    # GEMINI TOOL CALLING
    # =========================
    def fc_extract_transcript(self) -> dict:
        """Extract the full raw transcript from the current video.

        Use this first when transcript is missing. This tool reads the video/audio and stores
        the complete raw transcript in agent state for later topic search and final learning output.
        """
        transcript = self.tool_extract_transcript()
        raw_transcript = getattr(self.state, "raw_transcript", transcript) or ""
        return {
            "success": True,
            "transcript_length": len(raw_transcript),
            "preview": raw_transcript[:600],
        }

    def fc_detect_topic(self, user_request: str) -> dict:
        """Detect the specific topic the user wants to check inside the video.

        Call this after transcript exists. Input must be the original normalized user request.
        Returns the extracted topic or null when the request is generic.
        """
        topic = self.tool_detect_topic(user_request)
        return {
            "success": True,
            "topic": topic,
        }

    def fc_search_topic_in_transcript(self, topic: str) -> dict:
        """Search the detected topic in the raw transcript.

        Always search in the raw transcript, not the summary. Use this only after a non-empty
        topic has been detected.
        """
        result = self.tool_search_topic(topic)
        return {
            "success": True,
            "result": result,
        }

    def fc_lookup_wikipedia(self, query: str) -> dict:
        """Look up background knowledge on Wikipedia for a topic already found in the transcript.

        Use this only when the user asks for extra explanation, context, background knowledge,
        or deeper understanding beyond the raw transcript.
        """
        results = self.tool_lookup_wikipedia(query)
        return {
            "success": True,
            "results": results,
        }

    def fc_finalize_research(self) -> dict:
        """Finalize the tool-calling phase when enough information has been collected.

        Call this only after the required transcript/topic/background steps are complete.
        """
        return {
            "success": True,
            "has_transcript": bool(self.state.transcript),
            "detected_topic": self.state.detected_topic,
            "has_topic_result": self.state.topic_result is not None,
            "has_external_context": bool(self.state.external_context),
        }

    def _run_agent_loop_with_gemini_tool_calling(self, user_request: str):
        self._log("agent_mode=gemini_function_calling_manual")
        
        #Ghi log
        self._workflow_event(
            "agent_loop_started",
            mode="manual_function_calling",
            user_request=user_request,
            state=self._state_snapshot(),
        )
        
        #Kiểm tra người dùng có cần kiểm tra thông tin bên ngoài không
        needs_external = self._infer_external_need_fallback(user_request, self.state.detected_topic)

        orchestration_prompt = f"""
[ROLE]
Bạn là autonomous agent phân tích video bài giảng bằng Gemini function calling.

[GOAL]
Hoàn thành việc thu thập dữ liệu cần thiết trước khi hệ thống tạo LearningOutput JSON ở bước sau.

[IMPORTANT]
- Đây là pha tool-calling, KHÔNG phải pha trả lời cuối.
- Bạn phải ưu tiên gọi tool thay vì tự suy đoán.
- Nếu chưa có transcript thì phải gọi fc_extract_transcript.
- Nếu đã có transcript nhưng chưa có topic thì phải gọi fc_detect_topic với nguyên user request.
- Nếu có topic nhưng chưa search trong transcript thì phải gọi fc_search_topic_in_transcript.
- Chỉ gọi fc_lookup_wikipedia khi user thực sự cần kiến thức nền / giải thích thêm.
- Khi đã đủ dữ liệu, hãy gọi fc_finalize_research.
- Sau khi gọi fc_finalize_research, chỉ trả lời ngắn: READY.

[AVAILABLE TOOLS]
- fc_extract_transcript
- fc_detect_topic
- fc_search_topic_in_transcript
- fc_lookup_wikipedia
- fc_finalize_research

[STATE SNAPSHOT]
- has_transcript: {bool(self.state.transcript)}
- detected_topic: {json.dumps(self.state.detected_topic, ensure_ascii=False)}
- has_topic_result: {self.state.topic_result is not None}
- has_external_context: {bool(self.state.external_context)}
- needs_external_context: {needs_external}
- recent_observations: {json.dumps(self.state.observations[-6:], ensure_ascii=False)}

[USER REQUEST]
{user_request}
"""

        contents = [types.Content(role="user", parts=[types.Part(text=orchestration_prompt)])]
        config = types.GenerateContentConfig(
            tools=[
                self.fc_extract_transcript,
                self.fc_detect_topic,
                self.fc_search_topic_in_transcript,
                self.fc_lookup_wikipedia,
                self.fc_finalize_research,
            ],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        max_tool_rounds = 8
        last_response = None

        for round_index in range(1, max_tool_rounds + 1):
            #Ghi log
            self._workflow_event(
                "model_turn_started",
                round_index=round_index,
                state=self._state_snapshot(),
            )
            response = self._generate_with_retry(
                contents=contents,
                config=config,
                model_name=self.task_models["planner"],
            )
            last_response = response
            response_text = (getattr(response, "text", "") or "").strip()
            function_calls = self._extract_function_calls_from_response(response)

            self._workflow_event(
                "model_turn_completed",
                round_index=round_index,
                response_text=response_text[:2000],
                function_calls=[
                    {
                        "id": getattr(fc, "id", None),
                        "name": getattr(fc, "name", None),
                        "args": dict(getattr(fc, "args", None) or {}),
                    }
                    for fc in function_calls
                ],
                state=self._state_snapshot(),
            )

            if not function_calls:
                self._workflow_event(
                    "agent_loop_stopped_no_function_call",
                    round_index=round_index,
                    response_text=response_text[:2000],
                    state=self._state_snapshot(),
                )
                break

            contents.append(response.candidates[0].content)

            saw_finalize = False
            function_response_parts = []
            for function_call in function_calls:
                execution = self._execute_function_call(function_call, user_request=user_request)
                function_response_parts.append(
                    types.Part.from_function_response(
                        name=execution["name"],
                        response={"result": execution["response"]},
                    )
                )
                if execution["name"] == "fc_finalize_research":
                    saw_finalize = True

            contents.append(types.Content(role="user", parts=function_response_parts))

            if saw_finalize:
                self._workflow_event(
                    "agent_loop_finalize_requested",
                    round_index=round_index,
                    state=self._state_snapshot(),
                )
                break

        self._workflow_event(
            "agent_loop_finished",
            final_text=(getattr(last_response, "text", "") or "").strip()[:2000] if last_response else "",
            state=self._state_snapshot(),
        )
        return {
            "topic_result": self.state.topic_result,
            "external_context": self.state.external_context,
            "workflow_logs": self.state.workflow_logs,
        }

    # =========================
    # REACT PLANNER
    # =========================
    def tool_plan_next_action(self, user_request: str) -> AgentAction:
        self._log("tool=planner")

        prompt = f"""
[ROLE]
Bạn là planner của một autonomous agent phân tích video bài giảng.

[GOAL]
Chọn chính xác 1 action tiếp theo để hoàn thành yêu cầu người dùng.

[WORKING MODE]
Bạn phải suy luận theo mô hình ReAct:
Reason → Act → Observe → lặp lại

Bạn KHÔNG được trả lời trực tiếp.
Bạn CHỈ chọn action tiếp theo.

[AVAILABLE ACTIONS]
- extract_transcript
- detect_topic
- search_topic_in_transcript
- lookup_wikipedia
- finalize

(Lưu ý: save_to_database KHÔNG được dùng trong vòng lặp planner)

[CURRENT STATE]
- has_transcript: {bool(self.state.transcript)}
- detected_topic: {json.dumps(self.state.detected_topic, ensure_ascii=False)}
- topic_result: {json.dumps(self.state.topic_result, ensure_ascii=False)}
- has_external_context: {bool(self.state.external_context)}
- recent_observations: {json.dumps(self.state.observations[-6:], ensure_ascii=False)}

[DECISION LOGIC - BẮT BUỘC TUÂN THỦ]

1. Nếu chưa có transcript:
   → PHẢI chọn extract_transcript

2. Nếu đã có transcript nhưng chưa có topic:
   → PHẢI chọn detect_topic

3. Nếu có topic nhưng chưa search:
   → PHẢI chọn search_topic_in_transcript

4. Nếu topic đã tìm thấy:
   - Nếu user cần hiểu sâu / thêm kiến thức:
     → chọn lookup_wikipedia
   - Nếu không:
     → finalize

5. Nếu topic không tìm thấy:
   - Có thể lookup_wikipedia nếu user cần
   - hoặc finalize

6. Chỉ finalize khi:
   - đã có transcript
   - đã xử lý đủ yêu cầu user

[EXTERNAL CONTEXT RULE]
Nếu user yêu cầu:
- hiểu sâu hơn
- giải thích thêm
- kiến thức nền
- ví dụ mở rộng

→ needs_external_context = true  
→ ưu tiên lookup_wikipedia trước finalize

[CONSTRAINTS]
- KHÔNG chọn action không hợp lệ
- KHÔNG bỏ qua bước bắt buộc
- KHÔNG lặp action vô nghĩa
- KHÔNG finalize quá sớm
- KHÔNG gọi save_to_database

[OUTPUT FORMAT]
Chỉ trả JSON hợp lệ:

{{
  "thought": "một câu suy nghĩ ngắn",
  "action": "extract_transcript | detect_topic | search_topic_in_transcript | lookup_wikipedia | finalize",
  "action_input": null,
  "reason": "giải thích ngắn gọn",
  "needs_external_context": true hoặc false
}}

[USER REQUEST]
{user_request}
"""

        try:
            res = self._generate_with_retry(
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                ),
                model_name=self.task_models["planner"],
            )
            data = self._safe_json_load(res.text, {})
            action = AgentAction.model_validate(data)

            if action.action == "save_to_database":
                return AgentAction(
                    thought="Lưu DB chỉ thực hiện sau khi đã có kết quả cuối.",
                    action="finalize",
                    reason="save_after_finalize_only",
                    needs_external_context=action.needs_external_context,
                )

            if action.action == "lookup_wikipedia":
                if not self.state.detected_topic:
                    return AgentAction(
                        thought="Chưa có topic để tra cứu ngoài.",
                        action="finalize",
                        reason="lookup_without_topic",
                        needs_external_context=action.needs_external_context,
                    )

                if not (
                    self.state.topic_result
                    and self.state.topic_result.get("co_tim_thay") is True
                ):
                    return AgentAction(
                        thought="Chưa có bằng chứng đủ mạnh rằng topic xuất hiện trong transcript.",
                        action="finalize",
                        reason="skip_external_when_topic_not_found",
                        needs_external_context=action.needs_external_context,
                    )

            return action

        except Exception:
            if not self.state.transcript:
                return AgentAction(
                    thought="Chưa có transcript.",
                    action="extract_transcript",
                    reason="fallback_no_transcript",
                    needs_external_context=False,
                )

            if self.state.detected_topic is None:
                return AgentAction(
                    thought="Chưa biết topic cần dò.",
                    action="detect_topic",
                    action_input=user_request,
                    reason="fallback_detect_topic",
                    needs_external_context=False,
                )

            if self.state.detected_topic and self.state.topic_result is None:
                return AgentAction(
                    thought="Cần tìm topic trong transcript.",
                    action="search_topic_in_transcript",
                    action_input=self.state.detected_topic,
                    reason="fallback_search_topic",
                    needs_external_context=False,
                )

            if (
                self.state.detected_topic
                and self.state.topic_result is not None
                and self.state.topic_result.get("co_tim_thay") is True
                and not self.state.external_context
                and self._infer_external_need_fallback(user_request, self.state.detected_topic)
            ):
                return AgentAction(
                    thought="Fallback: cần tra cứu ngoài để bổ sung ngữ cảnh.",
                    action="lookup_wikipedia",
                    action_input=self.state.detected_topic,
                    reason="fallback_lookup_wikipedia",
                    needs_external_context=True,
                )

            return AgentAction(
                thought="Đủ dữ liệu để kết thúc.",
                action="finalize",
                reason="fallback_finalize",
                needs_external_context=False,
            )

    # =========================
    # DISPATCHER (bỏ)
    # =========================
    def _dispatch(self, action: AgentAction, final_data: Optional[dict] = None):
        if action.action == "extract_transcript":
            return self.tools["extract_transcript"]()

        if action.action == "detect_topic":
            return self.tools["detect_topic"](action.action_input or "")

        if action.action == "search_topic_in_transcript":
            return self.tools["search_topic_in_transcript"](action.action_input or "")

        if action.action == "lookup_wikipedia":
            return self.tools["lookup_wikipedia"](action.action_input or "")

        if action.action == "save_to_database":
            if final_data is None:
                return {"success": False, "message": "No final data to save"}
            return self.tools["save_to_database"](final_data)

        if action.action == "finalize":
            return None

        raise RuntimeError(f"Unknown action: {action.action}")

    # =========================
    # AGENT LOOP
    # =========================
    def _run_agent_loop(self, user_request: str):
        step = 0
        while step < self.max_steps:
            step += 1

            action = self.tool_plan_next_action(user_request)

            self._log(
                f"step={step} thought={action.thought} action={action.action} input={action.action_input} needs_external={action.needs_external_context}"
            )

            if action.action == "detect_topic" and not action.action_input:
                action.action_input = user_request

            if action.action == "lookup_wikipedia" and not action.action_input:
                action.action_input = self.state.detected_topic

            if action.action == "search_topic_in_transcript" and not action.action_input:
                action.action_input = self.state.detected_topic

            # Guardrails tối thiểu để giữ tính agentic nhưng tránh action vô nghĩa
            if action.action == "search_topic_in_transcript" and not self.state.transcript:
                self._observe("Planner chọn search_topic_in_transcript nhưng chưa có transcript.")
                continue

            if action.action == "lookup_wikipedia" and not self.state.detected_topic:
                self._observe("Planner chọn lookup_wikipedia nhưng chưa có topic.")
                continue

            if action.action == "save_to_database":
                self._observe("Bỏ qua save_to_database trong loop; chỉ save sau finalize.")
                continue

            if action.action == "lookup_wikipedia":
                wiki_sources = self._dispatch(action)

                if self.state.topic_result is None and self.state.detected_topic:
                    self.state.topic_result = {
                        "chu_de": self.state.detected_topic,
                        "co_tim_thay": False,
                        "moc_thoi_gian": None,
                        "doan_lien_quan": None,
                        "thong_tin_mo_rong": [],
                        "nguon_wikipedia": [],
                    }

                if self.state.topic_result is not None:
                    self.state.topic_result["nguon_wikipedia"] = wiki_sources
                    self.state.topic_result["thong_tin_mo_rong"] = [
                        item.get("snippet")
                        for item in wiki_sources
                        if item.get("snippet")
                    ]

                continue

            if action.action == "finalize":
                inferred_need_external = bool(action.needs_external_context)

                if (
                    inferred_need_external
                    and self.state.detected_topic
                    and self.state.topic_result is not None
                    and self.state.topic_result.get("co_tim_thay") is True
                    and not self.state.external_context
                ):
                    action = AgentAction(
                        thought="Người dùng muốn biết thêm về chủ đề, cần bổ sung thông tin nền trước khi finalize.",
                        action="lookup_wikipedia",
                        action_input=self.state.detected_topic,
                        reason="finalize_guard_lookup_wikipedia",
                        needs_external_context=True,
                    )

                    wiki_sources = self._dispatch(action)
                    if self.state.topic_result is not None:
                        self.state.topic_result["nguon_wikipedia"] = wiki_sources
                        self.state.topic_result["thong_tin_mo_rong"] = [
                            item.get("snippet")
                            for item in wiki_sources
                            if item.get("snippet")
                        ]
                    continue

                break

            result = self._dispatch(action)
            self._observe(f"Đã thực hiện action={action.action}.")

            # result được giữ implicit qua state; observe để planner thấy vòng lặp có tiến triển
            if result is not None and action.action not in ["extract_transcript"]:
                self._observe(str(result)[:300])

        return {
            "topic_result": self.state.topic_result,
            "external_context": self.state.external_context,
        }

    # =========================
    # MAIN
    # =========================
    def run(self, user_request: str):
        user_request = self._normalize_user_request(user_request)

        if not self.state.video_bytes and not self.state.transcript:
            raise RuntimeError("Không có video hoặc transcript để xử lý")

        self.state.user_language = self._detect_user_language(user_request)
        self._observe(f"Ngôn ngữ người dùng: {self.state.user_language}")

        agent_output = self._run_agent_loop_with_gemini_tool_calling(user_request)
        topic_result = agent_output.get("topic_result")
        self._workflow_event("run_post_research", topic_result=topic_result, state=self._state_snapshot())

        long_transcript_summary = self._summarize_long_transcript()
        transcript_context = self._build_transcript_context_for_final(self.state.transcript or "")

        final_prompt = f"""
[ROLE]
Bạn là AI hỗ trợ học từ video bài giảng.

[GOAL]
Tạo kết quả học tập hoàn chỉnh dưới dạng JSON đúng theo schema `LearningOutput`.

[PRIMARY DATA SOURCES]
Bạn chỉ được sử dụng các nguồn dữ liệu sau, theo thứ tự ưu tiên:
1. Transcript video
2. Long transcript summary đã được tổng hợp từ toàn bộ transcript
3. Topic result đã được hệ thống phân tích
4. Agent observations
5. Thông tin nền từ Wikipedia (chỉ dùng để bổ sung, không được thay thế transcript)

[USER LANGUAGE]
Toàn bộ output phải sử dụng ngôn ngữ có mã ISO: {self.state.user_language}

[STRICT INSTRUCTIONS]
- Phải trả về đúng 1 JSON hợp lệ theo schema `LearningOutput`.
- Không được trả thêm giải thích, markdown, tiêu đề hoặc văn bản ngoài JSON.
- Không được bịa nội dung không có trong transcript.
- Phải bao quát toàn bộ nội dung video, không chỉ phần đầu transcript.
- Nếu transcript bao gồm nhiều giai đoạn/nhiều phần nội dung, phần `tom_tat` và `y_chinh` phải phản ánh đủ các phần quan trọng trên toàn bộ video.
- Nếu thông tin không đủ chắc chắn, hãy để trường phù hợp là null hoặc danh sách rỗng thay vì suy đoán.
- Nếu không có chủ đề cụ thể từ yêu cầu người dùng, `ket_qua_tim_chu_de` phải là null.
- Nếu transcript không có timestamp chắc chắn, `moc_thoi_gian` phải là null.
- Nếu có thông tin từ Wikipedia, chỉ dùng như phần mở rộng kiến thức nền trong `thong_tin_mo_rong` hoặc `nguon_wikipedia`.
- Không được dùng thông tin từ Wikipedia để thay thế nội dung chính của transcript.
- `ghi_chu_agent` phải mô tả ngắn gọn các bước chính mà agent đã thực hiện.

[FIELD REQUIREMENTS]
- `chu_de`: nêu chủ đề chính của bài giảng.
- `tom_tat`: là danh sách 3 đến 8 ý tóm tắt ngắn, rõ ràng, bám sát transcript.
- `y_chinh`: là danh sách 3 đến 8 ý chính quan trọng nhất của bài giảng.
- `cau_hoi_trac_nghiem`: tạo từ 1 đến 10 câu hỏi trắc nghiệm, mỗi câu gồm:
  - `cau_hoi`
  - `lua_chon` với đúng 4 đáp án A, B, C, D
  - `dap_an_dung` chỉ được là A, B, C hoặc D
- `ket_qua_tim_chu_de`: chỉ điền khi người dùng có yêu cầu tìm chủ đề cụ thể.
- `ghi_chu_agent`: tóm tắt ngắn gọn pipeline agent đã chạy.

[USER REQUEST]
{user_request}

[TRANSCRIPT CONTEXT]
{transcript_context}

[IMPORTANT]
Transcript context bên dưới được dựng để giảm thất thoát thông tin từ transcript dài.
Nếu transcript rất dài, context có thể gồm nhiều cửa sổ đại diện (đầu/giữa/cuối) kết hợp với LONG TRANSCRIPT SUMMARY.
Hãy dùng cả hai nguồn để bao quát toàn bộ video.

[LONG TRANSCRIPT SUMMARY]
{json.dumps(long_transcript_summary, ensure_ascii=False)}

[TOPIC RESULT]
{json.dumps(topic_result, ensure_ascii=False) if topic_result else "null"}

[AGENT OBSERVATIONS]
{json.dumps(self.state.observations, ensure_ascii=False)}
"""

        self._workflow_event("final_generation_started", state=self._state_snapshot())

        res = self._generate_with_retry(
            contents=final_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=LearningOutput,
            ),
            model_name=self.task_models["final"],
        )

        parsed = getattr(res, "parsed", None)
        if parsed is not None:
            data = parsed.model_dump()
        else:
            data = self._safe_json_load(res.text, {})

        if not isinstance(data, dict):
            data = {}

        self._workflow_event("final_generation_completed", parsed=data, state=self._state_snapshot())

        # Fix trường ket_qua_tim_chu_de có thể là None
        if topic_result:
            existing_topic_result = data.get("ket_qua_tim_chu_de")
            data["ket_qua_tim_chu_de"] = existing_topic_result if isinstance(existing_topic_result, dict) else {}

            for key, value in topic_result.items():
                if data["ket_qua_tim_chu_de"].get(key) in [None, "", [], {}] and value not in [None, "", [], {}]:
                    data["ket_qua_tim_chu_de"][key] = value

            if (
                not data["ket_qua_tim_chu_de"].get("nguon_wikipedia")
                and topic_result.get("nguon_wikipedia")
            ):
                data["ket_qua_tim_chu_de"]["nguon_wikipedia"] = topic_result["nguon_wikipedia"]

            if (
                not data["ket_qua_tim_chu_de"].get("thong_tin_mo_rong")
                and topic_result.get("thong_tin_mo_rong")
            ):
                data["ket_qua_tim_chu_de"]["thong_tin_mo_rong"] = topic_result["thong_tin_mo_rong"]

        # Nếu model không sinh đủ ghi_chu_agent thì fallback bằng logs
        if not data.get("ghi_chu_agent"):
            data["ghi_chu_agent"] = "; ".join(self.state.logs[-8:])

        db_status = None
        if self._extract_db_request(user_request):
            save_action = AgentAction(
                thought="Người dùng yêu cầu lưu DB.",
                action="save_to_database",
                reason="post_finalize_save",
            )
            db_status = self._dispatch(save_action, final_data=data)
            self._log(f"db_status={db_status}")

        return {
            "result": data,
            "logs": self.state.logs,
            "workflow_logs": self.state.workflow_logs,
            "db_status": db_status,
        }
