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

    observations: list[str] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)


# =========================
# AGENT
# =========================
class GeminiVideoAgent:
    def __init__(self, api_key: str, model: str):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.fallback_models = list(dict.fromkeys([
            model,
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
        ]))
        self.state = AgentState()

        self.inline_limit = 20 * 1024 * 1024
        self.file_limit = 2 * 1024 * 1024 * 1024
        self.max_steps = 4

        self.tools: dict[str, Callable[..., Any]] = {
            "extract_transcript": self.tool_extract_transcript,
            "detect_topic": self.tool_detect_topic,
            "search_topic_in_transcript": self.tool_search_topic,
            "lookup_wikipedia": self.tool_lookup_wikipedia,
            "save_to_database": self.tool_save,
        }

    # =========================
    # RETRY
    # =========================
    def _is_retryable_error(self, exc: Exception) -> bool:
        msg = str(exc).upper()
        return any(
            s in msg
            for s in [
                "503",
                "UNAVAILABLE",
                "RATE",
                "TIMEOUT",
                "RESOURCE_EXHAUSTED",
                "INTERNAL",
            ]
        )

    def _sleep(self, attempt: int):
        delay = min(15, 2 * (2 ** attempt)) + random.uniform(0, 1)
        self.state.logs.append(f"retry_sleep={delay:.2f}s")
        time.sleep(delay)

    def _generate_with_retry(self, *, contents, config=None):
        last_exc = None
        max_retries_per_model = 2

        for model_name in self.fallback_models:
            for attempt in range(max_retries_per_model):
                try:
                    self.state.logs.append(
                        f"model_call model={model_name} attempt={attempt+1}/{max_retries_per_model}"
                    )
                    return self.client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=config,
                    )
                except Exception as e:
                    last_exc = e
                    self.state.logs.append(f"model_error model={model_name} err={e}")

                    if not self._is_retryable_error(e):
                        raise RuntimeError(f"Model failed: {e}") from e

                    if attempt < max_retries_per_model - 1:
                        delay = min(20, 1.5 * (2 ** attempt)) + random.uniform(0, 1)
                        self.state.logs.append(f"retry_sleep={delay:.2f}s")
                        time.sleep(delay)

            self.state.logs.append(f"switch_model_from={model_name}")

        raise RuntimeError(f"Model failed after retries across fallbacks: {last_exc}") from last_exc

    # =========================
    # HELPERS
    # =========================
    def _normalize_user_request(self, user_request: str) -> str:
        return " ".join(user_request.strip().split())

    def _extract_db_request(self, user_request: str) -> bool:
        lower = user_request.lower()
        return any(k in lower for k in ["lưu", "save", "store", "database", "db"])

    def _should_lookup_external(self, user_request: str, topic: Optional[str]) -> bool:
        if not topic:
            return False

        lower = user_request.lower()
        signals = [
            "tìm thêm",
            "thông tin về nó",
            "tra cứu",
            "bổ sung",
            "wikipedia",
            "kiến thức nền",
        ]
        return any(s in lower for s in signals)

    def _safe_json_load(self, text: str, default):
        try:
            return json.loads(text)
        except Exception:
            return default

    def _observe(self, message: str):
        self.state.observations.append(message[:500])
        if len(self.state.observations) > 10:
            self.state.observations = self.state.observations[-10:]

    def _translate_to_user_language(self, text: str) -> str:
        if not text:
            return text

        prompt = f"""
    Dịch đoạn sau sang tiếng Việt.
    Chỉ trả về nội dung đã dịch, không giải thích.

    {text}
    """

        try:
            res = self._generate_with_retry(contents=prompt)
            return (res.text or "").strip()
        except Exception:
            return text  # fallback nếu lỗi
    
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
        self.state.logs.append("tool=extract_transcript")

        if self.state.transcript:
            return self.state.transcript

        ingest = self.ingest_video()
        if "error" in ingest:
            raise RuntimeError(ingest["error"])

        prompt = """
Hãy trích xuất transcript từ video.

Yêu cầu:
- Giữ nguyên ngôn ngữ nói trong video.
- Không tóm tắt.
- Không phân tích.
- Nếu nghe không rõ thì ghi [không rõ].
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
                )
            )
        else:
            res = self._generate_with_retry(
                contents=[ingest["file"], prompt]
            )

        self.state.transcript = (res.text or "").strip()[:10000]
        self._observe(f"Đã có transcript, độ dài {len(self.state.transcript)} ký tự.")
        return self.state.transcript

    # =========================
    # CORE TOOLS
    # =========================
    def tool_detect_topic(self, user_request: str) -> Optional[str]:
        self.state.logs.append("tool=detect_topic")

        prompt = f"""
Trích xuất chủ đề cụ thể mà người dùng muốn kiểm tra trong video.

Trả JSON:
{{
  "topic": "..."
}}

Nếu không có chủ đề cụ thể, trả:
{{
  "topic": null
}}

Yêu cầu người dùng:
{user_request}
"""

        res = self._generate_with_retry(
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            ),
        )

        data = self._safe_json_load(res.text, {"topic": None})
        topic = data.get("topic")
        if isinstance(topic, str):
            topic = topic.strip()

        self.state.detected_topic = topic or None
        self._observe(f"Topic phát hiện: {self.state.detected_topic}")
        return self.state.detected_topic

    def tool_search_topic(self, topic: str) -> dict:
        self.state.logs.append(f"tool=search_topic topic={topic}")

        text = self.state.transcript or ""
        found = topic.lower() in text.lower()

        related_excerpt = None
        if found:
            idx = text.lower().find(topic.lower())
            start = max(0, idx - 120)
            end = min(len(text), idx + len(topic) + 180)
            related_excerpt = text[start:end].strip()

        result = {
            "chu_de": topic,
            "co_tim_thay": found,
            "moc_thoi_gian": None,
            "doan_lien_quan": related_excerpt,
            "thong_tin_mo_rong": [],
            "nguon_wikipedia": [],
        }

        self.state.topic_result = result
        self._observe(f"Kết quả search topic: found={found}")
        return result

    def tool_lookup_wikipedia(self, query: str) -> list[dict]:
        self.state.logs.append(f"tool=lookup_wikipedia query={query}")

        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": 3,
            "srprop": "snippet|titlesnippet",
            "utf8": 1,
            "format": "json",
        }

        try:
            resp = requests.get(
                url,
                params=params,
                timeout=10,
                headers={"User-Agent": "AI-Video-Learning-Assistant/1.0"},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            self._observe(f"Wikipedia lookup lỗi: {e}")
            return []

        raw_results = data.get("query", {}).get("search", [])
        cleaned = []

        for item in raw_results:
            title = item.get("title", "").strip()
            snippet = item.get("snippet", "").replace(
                '<span class="searchmatch">', ""
            ).replace("</span>", "")
            page_url = (
                f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                if title else ""
            )
            translated_snippet = self._translate_to_user_language(snippet)

            cleaned.append(
                {
                    "title": title,
                    "snippet": translated_snippet,
                    "url": page_url,
                }
            )

        self.state.external_context = cleaned
        self._observe(f"Tra cứu Wikipedia được {len(cleaned)} kết quả.")
        return cleaned

    def tool_save(self, data):
        self.state.logs.append("tool=save_db")

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
    # REACT PLANNER
    # =========================
    def tool_plan_next_action(self, user_request: str) -> AgentAction:
        self.state.logs.append("tool=planner")

        prompt = f"""
Bạn là planner cho một autonomous agent.

Nhiệm vụ:
- Chọn đúng 1 action tiếp theo để hoàn thành yêu cầu người dùng.
- Làm theo kiểu ReAct: Reason -> Act.
- Không gọi lại action đã hoàn tất.
- Chỉ chọn trong danh sách action hợp lệ.

Action hợp lệ:
1. extract_transcript
2. detect_topic
3. search_topic_in_transcript
4. lookup_wikipedia
5. save_to_database
6. finalize

Trạng thái hiện tại:
- has_transcript: {bool(self.state.transcript)}
- detected_topic: {json.dumps(self.state.detected_topic, ensure_ascii=False)}
- has_topic_result: {bool(self.state.topic_result)}
- has_external_context: {bool(self.state.external_context)}
- observations: {json.dumps(self.state.observations[-6:], ensure_ascii=False)}
- should_lookup_external: {self._should_lookup_external(user_request, self.state.detected_topic)}
- should_save_db: {self._extract_db_request(user_request)}

Quy tắc:
- Nếu chưa có transcript -> extract_transcript
- Nếu chưa detect topic -> detect_topic
- Nếu có topic mà chưa search -> search_topic_in_transcript
- Nếu user muốn tìm thêm thông tin và chưa có external_context -> lookup_wikipedia
- Không lưu DB trong loop nếu chưa có kết quả cuối
- Nếu đủ dữ liệu -> finalize

Trả JSON:
{{
  "thought": "một câu ngắn",
  "action": "extract_transcript | detect_topic | search_topic_in_transcript | lookup_wikipedia | save_to_database | finalize",
  "action_input": null,
  "reason": "vì sao"
}}
"""

        try:
            res = self._generate_with_retry(
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                ),
            )
            data = self._safe_json_load(res.text, {})
            action = AgentAction.model_validate(data)

            if action.action == "lookup_wikipedia" and not self._should_lookup_external(
                user_request, self.state.detected_topic
            ):
                return AgentAction(
                    thought="Không cần tra cứu ngoài.",
                    action="finalize",
                    reason="external_lookup_not_needed",
                )

            if action.action == "save_to_database":
                return AgentAction(
                    thought="Lưu DB chỉ thực hiện sau khi đã có kết quả cuối.",
                    action="finalize",
                    reason="save_after_finalize_only",
                )

            return action
        except Exception:
            if not self.state.transcript:
                return AgentAction(
                    thought="Chưa có transcript.",
                    action="extract_transcript",
                    reason="fallback_no_transcript",
                )
            if self.state.detected_topic is None:
                return AgentAction(
                    thought="Chưa biết topic cần dò.",
                    action="detect_topic",
                    action_input=user_request,
                    reason="fallback_detect_topic",
                )
            if self.state.detected_topic and self.state.topic_result is None:
                return AgentAction(
                    thought="Cần tìm topic trong transcript.",
                    action="search_topic_in_transcript",
                    action_input=self.state.detected_topic,
                    reason="fallback_search_topic",
                )
            if (
                self.state.detected_topic
                and not self.state.external_context
                and self._should_lookup_external(user_request, self.state.detected_topic)
            ):
                return AgentAction(
                    thought="Cần tra cứu ngoài để bổ sung ngữ cảnh.",
                    action="lookup_wikipedia",
                    action_input=self.state.detected_topic,
                    reason="fallback_lookup_wikipedia",
                )

            return AgentAction(
                thought="Đủ dữ liệu để kết thúc.",
                action="finalize",
                reason="fallback_finalize",
            )

    # =========================
    # DISPATCHER
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

            self.state.logs.append(
                f"step={step} thought={action.thought} action={action.action} input={action.action_input}"
            )

            if action.action == "detect_topic" and not action.action_input:
                action.action_input = user_request

            if action.action == "lookup_wikipedia" and not action.action_input:
                action.action_input = self.state.detected_topic

            if action.action == "search_topic_in_transcript" and not action.action_input:
                action.action_input = self.state.detected_topic
    
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
                    ][:3]

                continue

            if action.action == "finalize":
                break

            self._dispatch(action)

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

        agent_output = self._run_agent_loop(user_request)
        topic_result = agent_output.get("topic_result")
        external_context = agent_output.get("external_context", [])

        final_prompt = f"""
Bạn là AI hỗ trợ học từ video bài giảng.

Hãy tạo đúng JSON theo schema LearningOutput.

Yêu cầu:
- Dựa trên transcript và dữ liệu đã thu thập.
- Toàn bộ output bằng tiếng Việt.
- Không bịa nội dung ngoài transcript.
- Nếu không có chủ đề cụ thể thì `ket_qua_tim_chu_de` là null.
- `ghi_chu_agent` mô tả ngắn các bước agent đã thực hiện.

Yêu cầu người dùng:
{user_request}

Transcript:
{(self.state.transcript or "")[:6000]}

Topic result:
{json.dumps(topic_result, ensure_ascii=False) if topic_result else "null"}

Wikipedia context:
{json.dumps(external_context, ensure_ascii=False)}

Agent observations:
{json.dumps(self.state.observations, ensure_ascii=False)}
"""

        res = self._generate_with_retry(
            contents=final_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=LearningOutput,
            ),
        )

        parsed = getattr(res, "parsed", None)
        if parsed is not None:
            data = parsed.model_dump()
        else:
            data = self._safe_json_load(res.text, {})

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
            self.state.logs.append(f"db_status={db_status}")

        return {
            "result": data,
            "logs": self.state.logs,
            "db_status": db_status,
        }