from __future__ import annotations

import json
import os
import re
import smtplib
import time
import tempfile

from pydantic import BaseModel
from typing import Literal

from dataclasses import dataclass, field
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Optional

from google import genai
from google.genai import types

from schemas import LearningOutput


@dataclass
class AgentState:
    transcript: Optional[str] = None
    video_bytes: Optional[bytes] = None
    video_mime: Optional[str] = None
    video_name: Optional[str] = None
    uploaded_file_name: str = ""
    uploaded_file_uri: str = ""
    save_requested: bool = False
    export_requested: bool = False
    export_format: str = "md"
    send_email_requested: bool = False
    email_to: str = ""
    filename_hint: str = ""
    logs: list[str] = field(default_factory=list)


class AgentLoopResponse(BaseModel):
    thought: str
    action: Literal["tool", "final"]
    tool_name: Optional[str] = None
    tool_args: Optional[dict[str, Any]] = None
    final_answer: Optional[LearningOutput] = None


class VideoLearningAgent:
    def __init__(self, api_key: str, model: str) -> None:
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.state = AgentState()
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.inline_video_limit = 20 * 1024 * 1024   # 20 MB
        self.file_api_limit = 2 * 1024 * 1024 * 1024 # 2 GB
        self.file_active_timeout_seconds = 300
        self.file_poll_interval_seconds = 5

    def reset_state(
        self,
        transcript: Optional[str] = None,
        video_bytes: Optional[bytes] = None,
        video_mime: Optional[str] = None,
        video_name: Optional[str] = None,
    ) -> None:
        self.state = AgentState(
            transcript=transcript,
            video_bytes=video_bytes,
            video_mime=video_mime,
            video_name=video_name,
            uploaded_file_name="",
            uploaded_file_uri="",
        )

    # =========================
    # Internal file helpers
    # =========================
    def _get_file_state_name(self, uploaded_file: Any) -> Optional[str]:
        state = getattr(uploaded_file, "state", None)
        if state is None:
            return None
        return getattr(state, "name", None) or str(state)

    def _wait_until_file_active(self, file_name: str) -> dict[str, Any]:
        waited = 0

        while waited <= self.file_active_timeout_seconds:
            uploaded = self.client.files.get(name=file_name)
            state_name = self._get_file_state_name(uploaded)
            self.state.logs.append(f"Polling file state: {file_name} -> {state_name}")

            if state_name == "ACTIVE":
                self.state.uploaded_file_name = uploaded.name or ""
                self.state.uploaded_file_uri = getattr(uploaded, "uri", "") or ""
                return {
                    "status": "success",
                    "file": uploaded,
                    "state": state_name,
                }

            if state_name == "FAILED":
                return {
                    "status": "error",
                    "message": "File upload đã thất bại trong quá trình xử lý của Gemini (state=FAILED).",
                }

            time.sleep(self.file_poll_interval_seconds)
            waited += self.file_poll_interval_seconds

        return {
            "status": "error",
            "message": f"Hết thời gian chờ file chuyển sang ACTIVE sau {self.file_active_timeout_seconds} giây.",
        }

    # =========================
    # Tools exposed to Gemini
    # =========================
    def ingest_video(self) -> dict[str, Any]:
        """
        Tự chọn chiến lược nạp video:
        - inline nếu video nhỏ
        - File API nếu video lớn
        """
        self.state.logs.append("Tool: ingest_video()")

        if not self.state.video_bytes or not self.state.video_mime:
            return {
                "status": "error",
                "message": "Không có video trong phiên làm việc.",
            }

        size = len(self.state.video_bytes)

        if size <= self.inline_video_limit:
            return {
                "status": "success",
                "strategy": "inline",
                "message": f"Video đủ nhỏ để xử lý inline ({size} bytes).",
            }

        if size > self.file_api_limit:
            return {
                "status": "error",
                "message": "Video vượt quá giới hạn File API 2GB.",
            }

        # Nếu đã upload rồi thì kiểm tra lại trạng thái trước khi tái sử dụng
        if self.state.uploaded_file_name:
            wait_result = self._wait_until_file_active(self.state.uploaded_file_name)
            if wait_result.get("status") == "success":
                return {
                    "status": "success",
                    "strategy": "file_api",
                    "message": "Đã có file upload sẵn và file đang ở trạng thái ACTIVE.",
                    "file_name": self.state.uploaded_file_name,
                    "file_uri": self.state.uploaded_file_uri,
                }
            return wait_result

        suffix = Path(self.state.video_name or "video.mp4").suffix or ".mp4"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(self.state.video_bytes)
            tmp_path = tmp.name

        try:
            uploaded = self.client.files.upload(file=tmp_path)

            self.state.uploaded_file_name = uploaded.name or ""
            self.state.uploaded_file_uri = getattr(uploaded, "uri", "") or ""

            wait_result = self._wait_until_file_active(self.state.uploaded_file_name)
            if wait_result.get("status") != "success":
                return wait_result

            return {
                "status": "success",
                "strategy": "file_api",
                "message": "Đã upload video qua File API và file đã ở trạng thái ACTIVE.",
                "file_name": self.state.uploaded_file_name,
                "file_uri": self.state.uploaded_file_uri,
            }
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def extract_transcript(self, language: str = "vi") -> dict[str, Any]:
        self.state.logs.append(f"Tool: extract_transcript(language={language})")

        if self.state.transcript:
            return {
                "status": "success",
                "message": "Transcript đã tồn tại trong phiên làm việc.",
                "preview": self.state.transcript[:400],
            }

        if not self.state.video_bytes or not self.state.video_mime:
            return {
                "status": "error",
                "message": "Không có video để trích xuất transcript.",
            }

        ingest_result = self.ingest_video()
        if ingest_result.get("status") != "success":
            return ingest_result

        prompt = (
            f"Hãy tạo transcript rõ ràng cho video bài giảng sau bằng ngôn ngữ {language}. "
            "Nếu có thể, hãy chèn mốc thời gian gần đúng theo định dạng [MM:SS] ở các đoạn quan trọng. "
            "Chỉ trả về transcript, không giải thích thêm."
        )

        strategy = ingest_result.get("strategy")

        try:
            if strategy == "inline":
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                inline_data=types.Blob(
                                    data=self.state.video_bytes,
                                    mime_type=self.state.video_mime,
                                ),
                                video_metadata=types.VideoMetadata(fps=1),
                            ),
                            types.Part(text=prompt),
                        ],
                    ),
                )
            elif strategy == "file_api":
                uploaded_file = self.client.files.get(name=self.state.uploaded_file_name)
                state_name = self._get_file_state_name(uploaded_file)

                if state_name != "ACTIVE":
                    wait_result = self._wait_until_file_active(self.state.uploaded_file_name)
                    if wait_result.get("status") != "success":
                        return wait_result
                    uploaded_file = wait_result["file"]

                try:
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=[
                            uploaded_file,
                            prompt,
                        ],
                    )
                except Exception as e:
                    error_text = str(e)
                    if "FAILED_PRECONDITION" in error_text:
                        self.state.logs.append(
                            "generate_content gặp FAILED_PRECONDITION, chờ file ACTIVE rồi thử lại 1 lần."
                        )
                        wait_result = self._wait_until_file_active(self.state.uploaded_file_name)
                        if wait_result.get("status") != "success":
                            return wait_result

                        uploaded_file = wait_result["file"]
                        response = self.client.models.generate_content(
                            model=self.model,
                            contents=[
                                uploaded_file,
                                prompt,
                            ],
                        )
                    else:
                        raise
            else:
                return {
                    "status": "error",
                    "message": f"Chiến lược video không hợp lệ: {strategy}",
                }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Lỗi khi trích xuất transcript: {str(e)}",
            }

        transcript = (response.text or "").strip()
        self.state.transcript = transcript

        return {
            "status": "success",
            "message": f"Đã trích xuất transcript bằng chiến lược {strategy}.",
            "preview": transcript[:400],
        }

    def search_topic_timestamp(self, topic: str) -> dict[str, Any]:
        """Tìm chủ đề trong transcript và trả về mốc thời gian gần đúng."""
        self.state.logs.append(f"Tool: search_topic_timestamp(topic={topic})")

        if not self.state.transcript:
            return {
                "status": "error",
                "message": "Chưa có transcript để tìm kiếm chủ đề.",
            }

        prompt = f"""
Bạn là công cụ tìm kiếm chủ đề trong transcript bài giảng.
Chủ đề cần tìm: {topic}

Yêu cầu:
- Nếu transcript có mốc thời gian dạng [MM:SS], hãy tận dụng để trả về moc_thoi_gian.
- Nếu không có mốc thời gian rõ ràng, hãy suy ra gần đúng nếu đủ cơ sở, nếu không thì để null.
- doan_lien_quan phải là đoạn ngắn nhất nhưng đủ nghĩa.
- Chỉ trả JSON hợp lệ theo schema.

Schema:
{{
  "chu_de": "string",
  "co_tim_thay": true,
  "moc_thoi_gian": "MM:SS hoặc null",
  "doan_lien_quan": "string hoặc null"
}}

Transcript:
{self.state.transcript[:20000]}
""".strip()

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": {
                    "type": "object",
                    "properties": {
                        "chu_de": {"type": "string"},
                        "co_tim_thay": {"type": "boolean"},
                        "moc_thoi_gian": {"type": ["string", "null"]},
                        "doan_lien_quan": {"type": ["string", "null"]},
                    },
                    "required": ["chu_de", "co_tim_thay", "moc_thoi_gian", "doan_lien_quan"],
                },
            },
        )
        result = json.loads(response.text)
        return {"status": "success", **result}

    def save_result(self, filename: str = "") -> dict[str, Any]:
        """Đánh dấu lưu JSON kết quả cuối."""
        self.state.logs.append(f"Tool: save_result(filename={filename})")
        self.state.save_requested = True
        self.state.filename_hint = filename or self.state.filename_hint
        return {
            "status": "scheduled",
            "message": "Kết quả cuối cùng sẽ được lưu sau khi phân tích xong.",
        }

    def export_report(self, format: str = "md", send_email: bool = False, email_to: str = "") -> dict[str, Any]:
        """Đánh dấu xuất báo cáo hoặc gửi email sau khi có kết quả cuối."""
        self.state.logs.append(
            f"Tool: export_report(format={format}, send_email={send_email}, email_to={email_to})"
        )
        self.state.export_requested = True
        self.state.export_format = (format or "md").lower()
        self.state.send_email_requested = send_email
        self.state.email_to = email_to or ""
        return {
            "status": "scheduled",
            "message": "Báo cáo sẽ được xuất sau khi phân tích xong.",
            "format": self.state.export_format,
            "send_email": self.state.send_email_requested,
        }

    # =========================
    # Internal helpers
    # =========================
    def _call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if tool_name == "extract_transcript" and self.state.transcript:
            return {
                "status": "success",
                "message": "Transcript đã có sẵn, không cần gọi lại.",
                "preview": self.state.transcript[:400],
            }

        if tool_name == "ingest_video" and self.state.uploaded_file_name:
            wait_result = self._wait_until_file_active(self.state.uploaded_file_name)
            if wait_result.get("status") == "success":
                return {
                    "status": "success",
                    "strategy": "file_api",
                    "message": "Video đã được ingest trước đó và file đang ở trạng thái ACTIVE.",
                    "file_name": self.state.uploaded_file_name,
                    "file_uri": self.state.uploaded_file_uri,
                }
            return wait_result

        tool_map = {
            "ingest_video": self.ingest_video,
            "extract_transcript": self.extract_transcript,
            "search_topic_timestamp": self.search_topic_timestamp,
            "save_result": self.save_result,
            "export_report": self.export_report,
        }

        if tool_name not in tool_map:
            return {
                "status": "error",
                "message": f"Tool không tồn tại: {tool_name}",
            }

        try:
            return tool_map[tool_name](**arguments)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Lỗi khi gọi tool {tool_name}: {str(e)}",
            }

    def _build_agent_loop_schema(self) -> dict[str, Any]:
        return AgentLoopResponse.model_json_schema()

    def _slugify(self, value: str) -> str:
        value = value.strip().lower()
        value = re.sub(r"[^a-z0-9-_]+", "-", value)
        value = re.sub(r"-+", "-", value).strip("-")
        return value or "result"

    def _base_name(self, result: LearningOutput) -> str:
        hint = self.state.filename_hint or result.chu_de or self.state.video_name or "learning-output"
        return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{self._slugify(hint)}"

    def _save_result_file(self, result: LearningOutput) -> Path:
        path = self.output_dir / f"{self._base_name(result)}.json"
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        self.state.logs.append(f"Saved JSON: {path}")
        return path

    def _render_markdown_report(self, result: LearningOutput) -> str:
        lines = [
            f"# Báo cáo học tập: {result.chu_de}",
            "",
            "## Tóm tắt",
        ]
        lines.extend([f"- {item}" for item in result.tom_tat])
        lines.extend(["", "## Ý chính"])
        lines.extend([f"- {item}" for item in result.y_chinh])
        lines.extend(["", "## Câu hỏi trắc nghiệm"])
        for idx, q in enumerate(result.cau_hoi_trac_nghiem, start=1):
            lines.extend(
                [
                    f"### Câu {idx}",
                    q.cau_hoi,
                    f"- A. {q.lua_chon.A}",
                    f"- B. {q.lua_chon.B}",
                    f"- C. {q.lua_chon.C}",
                    f"- D. {q.lua_chon.D}",
                    f"**Đáp án đúng:** {q.dap_an_dung}",
                    "",
                ]
            )
        if result.ket_qua_tim_chu_de:
            lines.extend(
                [
                    "## Kết quả tìm chủ đề",
                    f"- Chủ đề: {result.ket_qua_tim_chu_de.chu_de}",
                    f"- Có tìm thấy: {'Có' if result.ket_qua_tim_chu_de.co_tim_thay else 'Không'}",
                    f"- Mốc thời gian: {result.ket_qua_tim_chu_de.moc_thoi_gian or 'Không có'}",
                    f"- Đoạn liên quan: {result.ket_qua_tim_chu_de.doan_lien_quan or 'Không có'}",
                    "",
                ]
            )
        if result.ghi_chu_agent:
            lines.extend(["## Ghi chú agent", result.ghi_chu_agent, ""])
        return "\n".join(lines)

    def _export_report_file(self, result: LearningOutput) -> Path:
        format_name = self.state.export_format.lower()
        base = self.output_dir / self._base_name(result)
        if format_name == "html":
            md = self._render_markdown_report(result)
            html = (
                "<html><body><pre style='white-space: pre-wrap; font-family: Arial;'>"
                + md
                + "</pre></body></html>"
            )
            path = base.with_suffix(".html")
            path.write_text(html, encoding="utf-8")
        else:
            path = base.with_suffix(".md")
            path.write_text(self._render_markdown_report(result), encoding="utf-8")
        self.state.logs.append(f"Exported report: {path}")
        return path

    def _send_email_if_configured(self, report_path: Path) -> Optional[str]:
        if not self.state.send_email_requested or not self.state.email_to:
            return None

        smtp_host = os.getenv("SMTP_HOST")
        smtp_port = os.getenv("SMTP_PORT")
        smtp_user = os.getenv("SMTP_USER")
        smtp_password = os.getenv("SMTP_PASSWORD")
        smtp_sender = os.getenv("SMTP_SENDER", smtp_user or "")

        if not all([smtp_host, smtp_port, smtp_user, smtp_password, smtp_sender]):
            msg = "Email đã được yêu cầu nhưng chưa cấu hình đủ SMTP_* trong môi trường."
            self.state.logs.append(msg)
            return msg

        email = EmailMessage()
        email["Subject"] = "Báo cáo học tập từ AI Video Learning Assistant"
        email["From"] = smtp_sender
        email["To"] = self.state.email_to
        email.set_content(report_path.read_text(encoding="utf-8"))

        with smtplib.SMTP(smtp_host, int(smtp_port)) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(email)

        msg = f"Đã gửi email tới {self.state.email_to}."
        self.state.logs.append(msg)
        return msg

    def _build_system_instruction(
        self,
        summary_lines: Optional[int],
        key_points: Optional[int],
        quiz_count: Optional[int],
        language: str,
    ) -> str:
        constraints = [f"- Ngôn ngữ đầu ra: {language}"]

        if summary_lines is not None:
            constraints.append(f"- Tóm tắt đúng {summary_lines} dòng ngắn gọn, dễ hiểu.")
        else:
            constraints.append(
                "- Tự quyết định số dòng tóm tắt sao cho đủ ý, ngắn gọn và hợp lý với độ dài nội dung, thường trong khoảng 3–7 dòng."
            )

        if key_points is not None:
            constraints.append(f"- Trích xuất đúng {key_points} ý chính.")
        else:
            constraints.append(
                "- Tự quyết định số ý chính sao cho bao quát nội dung quan trọng, tránh trùng lặp, thường trong khoảng 4–8 ý."
            )

        if quiz_count is not None:
            constraints.append(
                f"- Tạo đúng {quiz_count} câu hỏi trắc nghiệm, mỗi câu có 4 lựa chọn A/B/C/D và 1 đáp án đúng."
            )
        else:
            constraints.append(
                "- Tự quyết định số câu hỏi trắc nghiệm phù hợp với nội dung, ưu tiên chất lượng hơn số lượng, thường trong khoảng 3–6 câu."
            )

        constraint_text = "\n".join(constraints)

        return f"""
Bạn là AI Video Learning Agent.

Chỉ chọn:
- action="tool" khi thật sự cần công cụ
- action="final" khi đã đủ dữ liệu

Tool:
- ingest_video
- extract_transcript
- search_topic_timestamp
- save_result
- export_report

Quy tắc:
- Nếu đã có transcript thì không gọi extract_transcript.
- Chỉ gọi search_topic_timestamp khi người dùng yêu cầu mốc thời gian/chủ đề.
- Chỉ gọi save_result/export_report khi người dùng yêu cầu.
- Ưu tiên hoàn tất trong ít bước nhất.
- Nếu đủ dữ liệu thì trả final ngay.

Rằng buộc đầu ra cuối:
Ngôn ngữ đầu ra: {language}
""".strip()

    def _build_effective_user_request(
        self,
        user_request: str,
        auto_save: bool,
        auto_export: bool,
        export_format: str,
        email_to: str,
        send_email: bool,
    ) -> str:
        extra_instructions: list[str] = []
        if auto_save:
            extra_instructions.append("Sau khi hoàn tất, hãy lưu kết quả JSON.")
        if auto_export:
            if send_email and email_to:
                extra_instructions.append(
                    f"Sau khi hoàn tất, hãy xuất báo cáo dạng {export_format} và gửi email tới {email_to}."
                )
            else:
                extra_instructions.append(f"Sau khi hoàn tất, hãy xuất báo cáo dạng {export_format}.")

        if not extra_instructions:
            return user_request.strip()

        return (user_request.strip() + "\n\nYêu cầu bổ sung từ giao diện:\n- " + "\n- ".join(extra_instructions)).strip()

    def _run_full_agentic_session(
        self,
        user_request: str,
        summary_lines: Optional[int],
        key_points: Optional[int],
        quiz_count: Optional[int],
        language: str,
    ) -> LearningOutput:
        system_instruction = self._build_system_instruction(
            summary_lines=summary_lines,
            key_points=key_points,
            quiz_count=quiz_count,
            language=language,
        )

        loop_schema = self._build_agent_loop_schema()

        state_context = {
            "co_transcript_san": bool(self.state.transcript),
            "co_video_san": bool(self.state.video_bytes),
            "video_name": self.state.video_name,
        }

        messages: list[str] = [
            "Người dùng yêu cầu:",
            user_request,
            "",
            "Trạng thái hiện tại:",
            json.dumps(state_context, ensure_ascii=False),
        ]

        max_steps = 5

        for step in range(max_steps):
            response = self.client.models.generate_content(
                model=self.model,
                contents="\n".join(messages),
                config={
                    "system_instruction": system_instruction,
                    "response_mime_type": "application/json",
                    "response_json_schema": loop_schema,
                },
            )

            data = AgentLoopResponse.model_validate_json(response.text)

            thought = data.thought
            action = data.action
            tool_name = data.tool_name
            tool_args = data.tool_args or {}
            final_answer = data.final_answer

            self.state.logs.append(f"Agent step {step + 1}: {thought}")

            if action == "tool":
                tool_result = self._call_tool(tool_name, tool_args)
                self.state.logs.append(
                    f"Tool executed: {tool_name} | args={tool_args} | result={tool_result}"
                )

                messages.append(f"\nBước {step + 1} - thought: {thought}")
                messages.append(f"Hành động: gọi tool {tool_name}")
                messages.append(f"Kết quả tool: {json.dumps(tool_result, ensure_ascii=False)}")
                continue

            if action == "final":
                if final_answer is None:
                    raise ValueError("Model trả action='final' nhưng không có final_answer.")
                return final_answer

            raise ValueError(f"Action không hợp lệ: {action}")

        raise RuntimeError("Agent vượt quá số bước tối đa nhưng vẫn chưa hoàn tất.")

    # =========================
    # Public entrypoint
    # =========================
    def process(
        self,
        user_request: str,
        transcript: Optional[str] = None,
        video_bytes: Optional[bytes] = None,
        video_mime: Optional[str] = None,
        video_name: Optional[str] = None,
        summary_lines: Optional[int] = None,
        key_points: Optional[int] = None,
        quiz_count: Optional[int] = None,
        language: str = "vi",
        auto_save: bool = False,
        auto_export: bool = False,
        export_format: str = "md",
        send_email: bool = False,
        email_to: str = "",
    ) -> dict[str, Any]:
        self.reset_state(
            transcript=transcript,
            video_bytes=video_bytes,
            video_mime=video_mime,
            video_name=video_name,
        )
        self.state.logs.append("Process started")

        if not self.state.transcript and not self.state.video_bytes:
            raise ValueError("Bạn cần cung cấp transcript hoặc video để hệ thống phân tích.")

        effective_request = self._build_effective_user_request(
            user_request=user_request,
            auto_save=auto_save,
            auto_export=auto_export,
            export_format=export_format,
            email_to=email_to,
            send_email=send_email,
        )

        result = self._run_full_agentic_session(
            user_request=effective_request,
            summary_lines=summary_lines,
            key_points=key_points,
            quiz_count=quiz_count,
            language=language,
        )

        files: dict[str, str] = {}
        if self.state.save_requested:
            files["json"] = str(self._save_result_file(result))

        if self.state.export_requested:
            report_path = self._export_report_file(result)
            files["report"] = str(report_path)
            email_status = self._send_email_if_configured(report_path)
        else:
            email_status = None

        self.state.logs.append("Process finished")

        return {
            "result": result.model_dump(),
            "logs": self.state.logs,
            "files": files,
            "email_status": email_status,
            "state": {
                "save_requested": self.state.save_requested,
                "export_requested": self.state.export_requested,
                "export_format": self.state.export_format,
                "send_email_requested": self.state.send_email_requested,
                "email_to": self.state.email_to,
                "has_transcript": bool(self.state.transcript),
            },
        }