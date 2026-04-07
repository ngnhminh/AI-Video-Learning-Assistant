from __future__ import annotations

from dotenv import load_dotenv

import os
import re
import json
import time
import tempfile
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from dataclasses import dataclass, field
from typing import Optional, Any

from google import genai
from google.genai import types

from schemas import LearningOutput, TopicSearchResult

load_dotenv()


# =========================
# STATE
# =========================

@dataclass
class AgentState:
    transcript: Optional[str] = None
    video_bytes: Optional[bytes] = None
    video_mime: Optional[str] = None
    video_name: Optional[str] = None
    uploaded_file_name: str = ""
    uploaded_file_uri: str = ""
    logs: list[str] = field(default_factory=list)


# =========================
# AGENT
# =========================

class GeminiVideoAgent:
    def __init__(self, api_key: str, model: str):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.state = AgentState()

        self.inline_limit = 20 * 1024 * 1024
        self.file_limit = 2 * 1024 * 1024 * 1024

    # =========================
    # FILE HANDLING
    # =========================

    def _wait_active(self, name: str):
        while True:
            f = self.client.files.get(name=name)
            state = getattr(f.state, "name", None)

            if state == "ACTIVE":
                return f

            if state == "FAILED":
                raise RuntimeError("File processing FAILED")

            time.sleep(5)

    def ingest_video(self):
        if not self.state.video_bytes:
            return {"error": "No video"}

        size = len(self.state.video_bytes)

        if size <= self.inline_limit:
            return {"strategy": "inline"}

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(self.state.video_bytes)
            path = tmp.name

        uploaded = self.client.files.upload(file=path)
        f = self._wait_active(uploaded.name)

        self.state.uploaded_file_name = f.name
        self.state.uploaded_file_uri = getattr(f, "uri", "")
        return {"strategy": "file_api"}

    # =========================
    # REAL TOOLS
    # =========================

    def tool_extract_transcript(self, language_hint: str = "same_as_user_request"):
        self.state.logs.append("Tool: extract_transcript")

        if self.state.transcript:
            return {
                "transcript": self.state.transcript,
                "source": "cached_transcript",
            }

        if not self.state.video_bytes:
            return {"error": "No transcript and no video provided"}

        ingest = self.ingest_video()

        prompt = f"""
Hãy trích xuất transcript từ video.

Yêu cầu:
- Giữ nguyên nội dung theo ngôn ngữ nói trong video.
- Không tóm tắt.
- Không phân tích.
- Chỉ trả transcript rõ ràng, dễ đọc.
- Nếu nghe không rõ một đoạn, có thể ghi [không rõ].

Gợi ý ngôn ngữ đầu ra: {language_hint}
"""

        if ingest.get("strategy") == "inline":
            response = self.client.models.generate_content(
                model=self.model,
                contents=types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            inline_data=types.Blob(
                                data=self.state.video_bytes,
                                mime_type=self.state.video_mime or "video/mp4",
                            ),
                            video_metadata=types.VideoMetadata(fps=1),
                        ),
                        types.Part(text=prompt),
                    ],
                ),
            )
        else:
            file = self.client.files.get(name=self.state.uploaded_file_name)
            response = self.client.models.generate_content(
                model=self.model,
                contents=[file, prompt],
            )

        text = (response.text or "").strip()
        self.state.transcript = text

        return {
            "transcript": text,
            "source": "video_extracted",
        }

    def tool_search_topic(self, topic: str):
        self.state.logs.append(f"Tool: search_topic(topic={topic})")

        if not self.state.transcript:
            return {
                "chu_de": topic,
                "co_tim_thay": False,
                "moc_thoi_gian": None,
                "doan_lien_quan": None,
            }

        prompt = f"""
Dựa trên transcript dưới đây, hãy xác định chủ đề "{topic}" có xuất hiện hay không.

Yêu cầu:
- Trả về JSON với đúng các khóa:
  - chu_de
  - co_tim_thay
  - moc_thoi_gian
  - doan_lien_quan
- `co_tim_thay` là true hoặc false.
- `moc_thoi_gian` dùng MM:SS nếu suy ra được, nếu không chắc thì để null.
- `doan_lien_quan` là đoạn ngắn bám sát transcript, nếu không có thì để null.
- Không bịa.

Transcript:
{self.state.transcript[:20000]}
"""

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=TopicSearchResult,
            ),
        )

        parsed = getattr(response, "parsed", None)
        if parsed is None:
            data = json.loads(response.text)
            parsed = TopicSearchResult.model_validate(data)

        return parsed.model_dump()

    # =========================
    # EMAIL (POST-PROCESS ONLY)
    # =========================

    def tool_send_email_report(
        self,
        to_email: str,
        subject: str,
        report_content: str,
        content_format: str = "plain",
    ):
        self.state.logs.append(f"Tool: send_email_report(to={to_email})")

        smtp_host = os.getenv("SMTP_HOST")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_username = os.getenv("SMTP_USERNAME")
        smtp_password = os.getenv("SMTP_PASSWORD")
        smtp_from = os.getenv("SMTP_FROM", smtp_username)

        if not smtp_host or not smtp_username or not smtp_password or not smtp_from:
            return {
                "success": False,
                "message": "Thiếu cấu hình SMTP trong biến môi trường.",
            }

        try:
            msg = MIMEMultipart()
            msg["From"] = smtp_from
            msg["To"] = to_email
            msg["Subject"] = subject

            mime_type = "html" if content_format == "html" else "plain"
            msg.attach(MIMEText(report_content, mime_type, "utf-8"))

            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                server.login(smtp_username, smtp_password)
                server.sendmail(smtp_from, to_email, msg.as_string())

            return {
                "success": True,
                "message": f"Đã gửi báo cáo tới {to_email}",
            }

        except Exception as exc:
            return {
                "success": False,
                "message": f"Gửi email thất bại: {str(exc)}",
            }

    def _build_email_content(self, result: dict) -> str:
        lines: list[str] = []

        lines.append("BÁO CÁO HỌC TẬP TỪ VIDEO BÀI GIẢNG")
        lines.append("")
        lines.append(f"Chủ đề: {result.get('chu_de', 'Không xác định')}")
        lines.append("")

        lines.append("Tóm tắt:")
        for item in result.get("tom_tat", []):
            lines.append(f"- {item}")
        lines.append("")

        lines.append("Ý chính:")
        for item in result.get("y_chinh", []):
            lines.append(f"- {item}")
        lines.append("")

        questions = result.get("cau_hoi_trac_nghiem", [])
        if questions:
            lines.append("Câu hỏi trắc nghiệm:")
            for idx, q in enumerate(questions, start=1):
                lines.append(f"{idx}. {q['cau_hoi']}")
                options = q["lua_chon"]
                lines.append(f"   A. {options['A']}")
                lines.append(f"   B. {options['B']}")
                lines.append(f"   C. {options['C']}")
                lines.append(f"   D. {options['D']}")
                lines.append(f"   Đáp án đúng: {q['dap_an_dung']}")
                lines.append("")

        topic_result = result.get("ket_qua_tim_chu_de")
        if topic_result:
            lines.append("Kết quả tìm chủ đề:")
            lines.append(f"- Chủ đề: {topic_result.get('chu_de')}")
            lines.append(f"- Có tìm thấy: {topic_result.get('co_tim_thay')}")
            lines.append(f"- Mốc thời gian: {topic_result.get('moc_thoi_gian')}")
            lines.append(f"- Đoạn liên quan: {topic_result.get('doan_lien_quan')}")
            lines.append("")

        if result.get("ghi_chu_agent"):
            lines.append("Ghi chú agent:")
            lines.append(result["ghi_chu_agent"])

        return "\n".join(lines)

    def _extract_email_request(self, user_request: str) -> Optional[str]:
        lower_req = user_request.lower()
        if "email" not in lower_req and "mail" not in lower_req and "gửi" not in lower_req:
            return None

        match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', user_request)
        if match:
            return match.group(0)

        return None

    # =========================
    # TOOL DECLARATIONS
    # =========================

    def _tools(self):
        return [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="extract_transcript",
                        description=(
                            "Trích xuất transcript từ video khi chưa có transcript. "
                            "Chỉ gọi tool này nếu transcript chưa có sẵn."
                        ),
                        parameters={
                            "type": "object",
                            "properties": {
                                "language_hint": {
                                    "type": "string",
                                    "description": "Ngôn ngữ gợi ý cho transcript/output",
                                }
                            },
                        },
                    ),
                    types.FunctionDeclaration(
                        name="search_topic",
                        description=(
                            "Tìm một chủ đề cụ thể trong transcript khi người dùng yêu cầu "
                            "xác định xem bài giảng có nhắc tới chủ đề đó hay không."
                        ),
                        parameters={
                            "type": "object",
                            "properties": {
                                "topic": {
                                    "type": "string",
                                    "description": "Chủ đề cần tìm trong transcript",
                                }
                            },
                            "required": ["topic"],
                        },
                    ),
                ]
            )
        ]

    # =========================
    # AGENT LOOP
    # =========================

    def _build_initial_contents(self, user_request: str):
        parts: list[Any] = [
            types.Part(text=f"Yêu cầu người dùng:\n{user_request}")
        ]

        if self.state.transcript:
            parts.append(
                types.Part(
                    text=(
                        "Transcript đã được cung cấp sẵn bởi người dùng. "
                        "Nếu cần dùng, hãy đọc transcript này thay vì gọi extract_transcript.\n\n"
                        f"{self.state.transcript[:30000]}"
                    )
                )
            )
        else:
            if self.state.video_bytes:
                parts.append(
                    types.Part(
                        text=(
                            "Hiện chưa có transcript nhưng có video đầu vào. "
                            "Nếu cần nội dung bài giảng, hãy gọi tool extract_transcript."
                        )
                    )
                )
            else:
                parts.append(
                    types.Part(
                        text="Không có transcript và cũng không có video."
                    )
                )

        return [
            types.Content(
                role="user",
                parts=parts,
            )
        ]

    def _system_instruction(self) -> str:
        return """
Bạn là AI hỗ trợ học từ video bài giảng.

Mục tiêu:
- Phân tích nội dung bài giảng.
- Tạo đầu ra học tập có cấu trúc.
- Chỉ dùng tool khi thật sự cần.

Quy tắc làm việc:
1. Nếu transcript đã có sẵn trong ngữ cảnh, KHÔNG gọi `extract_transcript`.
2. Nếu transcript chưa có mà có video, hãy gọi `extract_transcript`.
3. Nếu người dùng yêu cầu tìm một chủ đề cụ thể trong bài giảng, hãy tự xác định chủ đề đó từ yêu cầu và gọi `search_topic`.
4. Nếu người dùng không yêu cầu tìm chủ đề cụ thể, không cần gọi `search_topic`.
5. Sau khi đã có đủ dữ liệu, hãy trả ra kết quả cuối cùng theo đúng schema mong muốn.
6. Toàn bộ output phải cùng ngôn ngữ với yêu cầu của người dùng.
7. Không bịa nội dung ngoài transcript.
8. Không gọi bất kỳ tool gửi email nào. Việc gửi email sẽ được xử lý sau khi kết quả cuối cùng đã được tạo.

Schema đầu ra cần thỏa:
- chu_de
- tom_tat (3-8 ý)
- y_chinh (3-8 ý)
- cau_hoi_trac_nghiem (1-10 câu)
- ket_qua_tim_chu_de (null nếu user không yêu cầu)
- ghi_chu_agent

Ràng buộc quiz:
- Mỗi câu hỏi phải có đúng 4 lựa chọn A/B/C/D
- dap_an_dung chỉ được là A, B, C hoặc D
"""

    def _execute_tool(self, name: str, args: dict):
        if name == "extract_transcript":
            return self.tool_extract_transcript(**args)

        if name == "search_topic":
            return self.tool_search_topic(**args)

        return {"error": f"unknown tool: {name}"}

    def run(self, user_request: str) -> dict:
        contents = self._build_initial_contents(user_request)

        # BƯỚC 1: agent loop với tools, KHÔNG ép JSON/schema
        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                tools=self._tools(),
                system_instruction=self._system_instruction(),
            ),
        )

        max_steps = 4
        step = 0

        while step < max_steps:
            step += 1
            candidate = response.candidates[0]
            parts = candidate.content.parts

            function_calls = [
                part.function_call
                for part in parts
                if getattr(part, "function_call", None)
            ]

            if not function_calls:
                break

            assistant_content = candidate.content
            tool_response_parts: list[Any] = []

            for fc in function_calls:
                name = fc.name
                args = dict(fc.args or {})
                result = self._execute_tool(name, args)

                tool_response_parts.append(
                    types.Part.from_function_response(
                        name=name,
                        response={"result": result},
                    )
                )

            tool_content = types.Content(
                role="tool",
                parts=tool_response_parts,
            )

            contents = [
                *contents,
                assistant_content,
                tool_content,
            ]

            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    tools=self._tools(),
                    system_instruction=self._system_instruction(),
                ),
            )

        # BƯỚC 2: gọi riêng để ép output về schema JSON
        final_prompt = """
Hãy tạo kết quả cuối cùng theo đúng schema LearningOutput.

Yêu cầu:
- Dựa trên toàn bộ ngữ cảnh hội thoại và kết quả tool đã có.
- Không gọi thêm tool.
- Trả đúng JSON theo schema.
- `ket_qua_tim_chu_de` là null nếu người dùng không yêu cầu tìm chủ đề cụ thể.
- Toàn bộ output phải cùng ngôn ngữ với yêu cầu của người dùng.
"""

        final_response = self.client.models.generate_content(
            model=self.model,
            contents=[
                *contents,
                types.Content(
                    role="user",
                    parts=[types.Part(text=final_prompt)],
                ),
            ],
            config=types.GenerateContentConfig(
                system_instruction=self._system_instruction(),
                response_mime_type="application/json",
                response_schema=LearningOutput,
            ),
        )

        parsed = getattr(final_response, "parsed", None)

        if parsed is None:
            try:
                data = json.loads(final_response.text)
                parsed = LearningOutput.model_validate(data)
            except Exception as exc:
                raise RuntimeError(
                    f"Model không trả được output hợp lệ theo schema. Raw response: {final_response.text}"
                ) from exc

        result = parsed.model_dump()

        if not result.get("ghi_chu_agent"):
            result["ghi_chu_agent"] = "; ".join(self.state.logs) if self.state.logs else "Không dùng tool"

        # BƯỚC 3: gửi email sau khi đã có LearningOutput cuối
        email_status = None
        to_email = self._extract_email_request(user_request)

        if to_email:
            email_content = self._build_email_content(result)
            email_status = self.tool_send_email_report(
                to_email=to_email,
                subject=f"Báo cáo học tập: {result.get('chu_de', 'Video bài giảng')}",
                report_content=email_content,
                content_format="plain",
            )
            self.state.logs.append(f"Post-process email status: {email_status}")

        return {
            "result": result,
            "files": {},
            "logs": self.state.logs,
            "email_status": email_status,
        }