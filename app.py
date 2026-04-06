from __future__ import annotations

import json
import mimetypes
import os
from pathlib import Path
from dotenv import load_dotenv

import streamlit as st

from agent import VideoLearningAgent

load_dotenv()
api_key=os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="AI Video Learning Assistant", page_icon="🎓", layout="wide")

st.title("🎓 AI Video Learning Assistant")
st.caption("MVP theo hướng agentic: Gemini API + Function Calling + Structured Output + Streamlit")

with st.sidebar:
    model = st.text_input("Model", value="gemini-2.5-flash")
    language = st.selectbox("Ngôn ngữ đầu ra", ["vi", "en"], index=0)
    
    summary_lines = None
    key_points = None
    quiz_count = None
    
    auto_save = st.checkbox("Tự lưu JSON kết quả", value=False)
    auto_export = st.checkbox("Tự xuất báo cáo", value=False)
    export_format = st.selectbox("Định dạng báo cáo", ["md", "html"], index=0)
    
    send_email = st.checkbox("Gửi email báo cáo", value=False)
    email_to = st.text_input("Email nhận báo cáo", value="")
    
left, right = st.columns([1, 1])

with left:
    st.subheader("Đầu vào")
    user_request = st.text_area(
        "Yêu cầu của bạn",
        height=120,
        value=(
            "Hãy phân tích bài giảng này, tóm tắt ngắn gọn, trích xuất các ý chính quan trọng, "
            "tạo câu hỏi trắc nghiệm phù hợp và nếu có phần nói về định luật Newton thì cho biết mốc thời gian."
        ),
    )

    transcript_file = st.file_uploader("Tải transcript (.txt, .md)", type=["txt", "md"])
    transcript_text = st.text_area("Hoặc dán transcript", height=220)

    video_file = st.file_uploader(
        "Tải video bài giảng (MVP hỗ trợ inline video nhỏ, ưu tiên dưới 20MB)",
        type=["mp4", "mov", "avi", "webm", "mpeg", "mpg"],
    )

with right:
    st.subheader("Mô tả kiến trúc")
    st.markdown(
        """
- **Prompt + Structured Output**: Gemini tạo `summary`, `key_points`, `quiz`.
- **Function Calling**: agent chỉ gọi tool khi cần cho transcript, tìm chủ đề, lưu kết quả, xuất báo cáo.
- **Agent Workflow**: tiếp nhận yêu cầu → dùng tool nếu cần → tổng hợp đầu ra JSON cuối.
        """
    )

run = st.button("🚀 Chạy hệ thống", type="primary", use_container_width=True)

if run:
    if not api_key:
        st.error("Bạn cần nhập Gemini API Key trước khi chạy.")
        st.stop()

    transcript = transcript_text.strip()
    if transcript_file is not None:
        transcript = transcript_file.read().decode("utf-8", errors="ignore").strip()

    video_bytes = None
    video_mime = None
    video_name = None
    if video_file is not None:
        video_bytes = video_file.read()
        video_name = video_file.name
        video_mime = video_file.type or mimetypes.guess_type(video_name)[0] or "video/mp4"

    if not transcript and not video_bytes:
        st.error("Bạn cần cung cấp transcript hoặc video.")
        st.stop()

    with st.spinner("Đang chạy agent..."):
        try:
            agent = VideoLearningAgent(api_key=api_key, model=model)
            payload = agent.process(
                user_request=user_request,
                transcript=transcript or None,
                video_bytes=video_bytes,
                video_mime=video_mime,
                video_name=video_name,
                summary_lines=summary_lines,
                key_points=key_points,
                quiz_count=quiz_count,
                language=language,
                auto_save=auto_save,
                auto_export=auto_export,
                export_format=export_format,
                send_email=send_email,
                email_to=email_to
            )
        except Exception as exc:
            st.exception(exc)
            st.stop()

    st.success("Xử lý thành công")

    result = payload["result"]

    st.subheader("Phản hồi của agent")
    st.info(result.get("ghi_chu_agent") or "Không có ghi chú agent.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Tóm tắt")
        for line in result["tom_tat"]:
            st.write(f"- {line}")

        st.subheader("Ý chính")
        for line in result["y_chinh"]:
            st.write(f"- {line}")

    with c2:
        st.subheader("Quiz")
        for idx, q in enumerate(result["cau_hoi_trac_nghiem"], start=1):
            st.markdown(f"**Câu {idx}. {q['cau_hoi']}**")
            options = q["lua_chon"]
            st.write(f"A. {options['A']}")
            st.write(f"B. {options['B']}")
            st.write(f"C. {options['C']}")
            st.write(f"D. {options['D']}")
            st.caption(f"Đáp án đúng: {q['dap_an_dung']}")
            st.divider()

    if result.get("ket_qua_tim_chu_de"):
        st.subheader("Kết quả tìm chủ đề")
        topic = result["ket_qua_tim_chu_de"]
        st.json(topic)

    st.subheader("JSON đầu ra")
    result_json = json.dumps(result, ensure_ascii=False, indent=2)
    st.code(result_json, language="json")
    st.download_button(
        "⬇️ Tải JSON đầu ra",
        data=result_json.encode("utf-8"),
        file_name="learning_output.json",
        mime="application/json",
    )

    if payload["files"]:
        st.subheader("Tệp đã tạo")
        for label, path_str in payload["files"].items():
            path = Path(path_str)
            if path.exists():
                mime = "text/markdown" if path.suffix == ".md" else "text/html" if path.suffix == ".html" else "application/json"
                st.download_button(
                    f"⬇️ Tải {label}: {path.name}",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime=mime,
                )
                st.caption(str(path.resolve()))

    if payload.get("email_status"):
        st.warning(payload["email_status"])

    with st.expander("Nhật ký tool / workflow"):
        for log in payload["logs"]:
            st.write(f"- {log}")
