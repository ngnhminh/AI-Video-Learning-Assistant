from __future__ import annotations

import json
import mimetypes
import os
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st

from agent import GeminiVideoAgent

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

st.set_page_config(
    page_title="AI Video Learning Assistant",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 AI Video Learning Assistant")
st.caption("Gemini API + Structured Output + Streamlit")


# =========================
# SESSION STATE
# =========================
if "payload" not in st.session_state:
    st.session_state.payload = None

if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False

if "quiz_results" not in st.session_state:
    st.session_state.quiz_results = None


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.subheader("Cấu hình")
    model = st.selectbox(
        "Model",
        [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
        ],
        index=0,
    )

# =========================
# MAIN INPUT
# =========================
left, right = st.columns([1, 1])

with left:
    st.subheader("Đầu vào")

    user_request = st.text_area(
        "Yêu cầu của bạn",
        height=140,
        value=(
            "Hãy phân tích bài giảng này, tóm tắt ngắn gọn, trích xuất các ý chính quan trọng, "
            "tạo câu hỏi trắc nghiệm phù hợp và nếu có phần nói về định luật Newton thì cho biết mốc thời gian."
        ),
    )

    video_file = st.file_uploader(
        "Tải video bài giảng",
        type=["mp4", "mov", "avi", "webm", "mpeg", "mpg"],
    )

with right:
    st.subheader("Mô tả")
    st.markdown(
        """
- Nếu có **transcript**, hệ thống ưu tiên dùng transcript.
- Nếu không có transcript mà có **video**, agent sẽ tự trích transcript.
- Kết quả trả về theo **schema có cấu trúc**:
  - `chu_de`
  - `tom_tat`
  - `y_chinh`
  - `cau_hoi_trac_nghiem`
  - `ket_qua_tim_chu_de`
  - `ghi_chu_agent`
- Phần quiz cho phép người dùng **chọn đáp án** và **nộp bài để chấm**.
        """
    )

run = st.button("🚀 Chạy hệ thống", type="primary", use_container_width=True)


# =========================
# RUN AGENT
# =========================
if run:
    if not api_key:
        st.error("Bạn cần cấu hình GEMINI_API_KEY trong file .env.")
        st.stop()

    video_bytes = None
    video_mime = None
    video_name = None

    if video_file is not None:
        video_bytes = video_file.read()
        video_name = video_file.name
        video_mime = video_file.type or mimetypes.guess_type(video_name)[0] or "video/mp4"

    if not video_bytes:
        st.error("Bạn cần cung cấp video.")
        st.stop()

    with st.spinner("Đang chạy agent..."):
        try:
            agent = GeminiVideoAgent(api_key=api_key, model=model)

            # Đồng bộ dữ liệu vào state của agent
            agent.state.video_bytes = video_bytes
            agent.state.video_mime = video_mime
            agent.state.video_name = video_name

            payload = agent.run(
                user_request=user_request,
            )

            st.session_state.payload = payload
            st.session_state.quiz_submitted = False
            st.session_state.quiz_results = None

        except Exception as exc:
            st.exception(exc)
            st.stop()


# =========================
# DISPLAY RESULT
# =========================
if st.session_state.payload:
    payload = st.session_state.payload
    result = payload["result"]

    st.success("Xử lý thành công")

    st.subheader("Phản hồi của agent")
    st.info(result.get("ghi_chu_agent") or "Không có ghi chú agent.")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Chủ đề")
        st.write(result.get("chu_de", "Không xác định"))

        st.subheader("Tóm tắt")
        for line in result.get("tom_tat", []):
            st.write(f"- {line}")

        st.subheader("Ý chính")
        for line in result.get("y_chinh", []):
            st.write(f"- {line}")

    with c2:
        st.subheader("Quiz")

        questions = result.get("cau_hoi_trac_nghiem", [])

        if questions:
            with st.form("quiz_form"):
                st.write("Chọn đáp án cho từng câu hỏi rồi nhấn **Nộp bài**.")

                for idx, q in enumerate(questions, start=1):
                    st.markdown(f"**Câu {idx}. {q['cau_hoi']}**")
                    options = q["lua_chon"]

                    option_labels = [
                        f"A. {options['A']}",
                        f"B. {options['B']}",
                        f"C. {options['C']}",
                        f"D. {options['D']}",
                    ]

                    st.radio(
                        f"Chọn đáp án câu {idx}",
                        option_labels,
                        index=None,
                        key=f"quiz_{idx}",
                    )
                    st.divider()

                submitted = st.form_submit_button("Nộp bài", use_container_width=True)

            if submitted:
                score = 0
                total = len(questions)
                detailed_results = []

                for idx, q in enumerate(questions, start=1):
                    correct_key = q["dap_an_dung"].strip().upper()
                    options = q["lua_chon"]

                    selected_label = st.session_state.get(f"quiz_{idx}")
                    selected_key = None
                    if selected_label:
                        selected_key = selected_label.split(".")[0].strip().upper()

                    is_correct = selected_key == correct_key
                    if is_correct:
                        score += 1

                    detailed_results.append(
                        {
                            "cau": idx,
                            "question": q["cau_hoi"],
                            "selected": selected_key,
                            "correct": correct_key,
                            "is_correct": is_correct,
                            "selected_text": options.get(selected_key, "Chưa chọn") if selected_key else "Chưa chọn",
                            "correct_text": options.get(correct_key, ""),
                        }
                    )

                st.session_state.quiz_submitted = True
                st.session_state.quiz_results = {
                    "score": score,
                    "total": total,
                    "details": detailed_results,
                }

        else:
            st.info("Không có câu hỏi trắc nghiệm.")

    # =========================
    # QUIZ RESULT
    # =========================
    if st.session_state.quiz_submitted and st.session_state.quiz_results:
        quiz_results = st.session_state.quiz_results

        st.subheader("Kết quả chấm bài")
        st.metric("Điểm số", f"{quiz_results['score']}/{quiz_results['total']}")

        for item in quiz_results["details"]:
            if item["is_correct"]:
                st.success(
                    f"Câu {item['cau']}: Đúng\n\n"
                    f"Bạn chọn: {item['selected']} - {item['selected_text']}"
                )
            else:
                st.error(
                    f"Câu {item['cau']}: Sai\n\n"
                    f"Bạn chọn: {item['selected'] or 'Chưa chọn'} - {item['selected_text']}\n\n"
                    f"Đáp án đúng: {item['correct']} - {item['correct_text']}"
                )

    # =========================
    # TOPIC SEARCH
    # =========================
    if result.get("ket_qua_tim_chu_de"):
        st.subheader("Kết quả tìm chủ đề")
        st.json(result["ket_qua_tim_chu_de"])

    # =========================
    # RAW JSON
    # =========================
    st.subheader("JSON đầu ra")
    result_json = json.dumps(result, ensure_ascii=False, indent=2)
    st.code(result_json, language="json")

    st.download_button(
        "⬇️ Tải JSON đầu ra",
        data=result_json.encode("utf-8"),
        file_name="learning_output.json",
        mime="application/json",
    )

    # =========================
    # FILES IF ANY
    # =========================
    if payload.get("files"):
        st.subheader("Tệp đã tạo")
        for label, path_str in payload["files"].items():
            path = Path(path_str)
            if path.exists():
                mime = (
                    "text/markdown"
                    if path.suffix == ".md"
                    else "text/html"
                    if path.suffix == ".html"
                    else "application/json"
                )
                st.download_button(
                    f"⬇️ Tải {label}: {path.name}",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime=mime,
                )
                st.caption(str(path.resolve()))

    # =========================
    # LOGS
    # =========================
    if payload.get("logs"):
        with st.expander("Nhật ký tool / workflow"):
            for log in payload["logs"]:
                st.write(f"- {log}")