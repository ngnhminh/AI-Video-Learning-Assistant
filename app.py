from __future__ import annotations

import json
import mimetypes
import os
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st
from pymongo import MongoClient

from agent import GeminiVideoAgent

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "ai_learning")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "video_results")

st.set_page_config(
    page_title="AI Video Learning Assistant",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 AI Video Learning Assistant")
st.caption("Gemini API + Structured Output + Streamlit")


# =========================
# DB HELPERS
# =========================
def get_mongo_collection():
    if not MONGO_URI:
        return None
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    return db[MONGO_COLLECTION]


def load_saved_lessons(limit: int = 20):
    collection = get_mongo_collection()
    if collection is None:
        return []

    docs = list(
        collection.find({}, {"data": 1, "created_at": 1})
        .sort("created_at", -1)
        .limit(limit)
    )
    return docs


def format_lesson_label(doc: dict) -> str:
    data = doc.get("data", {})
    topic = data.get("chu_de", "Không rõ chủ đề")
    created_at = doc.get("created_at")
    created_str = str(created_at) if created_at else "Không rõ thời gian"
    return f"{topic} | {created_str}"


def render_topic_result(topic_result: dict):
    st.subheader("Kết quả tìm chủ đề")

    st.write(f"**Chủ đề truy vấn:** {topic_result.get('chu_de', 'Không có')}")
    st.write(
        f"**Có tìm thấy trong transcript:** {'Có' if topic_result.get('co_tim_thay') else 'Không'}"
    )
    st.write(f"**Mốc thời gian:** {topic_result.get('moc_thoi_gian') or 'Không có'}")

    doan_lien_quan = topic_result.get("doan_lien_quan")
    st.write("**Đoạn liên quan:**")
    if doan_lien_quan:
        st.info(doan_lien_quan)
    else:
        st.caption("Không có đoạn liên quan.")

    thong_tin_mo_rong = topic_result.get("thong_tin_mo_rong", [])
    if thong_tin_mo_rong:
        st.markdown("#### Thông tin mở rộng")
        for idx, item in enumerate(thong_tin_mo_rong, start=1):
            st.write(f"{idx}. {item}")

    nguon_wikipedia = topic_result.get("nguon_wikipedia", [])
    if nguon_wikipedia:
        st.markdown("#### Nguồn Wikipedia")
        for idx, src in enumerate(nguon_wikipedia, start=1):
            title = src.get("title", "Không có tiêu đề")
            url = src.get("url", "")
            snippet = src.get("snippet", "")

            if url:
                st.markdown(f"**{idx}. [{title}]({url})**")
            else:
                st.markdown(f"**{idx}. {title}**")

            if snippet:
                st.caption(snippet)


def render_result(result: dict):
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

    if st.session_state.get("quiz_submitted") and st.session_state.get("quiz_results"):
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

    if result.get("ket_qua_tim_chu_de"):
        render_topic_result(result["ket_qua_tim_chu_de"])

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

    page_mode = st.radio(
        "Chế độ",
        ["Bài học mới", "Bài học đã lưu"],
        index=0,
    )


# =========================
# SAVED LESSONS MODE
# =========================
if page_mode == "Bài học đã lưu":
    st.subheader("📚 Bài học đã lưu trong database")

    saved_docs = load_saved_lessons()

    if not MONGO_URI:
        st.warning("Bạn chưa cấu hình MONGO_URI trong .env")
    elif not saved_docs:
        st.info("Chưa có bài học nào trong database.")
    else:
        options = {format_lesson_label(doc): doc for doc in saved_docs}
        selected_label = st.selectbox("Chọn bài học", list(options.keys()))
        selected_doc = options[selected_label]
        result = selected_doc.get("data", {})

        st.success("Đã tải bài học từ database")
        render_result(result)

    st.stop()


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
            "Hãy phân tích bài giảng này, tóm tắt ngắn gọn, trích xuất các ý chính quan trọng, tạo câu hỏi trắc nghiệm phù hợp và nếu có phần nói về Triều đại Âu Lạc thì cho biết nó được nhắc ở phút bao nhiêu ở video. Tìm thêm thông tin nền về chủ đề đó từ Wikipedia và lưu vào database."
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
- Chế độ **Bài học mới**: tải video và phân tích.
- Chế độ **Bài học đã lưu**: xem lại các bài đã lưu trong database.
- Kết quả trả về theo schema có cấu trúc.
- Phần quiz cho phép chọn đáp án và nộp bài.
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
    render_result(result)

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

    if payload.get("db_status"):
        status = payload["db_status"]
        if status.get("success"):
            st.success("Đã lưu vào database")
        else:
            st.error(status.get("message"))

    if payload.get("logs"):
        with st.expander("Nhật ký tool / workflow"):
            for log in payload["logs"]:
                st.write(f"- {log}")