# Lab 4: function calling thủ công - model chọn hàm, chương trình chạy hàm
# rồi gửi kết quả lại cho model
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

# transcript mẫu nhóm tự soạn cho bài lab (không lấy từ video thật)
TRANSCRIPT = {
    "[00:05]": "Văn Lang là nhà nước đầu tiên của người Việt.",
    "[00:20]": "Năm 938, Ngô Quyền đánh tan quân Nam Hán trên sông Bạch Đằng.",
    "[00:41]": "Năm 1010, Lý Thái Tổ dời đô về Thăng Long.",
}


def tim_moc_thoi_gian(tu_khoa: str) -> dict:
    """Hàm của chương trình: tìm các câu chứa từ khóa và trả về mốc thời gian."""
    hits = [{"moc": m, "cau": c} for m, c in TRANSCRIPT.items()
            if tu_khoa.lower() in c.lower()]
    return {"tu_khoa": tu_khoa, "ket_qua": hits}


khai_bao = types.FunctionDeclaration(                # mô tả hàm để model biết khi nào cần gọi
    name="tim_moc_thoi_gian",
    description="Tìm mốc thời gian trong transcript video theo từ khóa",
    parameters=types.Schema(type="OBJECT", properties={"tu_khoa": types.Schema(type="STRING")},
                            required=["tu_khoa"]),
)
config = types.GenerateContentConfig(
    tools=[types.Tool(function_declarations=[khai_bao])], temperature=0,
    thinking_config=types.ThinkingConfig(thinking_budget=0))

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
cau_hoi = "Trong video, đoạn nói về Bạch Đằng ở phút nào?"
history = [types.Content(role="user", parts=[types.Part(text=cau_hoi)])]

for vong in range(1, 4):                             # giới hạn số vòng để chắc chắn dừng
    response = client.models.generate_content(model="gemini-2.5-flash", contents=history,
                                              config=config)
    calls = response.function_calls or []
    if not calls:                                    # không gọi hàm nữa -> câu trả lời cuối
        print(f"Vòng {vong}: model trả lời -> {response.text.strip()}")
        break
    history.append(response.candidates[0].content)   # giữ lượt model yêu cầu gọi hàm
    for call in calls:
        ket_qua = tim_moc_thoi_gian(**call.args)     # chương trình tự chạy hàm
        print(f"Vòng {vong}: model gọi {call.name}({dict(call.args)})")
        print(f"         kết quả hàm: {ket_qua}")
        history.append(types.Content(role="user", parts=[
            types.Part.from_function_response(name=call.name, response=ket_qua)]))
