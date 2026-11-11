# Lab 1: gọi Gemini API lần đầu, đặt cấu hình sinh và đọc thông tin token
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()                                        # nạp GEMINI_API_KEY từ tệp .env
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-2.5-flash"

prompt = "Giải thích Gemini API là gì trong đúng 3 câu, dành cho sinh viên năm 3."
dem = client.models.count_tokens(model=MODEL, contents=prompt)
print("Token của prompt (đếm trước khi gửi):", dem.total_tokens)

response = client.models.generate_content(
    model=MODEL,
    contents=prompt,
    config=types.GenerateContentConfig(
        system_instruction=("Bạn là trợ giảng môn Các công nghệ lập trình hiện đại. "
                            "Trả lời bằng tiếng Việt."),
        temperature=0.2,                             # thấp: câu trả lời ít ngẫu nhiên
        max_output_tokens=512,
        # tắt thinking vì câu hỏi đơn giản
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    ),
)

print("Trả lời:\n" + response.text.strip())
u = response.usage_metadata
print("model_version:", response.model_version)
print("finish_reason:", response.candidates[0].finish_reason.name)
print("prompt_token_count:", u.prompt_token_count)
print("candidates_token_count:", u.candidates_token_count)
print("total_token_count:", u.total_token_count)
