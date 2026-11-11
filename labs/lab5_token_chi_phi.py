# Lab 5: đếm token trước khi gửi video và tính chi phí từ usage_metadata sau khi gọi
import os
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import types

# USD cho 1 triệu token, gemini-2.5-flash, bảng giá Standard tra cứu ngày 24/10/2026
GIA = {"input": 0.30, "audio": 1.00, "output": 2.50}
VIDEO = sys.argv[1] if len(sys.argv) > 1 else os.path.join("labs", "video_mau.mp4")
MODEL = "gemini-2.5-flash"

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
noi_dung = [types.Part.from_bytes(data=open(VIDEO, "rb").read(), mime_type="video/mp4"),
            "Tóm tắt video trong 3 gạch đầu dòng bằng tiếng Việt."]

truoc = client.models.count_tokens(model=MODEL, contents=noi_dung)
print("Đếm trước khi gửi:", truoc.total_tokens, "token đầu vào")

response = client.models.generate_content(
    model=MODEL, contents=noi_dung,
    config=types.GenerateContentConfig(
        temperature=0, thinking_config=types.ThinkingConfig(thinking_budget=0)),
)
print(response.text.strip())

u = response.usage_metadata
chi_phi = 0.0
for d in u.prompt_tokens_details:                    # tách token đầu vào theo loại dữ liệu
    gia = GIA["audio"] if d.modality.name == "AUDIO" else GIA["input"]
    chi_phi += d.token_count * gia / 1_000_000
    print(f"{d.modality.name:7s}{d.token_count:7d} token x {gia:.2f} USD/1M")
# token đầu ra gồm cả token suy nghĩ (nếu có)
ra = (u.candidates_token_count or 0) + (u.thoughts_token_count or 0)
chi_phi += ra * GIA["output"] / 1_000_000
print(f"{'OUTPUT':7s}{ra:7d} token x {GIA['output']:.2f} USD/1M")
print(f"Chi phí lượt gọi: {chi_phi:.6f} USD")
print(f"Nếu xử lý 100 video như vậy mỗi ngày: {chi_phi * 100:.4f} USD/ngày")
