# Lab 3: gửi video cho Gemini (inline hoặc Files API) và trích transcript có mốc thời gian
# Chạy: python labs\lab3_video_transcript.py [đường_dẫn_video] [inline|files]
import os
import sys
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

VIDEO = sys.argv[1] if len(sys.argv) > 1 else os.path.join("labs", "video_mau.mp4")
INLINE_LIMIT = 20 * 1024 * 1024                      # như agent.py: tối đa 20 MiB thì gửi inline
PROMPT = ("Chép lại lời nói trong video bằng tiếng Việt. "
          "Mỗi câu một dòng, bắt đầu bằng mốc thời gian [mm:ss].")

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
size = os.path.getsize(VIDEO)
mode = sys.argv[2] if len(sys.argv) > 2 else ("inline" if size <= INLINE_LIMIT else "files")
print(f"Video: {VIDEO} ({size / 1024 / 1024:.1f} MiB) -> cách gửi: {mode}")

uploaded = None
if mode == "inline":
    video_part = types.Part.from_bytes(data=open(VIDEO, "rb").read(), mime_type="video/mp4")
else:
    uploaded = client.files.upload(file=VIDEO)
    while uploaded.state.name == "PROCESSING":       # chờ Files API xử lý xong video
        time.sleep(3)
        uploaded = client.files.get(name=uploaded.name)
    print("Files API:", uploaded.name, "| trạng thái:", uploaded.state.name)
    video_part = types.Part.from_uri(file_uri=uploaded.uri, mime_type=uploaded.mime_type)

t0 = time.time()
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[video_part, PROMPT],
    config=types.GenerateContentConfig(
        temperature=0, thinking_config=types.ThinkingConfig(thinking_budget=0)),
)
print(f"Thời gian gọi model: {time.time() - t0:.1f} giây")
lines = [x for x in response.text.strip().splitlines() if x.strip()]
print(f"Transcript có {len(lines)} dòng. 5 dòng đầu:")
print("\n".join(lines[:5]))
for d in response.usage_metadata.prompt_tokens_details:
    print("Token đầu vào", d.modality.name, d.token_count)
if uploaded:
    client.files.delete(name=uploaded.name)          # xóa tệp đã tải lên sau khi dùng
    print("Đã xóa", uploaded.name)
