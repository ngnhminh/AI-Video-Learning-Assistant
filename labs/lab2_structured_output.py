# Lab 2: structured output - yêu cầu Gemini trả JSON đúng schema Pydantic, rồi kiểm tra lại
import os
from typing import List, Literal

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError


class CauHoi(BaseModel):
    cau_hoi: str
    lua_chon: List[str] = Field(min_length=4, max_length=4)
    dap_an_dung: Literal["A", "B", "C", "D"]


class BaiHoc(BaseModel):
    chu_de: str
    y_chinh: List[str] = Field(min_length=3, max_length=5)
    cau_hoi: List[CauHoi] = Field(min_length=2, max_length=2)


DOAN_VAN = ("Năm 938, Ngô Quyền đánh tan quân Nam Hán trên sông Bạch Đằng "
            "bằng trận địa cọc gỗ cắm dưới lòng sông. "
            "Chiến thắng này chấm dứt thời kỳ Bắc thuộc kéo dài hơn một nghìn năm. "
            "Năm 939, Ngô Quyền xưng vương và đóng đô ở Cổ Loa.")

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Tạo bài học ngắn bằng tiếng Việt từ đoạn văn sau:\n" + DOAN_VAN,
    config=types.GenerateContentConfig(
        response_mime_type="application/json",       # bắt buộc trả JSON
        response_schema=BaiHoc,                      # JSON phải theo schema này
        temperature=0,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    ),
)
print("JSON model trả về:\n" + response.text)
bai_hoc = BaiHoc.model_validate_json(response.text)  # kiểm tra lại phía chương trình
print("Hợp lệ. Chủ đề:", bai_hoc.chu_de)
print("Số ý chính:", len(bai_hoc.y_chinh), "| số câu hỏi:", len(bai_hoc.cau_hoi))

# JSON đúng cú pháp nhưng sai schema: chỉ 1 ý chính, không có câu hỏi
sai = '{"chu_de": "Bạch Đằng", "y_chinh": ["Năm 938"], "cau_hoi": []}'
try:
    BaiHoc.model_validate_json(sai)
except ValidationError as e:
    print("JSON sai schema bị chặn,", e.error_count(), "lỗi:")
    for err in e.errors():
        print(" -", err["loc"], err["msg"])
