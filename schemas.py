from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class QuizOptions(BaseModel):
    A: str = Field(description="Phương án A")
    B: str = Field(description="Phương án B")
    C: str = Field(description="Phương án C")
    D: str = Field(description="Phương án D")


class QuizQuestion(BaseModel):
    cau_hoi: str = Field(description="Nội dung câu hỏi trắc nghiệm")
    lua_chon: QuizOptions = Field(description="4 phương án lựa chọn")
    dap_an_dung: str = Field(description="Đáp án đúng, chỉ nhận A/B/C/D")


class TopicSearchResult(BaseModel):
    chu_de: str = Field(description="Chủ đề được truy vấn")
    co_tim_thay: bool = Field(description="Có tìm thấy chủ đề trong nội dung hay không")
    moc_thoi_gian: Optional[str] = Field(
        default=None,
        description="Mốc thời gian theo định dạng MM:SS nếu có",
    )
    doan_lien_quan: Optional[str] = Field(
        default=None,
        description="Đoạn nội dung liên quan đến chủ đề nếu có",
    )


class LearningOutput(BaseModel):
    chu_de: str = Field(description="Chủ đề chính của bài giảng")
    tom_tat: List[str] = Field(
        min_length=3,
        max_length=8,
        description="Danh sách các câu tóm tắt nội dung",
    )
    y_chinh: List[str] = Field(
        min_length=3,
        max_length=8,
        description="Danh sách ý chính của bài giảng",
    )
    cau_hoi_trac_nghiem: List[QuizQuestion] = Field(
        min_length=1,
        max_length=10,
        description="Danh sách câu hỏi trắc nghiệm",
    )
    ket_qua_tim_chu_de: Optional[TopicSearchResult] = Field(
        default=None,
        description="Kết quả tìm kiếm theo chủ đề nếu người dùng có yêu cầu",
    )
    ghi_chu_agent: Optional[str] = Field(
        default=None,
        description="Ghi chú ngắn về các bước agent đã thực hiện",
    )
