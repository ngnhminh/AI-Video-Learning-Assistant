# Hands-on lab: thực hành Gemini API

Năm bài thực hành đi kèm Chương 9 của báo cáo nhóm 17. Mỗi bài là một tệp Python độc lập.

## Chuẩn bị

1. Cài môi trường như mục 3.2 của báo cáo (`.venv`, `pip install -r requirements.txt`).
2. Tạo tệp `.env` ở thư mục gốc dự án, chứa `GEMINI_API_KEY=<khóa của bạn>`.
3. Tạo video mẫu 60 giây (bài 3 và 5 cần):

```
ffmpeg -ss 0 -t 60 -i "<video bài giảng>.mp4" -c copy labs\video_mau.mp4
```

## Chạy (từ thư mục gốc dự án)

| Bài | Lệnh | Nội dung |
|---|---|---|
| 1 | `.venv\Scripts\python.exe labs\lab1_goi_api.py` | Gọi API, cấu hình sinh, đọc usage |
| 2 | `.venv\Scripts\python.exe labs\lab2_structured_output.py` | JSON theo schema Pydantic |
| 3 | `.venv\Scripts\python.exe labs\lab3_video_transcript.py` | Gửi video inline, trích transcript |
| 3 | `.venv\Scripts\python.exe labs\lab3_video_transcript.py labs\video_mau.mp4 files` | Gửi video qua Files API |
| 4 | `.venv\Scripts\python.exe labs\lab4_function_calling.py` | Function calling thủ công |
| 5 | `.venv\Scripts\python.exe labs\lab5_token_chi_phi.py` | Đếm token và tính chi phí |

Chạy cả năm bài với video 60 giây tốn dưới 0,03 USD (gemini-2.5-flash, giá tra cứu ngày 24/10/2026).
