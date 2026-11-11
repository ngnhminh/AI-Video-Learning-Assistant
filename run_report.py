"""Video illustrations and printable evidence, with no additional model calls."""
import base64
import html
import json
import io
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def video_illustrations(directory, video_bytes):
    meta={"frames":[], "duration_s":None, "width":None, "height":None}
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        return {**meta, "note":"Chưa tạo được ảnh: cần ffmpeg và ffprobe trên PATH."}
    try:
        with tempfile.TemporaryDirectory(prefix="video_evidence_") as tmp:
            source=Path(tmp)/"input.mp4"
            source.write_bytes(video_bytes)
            flags={"creationflags":getattr(subprocess,"CREATE_NO_WINDOW",0)}
            info=json.loads(subprocess.check_output(["ffprobe","-v","error","-show_format","-show_streams","-of","json",str(source)],timeout=30,**flags))
            stream=next(s for s in info["streams"] if s.get("codec_type")=="video")
            duration=float(info["format"]["duration"])
            meta.update(duration_s=duration,width=stream["width"],height=stream["height"])
            for index,fraction in enumerate([.1,.5,.9],1):
                at=round(duration*fraction,3)
                name=f"video_frame_{index}.jpg"
                subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y","-ss",str(at),"-i",str(source),"-frames:v","1","-vf","scale=640:-2",str(directory/name)],check=True,timeout=30,capture_output=True,**flags)
                meta["frames"].append({"file":name,"at_s":at})
    except (OSError,ValueError,KeyError,StopIteration,subprocess.SubprocessError):
        meta["note"]="Chưa trích được đầy đủ khung hình; mở video gốc để kiểm tra."
    return meta


def load_review(directory):
    path=directory/"quality_review.json"
    if path.exists():return json.loads(path.read_text(encoding="utf-8"))
    return {"scope":"Chưa đánh giá nội dung với nguồn tham chiếu. Schema hợp lệ không chứng minh nội dung đúng.","rows":[]}


def request_rows(summary):
    return [{"#":c["call_id"],"Model":c.get("model_version") or c["model_requested"],
             "Trạng thái":c["status"],"Giây":c.get("latency_s"),
             "Input":(c.get("usage") or {}).get("prompt_token_count"),
             "Output":c.get("output_tokens_for_cost"),"Thinking":c.get("thinking_tokens_for_cost"),
             "USD ước tính":f"{c['estimated_standard_usd']:.8f}" if c["estimated_standard_usd"] is not None else "Chưa biết"} for c in summary["calls"]]


def evidence_zip(directory):
    buffer=io.BytesIO()
    names={"requests.json","run.json","payload.json","transcript.json","summary.json",
           "quality_review.json","schema_checks.json","report.html"}
    with ZipFile(buffer,"w",ZIP_DEFLATED) as archive:
        for path in sorted(directory.iterdir()):
            if path.is_file() and (path.name in names or re.fullmatch(r"(?:request|response)_\d+\.json|video_frame_\d+\.jpg",path.name)):
                archive.write(path,path.name)
    return buffer.getvalue()


def table(rows):
    if not rows:return ""
    esc=lambda x:html.escape(str(x) if x is not None else "Chưa biết")
    heads=list(dict.fromkeys(k for row in rows for k in row))
    return "<table><thead><tr>"+"".join(f"<th>{esc(k)}</th>" for k in heads)+"</tr></thead><tbody>"+"".join("<tr>"+"".join(f"<td>{esc(row.get(k))}</td>" for k in heads)+"</tr>" for row in rows)+"</tbody></table>"


def build_html(payload, directory):
    esc=lambda x:html.escape(str(x))
    summary=payload["measurement_summary"]
    run=payload["run_info"]
    result=payload["result"]
    review=load_review(directory)
    cost=summary["estimated_standard_usd"]
    fragments=["<!doctype html><html lang='vi'><meta charset='utf-8'><title>Minh chứng chạy video</title>",
        "<style>body{font:15px Arial,sans-serif;line-height:1.45;color:#202124;background:white;max-width:1050px;margin:32px auto;padding:16px}h1,h2{color:#172b4d}table{border-collapse:collapse;width:100%;margin:16px 0}td,th{border:1px solid #cbd5e1;padding:7px;text-align:left}th{background:#eef2f6}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f8fa;padding:12px;font-size:12px}.frames{display:flex;gap:12px}.frames figure{margin:0;width:32%}.frames img{width:100%}.note{color:#52606d}tr,figure{break-inside:avoid}@media print{body{margin:0;padding:0}button{display:none}thead{display:table-header-group}}</style>",
        "<button onclick='window.print()'>In báo cáo</button><h1>Minh chứng chạy video</h1>",
        f"<p>Mã lượt: {esc(run['run_id'])}<br>Video: {esc(run['video_name'])}<br>Schema: {'Hợp lệ' if payload.get('schema_valid') else 'Chưa đạt'}</p>",
        f"<p><b>Backend: {run['elapsed_s']:.4f} giây · {summary['requests']} request · Chi phí token ước tính: {f'{cost:.8f} USD' if cost is not None else 'Chưa biết'}</b></p>",
        f"<p>Input: {summary['total_prompt_tokens']} (gồm {summary.get('total_cached_tokens',0)} cache) · Output: {summary['total_output_tokens']} · Thinking: {summary['total_thinking_tokens']}</p>",
        "<h2>Hình ảnh từ video đầu vào</h2><div class='frames'>"]
    for frame in payload.get("video_preview",{}).get("frames",[]):
        data=base64.b64encode((directory/frame["file"]).read_bytes()).decode()
        fragments.append(f"<figure><img src='data:image/jpeg;base64,{data}'><figcaption>Giây {frame['at_s']:.3f} của clip</figcaption></figure>")
    fragments.extend(["</div>",f"<p>{esc(payload.get('video_preview',{}).get('note','Khung hình trích tự động; không phải kiểm chứng độ chính xác timestamp của transcript.'))}</p>",
        "<h2>Cấu hình từng request</h2>",table([{"#":c["call_id"],"Model":c["model_requested"],**c.get("generation_config",{})} for c in summary["calls"]]),
        "<h2>Usage thực tế và độ trễ</h2>",table(request_rows(summary)),
        "<h2>Đơn giá và phép tính</h2>",table([{"Model":model,**rates} for model,rates in summary["prices_per_million_tokens"].items()]),
        f"<p>Đơn vị: USD/1 triệu token; kiểm tra {esc(summary['price_checked'])}. <a href='{esc(summary['price_source'])}'>Bảng giá Google</a></p>"])
    for call in summary["calls"]:
        fragments.append(f"<p><b>Request {call['call_id']}</b></p><pre>{esc(call['cost_formula'])}</pre>")
    if cost is not None:
        fragments.append("<pre>Tổng = "+" + ".join(f"{c['estimated_standard_usd']:.8f}" for c in summary['calls'])+f" = {cost:.8f} USD</pre>")
    fragments.extend([f"<p class='note'>{esc(summary['note'])}</p>","<h2>Kiểm tra chất lượng</h2>",
        f"<p>Schema: hợp lệ; {len(result['tom_tat'])} ý tóm tắt, {len(result['y_chinh'])} ý chính, {len(result['cau_hoi_trac_nghiem'])} câu hỏi.</p>",
        f"<p>{esc(review['scope'])}</p>",table(review.get("rows",[])),
        "<h2>Kết quả học liệu</h2><pre>",esc(json.dumps(result,ensure_ascii=False,indent=2)),"</pre>",
        "<h2>Transcript</h2><pre>",esc(payload.get("transcript","")),"</pre>",
        "<h2>Trace công cụ</h2><pre>",esc(json.dumps(payload.get("workflow_logs",[]),ensure_ascii=False,indent=2)),"</pre></html>"])
    content="\n".join(fragments)
    (directory/"report.html").write_text(content,encoding="utf-8")
    return content


def render_measurements(payload):
    import streamlit as st
    from summarize_evidence import summarize_calls
    directory=Path(payload["evidence_dir"])
    run=payload.get("run_info") or json.loads((directory/"run.json").read_text(encoding="utf-8"))
    summary=summarize_calls(payload["request_metrics"],directory.name,run.get("elapsed_s"))
    payload={**payload,"run_info":run,"measurement_summary":summary}
    st.subheader("Thông số đo của video")
    st.caption(f"Lượt {directory.name} · {run.get('video_name','')} · Schema: hợp lệ sau khi ghép dữ liệu")
    top=st.columns(3)
    top[0].metric("Thời gian backend",f"{run['elapsed_s']:.2f} s")
    top[1].metric("Request thành công",f"{summary['successful_requests']}/{summary['requests']}")
    cost=summary["estimated_standard_usd"]
    top[2].metric("Chi phí token ước tính",f"${cost:.8f}" if cost is not None else "Chưa biết")
    bottom=st.columns(3)
    for col,label,key in zip(bottom,["Input token","Output token","Thinking token"],["total_prompt_tokens","total_output_tokens","total_thinking_tokens"]):
        col.metric(label,f"{summary[key]:,}" if summary[key] is not None else "Chưa đủ usage")
    st.caption("Backend đo đến khi tạo xong học liệu, chưa gồm tạo ảnh và báo cáo. Output và thinking được tính chung theo đơn giá output.")
    st.caption(f"Input đã bao gồm {summary.get('total_cached_tokens',0)} token cache; cache không được cộng thêm lần nữa.")
    preview=payload.get("video_preview",{})
    frames=preview.get("frames",[])
    if frames:
        st.image([str(directory/f["file"]) for f in frames],caption=[f"Khung hình tại {f['at_s']:.2f} giây" for f in frames],width=260)
    else:st.info(preview.get("note","Chưa có ảnh minh họa cho lượt cũ."))
    st.subheader("Usage và chi phí từng request")
    st.table(request_rows(summary))
    st.subheader("Đơn giá và phép tính chi phí")
    st.caption(f"USD/1 triệu token · Standard trả phí · Kiểm tra {summary['price_checked']}")
    st.table([{"Model":m,"Văn bản/hình/video":p['input'],"Âm thanh":p['audio'],"Output + thinking":p['output'],"Cache văn bản":p['cache']} for m,p in summary['prices_per_million_tokens'].items()])
    st.write("Chi phí request = Σ(token từng loại × đơn giá) / 1.000.000; chi phí video = tổng chi phí các request.")
    for call in summary["calls"]:
        st.caption(f"Request {call['call_id']} · {call.get('model_version') or call['model_requested']}")
        st.code(call["cost_formula"],language=None,wrap_lines=True)
        if call.get("output_count_basis","")!="usage_metadata":st.caption(call.get("output_count_basis") or call.get("cost_note"))
    if cost is not None:
        st.code("Tổng = "+" + ".join(f"{c['estimated_standard_usd']:.8f}" for c in summary["calls"])+f" = {cost:.8f} USD",language=None,wrap_lines=True)
    st.caption(summary["note"])
    st.markdown(f"[Nguồn đơn giá Google]({summary['price_source']})")
    st.subheader("Cấu hình đã dùng")
    st.table([{"#":c["call_id"],"Model":c["model_requested"],**c.get("generation_config",{})} for c in summary["calls"]])
    st.subheader("Kiểm tra chất lượng đầu ra")
    result=payload["result"]
    st.write(f"Schema hợp lệ · {len(result['tom_tat'])} ý tóm tắt · {len(result['y_chinh'])} ý chính · {len(result['cau_hoi_trac_nghiem'])} câu trắc nghiệm")
    review=load_review(directory)
    st.caption(review["scope"])
    if review.get("rows"):st.table(review["rows"])
    st.button("Làm mới đánh giá",help="Đọc lại đánh giá đã lưu; không gọi Gemini.")
    report=build_html(payload,directory)
    st.download_button("Tải báo cáo để in (HTML kèm hình ảnh)",report.encode("utf-8"),file_name=f"{directory.name}.html",mime="text/html")
    st.download_button("Tải thông số và phép tính (JSON)",json.dumps(summary,ensure_ascii=False,indent=2).encode("utf-8"),file_name="measurement_summary.json",mime="application/json")
    st.download_button("Tải trace và kết quả",json.dumps(payload,ensure_ascii=False,indent=2).encode("utf-8"),file_name="run_evidence.json",mime="application/json")
    st.download_button("Tải toàn bộ request response và hình ảnh (ZIP)",evidence_zip(directory),file_name=f"{directory.name}_evidence.zip",mime="application/zip")
