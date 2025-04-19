from model.model_stt.stt import extract_video_id, get_youtube_caption

def run_stt(url: str) -> list:
    video_id = extract_video_id(url)
    text = get_youtube_caption(video_id)
    return text.split()
