from model.model_stt.stt import get_subtitles_or_transcribe

def run_stt(url: str) -> list:
    texts = get_subtitles_or_transcribe(url)
    return texts
