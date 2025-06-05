import os, subprocess
from youtube_transcript_api import YouTubeTranscriptApi
from groq import Groq
from urllib.parse import urlparse, parse_qs
import io
from google.cloud import storage
from io import BytesIO

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/hojin/Downloads/gabolgido-8f7f3309efa3.json"

#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/gynovzs/gcs_keys/gabolgido-8f7f3309efa3.json"

# Groq API 설정
api_key = os.getenv('GROQ_API_KEY')
    
def extract_video_id(url):
    parsed_url = urlparse(url)
    if parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]
    elif 'youtube.com' in parsed_url.hostname:
        if parsed_url.path.startswith("/watch"):
            return parse_qs(parsed_url.query)['v'][0]
        elif parsed_url.path.startswith("/shorts/"):
            return parsed_url.path.split("/")[2]
    raise ValueError("올바르지 않은 YouTube URL입니다.")

def get_youtube_caption(video_id, language='ko'):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[language, 'en'])
        texts = [entry['text'].strip() for entry in transcript]
        return " ".join(texts)
    except Exception as e:
        print("❌ 자막 불러오기 실패:", str(e))
        return None

def format_timestamp(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 100)
    return f"{m:02}:{s:02}.{ms:02}"

def stream_audio_from_gcs_to_groq(bucket_name, blob_path):
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    # 바이너리 형태로 메모리에 로드
    audio_bytes = io.BytesIO()
    blob.download_to_file(audio_bytes)
    audio_bytes.seek(0)

    return audio_bytes

def stream_audio_to_gcs(url, bucket_name, destination_blob_name):
    try:
        # yt-dlp + ffmpeg 조합으로 오디오 추출 → stdout으로 받기
        cmd = [
            "yt-dlp",
            "-f", "bestaudio",
            "--extract-audio",
            "--audio-format", "mp3",
            "--audio-quality", "0",
            "--output", "-",  # stdout으로 출력
            "--quiet",
            url
        ]

        # subprocess에서 stdout으로 pipe
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # mp3 바이트를 메모리로 읽어들임
        audio_data = BytesIO()
        while True:
            chunk = process.stdout.read(1024)
            if not chunk:
                break
            audio_data.write(chunk)

        audio_data.seek(0)  # 스트림 포인터 처음으로 이동

        # GCS에 업로드
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_file(audio_data, content_type="audio/mpeg")
        print(f"✅ 메모리에서 GCS로 업로드 완료: gs://{bucket_name}/{destination_blob_name}")

    except Exception as e:
        print("❌ 업로드 실패:", str(e))

def transcribe_with_groq(audio_bytes):
    try:
        client = Groq(api_key=api_key)
        audio_file = audio_bytes
        audio_file.name = "audio.mp3"
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3-turbo",
            response_format="verbose_json",
            language="ko"
        )
        segments = transcription.model_dump().get("segments", [])
        texts = [segment['text'].strip() for segment in segments]
        return " ".join(texts)
    
    except Exception as e:
        print(f"❌ Groq 전사 실패: {str(e)}")
        return None

def get_subtitles_or_transcribe(url, language='ko'):
    
    video_id = extract_video_id(url)
    # print("🔎 자막 추출 시도 중...")
    captions = get_youtube_caption(video_id, language)
    if captions: # print("✅ 자체 자막이 존재합니다.")

        print('STT 코드->자체 자막:{}'.format(captions))

        return captions.split()

    else: # print("❌ 자체 자막이 없다면, Groq Whisper로 음성 인식 중...")
        
        stream_audio_to_gcs(url, 'gaboljido_1', 'tmp/tmp_audio.mp3')
        audio_bytes = stream_audio_from_gcs_to_groq('gaboljido_1', 'tmp/tmp_audio.mp3')

        if not audio_bytes:
            print("❌ 오디오 다운로드 실패로 전사 불가")
            return None

        transcription = transcribe_with_groq(audio_bytes)

        print('STT 코드->STT:{}'.format(transcription))

        return transcription.split()

def save_transcription(text, output_file="transcription.txt"):
    if text:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"✅ 텍스트가 {output_file}에 저장되었습니다.")
    else:
        print("❌ 저장할 텍스트가 없습니다.")



# main: get_subtitles_or_transcribe(url, language='ko')

# main: get_subtitles_or_transcribe(url, language='ko')

