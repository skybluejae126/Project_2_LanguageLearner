import requests
import os
from rvc_python.infer import RVCInference

# ============================
# 설정 파일 (API URL 등)
# ============================
VOICEVOX_API_URL = "http://localhost:50021"
RVC_OUTPUT_DIR = "RVC_output"
os.makedirs(RVC_OUTPUT_DIR, exist_ok=True)
VOX_OUTPUT_DIR = "VOX_output"
os.makedirs(VOX_OUTPUT_DIR, exist_ok=True)

# ============================
# 1. TTS 생성 함수 (VoiceVox)
# ============================
def generate_tts(text, speaker=11, speed=0.7, output_path="VOX_output/tts_output.wav"):
    params = {"text": text, "speaker": speaker}
    res1 = requests.post(f"{VOICEVOX_API_URL}/audio_query", params=params)
    if res1.status_code != 200:
        raise Exception("TTS Audio Query 요청 실패")
    
    audio_query = res1.json()
    audio_query["speedScale"] = speed  # 속도 조절 추가

    res2 = requests.post(f"{VOICEVOX_API_URL}/synthesis", params=params, json=audio_query)
    if res2.status_code != 200:
        raise Exception("TTS Synthesis 요청 실패")
    
    with open(output_path, "wb") as f:
        f.write(res2.content)
    
    return output_path

# ============================
# 2. RVC 변환 함수
# ============================
def convert_voice_rvc(input_wav: str, output_wav: str, model_path: str, device="cuda:0"):
    rvc = RVCInference(device=device)  # RVC 모델 초기화
    rvc.load_model(model_path)  # 모델 로드
    rvc.infer_file(input_wav, output_wav)  # 음성 변환 실행

    print(f"✅ 변환 완료! 저장된 파일: {output_wav}")
    return output_wav  # 변환된 파일 경로 반환

# ============================
# 실행 코드 (TTS → RVC)
# ============================
if __name__ == "__main__":
    text = "明日は月曜日なので学校に行かなければなりません。"
    
    print("[1] TTS 생성 중...")
    tts_audio = generate_tts(text)
    print(f"TTS 저장 완료: {tts_audio}")
    
    print("[2] RVC 변환 중...")
    rvc_audio = convert_voice_rvc("VOX_output/tts_output.wav", "RVC_output/rvc_output.wav", "models/mitest.pth")
    print(f"RVC 변환 완료: {rvc_audio}")
