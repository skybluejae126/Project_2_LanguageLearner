import numpy as np
import librosa
import torch
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean, cosine
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import matplotlib.pyplot as plt

# Wav2Vec2 모델 로드
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

# ============================
# 1. 오디오 로드 및 정규화
# ============================
def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    return y

def normalize_audio(y, target_dB=-20):
    """
    오디오 볼륨을 일정하게 정규화 (RMS 기준)
    """
    rms = np.sqrt(np.mean(y**2))
    scalar = 10**(target_dB / 20) / (rms + 1e-6)
    return y * scalar

# ============================
# 2. 특성 추출 (Wav2Vec2 사용)
# ============================
def extract_features(audio):
    """
    Wav2Vec2 모델을 사용하여 음성 임베딩을 추출
    """
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state
    return outputs.squeeze(0).numpy()

# ============================
# 3. DTW 거리 계산 (정규화 포함)
# ============================
def calculate_dtw_distance(ref_audio, test_audio):
    """
    MFCC 기반 DTW 거리 계산 (길이 정규화 포함)
    """
    ref_mfcc = librosa.feature.mfcc(y=ref_audio, sr=16000, n_mfcc=13)
    test_mfcc = librosa.feature.mfcc(y=test_audio, sr=16000, n_mfcc=13)
    
    distance, _ = fastdtw(ref_mfcc.T, test_mfcc.T, dist=euclidean)
    normalized_distance = distance / len(test_mfcc.T)  # 정규화
    return normalized_distance

# ============================
# 4. 코사인 유사도 계산 (구간별)
# ============================
def calculate_cosine_similarity(ref_audio, test_audio, num_segments=10):
    """
    Wav2Vec2 기반 음성 임베딩을 사용한 코사인 유사도 계산 (구간별)
    """
    ref_features = extract_features(ref_audio)
    test_features = extract_features(test_audio)
    
    segment_size = min(len(ref_features), len(test_features)) // num_segments
    cosine_similarities = []

    for i in range(num_segments):
        start = i * segment_size
        end = (i + 1) * segment_size
        if end > len(ref_features) or end > len(test_features):
            break
        
        ref_segment = ref_features[start:end].mean(axis=0)
        test_segment = test_features[start:end].mean(axis=0)
        
        similarity = 1 - cosine(ref_segment, test_segment)  # 코사인 유사도
        cosine_similarities.append(similarity)
    
    return np.mean(cosine_similarities), cosine_similarities

# ============================
# 5. 그래프 출력 (영어 라벨 유지)
# ============================
def plot_waveforms(ref_audio, test_audio):
    """
    원본 음성과 테스트 음성의 파형을 비교하는 그래프 출력
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(ref_audio, label="Reference Voice")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    plt.subplot(2, 1, 2)
    plt.plot(test_audio, label="Test Voice", color="orange")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    plt.show()

# ============================
# 6. 점수 계산 
# ============================
def score_calculate(dtw_score, cos_score):
    """
    DTW 거리와 코사인 유사도를 이용한 점수 도출출
    """
    calculated_score = (100 / (1 + (dtw_score / (3.5) ** (cos_score - 1)))) * cos_score * 100
    calculated_score = round(calculated_score, 2)
    if calculated_score < 50:
        calculated_score *= 0.6
    else:
        calculated_score *= 1.05
    calculated_score = round(calculated_score, 2)

    return calculated_score

# ============================
# 7. 실행 (오디오 로드 → 분석 → 출력)
# ============================
def main(ref_path, test_path):
    ref_audio = normalize_audio(load_audio(ref_path))
    test_audio = normalize_audio(load_audio(test_path))
    
    dtw_distance = calculate_dtw_distance(ref_audio, test_audio)
    cosine_avg, cosine_segments = calculate_cosine_similarity(ref_audio, test_audio)
    final_score = score_calculate(dtw_distance, cosine_avg)

    print(f"DTW Normalized Score: {dtw_distance}")
    print(f"Cosine Similarity (Average): {cosine_avg}")
    print(f"Final Score (100/100) : {final_score}")
    
    plot_waveforms(ref_audio, test_audio)

if __name__ == "__main__":
    main("RVC_output/rvc_output.wav", "User_voice/myvoice.wav")
