"""
app.py
- FastAPI 실시간 예측 API
- 요청 데이터 누적 후 KS-test 기반 drift 감지
- drift 발생 시 incoming.npy 저장 및 자동 재학습 트리거
"""

from fastapi import FastAPI
import numpy as np
import joblib
from scipy.stats import ks_2samp
import subprocess

app = FastAPI()

# ---------------------------------------------------------
# 1. 모델과 기준(reference) 데이터 로드
# ---------------------------------------------------------
model = joblib.load("model.pkl")
reference = np.load("reference.npy").ravel()   # KS-test 용 1차원 데이터
incoming = []  # 운영 데이터 누적 리스트

# ---------------------------------------------------------
# 2. 예측 엔드포인트
# ---------------------------------------------------------
@app.get("/predict")
def predict(value: float):
    """
    입력된 value 값에 대해 예측을 수행하고,
    incoming 데이터를 누적하여 drift 여부를 반환한다.
    """

    incoming.append(value)

    drift_detected = False  # 기본값

    # ---------------------------------------------------------
    # Drift 감지: 50개 이상 쌓이면 KS-test 수행
    # ---------------------------------------------------------
    if len(incoming) > 50:
        stat, p = ks_2samp(reference, incoming)
        drift_detected = bool(p < 0.05)  # numpy.bool → python bool 변환

        if drift_detected:
            print(" Drift 감지 → incoming.npy 저장")
            np.save("incoming.npy", np.array(incoming))

            # 자동 재학습 프로세스 실행
            subprocess.Popen(["python3", "train_retrain.py"])

    # ---------------------------------------------------------
    # 모델 예측 수행
    # ---------------------------------------------------------
    pred = model.predict([[value]])  # 입력 shape(1,1) 유지 필수

    return {
        "value": value,
        "prediction": int(pred[0]),
        "drift_detected": drift_detected,
        "sample_size": len(incoming)
    }
