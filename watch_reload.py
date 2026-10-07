"""
watch_reload.py
- model.pkl 파일의 수정시간(mtime)을 모니터링
- 변경되면 FastAPI 서버를 자동 재시작하여 새로운 모델 즉시 반영
"""

import os
import time
import subprocess

MODEL_FILE = "model.pkl"

# ---------------------------------------------------------
# model.pkl의 수정 시간 반환 함수
# ---------------------------------------------------------
def get_ts():
    return os.path.getmtime(MODEL_FILE)

print("model.pkl 변경 감지 시작")

last_ts = get_ts()

# ---------------------------------------------------------
# FastAPI 최초 실행
# ---------------------------------------------------------
process = subprocess.Popen([
    "uvicorn", "app:app",
    "--host", "0.0.0.0",
    "--port", "8000"
])

# ---------------------------------------------------------
# 파일 변경 감지 루프
# ---------------------------------------------------------
while True:
    time.sleep(1)
    new_ts = get_ts()

    if new_ts != last_ts:  # 변경됨
        print("model.pkl 변경 감지 → FastAPI 재시작")

        process.kill()
        process = subprocess.Popen([
            "uvicorn", "app:app",
            "--host", "0.0.0.0",
            "--port", "8000"
        ])

        last_ts = new_ts
        print(" FastAPI 재시작 완료 (새 모델 적용됨)")
