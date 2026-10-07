import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

print(" 재학습 시작")

# ---------------------------------------------------------
# 1. incoming 데이터 로드
# ---------------------------------------------------------
incoming = np.load("incoming.npy")
X = incoming.reshape(-1, 1)           # 2차원으로 변환
y = (X[:,0] > 0).astype(int)          # threshold 기반 라벨 생성

# ---------------------------------------------------------
# 2. 단일 클래스 여부 체크
# ---------------------------------------------------------
unique_classes = np.unique(y)
print("incoming 데이터 클래스:", unique_classes)

if len(unique_classes) < 2:
    print("단일 클래스 → 재학습 스킵")
    exit(0)

# ---------------------------------------------------------
# 3. 모델 재학습
# ---------------------------------------------------------
model = LogisticRegression()
model.fit(X, y)

# ---------------------------------------------------------
# 4. 새로운 모델 저장
# ---------------------------------------------------------
joblib.dump(model, "model.pkl")

print(" 재학습 완료 → model.pkl 업데이트됨")
