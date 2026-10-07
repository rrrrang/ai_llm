import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 1. 학습 데이터 생성
# ---------------------------------------------------------
# 정규분포를 따르는 실수 데이터 200개 생성
X = np.random.randn(200,1)

# threshold(0)를 기준으로 이진 라벨 생성
y = (X[:,0] > 0).astype(int)

# ---------------------------------------------------------
# 2. 모델 학습
# ---------------------------------------------------------
model = RandomForestClassifier()
model.fit(X, y)

# ---------------------------------------------------------
# 3. 학습된 모델과 레퍼런스 데이터 저장
# ---------------------------------------------------------
joblib.dump(model, "model.pkl")      # 모델 저장
np.save("reference.npy", X)          # 학습 당시 데이터 분포 저장

print(" 모델 학습 완료: model.pkl / reference.npy 생성됨")

# ---------------------------------------------------------
# 4. 학습 데이터 분포 시각화
# ---------------------------------------------------------
# 데이터 로드
reference = np.load("reference.npy")

# 분포 시각화
plt.figure(figsize=(10,5))

# 히스토그램 + KDE
sns.histplot(reference[:,0], bins=30, kde=True)

plt.title("Reference Data Distribution (Training Data)")
plt.xlabel("Feature Value")
plt.ylabel("Frequency")

plt.grid()
plt.show()
