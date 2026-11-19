import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

plt.rcParams["font.family"] = "Noto Sans CJK KR"

X = pd.read_csv("raw/secom.data", sep=" ", header=None)
y = pd.read_csv("raw/secom_labels.data", sep=" ", header=None, names=["label", "time"])
y = (y["label"] == 1).astype(int)

cut = int(len(X) * 0.8)
X_train, y_train = X.iloc[:cut], y.iloc[:cut]
keep = (X_train.isna().mean() <= 0.5) & (X_train.nunique() > 1)
X_train = X_train.loc[:, keep]

model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      LogisticRegression(class_weight="balanced", C=0.01, max_iter=5000))
model.fit(X_train, y_train)

coef = pd.Series(model[-1].coef_[0], index=[f"센서 {c}" for c in X_train.columns])
top = coef.reindex(coef.abs().sort_values(ascending=False).index)[:10]
print(top.round(3))

top = top[::-1]
colors = ["#d62728" if v > 0 else "#1f77b4" for v in top]
plt.figure(figsize=(8, 5))
plt.barh(top.index, top.values, color=colors)
plt.axvline(0, color="gray", linewidth=0.8)
plt.title("불량에 영향이 큰 센서 Top 10 (로지스틱 회귀 계수)")
plt.xlabel("빨강: 값이 클수록 불량 위험 ↑   /   파랑: 값이 클수록 불량 위험 ↓")
plt.tight_layout()
plt.savefig("top10_sensors.png", dpi=150)
