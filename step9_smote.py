import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import recall_score, average_precision_score, confusion_matrix
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE

X = pd.read_csv("raw/secom.data", sep=" ", header=None)
y = pd.read_csv("raw/secom_labels.data", sep=" ", header=None, names=["label", "time"])
y = (y["label"] == 1).astype(int)

cut = int(len(X) * 0.8)
X_train, X_test = X.iloc[:cut], X.iloc[cut:]
y_train, y_test = y.iloc[:cut], y.iloc[cut:]

keep = (X_train.isna().mean() <= 0.5) & (X_train.nunique() > 1)
X_train, X_test = X_train.loc[:, keep], X_test.loc[:, keep]

# imblearn의 파이프라인은 SMOTE를 학습(fit)할 때만 쓰고, 예측할 때는 건너뛴다
METHODS = {
    "처리 없음": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                   LogisticRegression(C=0.01, max_iter=5000)),
    "class_weight": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                          LogisticRegression(class_weight="balanced", C=0.01, max_iter=5000)),
    "SMOTE": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), SMOTE(random_state=42),
                                   LogisticRegression(C=0.01, max_iter=5000)),
}

rows = []
for name, new_model in METHODS.items():
    val_y, val_prob = [], []
    for tr, va in TimeSeriesSplit(n_splits=4).split(X_train):
        m = new_model().fit(X_train.iloc[tr], y_train.iloc[tr])
        val_prob += list(m.predict_proba(X_train.iloc[va])[:, 1])
        val_y += list(y_train.iloc[va])
    val_y, val_prob = np.array(val_y), np.array(val_prob)

    for t in np.arange(0.99, 0.0, -0.01):
        if recall_score(val_y, val_prob >= t) >= 0.9:
            break

    model = new_model().fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, prob >= t).ravel()
    rows.append({"방법": name, "PR-AUC": round(average_precision_score(y_test, prob), 3), "기준선": round(t, 2),
                 "불량 잡음": f"{tp}/{tp + fn}", "헛경보율": f"{fp / (fp + tn):.0%}"})

print("시험 데이터 (기준선은 검증 데이터에서 목표 재현율 90%로 골랐음)")
print(pd.DataFrame(rows).to_string(index=False))
