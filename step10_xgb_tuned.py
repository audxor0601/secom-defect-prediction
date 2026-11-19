from itertools import product

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import recall_score, average_precision_score, confusion_matrix

X = pd.read_csv("raw/secom.data", sep=" ", header=None)
y = pd.read_csv("raw/secom_labels.data", sep=" ", header=None, names=["label", "time"])
y = (y["label"] == 1).astype(int)

cut = int(len(X) * 0.8)
X_train, X_test = X.iloc[:cut], X.iloc[cut:]
y_train, y_test = y.iloc[:cut], y.iloc[cut:]

keep = (X_train.isna().mean() <= 0.5) & (X_train.nunique() > 1)
X_train, X_test = X_train.loc[:, keep], X_test.loc[:, keep]

ratio = (y_train == 0).sum() / (y_train == 1).sum()  # 학습 데이터의 합격 수 ÷ 불량 수

def validate(new_model):
    """학습 데이터 안의 시간순 교차검증으로만 확률을 모은다. 시험 데이터는 안 본다."""
    val_y, val_prob = [], []
    for tr, va in TimeSeriesSplit(n_splits=4).split(X_train):
        m = new_model().fit(X_train.iloc[tr], y_train.iloc[tr])
        val_prob += list(m.predict_proba(X_train.iloc[va])[:, 1])
        val_y += list(y_train.iloc[va])
    return np.array(val_y), np.array(val_prob)

def logistic():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(class_weight="balanced", C=0.01, max_iter=5000))

def xgb(params):
    # XGBoost는 빈 값을 스스로 처리하고 단위 차이에 영향을 안 받아서, step4처럼 전처리 없이 쓴다
    return lambda: XGBClassifier(**params, subsample=0.8, scale_pos_weight=ratio, random_state=42)

# 1. 튜닝: 후보 24개를 검증 PR-AUC로 비교
grid = []
for depth, n_est, lr, col in product([2, 3, 4], [100, 300], [0.05, 0.1], [0.3, 0.6]):
    params = {"max_depth": depth, "n_estimators": n_est, "learning_rate": lr, "colsample_bytree": col}
    val_y, val_prob = validate(xgb(params))
    grid.append({**params, "검증 PR-AUC": round(average_precision_score(val_y, val_prob), 3)})
grid = pd.DataFrame(grid).sort_values("검증 PR-AUC", ascending=False, kind="stable")
best = grid.iloc[0].drop("검증 PR-AUC").to_dict()
best = {k: (int(v) if k in ("max_depth", "n_estimators") else float(v)) for k, v in best.items()}

print(f"불량 가중치(scale_pos_weight): {ratio:.1f}")
print("검증 PR-AUC 상위 5개 설정 (후보", len(grid), "개 중)")
print(grid.head(5).to_string(index=False))

# 2. 최종 비교: 기준선은 검증에서 목표 재현율 90%로, 시험 데이터는 여기서 처음 쓴다
rows = []
for name, new_model in [("로지스틱 회귀", logistic), ("튜닝한 XGBoost", xgb(best))]:
    val_y, val_prob = validate(new_model)
    for t in np.arange(0.99, 0.0, -0.01):
        if recall_score(val_y, val_prob >= t) >= 0.9:
            break
    prob = new_model().fit(X_train, y_train).predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, prob >= t).ravel()
    rows.append({"모델": name, "검증 PR-AUC": round(average_precision_score(val_y, val_prob), 3),
                 "시험 PR-AUC": round(average_precision_score(y_test, prob), 3), "기준선": round(t, 2),
                 "불량 잡음": f"{tp}/{tp + fn}", "헛경보율": f"{fp / (fp + tn):.0%}"})

print()
print("최종 비교 (시험 데이터)")
print(pd.DataFrame(rows).to_string(index=False))
