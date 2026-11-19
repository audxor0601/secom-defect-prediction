import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import recall_score, confusion_matrix

X = pd.read_csv("raw/secom.data", sep=" ", header=None)
y = pd.read_csv("raw/secom_labels.data", sep=" ", header=None, names=["label", "time"])
y = (y["label"] == 1).astype(int)

cut = int(len(X) * 0.8)
X_train, X_test = X.iloc[:cut], X.iloc[cut:]
y_train, y_test = y.iloc[:cut], y.iloc[cut:]

keep = (X_train.isna().mean() <= 0.5) & (X_train.nunique() > 1)
X_train, X_test = X_train.loc[:, keep], X_test.loc[:, keep]

def new_model():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(class_weight="balanced", C=0.01, max_iter=5000))

val_y, val_prob = [], []
for tr, va in TimeSeriesSplit(n_splits=4).split(X_train):
    m = new_model().fit(X_train.iloc[tr], y_train.iloc[tr])
    val_prob += list(m.predict_proba(X_train.iloc[va])[:, 1])
    val_y += list(y_train.iloc[va])
val_y, val_prob = np.array(val_y), np.array(val_prob)

print("검증: 목표 재현율 → 기준선, 헛경보율")
chosen = {}
for target in [0.6, 0.7, 0.8, 0.9]:
    for t in np.arange(0.99, 0.0, -0.01):
        if recall_score(val_y, val_prob >= t) >= target:
            break
    false_alarm = (val_prob[val_y == 0] >= t).mean()
    chosen[target] = t
    print(f"  {target:.0%} → {t:.2f}, 헛경보 {false_alarm:.0%}")

model = new_model().fit(X_train, y_train)
prob = model.predict_proba(X_test)[:, 1]
print("시험:")
for target, t in chosen.items():
    pred = (prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    print(f"  목표 {target:.0%}: 불량 {tp}/{tp+fn} 잡음, 헛경보 {fp}/{fp+tn} ({fp/(fp+tn):.0%})")
