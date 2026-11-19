import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import fbeta_score, recall_score, confusion_matrix

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

candidates = np.arange(0.05, 0.95, 0.05)
scores = [fbeta_score(val_y, (np.array(val_prob) >= t).astype(int), beta=2) for t in candidates]
best = candidates[int(np.argmax(scores))]
print("검증으로 고른 기준선:", round(best, 2))

model = new_model().fit(X_train, y_train)
prob = model.predict_proba(X_test)[:, 1]
for t in [0.5, best]:
    pred = (prob >= t).astype(int)
    print(f"기준선 {t:.2f} → 재현율 {recall_score(y_test, pred):.2f}")
    print(confusion_matrix(y_test, pred))
