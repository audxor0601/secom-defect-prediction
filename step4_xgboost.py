import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, average_precision_score, confusion_matrix

X = pd.read_csv("raw/secom.data", sep=" ", header=None)
y = pd.read_csv("raw/secom_labels.data", sep=" ", header=None, names=["label", "time"])
y = (y["label"] == 1).astype(int)

cut = int(len(X) * 0.8)
X_train, X_test = X.iloc[:cut], X.iloc[cut:]
y_train, y_test = y.iloc[:cut], y.iloc[cut:]

keep = (X_train.isna().mean() <= 0.5) & (X_train.nunique() > 1)
X_train, X_test = X_train.loc[:, keep], X_test.loc[:, keep]

ratio = (y_train == 0).sum() / (y_train == 1).sum()
model = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.5,
                      scale_pos_weight=ratio, random_state=42)
model.fit(X_train, y_train)
prob = model.predict_proba(X_test)[:, 1]
pred = (prob >= 0.5).astype(int)

print("불량 가중치:", round(ratio, 1))
print("재현율:", round(recall_score(y_test, pred), 2))
print("PR-AUC:", round(average_precision_score(y_test, prob), 3))
print(confusion_matrix(y_test, pred))
