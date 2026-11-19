import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, average_precision_score, confusion_matrix

X = pd.read_csv("raw/secom.data", sep=" ", header=None)
y = pd.read_csv("raw/secom_labels.data", sep=" ", header=None, names=["label", "time"])
y = (y["label"] == 1).astype(int)

cut = int(len(X) * 0.8)
X_train, X_test = X.iloc[:cut], X.iloc[cut:]
y_train, y_test = y.iloc[:cut], y.iloc[cut:]

keep = (X_train.isna().mean() <= 0.5) & (X_train.nunique() > 1)
X_train, X_test = X_train.loc[:, keep], X_test.loc[:, keep]
print("남은 센서:", keep.sum())

imputer = SimpleImputer(strategy="median").fit(X_train)
scaler = StandardScaler().fit(imputer.transform(X_train))
X_train = scaler.transform(imputer.transform(X_train))
X_test = scaler.transform(imputer.transform(X_test))

model = LogisticRegression(class_weight="balanced", C=0.01, max_iter=5000)
model.fit(X_train, y_train)
prob = model.predict_proba(X_test)[:, 1]
pred = (prob >= 0.5).astype(int)

print("시험 데이터 불량 수:", y_test.sum(), "/", len(y_test))
print("재현율:", round(recall_score(y_test, pred), 2))
print("PR-AUC:", round(average_precision_score(y_test, prob), 3), "(아무렇게나 찍으면", round(y_test.mean(), 3), ")")
print("혼동행렬 [[합격→합격, 합격→불량], [불량→합격, 불량→불량]]:")
print(confusion_matrix(y_test, pred))
