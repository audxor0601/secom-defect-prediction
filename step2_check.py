import pandas as pd

X = pd.read_csv("raw/secom.data", sep=" ", header=None)
y = pd.read_csv("raw/secom_labels.data", sep=" ", header=None, names=["label", "time"])

print("제품 수, 센서 수:", X.shape)
print("합격(-1)/불량(1) 개수:", y["label"].value_counts().to_dict())
print("불량 비율:", round((y["label"] == 1).mean() * 100, 1), "%")

missing = X.isna().mean()
print("빈 값이 하나라도 있는 센서:", (missing > 0).sum())
print("빈 값이 50% 넘는 센서:", (missing > 0.5).sum())
print("값이 하나뿐인 센서:", (X.nunique() <= 1).sum())

t = pd.to_datetime(y["time"], format="%d/%m/%Y %H:%M:%S")
print("생산 기간:", t.min(), "~", t.max())
print("시간 순서대로 정렬돼 있나:", t.is_monotonic_increasing)
