import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

installed = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = [f for f in ["Noto Sans CJK KR", "AppleGothic"] if f in installed] or ["sans-serif"]

X = pd.read_csv("raw/secom.data", sep=" ", header=None)
y = pd.read_csv("raw/secom_labels.data", sep=" ", header=None, names=["label", "time"])
y = (y["label"] == 1).astype(int)

cut = int(len(X) * 0.8)
X_train, X_test = X.iloc[:cut], X.iloc[cut:]
y_train, y_test = y.iloc[:cut].to_numpy(), y.iloc[cut:].to_numpy()

keep = (X_train.isna().mean() <= 0.5) & (X_train.nunique() > 1)
X_train, X_test = X_train.loc[:, keep], X_test.loc[:, keep]

def new_model():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(class_weight="balanced", C=0.01, max_iter=5000))

# 기준선 고르기는 학습 데이터 안의 시간순 교차검증 결과만 본다 (step6과 같음)
val_y, val_prob = [], []
for tr, va in TimeSeriesSplit(n_splits=4).split(X_train):
    m = new_model().fit(X_train.iloc[tr], y_train[tr])
    val_prob += list(m.predict_proba(X_train.iloc[va])[:, 1])
    val_y += list(y_train[va])
val_y, val_prob = np.array(val_y), np.array(val_prob)

# 후보: 0 = 전부 재검사, 1 = 재검사 안 함, 그 사이는 0.01 간격
thresholds = np.round(np.arange(0, 1.001, 0.01), 2)

def flag(prob, t):
    if t >= 1:
        return np.zeros(len(prob), dtype=bool)  # 재검사 안 함
    return prob >= t  # t = 0이면 전부 재검사

def cost(y_true, prob, t, c):
    f = flag(prob, t)
    false_alarm = (f & (y_true == 0)).sum()
    missed = (~f & (y_true == 1)).sum()
    return false_alarm * 1 + missed * c

COSTS = [5, 10, 20, 30, 50, 100]
chosen = {}
for c in COSTS:
    # 비용이 같으면 기준선이 높은 쪽(재검사가 적은 쪽)을 고른다
    chosen[c] = min(thresholds, key=lambda t: (cost(val_y, val_prob, t, c), -t))

model = new_model().fit(X_train, y_train)
prob = model.predict_proba(X_test)[:, 1]
n, n_bad = len(y_test), int(y_test.sum())

rows = []
for c, t in chosen.items():
    f = flag(prob, t)
    caught = int((f & (y_test == 1)).sum())
    total = cost(y_test, prob, t, c)
    all_check = cost(y_test, prob, 0, c)  # 전수 재검사 = 합격품 전부가 헛경보
    rows.append({"C": c, "기준선": t, "재검사 수": f"{int(f.sum())}/{n}", "잡은 불량": f"{caught}/{n_bad}",
                 "놓친 불량": n_bad - caught, "비용": int(total), "전수 재검사 대비": f"{total / all_check:.0%}",
                 "_recheck": int(f.sum())})
table = pd.DataFrame(rows)

# 자체 확인: 전수 재검사 비용은 합격품 수, 재검사 안 함 비용은 불량 수 × C
assert cost(y_test, prob, 0, 10) == n - n_bad
assert cost(y_test, prob, 1, 10) == n_bad * 10

print("시험 데이터 (기준선은 검증 데이터로만 골랐음)")
print(table.drop(columns="_recheck").to_string(index=False))

fig, ax1 = plt.subplots(figsize=(8, 5))
x = range(len(COSTS))
ax1.plot(x, table["_recheck"], marker="o", color="#1f77b4", label="재검사 수")
ax1.set_xlabel("비용 비율 C (불량 1개 놓친 손해 ÷ 재검사 1번 비용)")
ax1.set_ylabel(f"재검사 수 (시험 {n}개 중)", color="#1f77b4")
ax1.set_ylim(0, n)
ax1.set_xticks(list(x), [str(c) for c in COSTS])
ax2 = ax1.twinx()
ax2.plot(x, table["놓친 불량"], marker="s", color="#d62728", label="놓친 불량")
ax2.set_ylabel(f"놓친 불량 (시험 불량 {n_bad}개 중)", color="#d62728")
ax2.set_ylim(0, n_bad)
for i, t in enumerate(table["기준선"]):
    ax1.annotate(f"기준선 {t:.2f}", (i, table["_recheck"][i]), textcoords="offset points", xytext=(0, 8),
                 ha="center", fontsize=8)
plt.title("비용 비율에 따라 고른 기준선의 시험 결과")
fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.88))
plt.tight_layout()
plt.savefig("cost_threshold.png", dpi=150)
