import pandas as pd
import matplotlib.pyplot as plt

# CSV oku
df = pd.read_csv("./eval/eval_results.csv")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 2 satır 3 kolonluk grid

# 1. Recall Hit
df["recall_hit"].value_counts().plot(kind="bar", color=["skyblue", "salmon"], ax=axes[0,0])
axes[0,0].set_title("Recall Hit Dağılımı")
axes[0,0].set_ylabel("Soru Sayısı")

# 2. Groundedness
df["groundedness"].dropna().plot(kind="hist", bins=10, color="purple", alpha=0.7, ax=axes[0,1])
axes[0,1].set_title("Groundedness Dağılımı")
axes[0,1].set_xlabel("Groundedness Skoru")

# 3. Latency
axes[0,2].boxplot(df["latency"], vert=False)
axes[0,2].set_title("Latency Dağılımı")
axes[0,2].set_xlabel("Saniye")

# 4. Answerable dağılımı
df["is_answerable"].value_counts().plot(kind="bar", color=["green", "red"], ax=axes[1,0])
axes[1,0].set_title("Answerable Dağılımı")
axes[1,0].set_ylabel("Soru Sayısı")

# 5. Unanswerable doğruluk (varsa)
if df["unanswerable_correct"].notna().any():
    df["unanswerable_correct"].value_counts().plot(kind="bar", color=["blue", "orange"], ax=axes[1,1])
    axes[1,1].set_title("Unanswerable Doğruluk")
    axes[1,1].set_ylabel("Soru Sayısı")
else:
    axes[1,1].set_axis_off()

# Boş kalan subplot
axes[1,2].set_axis_off()

plt.tight_layout()
plt.savefig("./eval/eval_report.png", dpi=150)  # PNG olarak kaydet
plt.show()
