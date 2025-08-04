import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- Step 1: Load Data ---
real = pd.read_csv("data/preprocessed_2D.csv")
ctgan = pd.read_csv("data/CTGAN_generated_data.csv")
taegan = pd.read_csv("data/TAEGAN_generated_data_2D.csv")
transformer = pd.read_csv("data/TransformersGAN_generated_data_2D.csv")

# --- Step 2: Add Pollution Classification Target ---
def add_pollution_class(df):
    df = df.copy()
    df["PollutionIndex"] = (
        df["CO(GT)"].clip(lower=0) +
        df["C6H6(GT)"].clip(lower=0) +
        df["NMHC(GT)"].clip(lower=0)
    )
    df["PollutionClass"] = pd.qcut(df["PollutionIndex"], q=3, labels=[0, 1, 2])
    df = df.drop(columns=["PollutionIndex"])
    return df

real = add_pollution_class(real)
ctgan = add_pollution_class(ctgan)
taegan = add_pollution_class(taegan)
transformer = add_pollution_class(transformer)

# --- Step 3: Define Features ---
feature_cols = [col for col in real.columns if col != "PollutionClass"]
target_col = "PollutionClass"

# --- Step 4: Define Evaluation Function ---
def train_evaluate(X, y, label):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(f"\n📊 Results for: {label}")
    print(classification_report(y_test, y_pred))
    return {
        "Dataset": label,
        "Accuracy": report["accuracy"],
        "F1_macro": report["macro avg"]["f1-score"],
        "Precision_macro": report["macro avg"]["precision"],
        "Recall_macro": report["macro avg"]["recall"]
    }

# --- Step 5: Run Evaluations ---
results = []
results.append(train_evaluate(real[feature_cols], real[target_col], "Real Only"))
results.append(train_evaluate(pd.concat([real, ctgan])[feature_cols],
                              pd.concat([real, ctgan])[target_col], "Real + CTGAN"))
results.append(train_evaluate(pd.concat([real, taegan])[feature_cols],
                              pd.concat([real, taegan])[target_col], "Real + TAEGAN"))
results.append(train_evaluate(pd.concat([real, transformer])[feature_cols],
                              pd.concat([real, transformer])[target_col], "Real + TransformerGAN"))

# --- Step 6: Visualize Results ---
df_results = pd.DataFrame(results)
metrics = ["Accuracy", "F1_macro", "Precision_macro", "Recall_macro"]

plt.figure(figsize=(10, 6))
for i, metric in enumerate(metrics):
    plt.bar(
        [r + i * 0.2 for r in range(len(df_results))],
        df_results[metric],
        width=0.2,
        label=metric
    )

plt.xticks([r + 0.3 for r in range(len(df_results))], df_results["Dataset"], rotation=20)
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.title("Classifier Performance on Real vs Synthetic-augmented Data")
plt.legend()
plt.tight_layout()
plt.show()
