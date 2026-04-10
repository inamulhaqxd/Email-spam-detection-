# ============================================================
# STEP 1: Import Libraries
# ============================================================
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score
)

# ============================================================
# STEP 2: Load Dataset
# Place your Kaggle CSV file in the same folder as this script.
# Dataset columns: label, text, label_num
# ============================================================
print("📥 Loading dataset...")
df = pd.read_csv("emails.csv")   # 👈 rename your Kaggle file to emails.csv

print(f"✅ Dataset loaded! Shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst 3 rows:\n{df.head(3)}")

# ============================================================
# STEP 3: Data Cleaning
# ============================================================
# Keep only needed columns
df = df[["label", "text", "label_num"]].copy()

# Drop duplicates and missing values
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

print(f"\nAfter cleaning — shape: {df.shape}")
print(f"\nClass distribution:\n{df['label'].value_counts()}")
print(f"\nSpam %: {df['label_num'].mean()*100:.2f}%")

# ============================================================
# STEP 4: EDA — Exploratory Data Analysis
# ============================================================

# Plot 1: Class Distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

df["label"].value_counts().plot(
    kind="bar", ax=axes[0],
    color=["steelblue", "tomato"],
    edgecolor="black"
)
axes[0].set_title("Ham vs Spam Distribution", fontsize=13)
axes[0].set_xticklabels(["Ham (Not Spam)", "Spam"], rotation=0)
axes[0].set_ylabel("Count")

# Plot 2: Email Length Distribution
df["text_length"] = df["text"].apply(len)
df.groupby("label")["text_length"].plot(
    kind="kde", ax=axes[1], legend=True
)
axes[1].set_title("Email Length Distribution", fontsize=13)
axes[1].set_xlabel("Email Length (characters)")
axes[1].legend(["Ham", "Spam"])

plt.tight_layout()
plt.savefig("eda_plot.png", dpi=150)
plt.show()
print("📊 EDA plot saved!")

# ============================================================
# STEP 5: Feature Extraction using TF-IDF
# TF-IDF converts raw email text into numerical features
# that the model can understand.
# max_features=10000 means we use the top 10,000 words.
# ============================================================
print("\n🔤 Extracting TF-IDF features...")

tfidf = TfidfVectorizer(
    max_features=10000,
    stop_words="english",
    ngram_range=(1, 2)      # unigrams + bigrams
)

X = tfidf.fit_transform(df["text"])
y = df["label_num"]

print(f"✅ TF-IDF shape: {X.shape}")

# ============================================================
# STEP 6: Train-Test Split (80% / 20%)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")

# ============================================================
# STEP 7: Train Logistic Regression Model
# ============================================================
print("\n🤖 Training Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    C=1.0,
    random_state=42
)
model.fit(X_train, y_train)
print("✅ Model trained!")

# ============================================================
# STEP 8: Evaluate the Model
# ============================================================
y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc  = roc_auc_score(y_test, y_pred_prob)

print(f"\n📊 Model Evaluation:")
print(f"   Accuracy  : {accuracy * 100:.2f}%")
print(f"   ROC-AUC   : {roc_auc:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

# ============================================================
# STEP 9: Plot ROC Curve + Confusion Matrix
# ============================================================
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
cm = confusion_matrix(y_test, y_pred)

fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))

# ROC Curve
axes2[0].plot(fpr, tpr, color="darkorange", lw=2,
              label=f"ROC Curve (AUC = {roc_auc:.2f})")
axes2[0].plot([0, 1], [0, 1], color="gray", linestyle="--")
axes2[0].set_title("ROC Curve", fontsize=13)
axes2[0].set_xlabel("False Positive Rate")
axes2[0].set_ylabel("True Positive Rate")
axes2[0].legend(loc="lower right")
axes2[0].grid(True, alpha=0.3)

# Confusion Matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Ham", "Spam"]
)
disp.plot(ax=axes2[1], colorbar=False, cmap="Blues")
axes2[1].set_title("Confusion Matrix", fontsize=13)

plt.tight_layout()
plt.savefig("evaluation_plot.png", dpi=150)
plt.show()
print("📈 Evaluation plot saved!")

# ============================================================
# STEP 10: Top Spam Words
# ============================================================
feature_names = tfidf.get_feature_names_out()
coefficients  = model.coef_[0]

top_spam_idx = np.argsort(coefficients)[-20:]
top_ham_idx  = np.argsort(coefficients)[:20]

top_spam_words = [(feature_names[i], coefficients[i]) for i in top_spam_idx[::-1]]
top_ham_words  = [(feature_names[i], coefficients[i]) for i in top_ham_idx]

fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))

# Top Spam Words
spam_words, spam_scores = zip(*top_spam_words)
axes3[0].barh(spam_words, spam_scores, color="tomato", edgecolor="black")
axes3[0].set_title("Top 20 Spam Indicator Words", fontsize=12)
axes3[0].set_xlabel("Coefficient Score")
axes3[0].invert_yaxis()

# Top Ham Words
ham_words, ham_scores = zip(*top_ham_words)
axes3[1].barh(ham_words, ham_scores, color="steelblue", edgecolor="black")
axes3[1].set_title("Top 20 Ham Indicator Words", fontsize=12)
axes3[1].set_xlabel("Coefficient Score")
axes3[1].invert_yaxis()

plt.tight_layout()
plt.savefig("top_words_plot.png", dpi=150)
plt.show()
print("🔍 Top words plot saved!")

# ============================================================
# STEP 11: Save Model and Vectorizer
# ============================================================
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("\n✅ model.pkl and tfidf.pkl saved successfully!")
