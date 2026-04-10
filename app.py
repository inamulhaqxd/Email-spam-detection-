# ============================================================
# STEP 1: Import Libraries
# ============================================================
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score
)
from sklearn.model_selection import train_test_split

# ============================================================
# STEP 2: Page Configuration
# ============================================================
st.set_page_config(
    page_title="📧 Email Spam Detector",
    page_icon="📧",
    layout="wide"
)

# ============================================================
# STEP 3: Load Model and TF-IDF Vectorizer
# ============================================================
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    return model, tfidf

model, tfidf = load_artifacts()

# ============================================================
# STEP 4: Load Dataset for EDA
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv("emails.csv")
    df = df[["label", "text", "label_num"]].copy()
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df["text_length"] = df["text"].apply(len)
    df["word_count"]  = df["text"].apply(lambda x: len(x.split()))
    return df

# ============================================================
# STEP 5: Custom CSS
# ============================================================
st.markdown("""
<style>
    .spam-box {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 1.4em;
        font-weight: bold;
    }
    .ham-box {
        background: linear-gradient(135deg, #55efc4, #00b894);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 1.4em;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 10px 15px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# STEP 6: App Title
# ============================================================
st.title("📧 Email Spam Detector")
st.write("Paste any email text below to instantly detect if it's **Spam** or **Ham (Not Spam)** using Logistic Regression + TF-IDF.")
st.write("---")

# ============================================================
# STEP 7: Main Prediction Section
# ============================================================
st.subheader("🔍 Check Your Email")

# Example emails for quick testing
example_emails = {
    "Select an example...": "",
    "🚨 Spam Example 1 — Prize Winner": "Congratulations! You have been selected as our lucky winner. Click here to claim your FREE prize worth $1000. Limited time offer! Call now 1-800-WINNER",
    "🚨 Spam Example 2 — Urgent Money": "URGENT: Your account has been suspended. Verify your details immediately to avoid permanent closure. Click the link below NOW.",
    "✅ Ham Example 1 — Work Email": "Hi team, please find attached the meeting notes from yesterday's project discussion. Let me know if you have any questions or concerns.",
    "✅ Ham Example 2 — Personal": "Hey, just wanted to check in and see how you're doing. Are you free to catch up over coffee this weekend?",
}

selected = st.selectbox("💡 Try an example email:", list(example_emails.keys()))
example_text = example_emails[selected]

email_input = st.text_area(
    "✉️ Paste email text here:",
    value=example_text,
    height=200,
    placeholder="Paste the full email content here including subject line..."
)

col_btn1, col_btn2 = st.columns([1, 5])
predict_btn = col_btn1.button("🔍 Detect", type="primary")
clear_btn   = col_btn2.button("🗑️ Clear")

if clear_btn:
    email_input = ""
    st.rerun()

# ============================================================
# STEP 8: Prediction Result
# ============================================================
if predict_btn:
    if not email_input.strip():
        st.warning("⚠️ Please enter some email text first!")
    else:
        # Transform and predict
        input_tfidf  = tfidf.transform([email_input])
        prediction   = model.predict(input_tfidf)[0]
        probability  = model.predict_proba(input_tfidf)[0]

        spam_prob = round(probability[1] * 100, 2)
        ham_prob  = round(probability[0] * 100, 2)

        st.write("---")
        st.subheader("📊 Result")

        # Show result box
        if prediction == 1:
            st.markdown(
                f'<div class="spam-box">🚨 SPAM DETECTED! ({spam_prob}% confidence)</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="ham-box">✅ SAFE EMAIL (Ham) — {ham_prob}% confidence</div>',
                unsafe_allow_html=True
            )

        st.write("")

        # Confidence metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("🚨 Spam Probability",   f"{spam_prob}%")
        col2.metric("✅ Ham Probability",    f"{ham_prob}%")
        col3.metric("📝 Word Count",         len(email_input.split()))

        # Confidence bar
        st.write("**Spam Confidence:**")
        st.progress(int(spam_prob), text=f"{spam_prob}% spam probability")

# ============================================================
# STEP 9: Dataset Analysis Tabs
# ============================================================
st.write("---")
st.subheader("📊 Dataset Analysis & Model Performance")

with st.spinner("Loading analysis..."):
    df = load_data()

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Data Overview",
    "📉 ROC & Confusion Matrix",
    "🔍 Top Spam Words",
    "📋 Sample Emails"
])

# --- Tab 1: Data Overview ---
with tab1:
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("📧 Total Emails",    len(df))
    col_b.metric("✅ Ham Emails",      len(df[df["label"] == "ham"]))
    col_c.metric("🚨 Spam Emails",     len(df[df["label"] == "spam"]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df["label"].value_counts().plot(
        kind="bar", ax=axes[0],
        color=["steelblue", "tomato"], edgecolor="black"
    )
    axes[0].set_title("Ham vs Spam Distribution")
    axes[0].set_xticklabels(["Ham", "Spam"], rotation=0)
    axes[0].set_ylabel("Count")

    df.groupby("label")["text_length"].plot(kind="kde", ax=axes[1], legend=True)
    axes[1].set_title("Email Length by Category")
    axes[1].set_xlabel("Characters")
    axes[1].legend(["Ham", "Spam"])

    plt.tight_layout()
    st.pyplot(fig)

# --- Tab 2: ROC + Confusion Matrix ---
with tab2:
    X_all   = tfidf.transform(df["text"])
    y_all   = df["label_num"]
    _, X_te, _, y_te = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    y_pred_t = model.predict(X_te)
    y_prob_t = model.predict_proba(X_te)[:, 1]

    acc = accuracy_score(y_te, y_pred_t)
    auc = roc_auc_score(y_te, y_prob_t)

    c1, c2 = st.columns(2)
    c1.metric("✅ Accuracy",  f"{acc*100:.2f}%")
    c2.metric("📈 ROC-AUC",  f"{auc:.4f}")

    fpr, tpr, _ = roc_curve(y_te, y_prob_t)
    cm = confusion_matrix(y_te, y_pred_t)

    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    axes2[0].plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc:.2f}")
    axes2[0].plot([0,1],[0,1], color="gray", linestyle="--")
    axes2[0].set_title("ROC Curve")
    axes2[0].set_xlabel("False Positive Rate")
    axes2[0].set_ylabel("True Positive Rate")
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.3)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
    disp.plot(ax=axes2[1], colorbar=False, cmap="Blues")
    axes2[1].set_title("Confusion Matrix")
    plt.tight_layout()
    st.pyplot(fig2)

# --- Tab 3: Top Spam Words ---
with tab3:
    feature_names = tfidf.get_feature_names_out()
    coefficients  = model.coef_[0]
    top_spam_idx  = np.argsort(coefficients)[-20:][::-1]
    top_ham_idx   = np.argsort(coefficients)[:20]

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
    axes3[0].barh(
        [feature_names[i] for i in top_spam_idx],
        [coefficients[i] for i in top_spam_idx],
        color="tomato", edgecolor="black"
    )
    axes3[0].set_title("Top 20 Spam Indicator Words")
    axes3[0].invert_yaxis()

    axes3[1].barh(
        [feature_names[i] for i in top_ham_idx],
        [coefficients[i] for i in top_ham_idx],
        color="steelblue", edgecolor="black"
    )
    axes3[1].set_title("Top 20 Ham Indicator Words")
    axes3[1].invert_yaxis()

    plt.tight_layout()
    st.pyplot(fig3)

# --- Tab 4: Sample Emails ---
with tab4:
    st.write("**Sample Spam Emails:**")
    st.dataframe(
        df[df["label"] == "spam"][["label", "text"]].head(5).reset_index(drop=True),
        use_container_width=True
    )
    st.write("**Sample Ham Emails:**")
    st.dataframe(
        df[df["label"] == "ham"][["label", "text"]].head(5).reset_index(drop=True),
        use_container_width=True
    )

# ============================================================
# STEP 10: Footer
# ============================================================
st.write("---")
st.caption("📧 Email Spam Detector — Logistic Regression + TF-IDF | Educational use only")
