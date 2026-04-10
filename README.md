# 📧 Email Spam Detector

A machine learning web app built with **Streamlit** that detects whether an email is **Spam or Ham (Not Spam)** using **Logistic Regression** and **TF-IDF** text features.

---
**Preview** = **https://spam-mails-dataset-dfmzswyja65jaxgzewvbps.streamlit.app/**

## 🚀 App Features

* 🔍 **Instant spam detection** — paste any email and get result in seconds
* 📊 **Confidence score** — shows spam/ham probability percentage
* 💡 **Example emails** — pre-loaded spam and ham examples to try
* 📈 **Interactive dashboard** with:

  * Data Overview & Distribution
  * ROC Curve & Confusion Matrix
  * Top Spam/Ham Indicator Words
  * Sample Email Viewer
* 🎨 **Color-coded results** — red for spam, green for ham

---

## 📁 Project Structure

```
email-spam-detection/
│
├── train_model.py       # Train model and save .pkl files
├── app.py               # Streamlit web application
├── emails.csv           # Dataset (add manually)
├── model.pkl            # Trained model
├── tfidf.pkl            # TF-IDF vectorizer
├── requirements.txt     # Dependencies
└── README.md            # Project documentation
```

---

## 📊 Dataset

* **Source:** Kaggle Email Spam Dataset
* **Columns used:**

  * `label` → spam/ham
  * `text` → email content
  * `label_num` → 0 (ham), 1 (spam)
* **Download:** [https://www.kaggle.com/datasets/venky73/spam-mails-dataset](https://www.kaggle.com/datasets/venky73/spam-mails-dataset)

---

## 🤖 How It Works

1. User inputs email text
2. Text is converted into numerical features using **TF-IDF**
3. Features are passed to a **Logistic Regression model**
4. Model predicts:

   * Spam or Ham
   * Confidence score (%)

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/inamulhaqxd/email-spam-detection.git
cd email-spam-detection
```

### 2. Add Dataset

* Download from Kaggle
* Rename file to `emails.csv`
* Place in project folder

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train Model

```bash
python train_model.py
```

### 5. Run Application

```bash
streamlit run app.py
```

---

## 🌐 Deployment (Streamlit Cloud)

### Option A (Recommended)

* Train model locally
* Upload these files to GitHub:

  * `emails.csv`
  * `model.pkl`
  * `tfidf.pkl`

### Option B (Advanced)

* Train model inside `app.py` if files are missing

### Steps

1. Push code to GitHub
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Connect repo
4. Select `app.py`
5. Deploy

---

## 📦 Requirements

```
streamlit
scikit-learn
pandas
numpy
matplotlib
seaborn
```

---

## 🐙 GitHub Push Commands

```bash
git init
git add .
git commit -m "Initial commit - Email Spam Detector"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/email-spam-detection.git
git push -u origin main
```

---

## ⚠️ Disclaimer

This project is for educational purposes only and should not be used as a production-level spam filter.

---

## 👤 Author

**Your Name**
GitHub: [https://github.com/inamulhaqxd](https://github.com/inamulhaqxd)

