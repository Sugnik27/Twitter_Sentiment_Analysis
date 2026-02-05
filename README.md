# 🐦 Twitter Post Sentiment Analysis (Deep Learning)

An end-to-end **Sentiment Analysis application** that predicts the sentiment of tweets or text using **Deep Learning models (RNN, LSTM, GRU)**.  
The project covers the complete ML lifecycle — **data preprocessing, model training with hyperparameter tuning, and a Streamlit web app for real-time inference**.

---

## 🚀 Features

- Predicts sentiment into:
  - 🟢 **Positive**
  - 🟡 **Neutral**
  - 🔴 **Negative**
- Deep Learning models:
  - Simple RNN
  - LSTM
  - GRU
- Hyperparameter tuning using **Keras Tuner**
- Confidence score with class-wise probabilities
- Confidence-based neutral handling
- Color-coded Streamlit UI
- Batch prediction via CSV upload
- Modular, production-style project structure

---

## 🧠 Model Architecture

Each model follows this architecture:

- **Embedding Layer**
- **Recurrent Layer** (RNN / LSTM / GRU)
- **Dropout Layer**
- **Dense Softmax Output Layer**

**Loss Function:**  
- `sparse_categorical_crossentropy`

**Optimizer:**  
- Adam

**Evaluation Metric:**  
- Accuracy

---

## 📂 Project Structure

Twitter_post_sentiment_analysis/
│
├── src/
│ ├── app.py # Streamlit web app
│ ├── training.py # DL model training + tuning
│ ├── preprocessing.py # Text preprocessing & tokenization
│ ├── deployment.py # Model inference utilities
│ └── config.py # Centralized configuration
│
├── models/
│ ├── best_overall_model.keras
│ ├── tokenizer.pkl
│ └── label_encoder.pkl
│
├── notebooks/
│ └── data_cleaning.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- Keras Tuner
- NumPy
- Pandas
- Scikit-learn
- Streamlit

---

✍️ Example Inputs
🟢 Positive

“I’m genuinely happy with how things turned out today. Everything worked smoothly and the support was amazing.”

🟡 Neutral

“The report has been submitted and the files are available in the shared folder for review.”

🔴 Negative

“I’m extremely frustrated with how this situation has been handled. Nothing seems to improve despite repeated efforts.”


--

🎯 Confidence-Based Neutral Handling

If the model confidence is below a predefined threshold, the prediction is automatically treated as Neutral.
This prevents incorrect polarity assignment for ambiguous or informational texts.

--

📊 Output

Predicted sentiment

Confidence score

Class-wise probabilities

Color-coded UI:

🟢 Positive

🟡 Neutral

🔴 Negative

--

📌 Future Improvements

- BiLSTM with Attention mechanism

- Improved neutral class balancing

- Explainable AI (attention visualization)

- Dockerization

- Cloud deployment (Streamlit Cloud / Render)

- Multi-language sentiment analysis


