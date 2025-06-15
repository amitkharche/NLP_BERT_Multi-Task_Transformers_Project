# 🤖 BERT NLP Suite – Multi-Task Transformers Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![HuggingFace](https://img.shields.io/badge/HuggingFace-🤗-yellow.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)

---

## 📌 Project Overview

This project provides a **multi-task NLP pipeline** powered by **BERT** and the Hugging Face 🤗 Transformers library. It enables:

- ✅ **Sentiment Analysis**
- ✅ **Named Entity Recognition (NER)** 
- ✅ **Question Answering (QA)** 

It features:
- Custom training scripts
- Pre-trained model fine-tuning
- Streamlit-based interactive demo app
- Manual batching logic that avoids internal Keras errors (e.g., `unpack_x_y_sample_weight`)

---

## 🎯 Business Use Case

In real-world applications, NLP can help:

- 📊 Understand user **sentiment** from reviews, feedback, or social media
- 🏷️ Extract **key information** like names, places, and organizations from unstructured text
- ❓ Enable **conversational Q&A** over structured documents or FAQs

### 💡 Benefits
| Use Case              | Benefit                             |
|----------------------|-------------------------------------|
| Sentiment Analysis   | Brand monitoring, feedback mining   |
| Named Entity Recognition | Information extraction, resume parsing |
| Question Answering   | Instant support, document querying  |

---

## 🗂️ Project Structure

```plaintext
transformers-bert-nlp/
├── data/                            # Dataset placeholder
├── notebooks/                       # EDA & model experimentation notebooks
│   ├── 01_BERT_QA.ipynb
│   ├── 02_BERT_Sentiment.ipynb
│   └── 03_BERT_NER.ipynb
├── output/                          # Saved fine-tuned models
│   ├── bert_sentiment_custom/
│   └── bert_ner/
├── src/
│   ├── train.py                     # Training script for sentiment and NER
│   └── evaluate.py                  # Model evaluation script (optional)
├── streamlit_app/
│   └── app.py                       # Streamlit interface
├── requirements.txt                 # Python dependencies
├── LICENSE                          # MIT License
└── README.md                        # You are here
```

---

## ⚙️ Installation & Setup

### 🔧 Clone and Install

```bash
git clone https://github.com/amitkharche/NLP_BERT_Multi-Task_Transformers_Project.git
cd NLP_BERT_Multi-Task_Transformers_Project
pip install -r requirements.txt
```

### 📦 Requirements

- Python ≥ 3.8
- TensorFlow ≥ 2.8
- transformers ≥ 4.30
- datasets
- streamlit
- plotly
- pandas

---

## 🚀 How to Use

### 🎯 Model Training (Sentiment or NER)

Run from root folder:

```bash
python src/train.py --task sentiment
python src/train.py --task ner
```

> 💡 Automatically downloads the dataset (IMDB or CoNLL-2003), fine-tunes BERT, and saves the model under `/output`.

### 🌐 Launch Streamlit UI

```bash
streamlit run streamlit_app/app.py
```

Then open your browser at: [http://localhost:8501](http://localhost:8501)

---

## 🧠 Tasks Covered

### 🟢 Sentiment Analysis

- Dataset: IMDB Movie Reviews
- Model: `TFBertForSequenceClassification`
- Labels: POSITIVE / NEGATIVE
- Output: `output/bert_sentiment_custom/`

### 🟠 Named Entity Recognition (NER)

- Dataset: CoNLL-2003
- Model: `TFBertForTokenClassification`
- Entities: `PER`, `ORG`, `LOC`, `MISC`
- Output: `output/bert_ner/`

### 🔵 Question Answering

- Model: `pipeline("question-answering")` using `bert-base-uncased`
- Context + Question input in Streamlit
- No training script; uses pre-trained inference only

---

## 🧪 Sample Inputs

### 📝 Sentiment Analysis

> Text: "I absolutely loved the new Batman movie. It was thrilling!"

**→ Output**: `POSITIVE (Confidence: 97%)`

---

### 🏷 Named Entity Recognition

> Text: "Elon Musk is the CEO of Tesla, based in Palo Alto, California."

**→ Output**:
| Entity     | Type | Confidence |
|------------|------|------------|
| Elon Musk  | PER  | 98.4%      |
| Tesla      | ORG  | 95.2%      |
| Palo Alto  | LOC  | 93.6%      |
| California | LOC  | 92.1%      |

---

### ❓ Question Answering

> **Context**: "The Eiffel Tower is in Paris, France. It was built in 1889."

> **Question**: "When was the Eiffel Tower built?"

**→ Output**: `"1889"` with 98.2% confidence

---

## 📊 Model Evaluation (Optional)

You can add or run evaluation metrics (accuracy, F1, etc.) by extending `src/evaluate.py` (template stub provided).

---

## 🛠 Tools & Libraries

- 🤗 **Transformers** – Model architectures
- 🧠 **TensorFlow** – Training framework
- 📚 **datasets** – Prebuilt NLP datasets
- 🎛 **Streamlit** – Interactive user interface
- 📈 **Plotly** – Visualizations
- 🐍 **Python 3.8+**

---

## 📝 License

This project is licensed under the **MIT License** – free to use, modify, and distribute.

---

## 🤝 Let's Connect

Have questions, suggestions, or ideas to collaborate?

- 🌐 [LinkedIn](https://www.linkedin.com/in/amit-kharche)
- 🧠 [Medium](https://medium.com/@amitkharche14)
- 💻 [GitHub](https://github.com/amitkharche)

---

## 💬 Final Words

Whether you're a developer building customer-facing tools, a researcher exploring NLP models, or a student learning transformers — this project offers a ready-to-use suite for training, inference, and deployment. Clone it, customize it, and bring your NLP ideas to life.
