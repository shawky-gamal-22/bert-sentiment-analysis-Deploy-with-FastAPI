# 🧠 Google Play App Reviews Sentiment Classifier API

This project provides a full machine learning pipeline and an interactive API for classifying user sentiments in app reviews from the Google Play Store. The system is built using Python, Hugging Face Transformers, and FastAPI.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [API Usage](#api-usage)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Acknowledgements](#acknowledgements)

---

## 📖 Overview

This project aims to build a sentiment classifier that can analyze user reviews of productivity apps on the Google Play Store. The reviews are categorized as `positive`, `negative`, or `neutral`.

We built and fine-tuned a BERT model, then deployed the model using FastAPI so users can interact with the system via HTTP requests and receive predictions in real-time.

---

## 📂 Dataset

The dataset was scraped using the `google-play-scraper` library. It includes thousands of reviews from top productivity apps such as:

- Todoist
- Google Keep
- Microsoft To Do
- Evernote

Each review is labeled manually or using rule-based heuristics into one of the following classes:

- **Positive**
- **Neutral**
- **Negative**

---

## 🧠 Model Architecture

We used the `bert-base-uncased` model from Hugging Face Transformers, and added a classification head on top. The model was fine-tuned using the preprocessed review texts.

```python
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
```


## 🏋️ Training Process
Tokenizer: BERT tokenizer (bert-base-uncased)

Max Length: 128 tokens

Optimizer: AdamW

Learning Rate: 2e-5

Epochs: 3

Batch Size: 16

Evaluation metrics:

Accuracy

F1 Score (weighted)

The best model was saved and used for deployment.

## ⚡ API Usage
The API is built using FastAPI and can be accessed via:

➕ Endpoint: /predict
Method: POST

Request Body:

```json
نسخ
تحرير
{
  "review": "I love using Todoist for my daily tasks!"
}
```
Response:

```json
نسخ
تحرير
{
  "sentiment": "positive",
  "confidence": 0.95
}
```

## 🛠️ Installation
Clone the repo

``` bash
نسخ
تحرير
git clone https://github.com/your-username/app-reviews-sentiment-api.git
cd app-reviews-sentiment-api
Create a virtual environment
```
```bash
نسخ
تحرير
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
Install dependencies
```
```bash
نسخ
تحرير
pip install -r requirements.txt
```
