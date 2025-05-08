import json

import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertTokenizer

from .sentiment_classifier import SentimentClassifier

with open("config.json", "r") as json_file:
    config = json.load(json_file)


class Model:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = BertTokenizer.from_pretrained(config["BERT_MODEL"])

        classifier = SentimentClassifier(len(config["CLASS_NAMES"]))

        classifier.load_state_dict(torch.load(config["PRE_TRAINED_MODEL"], map_location=self.device))
        classifier = classifier.eval()
        self.classifier = classifier.to(self.device)


    def predict(self, text):
        encoded_review = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=config["MAX_LEN"],
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",
            return_token_type_ids=False,
        )

        input_ids = encoded_review["input_ids"].to(self.device)
        attention_mask = encoded_review["attention_mask"].to(self.device)

        with torch.no_grad():
            probabilities = F.softmax(self.classifier(input_ids, attention_mask), dim=1)

        confidence, predicted_class = torch.max(probabilities, dim=1)

        predict_class = predicted_class.cpu().item()
        probabilities = probabilities.flatten().cpu().numpy().tolist()

        return (
            config["CLASS_NAMES"][predict_class],
            confidence,
            dict(zip(config["CLASS_NAMES"], probabilities)),
        )
    



model = Model()

def get_model():
    return model 




