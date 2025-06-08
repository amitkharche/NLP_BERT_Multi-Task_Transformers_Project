from transformers import pipeline
from datasets import load_dataset
from sklearn.metrics import classification_report

def evaluate_sentiment():
    classifier = pipeline("sentiment-analysis", model="output/bert_sentiment")
    dataset = load_dataset("imdb", split="test[:200]")
    y_true = [1 if ex['label'] else 0 for ex in dataset]
    y_pred = [1 if classifier(ex['text'])[0]['label'] == 'POSITIVE' else 0 for ex in dataset]
    print(classification_report(y_true, y_pred))

def evaluate_ner():
    ner = pipeline("ner", model="output/bert_ner", aggregation_strategy="simple")
    sample = "Hugging Face is based in New York City and was founded by Julien Chaumond."
    print(ner(sample))

if __name__ == "__main__":
    evaluate_sentiment()
    evaluate_ner()
