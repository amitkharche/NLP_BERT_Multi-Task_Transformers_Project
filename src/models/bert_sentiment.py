from transformers import TFBertForSequenceClassification

def get_sentiment_model(model_name='bert-base-uncased', num_labels=2):
    model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model
