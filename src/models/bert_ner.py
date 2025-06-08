from transformers import TFBertForTokenClassification

def get_ner_model(model_name='bert-base-cased', num_labels=9):
    model = TFBertForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
    return model
