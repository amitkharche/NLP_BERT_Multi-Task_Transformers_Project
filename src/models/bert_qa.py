from transformers import TFBertForQuestionAnswering

def get_qa_model(model_name='bert-base-uncased'):
    model = TFBertForQuestionAnswering.from_pretrained(model_name)
    return model
