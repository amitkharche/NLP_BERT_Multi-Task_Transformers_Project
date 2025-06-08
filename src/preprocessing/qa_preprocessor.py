from datasets import load_dataset
from transformers import BertTokenizerFast

def preprocess_qa(model_name='bert-base-uncased'):
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    dataset = load_dataset("squad")

    def encode(example):
        return tokenizer(example['question'], example['context'], truncation=True, padding='max_length', max_length=384)

    return dataset.map(encode, batched=True)
