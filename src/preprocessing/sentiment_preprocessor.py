from transformers import BertTokenizerFast
from datasets import load_dataset

def preprocess_sentiment(model_name='bert-base-uncased', split='train'):
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    dataset = load_dataset("imdb", split=split)

    def encode(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

    return dataset.map(encode, batched=True)
