from transformers import BertTokenizerFast
from datasets import load_dataset

def preprocess_ner(model_name='bert-base-cased'):
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    dataset = load_dataset("conll2003")

    label_list = dataset['train'].features['ner_tags'].feature.names

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True, padding='max_length', max_length=128)
        labels = []
        for i, label in enumerate(examples['ner_tags']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(label[word_idx])
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    dataset = dataset.map(tokenize_and_align_labels, batched=True)
    return dataset, label_list
