from transformers import (TFBertForSequenceClassification, TFBertForTokenClassification,
                          BertTokenizerFast, create_optimizer)
from datasets import load_dataset
import tensorflow as tf
import argparse

def train_sentiment(model_name='bert-base-uncased', epochs=2, batch_size=8):
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    dataset = load_dataset("imdb")
    dataset = dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding='max_length', max_length=512),
                          batched=True)
    dataset.set_format(type='tensorflow', columns=['input_ids', 'attention_mask', 'label'])

    train = dataset['train'].to_tf_dataset(
        columns=['input_ids', 'attention_mask'],
        label_cols='label',
        shuffle=True,
        batch_size=batch_size)

    model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    optimizer, _ = create_optimizer(init_lr=2e-5, num_train_steps=len(train)*epochs, num_warmup_steps=0)
    model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])
    model.fit(train, epochs=epochs)
    model.save_pretrained('output/bert_sentiment')

def train_ner(model_name='bert-base-cased', epochs=3, batch_size=8):
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
                    label_ids.append(label[word_idx] if True else -100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    dataset = dataset.map(tokenize_and_align_labels, batched=True)
    dataset.set_format(type='tensorflow', columns=['input_ids', 'attention_mask', 'labels'])

    train = dataset['train'].to_tf_dataset(
        columns=['input_ids', 'attention_mask'],
        label_cols='labels',
        shuffle=True,
        batch_size=batch_size)

    model = TFBertForTokenClassification.from_pretrained(model_name, num_labels=len(label_list))
    optimizer, _ = create_optimizer(init_lr=3e-5, num_train_steps=len(train)*epochs, num_warmup_steps=0)
    model.compile(optimizer=optimizer, loss=model.compute_loss)
    model.fit(train, epochs=epochs)
    model.save_pretrained('output/bert_ner')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["sentiment", "ner"], required=True)
    args = parser.parse_args()

    if args.task == "sentiment":
        train_sentiment()
    elif args.task == "ner":
        train_ner()
