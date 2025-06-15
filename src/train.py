from transformers import TFBertForSequenceClassification, BertTokenizerFast, create_optimizer
from datasets import load_dataset
import tensorflow as tf

# ✅ Helper function to manually unpack a batch
def unpack_batch(batch):
    if isinstance(batch, (list, tuple)):
        if len(batch) == 3:
            x, y, sample_weight = batch
        elif len(batch) == 2:
            x, y = batch
            sample_weight = None
        else:
            raise ValueError(f"Unexpected batch length: {len(batch)}")
    else:
        raise TypeError("Batch is not a tuple/list.")
    return x, y, sample_weight

# ✅ Load and preprocess dataset
def load_imdb_dataset(model_name='bert-base-uncased'):
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    dataset = load_dataset("imdb")

    def tokenize(example):
        return tokenizer(example['text'], truncation=True, padding='max_length', max_length=512)

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type='tensorflow', columns=['input_ids', 'attention_mask', 'label'])

    train_dataset = dataset['train'].to_tf_dataset(
        columns=['input_ids', 'attention_mask'],
        label_cols='label',
        shuffle=True,
        batch_size=8
    )

    test_dataset = dataset['test'].to_tf_dataset(
        columns=['input_ids', 'attention_mask'],
        label_cols='label',
        shuffle=False,
        batch_size=8
    )

    return train_dataset, test_dataset

# ✅ Train function using manual batch unpacking
def train_model():
    model_name = 'bert-base-uncased'
    train_dataset, test_dataset = load_imdb_dataset(model_name)

    model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    optimizer, _ = create_optimizer(
        init_lr=2e-5,
        num_train_steps=len(train_dataset) * 2,
        num_warmup_steps=0
    )

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy()

    for epoch in range(2):
        print(f"\nEpoch {epoch + 1}")
        metric.reset_states()

        for step, batch in enumerate(train_dataset):
            x, y, sample_weight = unpack_batch(batch)

            with tf.GradientTape() as tape:
                logits = model(x, training=True).logits
                loss = loss_fn(y, logits, sample_weight=sample_weight)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            metric.update_state(y, logits)

            if step % 100 == 0:
                print(f"Step {step}: Loss = {loss.numpy():.4f}, Accuracy = {metric.result().numpy():.4f}")

    model.save_pretrained("output/bert_sentiment_custom")

if __name__ == "__main__":
    train_model()
