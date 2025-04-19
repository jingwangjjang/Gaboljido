import os
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

def load_slot_labels(path):
    with open(path, encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]

def load_examples(seq_in_path, seq_out_path):
    with open(seq_in_path, encoding='utf-8') as f_in, open(seq_out_path, encoding='utf-8') as f_out:
        inputs = [line.strip().split() for line in f_in.readlines()]
        labels = [line.strip().split() for line in f_out.readlines()]
    return inputs, labels

def tokenize_and_align_labels(inputs, labels, tokenizer, label2id):
    tokenized_inputs = tokenizer(
        inputs,
        is_split_into_words=True,
        truncation=True,
        padding=True,
        return_tensors='pt'
    )

    aligned_labels = []
    for i, label in enumerate(labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            else:
                label_ids.append(label2id[label[word_idx]] if label[word_idx].startswith("I-") else -100)
            previous_word_idx = word_idx
        aligned_labels.append(label_ids)

    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs

def load_dataset(seq_in_path, seq_out_path, tokenizer, label2id):
    inputs, labels = load_examples(seq_in_path, seq_out_path)
    encodings = tokenize_and_align_labels(inputs, labels, tokenizer, label2id)
    return Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": encodings["labels"]
    })

def main():
    base_path = "data"
    model_name = "KPF/KPF-bert-ner"
    slot_label_path = os.path.join(base_path, "slot_label.txt")

    train_seq_in = os.path.join(base_path, "KPF_bert_ner/train/seq.in")
    train_seq_out = os.path.join(base_path, "KPF_bert_ner/train/seq.out")
    test_seq_in = os.path.join(base_path, "KPF_bert_ner/test/seq.in")
    test_seq_out = os.path.join(base_path, "KPF_bert_ner/test/seq.out")


    # 1. Load labels and tokenizer
    slot_labels = load_slot_labels(slot_label_path)
    label2id = {label: i for i, label in enumerate(slot_labels)}
    id2label = {i: label for i, label in enumerate(slot_labels)}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(slot_labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    # 2. Load dataset
    train_dataset = load_dataset(train_seq_in, train_seq_out, tokenizer, label2id)
    eval_dataset = load_dataset(test_seq_in, test_seq_out, tokenizer, label2id)

    # 3. Training args
    training_args = TrainingArguments(
        output_dir="./ner_output",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=20,
        logging_dir="./logs",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
        logging_steps=100,  # 로그 빈도
        eval_steps=100,     # 중간 평가 주기
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )

    # 4. Train
    trainer.train()
    trainer.save_model("./ner_output")

if __name__ == "__main__":
    main()