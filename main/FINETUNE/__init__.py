from ..models import get_tokenizer, get_classifier, BASE_MODEL_DIR, FINETUNED_MODEL_DIR, LABEL_TO_ID
from ..data import (
    load_many_csv_datasets,
    DEFAULT_TRAIN_PATH,
    DEFAULT_CV_PATH,
    DEFAULT_TEST_PATH,
)
from ..finetuning import (
    MAX_LEN,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
)
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

def do_FINETUNE(train_path=None, cv_path=None, test_path=None, input_model_dir=None, output_model_dir=None):
    train_path = train_path or DEFAULT_TRAIN_PATH
    cv_path = cv_path or DEFAULT_CV_PATH
    test_path = test_path or DEFAULT_TEST_PATH
    input_model_dir = input_model_dir or BASE_MODEL_DIR
    output_model_dir = output_model_dir or FINETUNED_MODEL_DIR

    tokenizer = get_tokenizer(model_dir=input_model_dir)
    model = get_classifier(model_dir=input_model_dir)

    dataset = load_many_csv_datasets({
        "train": train_path,
        "cv": cv_path,
        "test": test_path,
    })

    def preprocess(batch):
        # After preprocessing,
        # new columns are added:
        # - input_ids
        # - attention_mask
        tokenized = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LEN)
        # tokenized = tokenized.rename_column("label", "labels")
        tokenized["labels"] = [LABEL_TO_ID[label] for label in batch["label"]]
        return tokenized
    
    encoded_datasets = dataset.map(preprocess, batched=True)

    # Why this is needed (GitHub Copilot suggested):
    # "Set the format of the datasets to PyTorch tensors.
    # This will ensure that the data is returned as PyTorch tensors
    # when we access the elements of the dataset
    # This is important for compatibility with PyTorch DataLoader
    # especially when using batching."
    for split in ["train", "cv", "test"]:
        encoded_datasets[split].set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    # -----------------------------
    # Define metrics
    # -----------------------------
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted"),
            "precision": precision_score(labels, preds, average="weighted"),
            "recall": recall_score(labels, preds, average="weighted"),
        }
    
    # -----------------------------
    # Training arguments
    # -----------------------------
    training_args = TrainingArguments(
        output_dir=output_model_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # -----------------------------
    # Trainer
    # -----------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_datasets["train"],
        eval_dataset=encoded_datasets["cv"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # -----------------------------
    # Train
    # -----------------------------
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model(output_model_dir)
    print(f"Fine-tuned model saved to {output_model_dir}")

    # -----------------------------
    # Optional: Evaluate on test set
    # -----------------------------
    test_results = trainer.evaluate(encoded_datasets["test"])
    print(f"Test set results: {test_results}")
