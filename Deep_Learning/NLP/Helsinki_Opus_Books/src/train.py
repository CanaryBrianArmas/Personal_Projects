# src/train.py

import os
from transformers import TrainingArguments, DataCollatorForSeq2Seq
import evaluate as evalu
from src.model import load_model_and_tokenizer
from src.data_loader import load_and_prepare_dataset
from src.custom_trainer import CustomSeq2SeqTrainer
from src import config

# Global: load model and tokenizer for use in the compute_metrics function.
model, tokenizer = load_model_and_tokenizer()

def compute_metrics(eval_preds):
    """
    Computes BLEU score using sacreBLEU.
    """
    metric = evalu.load("sacrebleu")
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels (ignored tokens) with tokenizer.pad_token_id.
    labels = [
        [label if label != -100 else tokenizer.pad_token_id for label in l] 
        for l in labels
    ]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # sacreBLEU expects a list of reference lists for each prediction.
    decoded_labels = [[l] for l in decoded_labels]
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}

def main():
    # Load and tokenize the dataset.
    tokenized_datasets = load_and_prepare_dataset(tokenizer)
    # Handle datasets with/without a dedicated validation split.
    if "validation" in tokenized_datasets:
        train_dataset = tokenized_datasets["train"]
        eval_dataset = tokenized_datasets["validation"]
    else:
        train_dataset = tokenized_datasets["train"]
        eval_dataset = tokenized_datasets["test"]

    # Data collator for dynamic padding.
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Set up training arguments.
    training_args = TrainingArguments(
        output_dir=config.TRAINING_ARGS["output_dir"],
        num_train_epochs=config.TRAINING_ARGS["num_train_epochs"],
        per_device_train_batch_size=config.TRAINING_ARGS["per_device_train_batch_size"],
        per_device_eval_batch_size=config.TRAINING_ARGS["per_device_eval_batch_size"],
        eval_strategy=config.TRAINING_ARGS["evaluation_strategy"],
        save_strategy=config.TRAINING_ARGS["save_strategy"],
        logging_dir=config.TRAINING_ARGS["logging_dir"],
        fp16=True
    )

    # Initialize the Trainer.
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Start training.
    trainer.train()

    # Save the final model.
    trainer.save_model("./final_model")
    print("Training complete. Model saved to 'final_model/'.")

if __name__ == "__main__":
    main()
