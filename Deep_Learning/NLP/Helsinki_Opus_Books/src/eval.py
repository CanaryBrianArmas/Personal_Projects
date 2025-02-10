# src/eval.py

from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from model import load_model_and_tokenizer
from data_loader import load_and_prepare_dataset
import config

def main():
    # Load model and tokenizer.
    model, tokenizer = load_model_and_tokenizer()

    # Prepare the dataset.
    tokenized_datasets = load_and_prepare_dataset(tokenizer)
    eval_dataset = tokenized_datasets.get("validation") or tokenized_datasets.get("test")
    if eval_dataset is None:
        print("No evaluation dataset available.")
        return

    # Data collator for dynamic padding.
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Define minimal training arguments for evaluation.
    training_args = TrainingArguments(
        output_dir=config.TRAINING_ARGS["output_dir"],
        per_device_eval_batch_size=config.TRAINING_ARGS["per_device_eval_batch_size"],
        predict_with_generate=True,
    )

    # Compute metrics similar to train.py.
    def compute_metrics(eval_preds):
        from datasets import load_metric
        metric = load_metric("sacrebleu")
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels with the pad token id.
        labels = [
            [label if label != -100 else tokenizer.pad_token_id for label in l]
            for l in labels
        ]
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # sacreBLEU expects a list of reference lists per prediction.
        decoded_labels = [[ref] for ref in decoded_labels]
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": result["score"]}

    # Set up the Trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Run evaluation.
    results = trainer.evaluate()
    print("Evaluation results:")
    for key, value in results.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
