# src/data_loader.py

from datasets import load_dataset
import config

def load_and_prepare_dataset(tokenizer, max_source_length=128, max_target_length=128):
    """
    Loads the Helsinki-NLP/opus_books dataset for the specified language pair and preprocesses it.
    """
    # Load the dataset with the given configuration (e.g., "en-fr")
    raw_datasets = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG)

    # Preprocessing function: tokenize source (English) and target (French) texts.
    def preprocess_function(examples):
        # Each example contains a "translation" field with keys "en" and "fr"
        inputs = [ex[config.SOURCE_LANG] for ex in examples["translation"]]
        targets = [ex[config.TARGET_LANG] for ex in examples["translation"]]
        model_inputs = tokenizer(inputs, max_length=max_source_length, truncation=True)
        # Tokenize target texts
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Map the preprocessing function to the dataset in batched mode.
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
    
    # If there is no dedicated validation split, create one.
    if "validation" not in tokenized_datasets:
        tokenized_datasets = tokenized_datasets["train"].train_test_split(test_size=0.1)
    return tokenized_datasets
