
# Model & Dataset configuration
MODEL_NAME = "Helsinki-NLP/opus-mt-en-es"
DATASET_NAME = "Helsinki-NLP/opus_books"
DATASET_CONFIG = "en-es"
NUM_TRAIN_EPOCHS= 3
PER_DEVICE_TRAIN_BATCH_SIZE= 8

# Language pair settings
SOURCE_LANG = "en"
TARGET_LANG = "fr"

# Training hyperparameters
TRAINING_ARGS = {
    "output_dir": "./results",
    "num_train_epochs": NUM_TRAIN_EPOCHS,
    "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
    "per_device_eval_batch_size": 8,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "logging_dir": "./logs"
}