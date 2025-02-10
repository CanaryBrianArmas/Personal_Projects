This a project I have made using the "en-es" subset from the Dataset: [Helsinki_opus_books](https://huggingface.co/datasets/Helsinki-NLP/opus_books)


## Setup

1. Clone or download the repository.
2. Install the dependencies:
    ```
    pip install -r requirements.txt
    ```

## Running the Training

To train the model, simply run: python src/train.py

The trained model and results will be saved in the output directory specified in `src/config.py`.

## Customization

- Change the model or language pair by modifying `src/config.py`.
- Adjust hyperparameters like batch size and number of epochs in `src/config.py`.
