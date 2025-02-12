This a project I have made using the "en-es" subset from the Dataset: [Helsinki_opus_books](https://huggingface.co/datasets/Helsinki-NLP/opus_books)


## Setup

If you are gonna run this locally, do as follows:

1. Create a virtual enviroment(If you are going to run this locally):
    ```
    conda create -n NAME python=VERSION
    ```  
2. Clone, download the repository.

3. Install the dependencies:
    ```
    pip install -r requirements.txt
    ```

## Running the Project

### Train the model
`python main.py train`

### Evaluate the model
`python main.py eval`

### Test the model
`python main.py test`

### Use the model
`python main.py predict`

The trained model and results will be saved in the output directory specified in `src/config.py`.

Instead, you can use the Main_Colab.ipynb file and just run the cells.

## Customization

- Change the model or language pair by modifying `src/config.py`.
- Adjust hyperparameters like batch size and number of epochs in `src/config.py`.
- Modify data preprocessing in `src/data_loader.py` as needed.
