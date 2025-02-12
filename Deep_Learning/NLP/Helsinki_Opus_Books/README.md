This a project I have made using the "en-es" subset from the Dataset: [Helsinki_opus_books](https://huggingface.co/datasets/Helsinki-NLP/opus_books)

You can just upload the Main_Colab.ipynb file in GoogleDrive and run it there, or, you can do this locally as follows:

## Setup

1. Create a virtual enviroment:
    ```
    conda create -n NAME python=VERSION --> I have used 3.9 version
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

## Customization

- Change the model or language pair by modifying `src/config.py`.
- Adjust hyperparameters like batch size and number of epochs in `src/config.py`.
- Modify data preprocessing in `src/data_loader.py` as needed.
