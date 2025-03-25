# ML_BreastCancer

I am going to solve a Breast Cancer Classification problem using ML. The dataset I'm gonna use is public in this same platform: [dataset](https://github.com/selva86/datasets/blob/master/BreastCancer.csv)

The folder structure is as follows:
1. src/data_sample: full data and data sample.
2. src/results_notebook: final notebook with all the code.
3. src/models: the models that have been created and saved.
4. src/utils: The functions I create.


## Setup

Clone the repository:
   ```bash
   git clone https://github.com/CanaryBrianArmas/Personal_Projects.git
   cd Personal_Projects/Machine_Learning/BreastCancer_ML
   ```
Or create Venv:
```bash
python -m venv breast_cancer_env
source breast_cancer_env/bin/activate 
```
#### On Windows: breast_cancer_env\Scripts\activate


## Install dependencies
```bash
pip install -r requirements.txt
```

## Run the project
```bash
jupyter notebook BreastCancer_ML.ipynb
```

