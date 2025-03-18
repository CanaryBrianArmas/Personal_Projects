import pandas as pd
url = "./data_sample/BreastCancer.csv"
full_df = pd.read_csv(url)
sample_df = full_df.sample(frac=0.2, random_state=42)  # Ajusta el frac para <5MB
sample_df.to_csv("./data_sample/BreastCancer_sample.csv", index=False)