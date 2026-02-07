import pandas as pd

def save_csv(file_path, data):
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print("Saved cleaned dataset:",
          df.shape,
          df["sentiment"].value_counts().to_dict())
