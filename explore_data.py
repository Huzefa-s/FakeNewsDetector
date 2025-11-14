import pandas as pd

# Loading datasets
df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")

# Adding a column to distinguish between fake and true news
df_fake["label"] = 0 # 0 for fake news
df_true["label"] = 1 # 1 for true news

df = pd.concat([df_fake, df_true], axis=0) # combining the datasets
df = df.sample(frac=1).reset_index(drop=True)  # shuffling the dataset

# Show info
print("Combined dataset created!")
print("Shape of dataset:", df.shape)
print("\nColumns:", df.columns)
print("\nLabel distribution:")
print(df["label"].value_counts())
print("\nSample data:")
print(df.head())

# Save combined dataset to a CSV file
output_file = "combined_news.csv"
df.to_csv(output_file, index=False)
print(f"Combined dataset saved as '{output_file}'")
