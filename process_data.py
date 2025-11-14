import pandas as pd
import re
import string
from nltk.corpus import stopwords
import nltk

# # Download NLTK stopwords, run only once
# nltk.download('stopwords')

# Loading combined dataset
df = pd.read_csv("combined_news.csv")

# Combine title and text columns
df["content"] = df["title"].astype(str) + " " + df["text"].astype(str)

stop_words = set(stopwords.words("english")) #Selecting English stopwords

def clean_text(text):
    text = text.lower() # Lowercase
    text = re.sub(f"[{re.escape(string.punctuation)}0-9]", " ", text) # Removing punctuation and numbers
    text = re.sub(r"\s+", " ", text).strip() # Removing extra spaces
    words = [word for word in text.split() if word not in stop_words] # Removing stopwords
    return " ".join(words)

#  Cleaning function 
df["clean_content"] = df["content"].apply(clean_text)
output_file = "cleaned_news.csv"
df.to_csv(output_file, index=False)

print(f"Cleaning complete! Saved as '{output_file}'")
print(df[["clean_content", "label"]].head())
