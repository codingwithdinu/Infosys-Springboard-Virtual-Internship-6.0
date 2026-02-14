import pandas as pd
from datasets import load_dataset
from urgency_rule_based import detect_urgency_rule

dataset = load_dataset("jason23322/high-accuracy-email-classifier")

train_df = pd.DataFrame(dataset["train"])

# Create urgency column
train_df["urgency"] = train_df["text"].apply(detect_urgency_rule)

train_df.to_csv("data/processed/train_with_urgency.csv", index=False)

print("Urgency labels created successfully.")
