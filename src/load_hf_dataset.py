from datasets import load_dataset

def load_hf_data():
    dataset = load_dataset("jason23322/high-accuracy-email-classifier")
    return dataset

if __name__ == "__main__":
    ds = load_hf_data()
    print(ds["train"][0])

