import json

from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


def main():
    dataset_name = "xsum"
    dataset = load_dataset(dataset_name)
    datasize = 200
    human_txts = list()

    dataloader = DataLoader(dataset["train"])
    dataloader = DataLoader(dataset["train"], shuffle=True)
    counter = 0
    with tqdm(dataloader) as pbar:
        for i, batch in enumerate(pbar):
            pbar.set_description(f"Creating datasets... ({counter}/{datasize})")
            if counter >= datasize:
                break
            train_data = batch["document"][0]
            if len(train_data) < 1000 or len(train_data) >= 1500:
                continue
            if 
            human_txts.append(train_data.replace("\n", " "))
            counter += 1

    with open(f"./txtdata/{dataset_name}_human_doc.txt", "w") as f:
        for data in human_txts:
            f.write(data + "\n")


if __name__ == "__main__":
    main()
