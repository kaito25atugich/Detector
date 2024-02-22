import json

from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


def gen_xsum():
    dataset_name = "xsum"
    dataset = load_dataset(dataset_name)
    datasize = 200
    human_txts = dict()

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
            human_txts[counter] = train_data
            counter += 1

    with open(f"./txtdata/{dataset_name}_human.json", "w") as f:
        json.dump(human_txts, f, ensure_ascii=False)


def gen_pubmed():
    dataset_name = "pubmed"
    dataset = load_dataset(dataset_name)
    datasize = 200
    human_txts = dict()
    title_txts = dict()

    dataloader = DataLoader(dataset["train"], shuffle=True)
    counter = 0
    with tqdm(dataloader) as pbar:
        for i, batch in enumerate(pbar):
            pbar.set_description(f"Creating datasets... ({counter}/{datasize})")
            if counter >= datasize:
                break
            abstract_text = batch["MedilineCitation"]["Article"]["Abstract"][
                "AbstractText"
            ]
            if abstract_text:
                train_data = abstract_text
                if len(train_data) < 1000 or len(train_data) >= 1500:
                    continue
                human_txts[counter] = train_data
                title_txts[counter] = batch["MedilineCitation"]["Article"][
                    "ArticleTitle"
                ]
                counter += 1

    with open(f"./txtdata/{dataset_name}_human.json", "w") as f:
        json.dump(human_txts, f, ensure_ascii=False)

    with open(f"./txtdata/{dataset_name}_abst_title.json", "w") as f:
        json.dump(title_txts, f, ensure_ascii=False)


def gen_hc3():
    """This function generates not only a human dataset also an ai-generated dataset(chatgpt)"""
    dataset_name = "xsum"
    dataset = load_dataset(dataset_name)
    datasize = 200
    human_txts = dict()
    ai_txts = dict()
    question_txts = dict()

    dataloader = DataLoader(dataset["train"], shuffle=True)
    counter = 0
    with tqdm(dataloader) as pbar:
        for i, batch in enumerate(pbar):
            pbar.set_description(f"Creating datasets... ({counter}/{datasize})")
            if counter >= datasize:
                break
            human_data = batch["human_answers"][0]
            ai_data = batch["chatgpt_answers"][0]
            check_length = lambda x: len(x) < 1000 or len(x) >= 1500
            if  check_length(human_data) and check_length(ai_data)
                continue
            human_txts[counter] = human_data
            ai_txts[counter] = ai_data
            question_txts[counter] = batch["question"]
            counter += 1

    with open(f"./txtdata/{dataset_name}_human.json", "w") as f:
        json.dump(human_txts, f, ensure_ascii=False)

    with open(f"./txtdata/{dataset_name}_ai.json", "w") as f:
        json.dump(human_txts, f, ensure_ascii=False)
    
    with open(f"./txtdata/{dataset_name}_question.json", "w") as f:
        json.dump(human_txts, f, ensure_ascii=False)

if __name__ == "__main__":
    gen_xsum()
    gen_pubmed()
    gen_hc3()
