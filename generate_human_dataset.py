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


def gen_wp():
    dataset_name = "euclaise/writingprompts"
    dataset = load_dataset(dataset_name)
    datasize = 200
    human_txts = dict()
    context_txts = dict()

    dataloader = DataLoader(dataset["train"], shuffle=True)
    counter = 0
    with tqdm(dataloader) as pbar:
        for i, batch in enumerate(pbar):
            pbar.set_description(f"Creating datasets... ({counter}/{datasize})")
            if counter >= datasize:
                break
            human_data = batch["story"][0]
            check_length = lambda x: len(x) < 1000 or len(x) >= 1500
            if check_length(human_data):
                continue
            human_txts[counter] = human_data
            context_txts[counter] = batch["prompt"]
            counter += 1
    dataset_splitted = dataset_name.split("/")[-1]
    with open(f"./txtdata/{dataset_splitted}_human.json", "w") as f:
        json.dump(human_txts, f, ensure_ascii=False)

    with open(f"./txtdata/{dataset_splitted}_prompt.json", "w") as f:
        json.dump(human_txts, f, ensure_ascii=False)


def gen_hc3():
    """This function generates not only a human dataset also an ai-generated dataset(chatgpt)"""
    dataset_name = "Hello-SimpleAI/HC3"
    dataset = load_dataset(dataset_name, name="all")
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
            human_data = batch.get("human_answers")
            ai_data = batch.get("chatgpt_answers")
            if not (human_data and ai_data):
                continue
            human_data = human_data[0][0]
            ai_data = ai_data[0][0]

            if "Too many requests in 1 hour." in ai_data:
                continue
            check_length = lambda x: len(x) < 1000 or len(x) >= 1500
            if check_length(human_data) and check_length(ai_data):
                continue
            human_txts[counter] = human_data
            ai_txts[counter] = ai_data
            question_txts[counter] = batch["question"]
            counter += 1

    dataset_splitted = dataset_name.split("/")[-1]
    with open(f"./txtdata/{dataset_splitted}_human.json", "w") as f:
        json.dump(human_txts, f, ensure_ascii=False)

    with open(f"./txtdata/{dataset_splitted}_ai.json", "w") as f:
        json.dump(human_txts, f, ensure_ascii=False)

    with open(f"./txtdata/{dataset_splitted}_prompt.json", "w") as f:
        json.dump(human_txts, f, ensure_ascii=False)


if __name__ == "__main__":
    gen_xsum()
    gen_wp()
    gen_hc3()
