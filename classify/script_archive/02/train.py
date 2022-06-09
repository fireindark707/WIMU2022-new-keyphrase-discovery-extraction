from src.dataset import simple_dataset,word_context_dataset, combine_train_test
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding, get_scheduler
import torch
import torch.nn as nn
import datetime
import json
import random
from datasets import Dataset
from tqdm import tqdm
from sklearn.metrics import f1_score

torch.manual_seed(42)

def load_data_pairs(data_path):
    """
    Load data from file
    """
    data_pairs = []
    with open(data_path, "r") as f:
        for line in f:
            line = line.strip()
            word,text, label = line.split("\t")
            data_pairs.append((word,text, label))
    return data_pairs

def get_dataset(train_data_path, test_data_path):
    """
    Get dataset
    """
    # Load data
    train_data_pairs = load_data_pairs(train_data_path)
    test_data_pairs = load_data_pairs(test_data_path)
    # Create dataset
    # train_dataset = simple_dataset(train_data_pairs)
    # test_dataset = simple_dataset(test_data_pairs)
    train_dataset = word_context_dataset(train_data_pairs)
    test_dataset = word_context_dataset(test_data_pairs)
    # Combine train and test dataset
    combined_dataset = combine_train_test(train_dataset, test_dataset)
    return combined_dataset

def train(model, train_dataloader, optimizer,lr_scheduler, device):
    """
    Train the model for one epoch of training
    """
    model.train()
    for batch in tqdm(train_dataloader):
        # to device
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

def evaluate(model, dev_dataloader, device):
    """
    Evaluate the model on dev set, using marcro-F1
    """
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(dev_dataloader):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            pred = torch.argmax(outputs.logits, dim=-1)
            preds.extend(pred.cpu().numpy())
            labels.extend(batch["labels"].cpu().numpy())
    macro_f1 = f1_score(labels, preds, average='macro')
    return macro_f1

def inference(model, dev_dataloader, device):
    """
    Inference the model on test set
    """
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(dev_dataloader):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            pred = torch.softmax(outputs.logits, dim=-1)
            preds.extend(pred.cpu().numpy())
    return preds

def save(model, tokenizer, output_dir):
    """
    Save the model and tokenizer
    """
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == '__main__':
    train_data_path = "/home/phoenix000/IT/new_word_discovery/data/train_data.txt"
    test_data_path = "/home/phoenix000/IT/new_word_discovery/data/test_data.txt"
    model_pth = "hfl/chinese-roberta-wwm-ext"
    num_labels = 4
    batch_size = 16
    lr = 1e-5
    epochs = 20
    max_length = 100
    output_dir = "models/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = get_dataset(train_data_path, test_data_path)

    tokenizer = AutoTokenizer.from_pretrained(model_pth)
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, pad_to_multiple_of=None, padding=True)

    def preprocess_function(examples):
        """
        Preprocess function for tokenizer
        """
        result = tokenizer(
            examples["text"],examples["context"], truncation=True, max_length=max_length, padding=False)
        result["labels"] = examples["label"]
        return result

    train_dataset = Dataset.from_dict(dataset['train'])
    dev_dataset = Dataset.from_dict(dataset['test'])
    train_dataset = train_dataset.map(
        preprocess_function, batched=True, remove_columns=list(dataset['train'].keys()))
    dev_dataset = dev_dataset.map(
        preprocess_function, batched=True, remove_columns=list(dataset['test'].keys()))
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size)
    dev_dataloader = torch.utils.data.DataLoader(
        dev_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_pth, num_labels=num_labels)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_training_steps=len(train_dataloader)*epochs
    lr_scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    best_score = 0
    for epoch in range(epochs):
        train(model, train_dataloader, optimizer,lr_scheduler, device)
        score = evaluate(model, dev_dataloader, device)
        print("{} epoch, score: {}".format(epoch, score))
        if score > best_score:
            save(model, tokenizer, output_dir)
            best_score = score
    print("Done!")
    