from train import inference,get_dataset
from src.utils import most_common_element
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification,DataCollatorWithPadding
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import f1_score, classification_report

if __name__ == '__main__':
    train_data_path = "/home/phoenix000/IT/new_word_discovery/data/train_data.txt"
    test_data_path = "/home/phoenix000/IT/new_word_discovery/data/test_data.txt"
    model_pth = "models/20220608-152933"
    max_length = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device("cpu")

    dataset = get_dataset(train_data_path, test_data_path)
    test_dataset = Dataset.from_dict(dataset["test"])

    tokenizer = AutoTokenizer.from_pretrained(model_pth)
    model = AutoModelForSequenceClassification.from_pretrained(model_pth)
    model.to(device)
    
    def preprocess_function(examples):
        """
        Preprocess function for tokenizer
        """
        result = tokenizer(
            examples["text"], truncation=True, padding=False, max_length=max_length)
        return result
    
    test_dataset = test_dataset.map(preprocess_function, batched=True, remove_columns=dataset["test"].keys())
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=None, padding=True)
    test_dataloaer = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=data_collator)
    preds = inference(model, test_dataloaer, device) # prediction result

    text_label = {}
    for i, text in enumerate(dataset["test"]["text"]):
        text_label[text] = dataset["test"]["label"][i]

    text_pred = {}
    for i in range(len(dataset["test"]["text"])):
        text = dataset["test"]["text"][i]
        label = dataset["test"]["label"][i]
        #argmax
        pred = preds[i].argmax()
        try:
            text_pred[text].append(pred)
        except KeyError:
            text_pred[text] = [pred]

    # select the most frequent pred for each text
    text_pred_final = {}
    for text in text_pred:
        text_pred_final[text] = most_common_element(text_pred[text])
    print(text_pred_final)
    # calculate f1 score
    preds = []
    labels = []
    for text in text_label.keys():
        pred = text_pred_final[text]
        label = text_label[text]
        preds.append(pred)
        labels.append(label)
    f1 = f1_score(labels, preds, average='macro')
    print("F1 score:", f1)
    print("ClassificationReport:",classification_report(labels, preds))