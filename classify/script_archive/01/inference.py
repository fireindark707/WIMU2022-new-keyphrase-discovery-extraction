from train import inference,get_dataset
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification,DataCollatorWithPadding
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import f1_score, classification_report

if __name__ == '__main__':
    train_data_path = "/home/phoenix000/IT/new_word_discovery/data/train_data.txt"
    test_data_path = "/home/phoenix000/IT/new_word_discovery/data/test_data.txt"
    model_pth = "models/20220608-134158"
    max_length = 10
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

    pred_list = []
    ground_truth = []
    for i in range(len(dataset["test"]["text"])):
        text = dataset["test"]["text"][i]
        label = dataset["test"]["label"][i]
        #argmax
        pred = preds[i].argmax()
        pred_list.append(pred)
        ground_truth.append(label)
    
    print(classification_report(ground_truth, pred_list))
    print(f1_score(ground_truth, pred_list, average='macro'))


