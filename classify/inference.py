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
    model_pth = "models/20220608-203647"
    max_length = 120
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
            examples["entailment"], truncation=True, padding=False, max_length=max_length)
        return result
    
    test_dataset = test_dataset.map(preprocess_function, batched=True, remove_columns=dataset["test"].keys())
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=None, padding=True)
    test_dataloaer = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=data_collator)
    preds = inference(model, test_dataloaer, device) # prediction result

    class example:
        entailment_templates = [
            "不是一个短语",
            "是关于其他的短语",
            "是关于工人、劳动的短语",
            "是关于实体的短语",
        ]
        def __init__(self, text):
            self.text = text
            self.pred = {0:0,1:0,2:0,3:0}
            self.n = 0

        def assign_score(self, template, score):
            """
            Assign score for the template
            """
            self.n += 1
            id = 0
            for i,temp in enumerate(self.entailment_templates):
                if temp in template:
                    id = i
                    break
            self.pred[id] += score
        
        def assign_label(self, template, label):
            """
            Assign label for the template
            """
            assert label == 1
            self.n += 1
            id = 0
            for i,temp in enumerate(self.entailment_templates):
                if temp in template:
                    id = i
                    break
            self.label = id

    test_examples = {}
    for i in range(len(dataset["test"]["text"])):
        text = dataset["test"]["text"][i]
        label = dataset["test"]["label"][i]
        template = dataset["test"]["entailment"][i]
        pred = preds[i][1]
        if text not in test_examples.keys():
            test_examples[text] = example(text)
        test_examples[text].assign_score(template, pred)
        if label == 1:
            test_examples[text].assign_label(template, label)

    labels = []
    preds = []
    for text, test_example in test_examples.items():
        test_example.pred = {k:v/test_example.n*4 for k,v in test_example.pred.items()}
        labels.append(test_example.label)
        pred = 0
        for k,v in test_example.pred.items():
            if v > pred:
                pred = v
                pred_id = k
        preds.append(pred_id)

    print(classification_report(labels, preds))
    print(f1_score(labels, preds, average='macro'))