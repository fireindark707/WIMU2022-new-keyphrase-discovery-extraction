from .utils import deduplicate
label2id = {"O":0,"General":1,"Labor":2,"Name":3}

def simple_dataset(pairs):
    """
    Simple dataset with train and test data
    """
    dataset = {"text": [], "label": []}
    pairs = [(p[0], p[2]) for p in pairs]
    pairs = deduplicate(pairs)
    for pair in pairs:
        dataset["text"].append(pair[0])
        dataset["label"].append(label2id[pair[1]])
    return dataset

def word_context_dataset(pairs,context_per_text = 3):
    """
    Simple dataset with train and test data
    """
    dataset = {"text": [], "context":[], "label": []}
    pairs = deduplicate(pairs)

    # Limit the number of samples
    new_pairs = []
    text_list = deduplicate([pair[0] for pair in pairs])
    for text in text_list:
        text_pairs = [p for p in pairs if p[0] == text]
        try:
            text_pairs = sorted(text_pairs, key=lambda x: len(x[1]))[:context_per_text]
        except IndexError:
            text_pairs = text_pairs
        new_pairs.extend(text_pairs)
    pairs = new_pairs

    for pair in pairs:
        dataset["text"].append(pair[0])
        dataset["context"].append(pair[1])
        dataset["label"].append(label2id[pair[2]])
    return dataset

def word_entailment_dataset(pairs):
    """
    Simple dataset with train and test data
    """
    dataset = {"text": [],"entailment":[], "label": []}
    pairs = [(p[0], p[2]) for p in pairs]
    pairs = deduplicate(pairs)

    entailment_templates = [
        "{}不是一个短语",
        "{}是关于其他的短语",
        "{}是关于工人、劳动的短语",
        "{}是关于实体的短语",
    ]

    for pair in pairs:
        label_id = label2id[pair[1]]
        for i,e_template in enumerate(entailment_templates):
            dataset["text"].append(pair[0])
            dataset["entailment"].append(e_template.format(pair[0]))
            if label_id == i:
                dataset["label"].append(1)
            else:
                dataset["label"].append(0)
    return dataset

def word_context_entailment_dataset(pairs,context_per_text = 3):
    """
    Simple dataset with train and test data
    """
    dataset = {"text": [], "context":[],"entailment":[], "label": []}
    pairs = deduplicate(pairs)

    # Limit the number of samples
    new_pairs = []
    text_list = deduplicate([pair[0] for pair in pairs])
    for text in text_list:
        text_pairs = [p for p in pairs if p[0] == text]
        try:
            text_pairs = sorted(text_pairs, key=lambda x: len(x[1]))[:context_per_text]
        except IndexError:
            text_pairs = text_pairs
        new_pairs.extend(text_pairs)
    pairs = new_pairs

    entailment_templates = [
        "{}不是一个短语",
        "{}是关于其他的短语",
        "{}是关于工人、劳动的短语",
        "{}是关于实体的短语",
    ]

    for pair in pairs:
        label_id = label2id[pair[2]]
        for i,e_template in enumerate(entailment_templates):
            dataset["text"].append(pair[0])
            dataset["context"].append(pair[1])
            dataset["entailment"].append(e_template.format(pair[0]))
            if label_id == i:
                dataset["label"].append(1)
            else:
                dataset["label"].append(0)
    return dataset

def word_context_entailment_extra_dataset(pairs,context_per_text = 3):
    """
    Simple dataset with train and test data
    """
    dataset = {"text": [], "context":[],"entailment":[], "label": []}
    pairs = deduplicate(pairs)

    # Limit the number of samples
    new_pairs = []
    text_list = deduplicate([pair[0] for pair in pairs])
    for text in text_list:
        text_pairs = [p for p in pairs if p[0] == text]
        try:
            text_pairs = sorted(text_pairs, key=lambda x: len(x[1]))[:context_per_text]
        except IndexError:
            text_pairs = text_pairs
        new_pairs.extend(text_pairs)
    pairs = new_pairs

    entailment_templates = [
        "{}不是一个短语",
        "{}是关于其他的短语",
        "{}是关于工人、劳动的短语",
        "{}是关于实体的短语",
        "{}是一个短语",
    ]

    for pair in pairs:
        label_id = label2id[pair[2]]
        for i,e_template in enumerate(entailment_templates):
            dataset["text"].append(pair[0])
            dataset["context"].append(pair[1])
            dataset["entailment"].append(e_template.format(pair[0]))
            if label_id == i:
                dataset["label"].append(1)
            elif i == 4 and label_id > 0:
                dataset["label"].append(1)
            else:
                dataset["label"].append(0)
    return dataset

def combine_train_test(train_data, test_data):
    """
    Combine train and test data
    """
    dataset = {"train": train_data, "test": test_data}
    return dataset