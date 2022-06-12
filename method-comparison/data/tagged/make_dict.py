import json

def load_dict(data_type):
    tmp_dict = {}

    for line in open(f"./{data_type}_annotated.txt", "r", encoding="utf-8"):
        line = line.split("\n")[0].split(" ")

        if line[1] not in tmp_dict:
            tmp_dict[line[1]] = []

        tmp_dict[line[1]].append(line[0])

    return tmp_dict


if __name__ == "__main__":
    my_dict = {}

    my_dict["train"] = load_dict("train")
    my_dict["test"] = load_dict("test")

    with open("./my_dict.json", "w", encoding="utf-8") as f:
        json.dump(my_dict, fp=f, ensure_ascii=False)

    print("done!")
