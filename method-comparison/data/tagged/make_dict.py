FILE_TYPE = "test"

if __name__ == '__main__':
    file_loc = f"./{FILE_TYPE}_annotated.txt"
    out_file = f"./{FILE_TYPE}_dict.txt"

    my_dict = []
    for line in open(file_loc, 'r', encoding='utf-8'):
        line = line.split('\n')[0].split(' ')
        if (line[-1] != "O"):
            my_dict.append(line[0])

    with open(out_file, 'w', encoding='utf-8') as f:
        for word in my_dict:
            print(word , file=f)

