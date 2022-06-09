import argparse
parser = argparse.ArgumentParser(
    description='Merges annotations from one file into another.')
parser.add_argument('-t', '--type', help='Annotate train and test with annotations or make new file for annotation',
                    choices=['annotate', 'new'], required=True)
args = parser.parse_args()


def read_file(path):
    """
    Reads a file and returns a dictionary with phrase as keys and annotation type as values.
    """
    phrase_annotation = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if ' ' in line:
                phrase, annotation = line.split(' ')
                phrase_annotation[phrase] = annotation
            else:
                phrase_annotation[line] = ''
    return phrase_annotation


def merge_from(annotated, original):
    """
    Merges the annotations from annotated into original.
    """
    for phrase, annotation in annotated.items():
        if phrase in original:
            original[phrase] = annotation
    return original


def find_empty_annotation(phrase_annotation):
    """
    Finds phrases with empty annotation.
    """
    empty_annotation = []
    for phrase, annotation in phrase_annotation.items():
        if annotation == '':
            empty_annotation.append(phrase)
    return empty_annotation


def write_file(path, phrase_annotation):
    """
    Writes a dictionary with phrase as keys and annotation type as values to a file.
    """
    with open(path, 'w') as f:
        for phrase, annotation in phrase_annotation.items():
            f.write(phrase + ' ' + annotation + '\n')


if __name__ == '__main__':
    annotated = read_file("annotation.txt")
    train = read_file("train.txt")
    test = read_file("test.txt")

    if args.type == 'annotate':
        train_annotated = merge_from(annotated, train)
        test_annotated = merge_from(annotated, test)

        write_file("train_annotated.txt", train_annotated)
        write_file("test_annotated.txt", test_annotated)
    elif args.type == 'new':
        train_empty = find_empty_annotation(train)
        test_empty = find_empty_annotation(test)
        train_test_empty = list(set(train_empty) | set(test_empty))

        for phrase in train_test_empty:
            if not phrase in annotated.keys():
                annotated[phrase] = ''

        write_file("new_annotation.txt", annotated)
