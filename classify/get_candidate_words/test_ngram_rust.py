from rust_ngram import extract_ngrams
import time
import os

text_pth = "train.csv"
max_n = 6
min_freq = 5

# count time between start and end
def count_time(start, end):
    return round(end - start, 2)

# using rust_ngram to get the frequency of ngrams
start = time.time()
results = extract_ngrams(text_pth, max_n, min_freq)
end = time.time()
print("Time Rust:", count_time(start, end))

with open("train_ngram.txt", "w") as f:
    results_sorted = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for result in results_sorted:
        f.write(result[0] + "\t" + str(result[1]) + "\n")