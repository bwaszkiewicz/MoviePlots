import pandas as pd


def create_empty_csv(filename):
    df = pd.DataFrame(columns=['ID', 'Words'])
    df.to_csv(filename, index=False, header=True)


def collect_all_words(words):
    words_list = list(words)
    all_words = words_list[0]

    for i in range(1, len(words_list)):
        all_words += " " + words_list[i]
    return all_words


def save_words(id, words, filename):
    words = set(words)
    all_words = collect_all_words(words)
    data = [id, all_words]

    outdata = [data]
    print(outdata)
    df = pd.DataFrame(outdata)

    with open(filename, 'a') as file:
        df.to_csv(file, index=False, header=True)
