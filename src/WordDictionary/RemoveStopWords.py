from nltk.corpus import stopwords


## deprecated
def remove_stop_words_from_dictionary(words):
    stopWordsSet = set(stopwords.words('english'))
    i = 0
    words_dictionary = {}
    for word in words.values():
        if word in stopWordsSet:
            continue
        else:
            words_dictionary[i] = word
            i += 1
    return words_dictionary

