from nltk.corpus import stopwords

def plots_word_dictionary(words):
    wordsSet = set(words)
    wordsSet = remove_stop_words(wordsSet)

    i = 0
    plot_dictionary = {}
    for word in wordsSet:
        plot_dictionary[word] = i
        i += 1
    return plot_dictionary


def remove_stop_words(words):
    stopWordsSet = set(stopwords.words('english'))
    cleanSet = set()
    for word in words:
        if word in stopWordsSet:
            continue
        else:
            cleanSet.add(word)
    return cleanSet

