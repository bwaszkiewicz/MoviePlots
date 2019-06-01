from nltk.corpus import stopwords

def plots_word_dictionary(words):
    wordsSet = set(words)
    i = 0
    plot_dictionary = {}
    for word in wordsSet:
        plot_dictionary[i] = word
        i += 1
    return plot_dictionary

