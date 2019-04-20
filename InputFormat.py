def create_plot_not_split(plot_splited):
    plot_not_split = ""
    for j, word in enumerate(plot_splited):
        if "," in word:
            plot_splited[j] = word.replace(",", "")
        if ":" in word:
            plot_splited[j] = word.replace(":", "")
        if '[' in word and ']' in word:
            first = word.find('[')
            second = word.find(']')
            plot_splited[j] = word[:first] + word[second + 1:]
        plot_not_split = plot_not_split + " " + plot_splited[j]
    return plot_not_split


def create_data(sentences, window_size):
    data = []
    for sentence in sentences:
        for word_index, word in enumerate(sentence):
            for nb_word in sentence[max(word_index - window_size, 0): min(word_index + window_size, len(sentence)) + 1]:
                if nb_word != word:
                    data.append([word, nb_word])
    return data


def create_words(plot_split):
    words = []
    for word in plot_split:
        if word != '.' and word != ',' and word != ':':
            if "." in word:
                word = word.replace(".", "")
            if ':' in word:
                word = word.replace(":", "")
            if "," in word:
                word = word.replace(",", "")
            if '[' in word and ']' in word:
                first = word.find('[')
                second = word.find(']')
                word = word[:first] + word[second + 1:]
        words.append(word)
    return words
