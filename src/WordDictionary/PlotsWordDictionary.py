def plots_word_dictionary(words):
    i = 0
    plot_dictionary = {}
    for word in words:
        if word in plot_dictionary:
            continue
        else:
            # plot_dictionary[word] = i
            plot_dictionary[i] = word
            i += 1
    return plot_dictionary

