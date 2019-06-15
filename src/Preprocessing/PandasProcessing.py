import time
import pandas as pd
import re

def genres_filtr(data):
    processed_data = pd.DataFrame(columns=['PlotCorrected', 'GenreCorrected'])

    start_time = time.time()
    for i in range(0, len(data)):
        correctedgenre = data.loc[i, :].GenreCorrected
        if correctedgenre == "thriller" \
                or correctedgenre == "science_fiction" \
                or correctedgenre == "romance" \
                or correctedgenre == "musical" \
                or correctedgenre == "horror" \
                or correctedgenre == "drama" \
                or correctedgenre == "crime" \
                or correctedgenre == "comedy" \
                or correctedgenre == "animation" \
                or correctedgenre == "adventure" \
                or correctedgenre == "action":
            processed_data = processed_data.append(data.loc[i, :], ignore_index=True)
        print("Processing: " + str(i) + "/" + str(len(data)))
    elapsed_time = time.time() - start_time
    print("#GenresFiltr Execution time: " + str(elapsed_time))

    # processed_data.to_csv('processed_data.csv', ',')
    return processed_data


def genres_normalization(data):
    data['GenreCorrected'] = data['GenreCorrected'].str.replace('action', '0')
    data['GenreCorrected'] = data['GenreCorrected'].str.replace('adventure', '1')
    data['GenreCorrected'] = data['GenreCorrected'].str.replace('animation', '2')
    data['GenreCorrected'] = data['GenreCorrected'].str.replace('comedy', '3')
    data['GenreCorrected'] = data['GenreCorrected'].str.replace('crime', '4')
    data['GenreCorrected'] = data['GenreCorrected'].str.replace('drama', '5')
    data['GenreCorrected'] = data['GenreCorrected'].str.replace('horror', '6')
    data['GenreCorrected'] = data['GenreCorrected'].str.replace('musical', '7')
    data['GenreCorrected'] = data['GenreCorrected'].str.replace('romance', '8')
    data['GenreCorrected'] = data['GenreCorrected'].str.replace('science_fiction', '9')
    data['GenreCorrected'] = data['GenreCorrected'].str.replace('thriller', '10')
    return data


def plot_normalization(data, words_dictionary):
    for j in range(0, len(data)):
        raw_plot_array = data.PlotCorrected[j].split()

        for i in range(0, len(raw_plot_array)):
            if raw_plot_array[i] in words_dictionary:
                raw_plot_array[i] = str(words_dictionary[raw_plot_array[i]])
            print("Index of word in dictionary: "+str(i)+"/"+str(len(words_dictionary)))
        data.PlotCorrected[j] = " ".join(raw_plot_array)
    data = plot_postnormalization_cleaning(data)
    return data

def plot_postnormalization_cleaning(data):
    # regex = r"[a-zA-Z]+"
    # data['PlotCorrected'] = re.sub("[a-zA-Z]+", "", data['PlotCorrected'].str)
    # str = re.sub("[a-zA-Z]+", "", str)
    data['PlotCorrected'] = data['PlotCorrected'].str.replace(r'[a-zA-Z]+', '')
    return data

def averaging_count_plot_words(data, numberOfInputWords):

    # words_in_data = 0
    # for i in range(0, len(data.PlotCorrected)):
    #     words_in_data += (len(data.PlotCorrected[i].split()))
    #
    # avarage_words = round(words_in_data / len(data.PlotCorrected))

    current_plot_words_list = []
    avarage_words = numberOfInputWords

    for i in range(0,len(data.PlotCorrected)):
        current_plot_words_list = data.PlotCorrected[i].split()
        if len(current_plot_words_list) < avarage_words:
            j = 0
            while len(current_plot_words_list) < avarage_words:
                current_plot_words_list.append(current_plot_words_list[j])
                j+=1
            data.PlotCorrected[i] = " ".join(current_plot_words_list)
        elif len(current_plot_words_list) > avarage_words:
            for k in range(avarage_words, len(current_plot_words_list)):
                del current_plot_words_list[avarage_words]
            data.PlotCorrected[i] = " ".join(current_plot_words_list)


    good = True
    basic_count = len(data.PlotCorrected[0].split())
    for i in range(1, len(data.PlotCorrected)):
        if basic_count != len(data.PlotCorrected[i].split()):
            good = False

    print("Po usrednieniu")
    print(good)


    return data


def normalization_test(data):
    list = []
    passed = True
    for i in range(0, len(data)):
        if bool(re.search("[^\d\s:]", data.PlotCorrected[i])):
            passed = False
            print("Bad normalization in "+str(i)+" line")
            list.append(i)

    print(list)
    return passed

