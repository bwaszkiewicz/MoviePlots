import pandas as pd
import tensorflow as tf
import numpy as np
import sklearn
# import json

from sklearn.utils import shuffle
from Preprocessing import GenresCleansing, PlotsCleansing, PandasProcessing, DataShuffle
from Plots import PlotLearning
from Classifications import NaiveBayesClassification
from WordDictionary import PlotsWordDictionary
from WordDictionary import RemoveStopWords

def prepare_csv(outputFileCSV, numberOfPlotsPerGenre, numberOfInputWords):

    # count of rows and cols
    movies = pd.read_csv('../input/wiki_movie_plots_deduped.csv', ',')
    nRow, nCol = movies.shape
    print(f'There are {nRow} rows and {nCol} columns')

    # creation of the column count for aggregation

    movies['Count'] = 1
    print("Number of different genres: "+str(movies[['Genre', 'Count']].groupby(['Genre']).count().shape[0]))

    movies = GenresCleansing.genres_cleansing(movies)
    moviesGenre = movies[['GenreCorrected', 'Count']].groupby(['GenreCorrected']).count()
    moviesGenre.to_csv('GenreCorrected.csv', ',')

    # preparing file with plots after correction

    movies = PlotsCleansing.plot_cleansing(movies)
    moviesPlot = movies[['PlotCorrected']].replace(r'\\n', ' ', regex=True)
    moviesPlot.to_csv('PlotCorrected.csv')


    clippedMoviesPanda = movies[['PlotCorrected', 'GenreCorrected']]

    rawProcessedMoviesPanda = PandasProcessing.genres_filtr(clippedMoviesPanda) # Only PlotCorrected, single Genres

    shuffledRawProcessedMoviesPanda = DataShuffle.cut_movies(rawProcessedMoviesPanda, numberOfPlotsPerGenre)

    if shuffledRawProcessedMoviesPanda is None:
        print("Too large count")
        exit(0)

    print(shuffledRawProcessedMoviesPanda)

    # rawProcessedMoviesPanda.to_csv('test.csv',',')
    # loop for make dictionary

    plots = shuffledRawProcessedMoviesPanda.PlotCorrected
    print("Plots count: "+ str(len(plots)))
    plotsString = plots[0]
    for i in range(1, len(plots)): #2
        plotsString += " "+plots[i]
        print("plot nubmer: "+str(i))

    # print(plotsString)  # single plot

    wordsDictionary = PlotsWordDictionary.plots_word_dictionary(plotsString.split())

    vocabulary_size = len(wordsDictionary)

    # print(wordsDictionary)

    # with open('file.txt', 'w') as file:
    #     file.write(json.dumps(wordsDictionary))  # use `json.loads` to do the reverse

    print("Words dictionary created!")

    genresStandardizedMoviesPanda = PandasProcessing.genres_normalization(shuffledRawProcessedMoviesPanda)
    standardizedMoviesPanda = PandasProcessing.plot_normalization(genresStandardizedMoviesPanda, wordsDictionary)

    standardizedMoviesPanda = PandasProcessing.averaging_count_plot_words(standardizedMoviesPanda, numberOfInputWords)

    standardizedMoviesPanda = shuffle(standardizedMoviesPanda)


    standardizedMoviesPanda.to_csv(outputFileCSV, ',')

    return vocabulary_size


