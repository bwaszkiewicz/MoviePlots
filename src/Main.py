import pandas as pd
import tensorflow as tf
import numpy as np
import sklearn
import json

# Includes
from tensorflow import keras
from Preprocessing import GenresCleansing, PlotsCleansing, PandasProcessing
from Plots import PlotLearning
from Classifications import NaiveBayesClassification
from WordDictionary import PlotsWordDictionary
from WordDictionary import RemoveStopWords


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
rawProcessedMoviesPanda.to_csv('test.csv',',')




# # loop for make dictionary
#
# plots = moviesPlot.PlotCorrected
# print("Plots count: "+ str(len(plots)))
# plotsString = plots[0]
# for i in range(1, 2): #len(plots)):
#     plotsString += " "+plots[i]
#     print("plot nubmer: "+str(i))
#
# # print(string)  # single plot
#
# wordsDictionary = PlotsWordDictionary.plots_word_dictionary(plotsString.split())
# wordsDictionary = RemoveStopWords.remove_stop_words_from_dictionary(wordsDictionary)
# print(wordsDictionary)
#
#
#
#
#
# vocabulary_size = len(wordsDictionary)
#
# # --------------------------------- wrzucone na przyszlosc --------------
# model = keras.Sequential()
# model.add(keras.layers.Embedding(vocabulary_size, 16))      # 16 wymiarow, parametry:(batch_size, sequence_length)
# model.add(keras.layers.GlobalAveragePooling1D())
# model.add(keras.layers.Dense(16, activation=tf.nn.relu))
# model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
# # model.add(keras.layers.Dense(3, activation='softmax')) - proponowane przy loss function = sparse_categorical_crossentropy
#
# model.summary()
#
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['acc'])
# # alternatywna loss function do sprobowania: sparse_categorical_crossentropy,
#
# # --------------------------------- wrzucone na przyszlosc --------------
#
#
# # with open('file.txt', 'w') as file:
# #     file.write(json.dumps(wordsDictionary))  # use `json.loads` to do the reverse
#
# # text_file = open("Output.txt", "w")
# # text_file.write(wordsDictionary)
# # text_file.close()
#
# #test = movies[['Title', 'PlotCorrected', 'GenreCorrected']]
# #test.to_csv('test.csv', ',')
#
# # Uncomment to Naive Bayes Classification!
# # NaiveBayesClassification.naive_bayes_classification(movies)
#
# # plt = PlotLearning.plot_learning_curve(estimator=multinomialNB, title="Test",X=x_test, y=y_test.crime, cv=5)
# # plt.show()
