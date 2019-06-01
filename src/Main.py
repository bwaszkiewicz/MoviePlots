import pandas as pd
import numpy as np
import sklearn
import json

# Includes

from Cleansing import GenresCleansing, PlotsCleansing
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
print(movies[['Genre', 'Count']].groupby(['Genre']).count().shape[0])

movies = GenresCleansing.genres_cleansing(movies)
moviesGenre = movies[['GenreCorrected', 'Count']].groupby(['GenreCorrected']).count()
moviesGenre.to_csv('GenreCorrected.csv', ',')

# preparing file with plots after correction

movies = PlotsCleansing.plot_cleansing(movies)
moviesPlot = movies[['PlotCorrected']].replace(r'\\n', ' ', regex=True)
moviesPlot.to_csv('PlotCorrected.csv')

plots = moviesPlot.PlotCorrected
print("Plots count: "+ str(len(plots)))
plotsString = plots[0]
for i in range(1, len(plots)):
    plotsString += " "+plots[i]
    print("plot nubmer: "+str(i))

# print(string)  # single plot

wordsDictionary = PlotsWordDictionary.plots_word_dictionary(plotsString.split())
wordsDictionary = RemoveStopWords.remove_stop_words_from_dictionary(wordsDictionary)
print(wordsDictionary)

# with open('file.txt', 'w') as file:
#     file.write(json.dumps(wordsDictionary))  # use `json.loads` to do the reverse

# text_file = open("Output.txt", "w")
# text_file.write(wordsDictionary)
# text_file.close()

#test = movies[['Title', 'PlotCorrected', 'GenreCorrected']]
#test.to_csv('test.csv', ',')

# Uncomment to Naive Bayes Classification!
# NaiveBayesClassification.naive_bayes_classification(movies)

# plt = PlotLearning.plot_learning_curve(estimator=multinomialNB, title="Test",X=x_test, y=y_test.crime, cv=5)
# plt.show()
