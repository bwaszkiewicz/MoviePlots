import pandas as pd
import numpy as np
import sklearn

# Includes

from Cleansing import GenresCleansing, PlotsCleansing
from Plots import PlotLearning
from Classifications import NaiveBayesClassification
from WordDictionary import PlotsWordDictionary

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
moviesPlot = movies[['PlotCorrected']].replace(r'\\n',' ', regex=True)
moviesPlot.to_csv('PlotCorrected.csv')

pd.set_option("display.max_colwidth", 10000)
totest = moviesPlot['PlotCorrected'].to_string(index=False)


wynik = PlotsWordDictionary.plots_word_dictionary(totest.split())



# text_file = open("Output.txt", "w")
# text_file.write(wynik.join(" "))
# text_file.close()

# test = movies[['Title', 'PlotCorrected', 'GenreCorrected']]
# test.to_csv('test.csv', ',')

# Uncomment to Naive Bayes Classification!
# NaiveBayesClassification.naive_bayes_classification(movies)

# plt = PlotLearning.plot_learning_curve(estimator=multinomialNB, title="Test",X=x_test, y=y_test.crime, cv=5)
# plt.show()
