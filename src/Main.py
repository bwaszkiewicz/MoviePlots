import pandas as pd
import numpy as np
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier


# Includes

from Cleansing import GenresCleansing, PlotsCleansing
from Plots import PlotLearning

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
moviesPlot = movies[['PlotCorrected']]
moviesPlot.to_csv('PlotCorrected.csv')

# -------------------------
# classification algorithms

# dummy classes
movies = pd.concat([movies, movies.GenreCorrected.str.get_dummies(sep='|')], axis=1)
print(f'Size of data set: {movies.shape} with dummies')

# split for train and test sets
moviesTrain, moviesTest = train_test_split(movies[movies.GenreCorrected != ''], random_state=42, test_size=0.20, shuffle=True)

# term frequency in plot (how many times it appears in the plot)
tfip = TfidfVectorizer(stop_words='english', smooth_idf=False, sublinear_tf=False, norm=None, analyzer='word')

x_train = tfip.fit_transform(moviesTrain.PlotCorrected)
x_test = tfip.transform(moviesTest.PlotCorrected)
trainRow = x_train.shape[0]
testRow = x_test.shape[0]
print(f'There are {trainRow} elements (rows) in the train set')
print(f'There are {testRow} elements (rows) in the test set')

type(x_train)
print(f'The corpus contains {len(x_train[0].toarray()[0])} words.')

# classes
y_train = moviesTrain[moviesTrain.columns[14:]] # !! TRZEBA SPRAWDZIC ILE KOLUMN POMINAC (testowo ustawiona wartosc 14)
y_test = moviesTest[moviesTest.columns[14:]]
print(f'There are {len(y_train.columns)} columns in y_train')
print(f'There are {len(y_test.columns)} columns in y_test')

# Multinomial Naive Bayes Classification
multinomialNB = OneVsRestClassifier(MultinomialNB(fit_prior=True, class_prior=None))
multinomialNB.fit(x_train, y_train.crime)
prediction = multinomialNB.predict(x_test)
print(f'Test accuracy is {sklearn.metrics.accuracy_score(y_test.crime, prediction)}')

# plt = PlotLearning.plot_learning_curve(estimator=multinomialNB, title="Test",X=x_test, y=y_test.crime, cv=5)
# plt.show()