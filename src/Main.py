import pandas as pd
import tensorflow as tf
import numpy as np
import sklearn

# Includes
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
from Preprocessing import Preprocessing, PandasProcessing


createCSV = True
outputFileCSV = "standardized.csv"
numberOfPlotsPerGenre = 200
numberOfInputWords = 250

if createCSV == True:
    Preprocessing.prepare_csv(outputFileCSV, numberOfPlotsPerGenre, numberOfInputWords)


##################################### OLA ################################################
standardizedData = pd.read_csv('standardized.csv', ',')
 # TESTY
print("Wynik testu normalizacji: ")
print(PandasProcessing.normalization_test(standardizedData))
print("(True - dane poprawne; False - dane nie poprawne")

#
#
#
# standardizedDataSize = len(standardizedData)
#
# for i, row in standardizedData.iterrows():
#     new = standardizedData.at[i, 'PlotCorrected'].split()
#     standardizedData.at[i, 'PlotCorrected'] = new
# #standardizedData['PlotCorrected']
# #standardizedData['PlotCorrected'] = standardizedData['PlotCorrected'].astype(float);
# #standardizedData['GenreCorrected'] = standardizedData['GenreCorrected'].astype(float);
#
# standardizedData['PlotCorrected'].tolist()
#
# x_data = standardizedData.PlotCorrected
# # y_data = standardizedData.GenreCorrected
# index = 0
#
# ####################################UJEDNOLICENIE DLUGOSCI - nie dziala
# # for x in x_data:
# #     if len(x) > 20:
# #      x = x[:20]
# #      x_data[index] = x
# #     index += 1
#
# for x in x_data:
#     index = 0
#     for z in x:
#         x[index] = float(z)
#         index += 1
#
# # index=0
# # for y in y_data:
# #     y_data[index] = float(y)
# #     index += 1
#
#
#
# x_train = x_data[:1000]
# # y_train = y_data[:1000]
# partial_x_train = x_data[1000:1200]
# # partial_y_train = y_data[1000:1200]
# x_test = x_data[1200:]
# # y_test = y_data[1200:]
#
#
#
# #
# # x_train = standardizedData.PlotCorrected[:1000]
# y_train = standardizedData.GenreCorrected[:1000]
# #
# # partial_x_train = standardizedData.PlotCorrected[1000:1200]
# partial_y_train = standardizedData.GenreCorrected[1000:1200]
# #
# # x_test = standardizedData.PlotCorrected[1200:]
# y_test = standardizedData.GenreCorrected[1200:]
#
# # for x in partial_x_train:
# #     index = 0
# #     for z in x:
# #         x[index] = float(z)
# #         index += 1
# #
# # for x in x_test:
# #     index = 0
# #     for z in x:
# #         x[index] = float(z)
# #         index += 1
# #
# index=0
# for y in partial_y_train:
#     partial_y_train[index] = float(y)
#     index += 1
#
# index=0
# for y in y_train:
#     y_train[index] = float(y)
#     index += 1
#
# print(f'{len(standardizedData)} movies in the standardized data')
# print(f'{len(x_train)} plots in the train set')
# print(f'{len(y_train)} genres in the train set')
#
# print(f'{len(x_test)} plots in the test set')
# print(f'{len(y_test)} genres in the test set')
#
#
# print(len(x_train[1]))
# print((x_train[5]))
#
# # vocabulary_size = len(wordsDictionary)
# vocabulary_size = 37649 # pomocniczo zeby nie puszczac calosci
# print(vocabulary_size)
#
#  # # --------------------------------- wrzucone na przyszlosc --------------
# model = keras.Sequential()
# model.add(keras.layers.Embedding(vocabulary_size, 16))      # 16 wymiarow, parametry:(batch_size, sequence_length)
# model.add(keras.layers.GlobalAveragePooling1D())
# model.add(keras.layers.Dense(16, activation=tf.nn.relu))
# model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
#  # # model.add(keras.layers.Dense(3, activation='softmax')) - proponowane przy loss function = sparse_categorical_crossentropy
#  #
# model.summary()
#  #
#
#
# # x_train = np.array(x_train)
# # partial_x_train = np.array(partial_x_train)
# # y_train = np.array(y_train)
# # partial_y_train = np.array(partial_y_train)
#
# enc = OneHotEncoder(sparse=False)
# #x_train = enc.fit_transform(x_train)
# #partial_x_train = enc.fit_transform(partial_x_train)
#
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
#  # # alternatywna loss function do sprobowania:categorical_crossentropy, sparse_categorical_crossentropy,
#  #
#  # # --------------------------------- wrzucone na przyszlosc --------------
#
# history = model.fit(partial_x_train, partial_y_train,  epochs=40, batch_size=512, validation_data=(x_train, y_train), verbose=1)
# #results = model.evaluate(x_test, y_test)
# #print(results)
#
# ##################################### OLA ################################################
#
#
#
# #
# #
# # # with open('file.txt', 'w') as file:
# # #     file.write(json.dumps(wordsDictionary))  # use `json.loads` to do the reverse
# #
# # # text_file = open("Output.txt", "w")
# # # text_file.write(wordsDictionary)
# # # text_file.close()
# #
# # #test = movies[['Title', 'PlotCorrected', 'GenreCorrected']]
# # #test.to_csv('test.csv', ',')
# #
# # # Uncomment to Naive Bayes Classification!
# # # NaiveBayesClassification.naive_bayes_classification(movies)
# #
# # # plt = PlotLearning.plot_learning_curve(estimator=multinomialNB, title="Test",X=x_test, y=y_test.crime, cv=5)
# # # plt.show()
