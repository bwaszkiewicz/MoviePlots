import pandas as pd
import tensorflow as tf
import numpy as np
import sklearn

# Includes
from tensorflow import keras
from Preprocessing import Preprocessing, PandasProcessing

#
# createCSV = True
# outputFileCSV = "standardized.csv"
# numberOfPlotsPerGenre = 200
# numberOfInputWords = 250
#
# if createCSV == True:
#     Preprocessing.prepare_csv(outputFileCSV, numberOfPlotsPerGenre, numberOfInputWords)


##################################### OLA ################################################
standardizedData = pd.read_csv('standardized.csv', ',')
 # TESTY
print("Wynik testu normalizacji: ")
print(PandasProcessing.normalization_test(standardizedData))
print("(True - dane poprawne; False - dane nie poprawne")




standardizedDataSize = len(standardizedData)

for i, row in standardizedData.iterrows():
    new = standardizedData.at[i, 'PlotCorrected'].split()
    standardizedData.at[i, 'PlotCorrected'] = new

standardizedData['PlotCorrected']

x_data = standardizedData.PlotCorrected

index = 0
for x in x_data:
    index = 0
    for z in x:
        x[index] = float(z)
        index += 1

x_train = x_data[:1200]
y_train = standardizedData.GenreCorrected[:1200]
x_test = x_data[1200:]
y_test = standardizedData.GenreCorrected[1200:]


index=0
for y in y_train:
    y_train[index] = float(y)
    index += 1

index=1200
for y in y_test:
    y_test[index] = float(y)
    index += 1




print(f'{len(standardizedData)} movies in the standardized data')
print(f'{len(x_train)} plots in the train set')
print(f'{len(y_train)} genres in the train set')

print(f'{len(x_test)} plots in the test set')
print(f'{len(y_test)} genres in the test set')

# vocabulary_size = len(wordsDictionary)
vocabulary_size = 37649  # pomocniczo zeby nie puszczac calosci


x_train = keras.preprocessing.sequence.pad_sequences(x_train, padding='post', maxlen=250)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, padding='post', maxlen=250)



 # # --------------------------------- MODEL --------------
model = keras.Sequential()

# model.add(keras.layers.Embedding(vocabulary_size, 16))      # 16 wymiarow, parametry:(batch_size, sequence_length)
model.add(keras.layers.Embedding(input_dim=vocabulary_size,output_dim= 512, input_length=250)) # model.add(keras.layers.Embedding(input_dim=vocabulary_size, output_dim=11, input_length=250))
model.add(keras.layers.GlobalAveragePooling1D())
# model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(512, input_shape=(250,), activation=tf.nn.relu))
model.add(keras.layers.Dense(512, activation=tf.nn.relu))
model.add(keras.layers.Dense(512, activation=tf.nn.relu))
model.add(keras.layers.Dense(11, activation='softmax'))  #- proponowane przy loss function = sparse_categorical_crossentropy
model.summary()
model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['acc'])
 # # alternatywna loss function do sprobowania:categorical_crossentropy, sparse_categorical_crossentropy, binary_crossentropy
history = model.fit(x_train, y_train, epochs=10, batch_size=200) # class_weight=10


# z mnista

# model = keras.Sequential()
# model.add(keras.layers.Dense(512, input_shape=(250,)))
# model.add(keras.layers.Activation('relu'))
# model.add(keras.layers.Dropout(0.2))
#
# model.add(keras.layers.Dense(512))
# model.add(keras.layers.Activation('relu'))
# model.add(keras.layers.Dropout(0.2))
#
# model.add(keras.layers.Dense(11))
# model.add(keras.layers.Activation('softmax'))
# model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
#
# history = model.fit(x_train, y_train,
#           batch_size=500, epochs=20,
#           verbose=2)




# print(f'{x_train.size} size plots in the train set')
# print(f'{y_train.size} size genres in the train set')
#
# print(f'{x_test.shape[0]}aaaaaaaaaaaaaaaaa')
# print(f'{y_test.shape[0]}aaaaaaaaaaaaaaaaa')


# model = keras.Sequential()
# model.add(keras.layers.Embedding(vocabulary_size, 20, input_length=250))
# model.add(keras.layers.Dropout(0.15))
# model.add(keras.layers.GlobalMaxPool1D())
# model.add(keras.layers.Dense(1, activation='sigmoid'))
#
# model.compile(class_mode='categorical', optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
#
# history = model.fit(x_train, y_train,
#                     class_weight=11,
#                     epochs=20,
#                     batch_size=500,
#                     validation_split=0.1)


results = model.evaluate(x_test, y_test, batch_size=100)
print(results)

for layer in model.layers:
    print(layer.output_shape)

##################################### OLA ################################################



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
