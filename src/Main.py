import pandas as pd
import tensorflow as tf
import numpy as np
import sklearn

# Includes
from tensorflow import keras
from Preprocessing import Preprocessing, PandasProcessing


createCSV = True
outputFileCSV = "standardized.csv"
numberOfPlotsPerGenre = 250
numberOfInputWords = 500
# vocabulary_size = len(wordsDictionary)
vocabulary_size = 41480 #37649  # pomocniczo zeby nie puszczac calosci

if createCSV == False:
    Preprocessing.prepare_csv(outputFileCSV, numberOfPlotsPerGenre, numberOfInputWords)


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


x_data = standardizedData.PlotCorrected

index = 0
for x in x_data:
    index = 0
    for z in x:
        x[index] = float(z)
        index += 1

x_train = x_data[:1750]
y_train = standardizedData.GenreCorrected[:1750]
x_test = x_data[1750:]
y_test = standardizedData.GenreCorrected[1750:]

index=0
for y in y_train:
    y_train[index] = float(y)
    index += 1

index=1750
for y in y_test:
    y_test[index] = float(y)
    index += 1


print(f'{len(standardizedData)} movies in the standardized data')
print(f'{len(x_train)} plots in the train set')
print(f'{len(y_train)} genres in the train set')

print(f'{len(x_test)} plots in the test set')
print(f'{len(y_test)} genres in the test set')


x_train = keras.preprocessing.sequence.pad_sequences(x_train, padding='post', maxlen=numberOfInputWords)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, padding='post', maxlen=numberOfInputWords)


 # # --------------------------------- MODEL --------------
model = keras.Sequential()

# model.add(keras.layers.Embedding(vocabulary_size, 16))      # 16 wymiarow, parametry:(batch_size, sequence_length)
model.add(keras.layers.Embedding(input_dim=vocabulary_size,output_dim= 512, input_length=numberOfInputWords)) # model.add(keras.layers.Embedding(input_dim=vocabulary_size, output_dim=11, input_length=250))
model.add(keras.layers.GlobalAveragePooling1D())
# model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(512, input_shape=(numberOfInputWords,), activation=tf.nn.relu))
# model.add(keras.layers.Dense(512, activation=tf.nn.relu))
model.add(keras.layers.Dense(512, activation=tf.nn.tanh))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(11, activation='softmax'))  #- proponowane przy loss function = sparse_categorical_crossentropy
model.summary()
model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['acc'])
 # # alternatywna loss function do sprobowania:categorical_crossentropy, sparse_categorical_crossentropy, binary_crossentropy

history = model.fit(x_train, y_train, epochs=5, batch_size=25) # class_weight=10

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
