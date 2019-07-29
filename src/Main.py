import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sklearn


# Includes
from tensorflow import keras
from Preprocessing import Preprocessing, PandasProcessing
import SettingsManager

# Read settings
createCSV, outputFileCSV, numberOfPlotsPerGenre, numberOfInputWords, vocabulary_size, epochsNumber, historyBatchSize, \
resultsBatchSize, learningFactor = SettingsManager.load_settings()

if createCSV == True:
    vocabulary_size = Preprocessing.prepare_csv(outputFileCSV, numberOfPlotsPerGenre, numberOfInputWords)
    SettingsManager.change_setting("vocabulary_size", vocabulary_size)

standardizedData = pd.read_csv(outputFileCSV, ',')
 # TESTY
print("Wynik testu normalizacji: ")
print(PandasProcessing.normalization_test(standardizedData))
print("(True - dane poprawne; False - dane nie poprawne")

# -- preparing data for the model

for i, row in standardizedData.iterrows():
    new = standardizedData.at[i, 'PlotCorrected'].split()
    standardizedData.at[i, 'PlotCorrected'] = new


x_data = standardizedData.PlotCorrected

x_dataSize = len(x_data)

index = 0
for x in x_data:
    index = 0
    for z in x:
        x[index] = float(z)
        index += 1

x_train = x_data[:int(x_dataSize*learningFactor)]
y_train = standardizedData.GenreCorrected[:int(x_dataSize*learningFactor)]
x_test = x_data[int(x_dataSize*learningFactor):]
y_test = standardizedData.GenreCorrected[int(x_dataSize*learningFactor):]

index=0
for y in y_train:
    y_train[index] = float(y)
    index += 1

index=int(x_dataSize*learningFactor)
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

# -- preparing the sequential model
model = keras.Sequential()
model.add(keras.layers.Embedding(input_dim=vocabulary_size,output_dim= 512, input_length=numberOfInputWords)) # model.add(keras.layers.Embedding(input_dim=vocabulary_size, output_dim=11, input_length=250))
model.add(keras.layers.GlobalAveragePooling1D())
# model.add(keras.layers.Dense(512, activation=tf.nn.tanh))
# model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(11, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['acc'])

history = model.fit(x_train, y_train, epochs=epochsNumber, batch_size=historyBatchSize)

results = model.evaluate(x_test, y_test, batch_size=resultsBatchSize)

# -- printing results
print(results)

for layer in model.layers:
    print(layer.output_shape)

history_dict = history.history
print(history_dict.keys())

acc = history_dict['acc']
loss = history_dict['loss']
epochs = range(1, len(acc) + 1)

# "bo" -> "blue dot"
plt.plot(epochs, loss, 'r', label='Training loss')
# b -> "solid blue line"
plt.plot(epochs, acc, 'g', label='Training accuracy')
plt.title('Training accuracy and loss')
plt.xlabel('Epochs')
plt.ylabel('Loss/Accuracy')
plt.legend()

plt.show()
