import tensorflow as tf
import numpy
import pandas
import InputFormat as IF
import SaveDefs as SD

# INIT
filename = 'wiki_movie_plots_deduped.csv'
data = pandas.read_csv(filename)  # current filename
plots = data.Plot
words = []
sentences = []
word2int = {}
int2word = {}
OUTPUT_FILENAME = "outWords.csv"

SD.create_empty_csv(OUTPUT_FILENAME)

for i in range(0, 2):
    plot = plots[i]  # single plot
    print(data.Title[i])

    plot = plot.lower()
    refactorPlot = ""
    for word in plot:
        if '—' in word:
            word = word.replace("—", " ")
        refactorPlot += word

    print(refactorPlot)
    plotSplit = refactorPlot.split()
    # CREATE plotNotSplit
    plotNotSplit = IF.create_plot_not_split(plotSplit)

    # CREATE WORDS
    words += IF.create_words(plotSplit)
    SD.save_words(i, words, OUTPUT_FILENAME)

    # raw sentences is a list of sentences.
    raw_sentences = plotNotSplit.split('.')
    for sentence in raw_sentences:
        sentences.append(sentence.split())

words = set(words)  # all duplicates are removed
vocab_size = len(words)  # gives the total number of unique words

for j, word in enumerate(words):
    word2int[word] = j
    int2word[j] = word

print("Words: ", words)
print("Sentences: ", sentences)

# CREATE DATA
WINDOW_SIZE = 2
data = IF.create_data(sentences, WINDOW_SIZE)

print("Data: ", data)


# function to convert numbers to one hot vectors
def to_one_hot(data_point_index, vocab_size):
    temp = numpy.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp


x_train = []  # input word
y_train = []  # output word
for data_word in data:
    x_train.append(to_one_hot(word2int[data_word[0]], vocab_size))
    y_train.append(to_one_hot(word2int[data_word[1]], vocab_size))
# convert them to numpy arrays
x_train = numpy.asarray(x_train)
y_train = numpy.asarray(y_train)

x = tf.placeholder(tf.float32, shape=(None, vocab_size))
y_label = tf.placeholder(tf.float32, shape=(None, vocab_size))

EMBEDDING_DIM = 5  # you can choose your own number
W1 = tf.Variable(tf.random_normal([vocab_size, EMBEDDING_DIM]))
b1 = tf.Variable(tf.random_normal([EMBEDDING_DIM]))  # bias
hidden_representation = tf.add(tf.matmul(x, W1), b1)

W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, vocab_size]))
b2 = tf.Variable(tf.random_normal([vocab_size]))
prediction = tf.nn.softmax(tf.add(tf.matmul(hidden_representation, W2), b2))

# Tensor flow

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
# define the loss function:
cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))
# define the training step:
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_loss)
n_iters = 10000
# train for n_iter iterations
for _ in range(n_iters):
    sess.run(train_step, feed_dict={x: x_train, y_label: y_train})
    #  print('loss is : ', sess.run(cross_entropy_loss, feed_dict={x: x_train, y_label: y_train}))

vectors = sess.run(W1 + b1)


def euclidean_dist(vec1, vec2):
    return numpy.sqrt(numpy.sum((vec1 - vec2) ** 2))


def find_closest(word_index, vectors):
    min_dist = 10000  # to act like positive infinity
    min_index = -1
    query_vector = vectors[word_index]
    for index, vector in enumerate(vectors):
        if euclidean_dist(vector, query_vector) < min_dist and not numpy.array_equal(vector, query_vector):
            min_dist = euclidean_dist(vector, query_vector)
            min_index = index
    return min_index


#print(int2word[find_closest(word2int['shot'], vectors)])
