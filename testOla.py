import tensorflow as tf
import numpy
import pandas

filename = 'wiki_movie_plots_deduped.csv'
data = pandas.read_csv(filename)
plots = data.Plot  # all plots

plot = plots[1]  # taking only one plot
plot = plot.split()
# print(plot)

plot_not_split = ""
for i, word in enumerate(plot):
    plot[i] = word.lower()
    if "," in word:
        plot[i] = word.replace(",", "")
    if ":" in word:
        plot[i] = word.replace(":", "")
    plot_not_split = plot_not_split + " " + plot[i]

# print(plot)
print(plot_not_split)

words = []
for word in plot:  # dividing plot into each word (all lowercase)
    if word != '.':
        if "." in word:
            word = word.replace(".", "")
        words.append(word.lower())

words = set(words)
# print(words)

word2int = {}
int2word = {}

plot_words_size = len(words)

for i, word in enumerate(words):
    word2int[word] = i
    int2word[i] = word

# print(word2int['fence'])
# print(int2word[word2int['fence']])

plot_sentences = plot_not_split.split(".")
sentences = []
for sentence in plot_sentences:  # dividing plot into sentences
    sentences.append(sentence.split())

# print(sentences)

data = []
WINDOW_SIZE = 2  # how many nearby words in sentence want to put to training data
for sentence in sentences:
    for i, word in enumerate(sentence):
        for nearby_word in sentence[max(i - WINDOW_SIZE, 0):  min(i + WINDOW_SIZE, len(sentence)) + 1]:
            if nearby_word != word:
                data.append([word, nearby_word])


# print(data)


def to_one_hot(data_point_index, size):  # converting numbers to one hot vectors
    temp = numpy.zeros(size)
    temp[data_point_index] = 1
    return temp


x_train = []  # input word
y_train = []  # output word

for data_word in data:
    x_train.append(to_one_hot(word2int[data_word[0]], plot_words_size))
    y_train.append(to_one_hot(word2int[data_word[1]], plot_words_size))

x_train = numpy.asarray(x_train)  # converting to numpy arrays
y_train = numpy.asarray(y_train)

# print(x_train)
# print(x_train.shape, y_train.shape)

x = tf.placeholder(tf.float32, shape=(None, plot_words_size))
y_label = tf.placeholder(tf.float32, shape=(None, plot_words_size))

EMBEDDING_DIM = 5
w1 = tf.Variable(tf.random_normal([plot_words_size, EMBEDDING_DIM]))
b1 = tf.Variable(tf.random_normal([EMBEDDING_DIM]))
hidden_representation = tf.add(tf.matmul(x, w1), b1)

w2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, plot_words_size]))
b2 = tf.Variable(tf.random_normal([plot_words_size]))
prediction = tf.nn.softmax(tf.add(tf.matmul(hidden_representation, w2), b2))

session = tf.Session()
init = tf.global_variables_initializer()
session.run(init)

cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))  # loss function

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_loss)    # training step
n_iterations = 10000

for _ in range(n_iterations):        # train for n iterations
    session.run(train_step, feed_dict={x: x_train, y_label: y_train})
    #print('loss is : ', session.run(cross_entropy_loss, feed_dict={x: x_train, y_label: y_train}))


vectors = session.run(w1 + b1)
# print(vectors)


def euclidean_dist(vec1, vec2):
    return numpy.sqrt(numpy.sum((vec1-vec2)**2))


def find_closest(word_index, vectors):
    min_dist = 10000
    min_index = -1
    query_vector = vectors[word_index]
    for index, vector in enumerate(vectors):
        if euclidean_dist(vector, query_vector) < min_dist and not numpy.array_equal(vector, query_vector):
            min_dist = euclidean_dist(vector, query_vector)
            min_index = index
    return min_index


print(int2word[find_closest(word2int['moon'], vectors)])
print(int2word[find_closest(word2int['night'], vectors)])
print(int2word[find_closest(word2int['the'], vectors)])
