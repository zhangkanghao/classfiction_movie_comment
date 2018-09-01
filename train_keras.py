import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 得到最常出现的10000个词.舍弃一些较为稀少的词
NUM_WORDS = 10000
(train_data, train_label), (test_data, test_label) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)


# transfer into multi-hot-encode
def multi_hot_sequence(sequences, dimension):
    result = np.zeros((len(sequences), dimension))
    for i, index in enumerate(sequences):
        result[i, index] = 1.0
    return result


train_data = multi_hot_sequence(train_data, NUM_WORDS)
test_data = multi_hot_sequence(test_data, NUM_WORDS)

# dictionary
word_index = keras.datasets.imdb.get_word_index(path='../datasets/imdb_word_index.json')
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2  # unknown
word_index['<UNUSED>'] = 3
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# padding sequences
train_data = keras.preprocessing.sequence.pad_sequences(sequences=train_data, maxlen=256, padding='post',
                                                        truncating='post',
                                                        value=word_index['<PAD>'])
test_data = keras.preprocessing.sequence.pad_sequences(sequences=test_data, maxlen=256, padding='post',
                                                       truncating='post',
                                                       value=word_index['<PAD>'])

# create validation data
x_val = train_data[:10000]
partial_x_train = train_data[10000:]
y_val = train_label[:10000]
patial_y_train = train_label[10000:]

# build model
vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit(x=partial_x_train,
                    y=patial_y_train,
                    batch_size=512,
                    epochs=40,
                    validation_data=(x_val, y_val),
                    verbose=1)

result = model.evaluate(test_data, test_label)
print(result)

# visible
history_dict = history.history
acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epoch = range(1, len(acc) + 1)

plt.plot(epoch, acc, 'bo', label='Training acc')
plt.plot(epoch, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()

plt.clf()
plt.plot(epoch, loss, 'bo', label='Training loss')
plt.plot(epoch, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
