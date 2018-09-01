import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt

NUM_WORDS = 10000
(train_data, train_label), (test_data, test_label) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)


def multi_hot_sequences(sequences, dimension):
    result = np.zeros((len(sequences), dimension))
    for (i, word_indices) in enumerate(sequences):
        result[i, word_indices] = 1.0
    return result


train_data = multi_hot_sequences(train_data, NUM_WORDS)
test_data = multi_hot_sequences(test_data, NUM_WORDS)


baseline_model=keras.Sequential([
    keras.layers.Dense(16,activation=tf.nn.relu,input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16,activation=tf.nn.relu),
    keras.layers.Dense(1,activation=tf.nn.sigmoid),
])
baseline_model.compile(optimizer=tf.train.AdamOptimizer(),
                       loss='binary_crossentropy',
                       metrics=['accuracy','binary_crossentropy'])
baseline_model.summary()
baseline_history=baseline_model.fit(train_data,train_label,256,20,validation_data=(test_data,test_label),verbose=2)

small_model=keras.Sequential([
    keras.layers.Dense(4,activation=tf.nn.relu,input_shape=(NUM_WORDS,)),
    keras.layers.Dense(4,activation=tf.nn.relu),
    keras.layers.Dense(1,activation=tf.nn.sigmoid)
])
small_model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy','binary_crossentropy'])
small_model.summary()
small_hisroty=small_model.fit(train_data,train_label,256,20,validation_data=(test_data,test_label),verbose=2)

bigger_model=keras.Sequential([
    keras.layers.Dense(64,activation=tf.nn.relu,input_shape=(NUM_WORDS,)),
    keras.layers.Dense(64,activation=tf.nn.relu),
    keras.layers.Dense(1,activation=tf.nn.sigmoid)
])
bigger_model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy','binary_crossentropy'])
bigger_model.summary()
bigger_history=bigger_model.fit(train_data,train_label,256,20,validation_data=(test_data,test_label),verbose=2)

#l2-norm:mean of square difference (mse)
l2_model=keras.Sequential([
    keras.layers.Dense(16,activation=tf.nn.relu,kernel_regularizer=keras.regularizers.l2(0.001),input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16,activation=tf.nn.relu,kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dense(1,activation=tf.nn.sigmoid)
])
l2_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy','binary_crossentropy'])
l2_model.summary()
l2_history=l2_model.fit(train_data,train_label,256,20,2,validation_data=(test_data,test_label))

dpt_model=keras.Sequential([
    keras.layers.Dense(16,activation=tf.nn.relu,input_shape=(NUM_WORDS,)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(16,activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1,activation=tf.nn.sigmoid)
])
dpt_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy','binary_crossentropy'])
dpt_model.summary()
dpt_history=dpt_model.fit(train_data,train_label,256,20,2,validation_data=(test_data,test_label))


def plot_history(histories,key='binary_crossentropy'):
    plt.figure(figsize=(10,8))
    for name,history in histories:
        val=plt.plot(history.epoch,history.history['val_'+key],'--',label=name.title()+' Val')
        plt.plot(history.epoch,history.history[key],color=val[0].get_color(),label=name.title()+' Train')
    plt.xlabel('epoch')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()
    plt.xlim([0,max(history.epoch)])

plot_history([('baseline',baseline_history),
              ('small',small_hisroty),
              ('bigger',bigger_history)])
plot_history([('baseline',baseline_history),
              ('l2',l2_history),
              ('dropout',dpt_history)])
plt.show()


