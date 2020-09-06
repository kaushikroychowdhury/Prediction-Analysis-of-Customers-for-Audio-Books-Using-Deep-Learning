import numpy as np
import tensorflow as tf

### CREATING THE ML ALGO

# Data
npz = np.load('Audiobooks_data_train.npz')

train_inputs = npz['inputs'].astype(np.float)
train_targets = npz['targets'].astype(np.int)

# targets should be int , because of sparse_categorical_crossentropy (we want one-hot encoding)

npz = np.load('Audiobooks_data_validation.npz')
validation_inputs , validation_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)
npz = np.load('Audiobooks_data_test.npz')
test_inputs , test_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)


### MODEL ( Outline, Optimizers, Loss, Early Stopping, Training

input_size = 10
output_size = 2
hidden_layer_size = 50

## Outlining the model

model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'),
    tf.keras.layers.Dense(output_size, activation='softmax')
])

## Optimizer and Loss Function

model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

## Training

batch_size = 100
max_epoch = 100

# set an early stopping mechanism .. let's set "patience = 2" to be a bit tolerant against random validation loss increase

early_stopping = tf.keras.callbacks.EarlyStopping(patience=2)

# fit the model

model.fit(train_inputs, train_targets, batch_size= batch_size, epochs= max_epoch,
          validation_data=(validation_inputs, validation_targets),
          callbacks=[early_stopping], verbose = 2)

### TEST THE MODEl

test_loss , test_accuracy = model.evaluate(test_inputs, test_targets)
print('Test loss : {0:.2f}    Test accuracy : {1:.2f}%'.format(test_loss, test_accuracy*100.))
