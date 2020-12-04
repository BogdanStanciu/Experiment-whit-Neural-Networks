import tensorflow as tf
import mitdeeplearning as mdl
import numpy as np 
import os 
import time 
import functools
from IPython import display as ipythondisplay
from tqdm import tqdm

"""
Using the MIT music dataset we will create a RNN to learning
the pattern in all the songs, so will be able to create new 
songs starting from one note
"""

# Function to convert all the songs string to a vectorized
# (numeric) representation
def vectorize_string(string):
    vectorized_output = np.array([char2index[char] for char in string])
    return vectorized_output

# Divide the text into example of sequences that will use
# during traning. Each sequence into the RNN will contain
# seq_length, define target sequence for each input sequence
def get_batch(vectorized_songs, seq_length, batch_size):
    # length of the vectorize songs string
    n = vectorized_songs.shape[0] - 1

    # Randomly choose the starting indices for the examples in the traning batch
    idx = np.random.choice(n - seq_length, batch_size)

    # Construct a list of input sequences for the traning batch
    input_batch = [vectorized_songs[i : i + seq_length] for i in idx]
    output_batch = [vectorized_songs[i+1 : i + seq_length + 1] for i in idx]

    x_batch = np.reshape(input_batch, [batch_size, seq_length])
    y_batch = np.reshape(output_batch, [batch_size, seq_length])
    
    return x_batch, y_batch

# Define a function for creating a LSTM layer with N un
def LSTM(rnn_units): 
    return tf.keras.layers.LSTM(
        rnn_units, 
        return_sequences=True, 
        recurrent_initializer='glorot_uniform',
        recurrent_activation='sigmoid', 
        stateful=True
    )

# Define a function for creating a model with N layers with differents types
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        LSTM(rnn_units), 
        tf.keras.layers.Dense(vocab_size)  
    ])

    return model

# Define a loss function for the model
def compute_loss(labels, logits): 
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    return loss



# Loading traning data from mit library
songs = mdl.lab1.load_training_data()
songs_joined = "\n\n".join(songs)

# Find a unique characters in the joined string
vocab = sorted(set(songs_joined))

# Define the numerical representation of text
# Create a mapping from char to unique index 
char2index = {u:i for i, u in enumerate(vocab)}

# Create a mapping from indexs to char
idx2char = np.array(vocab)

vectorized_songs = vectorize_string(songs_joined)


# Optimization parameters:
num_training_iterations = 2000  # Increase this to train longer
batch_size = 32  # Experiment between 1 and 64
seq_length = 100  # Experiment between 50 and 500
learning_rate = 5e-3  # Experiment between 1e-5 and 1e-1

# Model parameters: 
vocab_size = len(vocab)
embedding_dim = 256 
rnn_units = 2048  # Experiment between 1 and 2048

# Checkpoint location: 
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size)
optimizer = tf.keras.optimizers.Adam(learning_rate)


@tf.function
def train_step(x, y): 
    with tf.GradientTape() as tape:
        y_hat = model(x)

        # Compute loss 
        loss = compute_loss(y, y_hat)
        # Compute gradient
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


##################
# Begin training!#
##################

# Create plot for visualizing the loss
history = []
plotter = mdl.util.PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss')
if hasattr(tqdm, '_instances'): tqdm._instances.clear()

for iter in tqdm(range(num_training_iterations)): 
    # Grab a batch and propagate it throgh the network
    x_batch, y_batch = get_batch(vectorized_songs, seq_length, batch_size)    
    loss = train_step(x_batch, y_batch)

    history.append(loss.numpy().mean())
    plotter.plot(history)
    
    # Upload model with the changed weigths
    if iter % 100 == 0: 
        model.save_weights(checkpoint_prefix)

model.save_weights(checkpoint_prefix)
