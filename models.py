"""
Models
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

MAX_VOCAB_LENGTH = 10000
MAX_INPUT_LENGTH = 15


def Create_Text_Vectorizer(train_sentences):
    text_vectorizer = TextVectorization(max_tokens=MAX_VOCAB_LENGTH,
                            output_mode="int",
                            output_sequence_length=MAX_INPUT_LENGTH)
    text_vectorizer.adapt(train_sentences)
    return text_vectorizer


def Create_Embedding_Layer():
    embedding = layers.Embedding(input_dim=MAX_VOCAB_LENGTH, # set input shape
                            output_dim=128, # set size of embedding vector
                            embeddings_initializer="uniform", # default, intialize randomly
                            input_length=MAX_INPUT_LENGTH, # how long is each input
                            name="embedding_1")
    return embedding


class Model():
    """
    Class with multiple model architectures
    """
    def __init__(self) -> None:
        pass

    def Baseline(self):
        """
        Create and return a Baseline model
        """
        # TF-IDF, Multinomial Naive Bayes
        # Create tokenization and modelling pipeline
        model = Pipeline([
                            ("tfidf", TfidfVectorizer()), # convert words to numbers using tfidf
                            ("clf", MultinomialNB()) # model the text
        ])
        return model

    def Model_1(self, train_sentences):
        """
        Create and return a simple Dense Model
        """
        inputs = layers.Input(shape=(1,), dtype="string") # inputs are 1-dimensional strings
        x = Create_Text_Vectorizer(train_sentences)(inputs) # turn the input text into numbers
        x = Create_Embedding_Layer()(x) # create an embedding of the numerized text
        x = layers.GlobalAveragePooling1D()(x) # lower the dimensionality of the embedding (try running the model without this layer and see what happens)
        outputs = layers.Dense(1, activation="sigmoid")(x) # create the output layer, want binary outputs so use sigmoid activation
        model = tf.keras.Model(inputs, outputs, name="model_1_dense") # construct the model

        # Compile model
        model.compile(loss="binary_crossentropy",
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=["accuracy"])
        return model

    def Model_2(self, train_sentences):
        """
        Create and return a LSTM Model
        """
        inputs = layers.Input(shape=(1,), dtype="string")
        x = Create_Text_Vectorizer(train_sentences)(inputs) # turn the input text into numbers
        x = Create_Embedding_Layer()(x) # create an embedding of the numerized text
        x = layers.LSTM(32, return_sequences=True)(x) # return vector for each word in the text (you can stack RNN cells as long as return_sequences=True)
        x = layers.LSTM(32)(x) # return vector for whole sequence
        x = layers.Dense(32, activation="relu")(x) # optional dense layer on top of output of LSTM cell
        outputs = layers.Dense(1, activation="sigmoid")(x)
        model = tf.keras.Model(inputs, outputs, name="model_2_LSTM")

        # Compile model
        model.compile(loss="binary_crossentropy",
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=["accuracy"])
        return model

    def Model_3(self, train_sentences):
        """
        Create and return a model with GRU cell
        """
        inputs = layers.Input(shape=(1,), dtype="string")
        x = Create_Text_Vectorizer(train_sentences)(inputs) # turn the input text into numbers
        x = Create_Embedding_Layer()(x) # create an embedding of the numerized text
        x = layers.GRU(32, return_sequences=True)(x) # return vector for each word in the text (you can stack RNN cells as long as return_sequences=True)
        x = layers.GRU(32)(x) # return vector for whole sequence
        x = layers.Dense(32, activation="relu")(x) # optional dense layer on top of output of GRU cell
        outputs = layers.Dense(1, activation="sigmoid")(x)
        model = tf.keras.Model(inputs, outputs, name="model_3_GRU")

        # Compile model
        model.compile(loss="binary_crossentropy",
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=["accuracy"])
        return model


    def Model_4(self, train_sentences):
        """
        Create and return a Bidirectional RNN model
        """
        inputs = layers.Input(shape=(1,), dtype="string")
        x = Create_Text_Vectorizer(train_sentences)(inputs) # turn the input text into numbers
        x = Create_Embedding_Layer()(x) # create an embedding of the numerized text
        x = layers.Bidirectional(layers.LSTM(64))(x)
        x = layers.Dense(32, activation="relu")(x) # optional dense layer on top of output of LSTM cell
        outputs = layers.Dense(1, activation="sigmoid")(x)
        model = tf.keras.Model(inputs, outputs, name="model_4_BiDir")

        # Compile model
        model.compile(loss="binary_crossentropy",
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=["accuracy"])
        return model


    def Model_5(self, train_sentences):
        """
        Create and return a model with Conv1D layer
        """
        inputs = layers.Input(shape=(1,), dtype="string")
        x = Create_Text_Vectorizer(train_sentences)(inputs) # turn the input text into numbers
        x = Create_Embedding_Layer()(x) # create an embedding of the numerized text
        x = layers.Conv1D(filters=32, kernel_size=5, activation="relu")(x)
        x = layers.GlobalMaxPool1D()(x)
        x = layers.Dense(32, activation="relu")(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)
        model = tf.keras.Model(inputs, outputs, name="model_5_Conv1D")

        # Compile model
        model.compile(loss="binary_crossentropy",
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=["accuracy"])
        return model