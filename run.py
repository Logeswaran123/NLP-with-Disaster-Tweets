import argparse
import tensorflow as tf

from models import Model
from preprocess import load_and_preprocess
from utils import calculate_results, create_tensorboard_callback

SAVE_DIR = "model_logs"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--data", required=True, default="dataset",
                        help="Path to dataset dir", type=str)
    args = parser.parse_args()
    dataset_path = args.data


    # Load and preprocess training and test data
    train_sentences, val_sentences, train_labels, val_labels = load_and_preprocess(dataset_path)

    models = Model()

    # Create Baseline model
    baseline_model = models.Baseline()

    # Fit model on training data
    baseline_model.fit(train_sentences, train_labels)

    # Predict on validation data and calculate scores
    baseline_preds = baseline_model.predict(val_sentences)
    baseline_results = calculate_results(y_true=val_labels, y_pred=baseline_preds)
    print("\nBaseline model Results:\n", baseline_results)
    print("\n-----------------------------------------------------\n")


    # Create Dense model
    model_1 = models.Model_1(train_sentences)

    # Fit model on training data
    model_1_history = model_1.fit(train_sentences, # input sentences can be a list of strings due to text preprocessing layer built-in model
                            train_labels,
                            epochs=5,
                            validation_data=(val_sentences, val_labels),
                            callbacks=[create_tensorboard_callback(dir_name=SAVE_DIR, 
                                                                    experiment_name="simple_dense_model")])
    model_1.evaluate(val_sentences, val_labels)

    # Predict on validation data and calculate scores
    model_1_pred_probs = model_1.predict(val_sentences)
    model_1_results = calculate_results(y_true=val_labels, y_pred=tf.squeeze(tf.round(model_1_pred_probs)))
    print("\nDense model Results:\n", model_1_results)
    print("\n-----------------------------------------------------\n")


    # Create LSTM model
    model_2 = models.Model_2(train_sentences)

    # Fit model on training data
    model_2_history = model_2.fit(train_sentences, # input sentences can be a list of strings due to text preprocessing layer built-in model
                            train_labels,
                            epochs=5,
                            validation_data=(val_sentences, val_labels),
                            callbacks=[create_tensorboard_callback(dir_name=SAVE_DIR, 
                                                                    experiment_name="lstm_model")])
    model_2.evaluate(val_sentences, val_labels)

    # Predict on validation data and calculate scores
    model_2_pred_probs = model_2.predict(val_sentences)
    model_2_results = calculate_results(y_true=val_labels, y_pred=tf.squeeze(tf.round(model_2_pred_probs)))
    print("\nLSTM model Results:\n", model_2_results)
    print("\n-----------------------------------------------------\n")


    # Create RNN model with GRU
    model_3 = models.Model_3(train_sentences)

    # Fit model on training data
    model_3_history = model_3.fit(train_sentences, # input sentences can be a list of strings due to text preprocessing layer built-in model
                            train_labels,
                            epochs=5,
                            validation_data=(val_sentences, val_labels),
                            callbacks=[create_tensorboard_callback(dir_name=SAVE_DIR, 
                                                                    experiment_name="gru_model")])
    model_3.evaluate(val_sentences, val_labels)

    # Predict on validation data and calculate scores
    model_3_pred_probs = model_3.predict(val_sentences)
    model_3_results = calculate_results(y_true=val_labels, y_pred=tf.squeeze(tf.round(model_3_pred_probs)))
    print("\nGRU model Results:\n", model_3_results)
    print("\n-----------------------------------------------------\n")


    # Create Bidirectional RNN
    model_4 = models.Model_4(train_sentences)

    # Fit model on training data
    model_4_history = model_4.fit(train_sentences, # input sentences can be a list of strings due to text preprocessing layer built-in model
                            train_labels,
                            epochs=5,
                            validation_data=(val_sentences, val_labels),
                            callbacks=[create_tensorboard_callback(dir_name=SAVE_DIR, 
                                                                    experiment_name="bidir_model")])
    model_4.evaluate(val_sentences, val_labels)

    # Predict on validation data and calculate scores
    model_4_pred_probs = model_4.predict(val_sentences)
    model_4_results = calculate_results(y_true=val_labels, y_pred=tf.squeeze(tf.round(model_4_pred_probs)))
    print("\nBidirectional model Results:\n", model_4_results)
    print("\n-----------------------------------------------------\n")