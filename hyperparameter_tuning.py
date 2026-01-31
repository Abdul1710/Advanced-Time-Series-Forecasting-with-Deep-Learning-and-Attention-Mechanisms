import optuna
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------
# LOAD DATA
# -----------------------------
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")

time_steps = X_train.shape[1]
n_features = X_train.shape[2]

# -----------------------------
# ATTENTION LAYER
# -----------------------------
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, features, hidden):
        hidden = tf.expand_dims(hidden, 1)
        score = self.V(tf.nn.tanh(
            self.W1(features) + self.W2(hidden)
        ))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector

# -----------------------------
# OBJECTIVE FUNCTION
# -----------------------------
def objective(trial):
    lstm_units = trial.suggest_int("lstm_units", 32, 128, step=32)
    attn_units = trial.suggest_int("attention_units", 16, 64, step=16)
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

    inputs = Input(shape=(time_steps, n_features))
    lstm_out, state_h, _ = LSTM(
        lstm_units, return_sequences=True, return_state=True
    )(inputs)

    context = BahdanauAttention(attn_units)(lstm_out, state_h)
    x = Dropout(dropout)(context)
    output = Dense(1)(x)

    model = Model(inputs, output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="mse"
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )

    return min(history.history["val_loss"])

# -----------------------------
# RUN STUDY
# -----------------------------
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

print("Best trial:")
print(study.best_trial.params)
print("Best validation loss:", study.best_value)
