import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------
# SEEDS
# -----------------------------
tf.random.set_seed(42)
np.random.seed(42)

# -----------------------------
# LOAD DATA
# -----------------------------
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# -----------------------------
# BAHADANAU ATTENTION (CUSTOM)
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
        return context_vector, attention_weights

# -----------------------------
# MODEL ARCHITECTURE
# -----------------------------
time_steps = X_train.shape[1]
n_features = X_train.shape[2]

inputs = Input(shape=(time_steps, n_features))

# Encoder
lstm_out, state_h, state_c = LSTM(
    64, return_sequences=True, return_state=True
)(inputs)

# Attention
attention = BahdanauAttention(32)
context_vector, attention_weights = attention(lstm_out, state_h)

# Output
x = Dropout(0.2)(context_vector)
output = Dense(1)(x)

model = Model(inputs=inputs, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse"
)

model.summary()

# -----------------------------
# TRAIN
# -----------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# -----------------------------
# EVALUATE
# -----------------------------
y_pred = model.predict(X_test).flatten()

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print("Attention LSTM Results")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAPE : {mape:.2f}%")

# -----------------------------
# SAVE ATTENTION WEIGHTS
# -----------------------------
attention_model = Model(
    inputs=inputs,
    outputs=attention_weights
)

att_weights = attention_model.predict(X_test)
np.save("attention_weights.npy", att_weights)
