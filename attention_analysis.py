import numpy as np
import matplotlib.pyplot as plt

att_weights = np.load("attention_weights.npy")

# Average attention across samples
mean_attention = att_weights.mean(axis=0).squeeze()

plt.figure()
plt.plot(mean_attention)
plt.xlabel("Time Steps (Lookback)")
plt.ylabel("Attention Weight")
plt.title("Average Attention Distribution")
plt.show()
