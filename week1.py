import numpy as np
import matplotlib.pyplot as plt
import time
from numba import njit

N_GAMBLERS = 20_000
T_ROUNDS = 2_000
START_WEALTH = 100

def numpy_sim(n, t, start):
    flips = np.random.choice([-1, 1], size=(n, t))
    wealth = np.concatenate([np.full((n, 1), start), start + np.cumsum(flips, axis=1)], axis=1)
    bankrupt = np.maximum.accumulate(wealth <= 0, axis=1)
    wealth[bankrupt] = 0
    return wealth

@njit
def numba_sim(n, t, start):
    wealth = np.zeros((n, t + 1))
    wealth[:, 0] = start
    for i in range(n):
        w = float(start)
        for j in range(1, t + 1):
            if w > 0:
                w += 1 if np.random.random() > 0.5 else -1
                w = max(0, w)
            wealth[i, j] = w
    return wealth

print("NumPy:")
t0 = time.time()
np_paths = numpy_sim(N_GAMBLERS, T_ROUNDS, START_WEALTH)
print(f"{time.time() - t0:.4f}s\n")

print("Numba:")
t0 = time.time()
nb_paths = numba_sim(N_GAMBLERS, T_ROUNDS, START_WEALTH)
print(f"{time.time() - t0:.4f}s\n")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

rounds = np.arange(T_ROUNDS + 1)
for i in range(100):
    ax1.plot(rounds, np_paths[i], 'gray', alpha=0.1)

mean_path = np_paths.mean(axis=0)
final = np_paths[:, -1]
winner = final.argmax()
loser = final.argmin()

ax1.plot(rounds, mean_path, 'r--', lw=2, label='Mean')
ax1.plot(rounds, np_paths[winner], 'g', lw=1.5, label='Winner')
ax1.plot(rounds, np_paths[loser], 'orange', lw=1.5, label='Loser')
ax1.set_xlabel("Rounds")
ax1.set_ylabel("Bankroll ($)")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.hist(final, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
mean_val = final.mean()
median_val = np.median(final)
ax2.axvline(mean_val, color='red', ls='--', lw=2, label=f'Mean: ${mean_val:.0f}')
ax2.axvline(median_val, color='green', ls=':', lw=2, label=f'Median: ${median_val:.0f}')
ax2.set_xlabel("Final Bankroll ($)")
ax2.set_ylabel("Count")
ax2.legend()

plt.tight_layout()
plt.show()

