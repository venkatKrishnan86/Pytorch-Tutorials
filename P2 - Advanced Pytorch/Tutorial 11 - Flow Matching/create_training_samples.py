import numpy as np
import torch
import matplotlib.pyplot as plt

x_min = -4
x_max = 4
N = 1000000

original_dist = np.array([
    np.random.uniform(x_min, x_max, N),
    np.random.uniform(x_min, x_max, N)
]).reshape(N, 2)

samples = []
counter = 0

while counter < N:
    point = np.random.uniform(x_min, x_max, 2)
    if np.sum(np.floor(point)) % 2 == 0:
        samples.append(point)
        counter += 1

samples = np.array(samples)
plt.plot(samples[:, 0], samples[:, 1], 'o', markersize=1, alpha=0.5)
plt.xlim(x_min, x_max)
plt.ylim(x_min, x_max)
plt.title('Training Samples')
plt.savefig('true_samples.png')
plt.show()

# Save the samples to a file
torch.save(torch.tensor(samples, dtype=torch.float32), 'training_samples.pt')

