import numpy as np
import torch
import matplotlib.pyplot as plt
from model import MLP

flow_matcher = MLP().to('cuda' if torch.cuda.is_available() else 'cpu')
flow_matcher.load_state_dict(torch.load("model_checkpoint.pth", map_location='cpu'))
flow_matcher.eval()

num_steps = 500
batch_size = 32
predicted_samples = np.array([[0, 0]], dtype=np.float32)  # Initialize with a single point

for i in range(310):
    x_t = torch.randn(batch_size, 2).to('cuda' if torch.cuda.is_available() else 'cpu')
    for t in torch.linspace(0, 1, num_steps):
        new_x = flow_matcher(x_t, t.expand(x_t.shape[0]).to(x_t.device))
        x_t += new_x/num_steps
    predicted_samples = np.concatenate([predicted_samples, x_t.cpu().detach().numpy()], axis=0)
    if i % 100 == 0:
        print(f'Step {i+1}, Predicted Samples: {predicted_samples.shape[0]}')

x_min = -4
x_max = 4
predicted_samples = predicted_samples[1:]  # Remove the initial point

plt.plot(predicted_samples[:, 0], predicted_samples[:, 1], 'o', markersize=1, alpha=0.5)
plt.xlim(x_min, x_max)
plt.ylim(x_min, x_max)
plt.title('Predicted Samples')
plt.savefig('predicted_samples.png')
plt.show()