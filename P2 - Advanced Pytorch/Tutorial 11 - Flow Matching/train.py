import torch
from model import MLP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_data = torch.load('training_samples.pt').to(device)
model = MLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999975)
num_steps = 250_000
batch_size = 128

losses = []

for step in range(num_steps):
    model.train()

    x_1 = train_data[torch.randint(0, train_data.shape[0], (batch_size,))].to(device)
    x_0 = torch.randn_like(x_1).to(device)
    target = x_1 - x_0
    t = torch.rand(x_0.shape[0], device=device)
    x_t = x_0 + t[:, None] * target

    output = model(x_t, t)

    optimizer.zero_grad()
    loss = torch.mean((output - target) ** 2)

    loss.backward()
    optimizer.step()
    scheduler.step()

    losses.append(loss.item())

    if step % 1000 == 0:
        print(f'Step {step}, Loss: {loss.item()}, LR: {scheduler.get_last_lr()[0]}')
        torch.save(model.state_dict(), 'model_checkpoint.pth')  # Save the model checkpoint
