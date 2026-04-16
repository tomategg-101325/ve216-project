import torch
import torch.nn as nn
import numpy as np

import global_params as params
import global_functions as funcs
from separator_model import SignalSeparator

# Generate training batch
def generate_batch(batch_size, noise_level=0.0):
    t0 = np.random.uniform(0, params.T_beat, size=(batch_size, 1))
    A1 = np.random.uniform(0.01, 1, size=(batch_size, 1))
    A2 = np.random.uniform(0.01, 1, size=(batch_size, 1))
    
    t = t0 + np.arange(params.N) / params.fs   # shape (batch_size, N)
    y = funcs.get_s1(A1, t) + funcs.get_s2(A2, t) + np.random.normal(0.0, noise_level, size=params.N)
    
    t_ref = t0 + (params.N - 1) / params.fs   # end of window to maintain causality
    s1 = funcs.get_s1(A1, t_ref)
    s2 = funcs.get_s2(A2, t_ref)
    targets = np.hstack([s1, s2, A1, A2])
    
    return torch.tensor(y, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

# Training loop
model = SignalSeparator(params.N)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

epochs = 2000
log_period = 50

for epoch in range(epochs):
    x, y_true = generate_batch(256, noise_level=0.02)
    y_pred = model(x)
    loss = criterion(y_pred, y_true)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # verbose
    if (epoch + 1) % log_period == 0:
        print(f'Epoch {epoch + 1}/{epochs}, loss: {loss.item():6f}')

# Export model weights
torch.save(model.state_dict(), 'separator_model_weights.pth')

print('Training complete!')

