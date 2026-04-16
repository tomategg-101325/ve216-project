import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

import global_params as params
import global_functions as funcs
from separator_model import SignalSeparator

# Read model from file
model = SignalSeparator(params.N)
model.load_state_dict(torch.load('separator_model_weights.pth'))
model.eval()

# Generate test data
A1, A2 = float(input('A1 = ')), float(input('A2 = '))
current_time = params.N / params.fs
random_size = float(input('Noise level: ')) # add noise to simulate reality
starting_t = (1 + np.arange(params.N)) / params.fs
sampled_signal = funcs.get_s1(A1, starting_t) + funcs.get_s2(A2, starting_t) + np.random.uniform(-random_size, random_size, size=params.N)

plot_times = np.array([]);
plot_s1 = np.array([]);
plot_s2 = np.array([]);

def simulation_step(current_time, sampled_signal, plot_times, plot_s1, plot_s2):
    # step the simulation

    s1, s2, a1, a2 = model.forward(torch.tensor(sampled_signal, dtype=torch.float32)).detach().numpy()
    next_plot_times = np.append(plot_times, [current_time])
    next_plot_s1 = np.append(plot_s1, [s1])
    next_plot_s2 = np.append(plot_s2, [s2])

    # sliding window of sampled signal
    next_time = current_time + 1/params.fs
    next_sampled_signal = np.delete(sampled_signal, 0)
    next_sampled_signal = np.append(next_sampled_signal, [funcs.get_s1(A1, next_time) + funcs.get_s2(A2, next_time) + np.random.uniform(-random_size, random_size)])

    return next_time, next_sampled_signal, next_plot_times, next_plot_s1, next_plot_s2, a1, a2

# Simulation
rounds = 1000
for i in range(rounds):
    current_time, sampled_signal, plot_times, plot_s1, plot_s2, a1, a2 = simulation_step(current_time, sampled_signal, plot_times, plot_s1, plot_s2)

print('Simulation completed!')
print(f'A1 = {a1:.3f}\nA2 = {a2:.3f}')

# Plotting
plot_s1_ref = funcs.get_s1(A1, plot_times) # get reference waveform
plot_s2_ref = funcs.get_s2(A2, plot_times)

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)

ax1.plot(plot_times*1e6, plot_s1_ref, color=(0.5, 0.5, 0.5), linewidth=2, linestyle = 'dashed')
ax1.plot(plot_times*1e6, plot_s1, color=(1.0, 0.2, 0.2), linewidth=1)
ax2.plot(plot_times*1e6, plot_s2_ref, color=(0.5, 0.5, 0.5), linewidth=2, linestyle = 'dashed')
ax2.plot(plot_times*1e6, plot_s2, color=(0.05, 0.75, 0.3), linewidth=1)
ax1.set_title(f'A1 = {A1:.3f}, f = {params.f / 1000} kHz')
ax2.set_title(f'A2 = {A2:.3f}, f = {(params.f+params.df) / 1000} kHz')

ax1.set_ylabel('amplitude')
ax2.set_ylabel('amplitude')
ax2.set_xlabel('time (us)')

plt.savefig('separator_simulation.png', dpi=300)

