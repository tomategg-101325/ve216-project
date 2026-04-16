f = 100e3           # 100 kHz
df = 20e3           # 20 kHz
fs = 10e6           # sampling rate 10 MHz (above Nyquist)
T_beat = 1.0 / df
N = int(T_beat * fs)

