import numpy as np
from scipy.io.wavfile import write

# Parameters
sample_rate = 48000
duration = 1  # seconds
t = np.linspace(0, duration, int(sample_rate * duration), False)
fc = 1000  # 1 kHz analog frequency
deviation = 5000  # 5 kHz deviation
fsc = 67000  # 67 kHz subcarrier

# Analog signal (1 kHz sine)
analog = np.sin(2 * np.pi * fc * t)
analog_fm = deviation * np.cumsum(analog) / sample_rate

# Digital signal (8-PSK subcarrier)
symbols = np.exp(2j * np.pi * np.random.choice(8, size=int(len(t) / (sample_rate / symbol_rate))) / 8)
digital_base = np.zeros(len(t), dtype=complex)
for i, sym in enumerate(symbols):
    digital_base[i * int(sample_rate / symbol_rate):(i + 1) * int(sample_rate / symbol_rate)] = sym
digital_fm = np.real(digital_base * np.exp(2j * np.pi * fsc * t / sample_rate))
digital_fm = digital_fm / np.max(np.abs(digital_fm)) * 2500

# Composite FM baseband
total_phase = (analog_fm + digital_fm) / (deviation + 2500)
iq_signal = np.exp(1j * 2 * np.pi * total_phase)

# Save as .wav
iq_i = (iq_signal.real * 32767).astype(np.int16)
iq_q = (iq_signal.imag * 32767).astype(np.int16)
write('fm_baseband.wav', sample_rate, np.column_stack((iq_i, iq_q)))
print("FM baseband .wav generated.")