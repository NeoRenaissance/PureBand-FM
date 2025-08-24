import numpy as np
from scipy.signal import firwin, lfilter, hilbert

# Parameters
fs = 24000  # AM baseband
duration = 10
t = np.arange(0, duration, 1/fs)
fc = 1000  # 1 kHz analog
digital_rate = 100000  # Adjusted to 100 kbps
symbol_rate = digital_rate / 2  # QPSK
samples_per_symbol = int(fs / symbol_rate)
fec_rate = 1/3  # Robust FEC
effective_rate = digital_rate * fec_rate
min_signal = -120

# Generate AM signal
signal = np.sin(2 * np.pi * fc * t).astype(np.float64)
symbols = np.exp(2j * np.pi * np.random.choice(4, size=int(len(t) / samples_per_symbol)) / 4).astype(np.complex128)
digital_base = np.zeros(len(t), dtype=np.complex128)
for i, sym in enumerate(symbols): digital_base[i * samples_per_symbol:(i + 1) * samples_per_symbol] = sym
digital_fm = np.real(digital_base * np.exp(2j * np.pi * 7000 * t / fs))
am_signal = (1 + 0.5 * signal + 0.3 * digital_fm) * np.exp(1j * 2 * np.pi * 1380e3 * t)

# Add interference and noise
interference = 5.6 * np.sin(2 * np.pi * 1000 * t + np.pi/4).astype(np.float64)  # 15 dB stronger
low_freq_noise = 0.2 * np.sin(2 * np.pi * 60 * t) + 0.1 * np.sin(2 * np.pi * 120 * t)
noise = np.random.normal(0, np.sqrt(10**-8), len(t)).astype(np.float64)
received = am_signal + interference + low_freq_noise + noise

# Co-channel canceller
interf_freq = 1000
notch_filter = firwin(101, [interf_freq - 50, interf_freq + 50], fs=fs, pass_zero=False, scale=False)
notched = lfilter(notch_filter, 1, received)
demod = np.angle(hilbert(notched)).astype(np.float64)

# Subsonic ANC (10 Hz)
ref_carrier = 0.05 * np.sin(2 * np.pi * 10 * t).astype(np.float64)
noise_est = lfilter([1], [1, -0.95], demod - ref_carrier * 0.1)
cleaned_signal = demod - noise_est

# Extract analog and digital
analog_filter = firwin(141, 7000, fs=fs, pass_zero=True)
analog_out = lfilter(analog_filter, 1, cleaned_signal)[:len(signal)-1]
digital_filter = firwin(101, [6900, 7100], fs=fs, pass_zero=False)
digital_out = lfilter(digital_filter, 1, cleaned_signal)
digital_out = digital_out * np.exp(-2j * np.pi * 7000 * t / fs)

# Analysis
analog_snr = 10 * np.log10(np.mean(signal**2) / np.var(analog_out - signal))
digital_snr = 10 * np.log10(np.mean(np.abs(digital_base)**2) / np.var(np.angle(digital_out) - np.angle(digital_base)))
ber = np.mean(np.abs(np.angle(digital_out[::samples_per_symbol]) - np.angle(digital_base[::samples_per_symbol])) > np.pi/4)

print(f"Analog SNR: {analog_snr:.1f} dB")
print(f"Digital SNR: {digital_snr:.1f} dB")
print(f"Bit Error Rate: {ber:.4f}")
print(f"Effective Data Rate: {effective_rate * (1 - ber) / 1000:.1f} kbps")