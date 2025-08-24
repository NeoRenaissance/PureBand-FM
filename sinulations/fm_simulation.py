import numpy as np
from scipy.signal import firwin, lfilter, hilbert

# Parameters
fs = 48000  # Hz
t = np.arange(0, 0.1, 1/fs)  # 0.1 second simulation
fc = 1000   # 1 kHz analog frequency
kd = 5000   # Analog deviation (5 kHz)
fsd = 67e3  # Digital subcarrier frequency (67 kHz)
data_rate = 200e3  # Digital data rate (200 kbps with FEC)
symbol_rate = data_rate / 3  # 8-PSK, 3 bits/symbol
samples_per_symbol = int(fs / symbol_rate)
fec_rate = 1/4  # FEC rate (LDPC + RS Hybrid)
effective_rate = data_rate * fec_rate
ultrasonic_freq = 25e3  # Ultrasonic noise detector (25 kHz)
mu = 0.01  # LMS step size

# Analog signal (1 kHz sine + 25 kHz reference)
analog = np.sin(2 * np.pi * fc * t) + 0.3 * np.sin(2 * np.pi * ultrasonic_freq * t)
analog_fm = kd * np.cumsum(analog) / fs

# Digital signal (8-PSK FBMC-OQAM with HLS mock)
symbols = np.exp(2j * np.pi * np.random.choice(8, size=int(len(t) / samples_per_symbol)) / 8).astype(np.complex128)
digital_base = np.zeros(len(t), dtype=complex)
for i, sym in enumerate(symbols):
    digital_base[i * samples_per_symbol:(i + 1) * samples_per_symbol] = sym
hls_overhead = np.zeros(len(t))
for i in range(0, len(t), int(fs * 0.1)):
    hls_overhead[i:i+int(fs*0.01)] = 0.1  # HLS burst
digital_fm = np.real(digital_base * np.exp(2j * np.pi * fsd * t) * (1 + hls_overhead))
digital_fm = digital_fm / np.max(np.abs(digital_fm)) * 2500  # 2.5 kHz deviation

# Composite FM signal
total_phase = (analog_fm + digital_fm) / (kd + 2500)
fm_signal = np.cos(2 * np.pi * total_phase).astype(np.float64)

# High Noise Model with Filters
fm_bp = firwin(101, [88e6/fs*2, 108e6/fs*2], fs=fs, pass_zero=False)
hf_lp = firwin(101, 120e6/fs*2, fs=fs)
lte_notch = 1 - firwin(101, [0.7e9/fs*2, 2.7e9/fs*2], fs=fs, pass_zero=False)
filtered_signal = lfilter(fm_bp * hf_lp * lte_notch, 1, fm_signal)

# LNA (20 dB gain, 1.5 dB NF)
lna_gain = 10**(20/10)
lna_noise = np.random.normal(0, np.sqrt(10**(-1.5/10)), len(t)) * lna_gain
received_pre_lna = filtered_signal + lna_noise

# Multipath and Interference
mp1_delay = int(5e-6 * fs)
mp2_delay = int(10e-6 * fs)
multipath1 = received_pre_lna[:-mp1_delay] * 0.316  # -10 dB
multipath2 = received_pre_lna[:-mp2_delay] * 0.178  # -15 dB
thermal_noise = np.random.normal(0, 0.01, len(t))  # -90 dBm
interference = np.random.normal(0, 0.02, len(t))  # -85 dBm
received = received_pre_lna + multipath1 + multipath2 + thermal_noise + interference

# Demodulate
demod_phase = np.unwrap(np.angle(hilbert(received) * np.exp(-2j * np.pi * fc * t)))
analog_out_raw = np.diff(demod_phase) * fs / (2 * np.pi * kd)

# Noise Detection from Ultrasonic (25 kHz)
ultrasonic_filter = firwin(101, [24e3, 26e3], fs=fs, pass_zero=False)
ultrasonic_signal = lfilter(ultrasonic_filter, 1, analog_out_raw)
expected_ultrasonic = 0.3 * np.sin(2 * np.pi * ultrasonic_freq * t[:len(analog_out_raw)])
noise_estimate = ultrasonic_signal - expected_ultrasonic

# LMS Adaptive Filter for ANC
error = np.zeros(len(noise_estimate))
w = np.zeros(len(noise_estimate))
for n in range(1, len(noise_estimate)):
    output = np.dot(w[:n], noise_estimate[:n][::-1])
    error[n] = analog_out_raw[n] - output
    w[:n] += mu * error[n] * noise_estimate[:n][::-1]
analog_out = analog_out_raw - w

# ANC for Digital
digital_filter = firwin(101, [66e3, 68e3], fs=fs, pass_zero=False)
digital_out_raw = lfilter(digital_filter, 1, demod_phase)
digital_out_raw = digital_out_raw * np.exp(-2j * np.pi * fsd * t / fs)
digital_out = digital_out_raw - w * (2500 / kd)  # Scale noise to digital level

# Analysis
analog_snr = 10 * np.log10(np.mean(analog[:len(analog_out)]**2) / np.var(analog_out - analog[:len(analog_out)]))
digital_snr = 10 * np.log10(np.mean(np.abs(digital_base)**2) / np.var(np.angle(digital_out) - np.angle(digital_base)))
ber = np.mean(np.abs(np.angle(digital_out[::samples_per_symbol]) - np.angle(digital_base[::samples_per_symbol])) > np.pi/4)

print(f"Analog SNR: {analog_snr:.1f} dB")
print(f"Digital SNR: {digital_snr:.1f} dB")
print(f"Bit Error Rate: {ber:.4f}")
print(f"Effective Data Rate: {effective_rate * (1 - ber) / 1000:.1f} kbps")