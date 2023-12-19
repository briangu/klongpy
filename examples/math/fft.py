import numpy as np

# Create a simple signal
signal = np.array([0.0, 1.0, 0.0, -1.0])

# Apply FFT to the signal
fft_result = np.fft.fft(signal)

# Display the FFT result
print("FFT Result:", fft_result)
