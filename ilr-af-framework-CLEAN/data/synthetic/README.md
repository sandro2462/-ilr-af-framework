# Synthetic data (TIMA / TimeGAN)

Place the TIMA-generated synthetic windows here (e.g. `synthetic_windows.npz`).
These data are generated and do not correspond to any real patient.

Expected format (mirrors the real windowed input, without any identifiers):
- X: float32 array, shape (n_windows, 30, 18)
- y: int8 array, shape (n_windows,)
- features: list of 18 feature names
