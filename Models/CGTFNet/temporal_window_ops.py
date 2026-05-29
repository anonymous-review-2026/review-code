import torch


def build_sliding_windows(bold_signal, window_length, stride):
    """
    bold_signal : (batchSize, N, T)
    output : (batchSize, (T-windowLength) // stride, N, windowLength )
    """
    T = bold_signal.shape[2]
    windowed_bold_signals = []
    sampling_end_points = []

    for window_index in range((T - window_length) // stride + 1):
        sampled_window = bold_signal[:, :, window_index * stride : window_index * stride + window_length]
        sampling_end_points.append(window_index * stride + window_length)
        sampled_window = torch.unsqueeze(sampled_window, dim=1)
        windowed_bold_signals.append(sampled_window)

    windowed_bold_signals = torch.cat(windowed_bold_signals, dim=1)
    return windowed_bold_signals, sampling_end_points
