import numpy as np

def spectrogram_padding(spectrogram, spec_len): 
    # data = np.copy(self.audio)

    # data = np.zeros((data.shape[0], data[0, 1].shape[0], max_len))
    if spectrogram.shape[1] > spec_len:
        start_cut = (spectrogram.shape[1] - spec_len) // 2 
        return spectrogram[:, start_cut:start_cut+spec_len]

    return np.pad(spectrogram, [[0, 0], [0, spec_len-spectrogram.shape[1]]])

    