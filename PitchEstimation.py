import numpy as np
from scipy.fft import fft
from typing import List
from scipy.signal import hanning, find_peaks, medfilt
import pickle


class PitchEstimation:
    def __init__(self,
                 frame_size_second: float = 0.03,
                 frame_step_second: float = 0.01,
                 num_f_sample: int = 16384,
                 median_size:int = 5,
                 min_f: float = 70,
                 max_f: float = 400,
                 harmony_threshold=4,
                 ):
        self.signal = None
        self.fs = None
        self.frame_size = None
        self.frame_step = None
        self.time = None
        self.voice_fft = None
        self.unvoice_fft = None
        self.median_size = median_size
        self.num_f_sample = num_f_sample
        self.frame_size_second = frame_size_second
        self.frame_step_second = frame_step_second
        self.min_f = min_f
        self.max_f = max_f
        self.count_harmony = []
        self.harmony_threshold = harmony_threshold

    def predict(self, signal: np.array, fs: np.array, voice_segments: List[tuple]):
        self.voice_fft = self.unvoice_fft = None
        self.signal = signal
        self.time = np.arange(len(self.signal))
        self.fs = fs
        self.frame_size = int(self.frame_size_second * self.fs)
        self.frame_step = int(self.frame_step_second * self.fs)
        res_f = []
        res_t = []
        for start, end in voice_segments:
            signal = self.signal[start: end]
            time = self.time[start: end]
            window = hanning(self.frame_size)
            seg_f = []
            seg_t = []
            for center in range(self.frame_size // 2, len(signal) - self.frame_size // 2, self.frame_step):
                frame = signal[center - self.frame_size // 2: center + self.frame_size // 2]
                frame = frame * window
                spectrum = (np.abs(fft(frame, self.num_f_sample)))
                spectrum = spectrum[:len(spectrum) // 2]
                f = self.__find_fo(spectrum)
                if f is not None:
                    seg_f.append(f)
                    seg_t.append(time[center])
                    if self.voice_fft is None and center > 5:
                        self.voice_fft = spectrum[:1000]
                else:
                    if self.unvoice_fft is None:
                        self.unvoice_fft = spectrum[:1000]
            seg_f = self.medfilt(seg_f, self.median_size)
            seg_f, seg_t = self.fix_last_pitch(seg_f, seg_t)
            res_f.extend(seg_f)
            res_t.extend(seg_t)
        return res_f, res_t

    def __find_peaks(self, arr: np.array):
        arr = self.medfilt(arr, 25)
        return find_peaks(arr)[0]

    def __find_fo(self, arr: np.array):
        peaks = self.__find_peaks(arr)
        peaks = peaks[:21]
        n = len(peaks)
        f_final = None
        max_harmony = 0
        for i in range(n - 2):
            f_tmp = (peaks[i + 1] - peaks[i])
            count_harmony = 0
            for j in range(i + 2, n):
                f_check = (peaks[j] - peaks[i])
                if f_check / f_tmp % 1 < 0.1 or 1 - f_check / f_tmp % 1 < 0.1:
                    count_harmony += 1
            if count_harmony >= max_harmony:
                f_final = f_tmp * self.fs / self.num_f_sample
                max_harmony = count_harmony
        self.count_harmony.append(max_harmony)
        if max_harmony < self.harmony_threshold:
            return None
        if not (70 < f_final < 400):
            return None
        return f_final

    @staticmethod
    def medfilt(arr: np.array, size: int):
        if size % 2 == 0:
            raise ValueError("Frame size must be odd")
        arr = np.concatenate([[0] * (size // 2), arr, [0] * (size // 2)])
        res = []
        n = len(arr)
        for center in range(size // 2, n - size // 2):
            frame = arr[center - size // 2: center + size // 2 + 1]
            res.append(sorted(frame)[size // 2])
        return res

    def fix_last_pitch(self, arr: list, time: list):
        res = arr[:-self.median_size]
        res_time = time[:-self.median_size]
        for f, t in zip(arr[-self.median_size:], time[-self.median_size:]):
            if abs(f - np.mean(res[-self.median_size:])) < 20:
                res.append(f)
                res_time.append(t)
        return res, res_time






