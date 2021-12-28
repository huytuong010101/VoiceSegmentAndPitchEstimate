import numpy as np


class VoiceOrUnvoice:
    def __init__(self, frame_size_second, frame_step_second, threshold=-1):
        self.frame_size_second = frame_size_second
        self.frame_step_second = frame_step_second
        self.threshold = threshold
        self.frame_size = None
        self.frame_step = None
        self.audio = None
        self.fs = None
        self.ste_sil = []
        self.ste_voice = []
        self.label = None

    def predict(self, audio: np.array, fs: int, label: dict = None):
        """
        This function use to segment the voice frame
        :param audio: vector audio signal
        :param fs: sample frequency
        :param labels: labels using for evaluate
        :return: list of pair which is time in sample of voice
        """
        self.label = label
        # Assign audio and frequence sample
        self.audio = audio / np.abs(np.max(audio))
        self.fs = fs
        # Convert something from second -> sample
        self.frame_size = int(self.frame_size_second * self.fs)
        self.frame_size = self.frame_size + (~self.frame_size & 1)
        self.frame_step = int(self.frame_step_second * self.fs)
        # Calculate STE value
        time_ste, mag_ste = self.__ste()
        # time_zcr, mag_zcr = self.__zcr()
        # self.__write_label(time_zcr, mag_zcr)
        return self.__find_voice(time_ste, mag_ste), (time_ste, mag_ste)

    def __ste(self, normalize: bool = True) -> tuple:
        """
        This function use to calc ste value
        :param normalize: need normalize?
        :return: (time axis, ste value)
        """
        time = []
        mag = []
        for center in range(self.frame_size // 2, len(self.audio) - self.frame_size // 2, self.frame_step):
            frame = self.audio[center - self.frame_size // 2: center + self.frame_size // 2 + 1]
            time.append(center)
            mag.append(np.sum(frame ** 2))
        time, mag = np.array(time), np.array(mag)
        if normalize:
            mag /= np.max(mag)
        return time, mag

    def __find_voice(self, time: np.array, mag: np.array):
        """
        This function use to segment the voice from STE value
        :param time: time axis
        :param mag: STE value
        :return: list of voice segment
        """
        res = []
        is_voice = mag > self.threshold
        i = 1
        start = None
        voice_duration = int(0.1 * self.fs / self.frame_step)
        # Loop all frame of STE
        while i < len(is_voice):
            # voice -> uv/sil
            if all(is_voice[i - voice_duration:i]) and not is_voice[i] and start is not None:
                res.append((start, time[i]))
                start = None
            # unv/sil -> voice
            if not is_voice[i - 1] and all(is_voice[i:i+voice_duration]):
                start = time[i]
            i += 1
        return res

    def __write_label(self, time: np.array, mag: np.array):
        """
        This function use to debug, please ignore it
        :param time:
        :param mag:
        :return:
        """
        for index, mag in zip(time, mag):
            is_sil = False
            for start, end in self.label["sil"]:
                if start < index / self.fs < end:
                    self.ste_sil.append(mag)
                    is_sil = True
                    break
            if not is_sil:
                self.ste_voice.append(mag)
        tmp = np.array([self.ste_sil, self.ste_voice])
        np.save("v_u_distribution_zcr.npy", tmp)

    def __zcr(self, normalize: bool = True) -> tuple:
        time = []
        mag = []
        for center in range(self.frame_size // 2, len(self.audio) - self.frame_size // 2, self.frame_step):
            frame = self.audio[center - self.frame_size // 2: center + self.frame_size // 2 + 1]
            time.append(center)
            mag.append(np.sum(np.abs(np.sign(frame[:-1]) - np.sign(frame[1:]))))
        time, mag = np.array(time), np.array(mag)
        if normalize:
            mag /= np.max(mag)
        return time, mag


