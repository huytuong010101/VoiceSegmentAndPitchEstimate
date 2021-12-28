from scipy.io import wavfile
from matplotlib import pyplot as plt
import os
import numpy as np
from utils import load_label, plot_voice_segment, plot_pitch_estimate, eval_segment
from VoiceOrUnvoice import VoiceOrUnvoice
from PitchEstimation import PitchEstimation

SAMPLE_DIR = "TinHieuKiemThu"

if __name__ == '__main__':
    voice_detection = VoiceOrUnvoice(
        frame_size_second=0.02,
        frame_step_second=0.01,
        threshold=0.002
    )
    pitch_estimation = PitchEstimation(
        frame_size_second=0.02,
        frame_step_second=0.01,
        num_f_sample=16384,
        median_size=7,
        harmony_threshold=5
    )
    # List all input file
    wav_files = os.listdir(SAMPLE_DIR)
    wav_files = [os.path.join(SAMPLE_DIR, item) for item in wav_files if item.endswith(".wav")]
    for index, wav_file in enumerate(wav_files):
        # Init figure
        plt.rcParams["figure.figsize"] = (20, 15)
        print("======================================")
        print("Processing on", wav_file)
        plt.figure(wav_file)
        # Read audio and label
        fs, signal = wavfile.read(filename=wav_file)
        label = load_label(wav_file.split(".")[0] + ".lab")
        # Voice segment
        voice_segment, (time, ste) = voice_detection.predict(signal, fs, label)
        plot_voice_segment(signal, fs, label, voice_segment, time, ste, plt)
        # Pitch estimte
        f, t = pitch_estimation.predict(signal, fs, voice_segment)
        plot_pitch_estimate(t, f, signal, plt)
        # Plot debug
        if pitch_estimation.voice_fft is not None:
            plt.subplot(5, 1, 4).set_xlabel("Hz")
            plt.subplot(5, 1, 4).set_ylabel("Db")
            plt.subplot(5, 1, 4).plot(pitch_estimation.voice_fft)
            plt.subplot(5, 1, 4).set_title("Spectrum of Voice frame")
        if pitch_estimation.unvoice_fft is not None:
            plt.subplot(5, 1, 5).set_xlabel("Hz")
            plt.subplot(5, 1, 5).set_ylabel("Db")
            plt.subplot(5, 1, 5).plot(pitch_estimation.unvoice_fft)
            plt.subplot(5, 1, 5).set_title("Spectrum of Silent frame")
        # Evaluate segemnt
        error_segment = eval_segment(voice_segment, label, fs)
        print("Error segment:", error_segment)
        # Evaluate F0
        mean = np.mean(f)
        std = np.std(f)
        plt.savefig("outputs\\" + str(index) + ".jpg")
        print(f"Mean: {mean}, error: {abs(mean - label['F0mean'])}")
        print(f"STD: {std}, error: {abs(std - label['F0std'])}")
        plt.tight_layout(h_pad=3)
        plt.subplot(5, 1, 2).set_title(wav_file)
        plt.show(block=False)
    # plt.figure(4)
    # plt.hist(pitch_estimation.count_harmony)
    # plt.ylabel("Number of harmony")
    # plt.xlabel("Harmony")
    # plt.savefig("hermony.jpg")
    plt.show()








