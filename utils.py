import numpy as np
from matplotlib import pyplot


def load_label(path: str) -> dict:
    with open(path) as f:
        lines = f.read().splitlines()
    labels = {}
    for line in lines:
        arr = line.split()
        if len(arr) == 3:
            # Time label
            start, end, label = float(arr[0]), float(arr[1]), arr[2]
            if label not in labels:
                labels[label] = []
            labels[label].append((start, end))
        elif len(arr) == 2:
            # F0 label
            label, val = arr[0], float(arr[1])
            labels[label] = val
    return labels


def plot_voice_segment(
        signal: np.array,
        fs: float,
        label: dict,
        voice_segment: list,
        time: np.array,
        ste: np.array,
        plt: pyplot
):
    # PLot audio and STE
    plt.subplot(5, 1, 1).set_xlabel("sample")
    plt.subplot(5, 1, 1).set_ylabel("STE value")
    plt.subplot(5, 1, 1).plot(time, ste, color="b")
    plt.subplot(5, 1, 1).set_title("STE value")
    plt.subplot(5, 1, 2).set_xlabel("sample")
    plt.subplot(5, 1, 2).set_ylabel("Amplitude")
    plt.subplot(5, 1, 2).plot(signal, color="b")
    plt.subplot(5, 1, 2).set_title("Audio")

    id_color = {"Ground truth": u"g", "Predict": u"r"}
    markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='') for color in id_color.values()]
    plt.subplot(5, 1, 2).legend(markers, id_color.keys(), numpoints=1)
    # Plot predict
    for start, end in voice_segment:
        plt.subplot(5, 1, 2).axvline(x=start, color="r")
        plt.subplot(5, 1, 2).axvline(x=end, color="r")
    # Plot label
    for key in label:
        if key != "sil" and key != "F0mean" and key != "F0std":
            for start, end in label[key]:
                plt.subplot(5, 1, 2).axvline(x=start * fs, color="g", alpha=0.5)
                plt.subplot(5, 1, 2).axvline(x=end * fs, color="g", alpha=0.5)


def plot_pitch_estimate(t: list, f: list, signal: np.array, plt: pyplot):
    t.insert(0, 0)
    f.insert(0, 0)
    t.append(len(signal))
    f.append(0)
    plt.subplot(5, 1, 3).set_xlabel("sample")
    plt.subplot(5, 1, 3).set_ylabel("Hz")
    plt.subplot(5, 1, 3).scatter(t, f, s=1)
    plt.subplot(5, 1, 3).set_title("F0 estimate")


def eval_segment(voice_segment: list, label: dict, fs: float):
    errors = []
    for start, end in voice_segment:
        min_start = min_end = 99999
        for key in label:
            if key != "sil" and key != "F0mean" and key != "F0std":
                min_start = min(min_start, abs(start / fs - label[key][0][0]))
                min_end = min(min_end, abs(end / fs - label[key][0][1]))
        errors.append(min_start)
        errors.append(min_end)
    return np.mean(errors)


if __name__ == "__main__":
    print(load_label(r"D:\DUT\XuLiTinHieuSo\Final-Project\samples\phone_F1.lab"))