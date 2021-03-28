import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

FIG_SIZE = (15, 10)

def make_waveform_plot(signal: np.ndarray, sample_rate: int):
    plt.figure(figsize=FIG_SIZE)
    librosa.display.waveplot(signal, sample_rate, alpha=0.4)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform")


def make_spectrum_plot(freq: np.ndarray, spectrum: np.ndarray):
    plt.figure(figsize=FIG_SIZE)
    plt.plot(freq, spectrum, alpha=0.4)
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    plt.title("Power spectrum")


def make_spectrogram_plot(spectrogram: np.ndarray, sample_rate: int, hop_length: int, log=False):
    if log:
        plt.figure(figsize=FIG_SIZE)
        librosa.display.specshow(librosa.amplitude_to_db(spectrogram), sr=sample_rate, hop_length=hop_length)
        plt.colorbar(format="%+2.0f dB")
        plt.title("Spectrogram (dB)")
    else:
        plt.figure(figsize=FIG_SIZE)
        librosa.display.specshow(spectrogram, sr=sample_rate, hop_length=hop_length)
        plt.colorbar()
        plt.title("Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Frequency")


def make_mfcc_plot(MFCCs: np.ndarray, sample_rate: int, hop_length: int):
    plt.figure(figsize=FIG_SIZE)
    librosa.display.specshow(MFCCs, sr=sample_rate, hop_length=hop_length)
    plt.xlabel("Time")
    plt.ylabel("MFCC coefficients")
    plt.colorbar()
    plt.title("MFCCs")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Script to visualize sound data for"
                                                     "machine learning project.")
    arg_parser.add_argument("--input_file",
                            "-t",
                            metavar="path",
                            dest="input_file",
                            required=True,
                            help="Specifies the file for plotting..")
    arg_parser.add_argument("--sample_rate",
                            "-s",
                            metavar="int",
                            dest="sample_rate",
                            type=int,
                            required=False,
                            default=22050,
                            help="Specifies the sample rate..")

    args = arg_parser.parse_args()
    input_file = args.input_file
    sample_rate = args.sample_rate
    signal, sample_rate = librosa.load(input_file, sample_rate)
    make_waveform_plot(signal, sample_rate)

    fft = np.fft.fft(signal)
    spectrum = np.abs(fft)

    freq = np.linspace(0, sample_rate, len(spectrum))

    left_spectrum = spectrum[:int(len(spectrum) / 2)]
    left_freq = freq[:int(len(spectrum) / 2)]

    make_spectrum_plot(left_freq, left_spectrum)

    hop_length = 512
    n_fft = 2048

    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)

    spectrogram = np.abs(stft)

    make_spectrogram_plot(spectrogram, sample_rate, hop_length)
    make_spectrogram_plot(spectrogram, sample_rate, hop_length, log=True)

    MFCCs = librosa.feature.mfcc(signal, sample_rate, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)

    make_mfcc_plot(MFCCs, sample_rate, hop_length)

    scaler = MinMaxScaler()
    transformed_mfccs = scaler.fit_transform(MFCCs)

    make_mfcc_plot(transformed_mfccs, sample_rate, hop_length)

    scaler = StandardScaler()
    transformed2_mfccs = scaler.fit_transform(MFCCs)

    make_mfcc_plot(transformed2_mfccs, sample_rate, hop_length)

    plt.show()
