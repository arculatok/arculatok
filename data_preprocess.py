import numpy as np
import librosa.display
import argparse
from pathlib import Path

HOP_LENGTH = 512
N_FFT = 2048
N_MFCC = 13
DATA_ROOT = "./train/audio/"

def parse_file_list(path_to_list):
    file_list = []
    with open(path_to_list, 'r') as f:
        for file_path in f:
            file_list.append(file_path.rstrip())
    return file_list


def preprocess(file_list, sample_rate, folder):
    for input_path in file_list:
        signal, sample_rate = librosa.load(DATA_ROOT + input_path, sample_rate)
        mfccs = librosa.feature.mfcc(signal, sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mfcc=N_MFCC)
        word_folder = input_path.split('/')[0]
        Path(folder + word_folder).mkdir(parents=True, exist_ok=True)
        np.savetxt(folder + input_path[:-4] + ".csv", mfccs, delimiter=";")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Data preprocessor script for speech recognition"
                                                     "machine learning project.")
    arg_parser.add_argument("--train_list",
                            "-t",
                            metavar="FILE",
                            dest="train_list",
                            required=True,
                            help="Specifies the files used in training..")
    arg_parser.add_argument("--val_list",
                            "-v",
                            metavar="FILE",
                            dest="val_list",
                            required=True,
                            help="Specifies the files used in validation..")
    arg_parser.add_argument("--",
                            "-e",
                            metavar="FILE",
                            dest="test_list",
                            required=True,
                            help="Specifies the files used in testing..")
    arg_parser.add_argument("--sample_rate",
                            "-s",
                            metavar="int",
                            dest="sample_rate",
                            type=int,
                            required=False,
                            default=22050,
                            help="Specifies the sample rate..")
    args = arg_parser.parse_args()

    train_list = parse_file_list(args.train_list)
    val_list = parse_file_list(args.val_list)
    test_list = parse_file_list(args.test_list)
    sample_rate = args.sample_rate

    train_folder = "./train_processed/"
    val_folder = "./val_processed/"
    test_folder = "./test_processed/"

    Path(train_folder).mkdir(parents=True, exist_ok=True)
    Path(val_folder).mkdir(parents=True, exist_ok=True)
    Path(test_folder).mkdir(parents=True, exist_ok=True)

    preprocess(train_list, sample_rate, train_folder)
    preprocess(val_list, sample_rate, val_folder)
    preprocess(test_list, sample_rate, test_folder)

