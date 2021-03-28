import numpy as np
import librosa.display
import argparse
from random import shuffle
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

TRAIN_LIST = "./training_list.txt"
TEST_LIST = "./testing_list.txt"
VAL_LIST = "./validation_list.txt"

ALL_TRAIN = 51088
ALL_TEST = 6835
ALL_VAL = 6798

TRAIN_FOLDER = "./train_processed/"
VAL_FOLDER = "./val_processed/"
TEST_FOLDER = "./test_processed/"

HOP_LENGTH = 512
N_FFT = 2048
N_MFCC = 13
DATA_ROOT = "./train/audio/"


def parse_file_list(path_to_list, num, limit):
    if num >= limit:
        print(f"LIMITS:\n"
              f"TRAIN: {ALL_TRAIN}\n"
              f"TEST: {ALL_TEST}\n"
              f"VALIDATION: {ALL_VAL}\n")
        raise ValueError("One or more of the limits were exceeded!")
    file_list = []
    with open(path_to_list, 'r') as f:
        for file_path in f:
            file_list.append(file_path.rstrip())
    if num < 1:
        return file_list
    else:
        shuffle(file_list)
        return file_list[:num]


def preprocess(file_list, folder, sample_rate):
    for input_path in file_list:
        print(input_path)
        signal, sample_rate = librosa.load(DATA_ROOT + input_path, sample_rate)
        mfccs = librosa.feature.mfcc(signal, sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mfcc=N_MFCC)
        scaler = MinMaxScaler()
        transformed_mfccs = scaler.fit_transform(mfccs)
        word_folder = input_path.split('/')[0]
        Path(folder + word_folder).mkdir(parents=True, exist_ok=True)
        np.savetxt(folder + input_path[:-4] + ".csv", transformed_mfccs, delimiter=";")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Data preprocessor script for speech recognition"
                                                     "machine learning project.")
    arg_parser.add_argument("--train_num",
                            "-t",
                            metavar=f"Integer <{ALL_TRAIN}",
                            dest="train_num",
                            required=False,
                            default=-1,
                            help="Specifies the number of files used in training. (Def: all)")
    arg_parser.add_argument("--val_num",
                            "-v",
                            metavar=f"Integer <{ALL_VAL}",
                            dest="val_num",
                            required=False,
                            default=-1,
                            help="Specifies the number of files used in validation. (Def: all)")
    arg_parser.add_argument("--test_num",
                            "-e",
                            metavar=f"Integer <{ALL_TEST}",
                            dest="test_num",
                            required=False,
                            default=-1,
                            help="Specifies the number of files used in testing. (Def: all)")
    arg_parser.add_argument("--sample_rate",
                            "-s",
                            metavar="Integer",
                            dest="sample_rate",
                            type=int,
                            required=False,
                            default=22050,
                            help="Specifies the sample rate.")
    args = arg_parser.parse_args()

    train_num = int(args.train_num)
    val_num = int(args.val_num)
    test_num = int(args.test_num)

    train_list = parse_file_list(TRAIN_LIST, train_num, ALL_TRAIN)
    test_list = parse_file_list(TEST_LIST, test_num, ALL_TEST)
    val_list = parse_file_list(VAL_LIST, val_num, ALL_VAL)

    sample_rate = args.sample_rate

    Path(TRAIN_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(VAL_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(TEST_FOLDER).mkdir(parents=True, exist_ok=True)

    preprocess(train_list, TRAIN_FOLDER, sample_rate)
    preprocess(test_list, TEST_FOLDER, sample_rate)
    preprocess(val_list, VAL_FOLDER, sample_rate)

