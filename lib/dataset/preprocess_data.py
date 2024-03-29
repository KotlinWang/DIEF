import os.path

import librosa
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pylab as pylab

from sklearn.utils import shuffle
import multiprocessing

parser = argparse.ArgumentParser()
parser.add_argument("--csv_file", type=str)
parser.add_argument("--data_dir", type=str)
parser.add_argument("--store_dir", type=str)
parser.add_argument("--num_works", type=int)
parser.add_argument("--split_rate", type=float)


def thread_preprocess(args, num_works, audios, date_type):
    l = []
    if args.num_works == 0:
        num = len(audios)
    else:
        num = len(audios) // args.num_works
    end = 0
    for i in range(num_works):
        star = end
        end = (i + 1) * num
        audio = audios[star:end]
        t = multiprocessing.Process(target=extract_features, args=(args, audio, date_type), daemon=True)
        l.append(t)
        t.start()
    if num * num_works < len(audios):
        audio = audios[end:]
        t = multiprocessing.Process(target=extract_features, args=(args, audio, date_type), daemon=True)
        l.append(t)
        t.start()
    for k in l:
        # k.terminate()
        k.join()


def save_data(cfg, data_list, data_type):
    with open(os.path.join(cfg.store_dir, 'json', '{}.txt'.format(data_type)), 'a') as file:
        file.writelines(data_list)
    file.close()
    print('Finish...')


def extract_features(args, audios, data_type):
    audio_names = list(audios.slice_file_name.unique())
    values = []
    for audio in tqdm(audio_names):
        entries = audios.loc[audios["slice_file_name"] == audio].to_dict(orient="records")

        data, sr = librosa.load("{}/fold{}/{}".format(args.data_dir, entries[0]["fold"],
                                                      audio))  # All audio all sampled to a sampling rate of 22050

        if data.shape[0] < 88200:
            # clip = np.concatenate((data, np.zeros((88200 - data.shape[0]))))
            clip = np.concatenate((data, [1e-5]*(88200 - data.shape[0])))
        elif data.shape[0] > 88200:
            clip = data[:88200]
        else:
            clip = data
        Pxx, freqs, bins, imm = pylab.specgram(clip, Fs=sr, cmap=pylab.get_cmap('jet'))
        pylab.axis('off')
        pylab.tight_layout(pad=0)
        pylab.savefig(os.path.join(args.store_dir, 'ImageBase', '{}.jpg'.format(audio)))
        pylab.close()
        pylab.cla()
        pylab.clf()
        np.savetxt(os.path.join(args.store_dir, 'AudioBase', '{}.txt'.format(audio)), clip, fmt="%f", delimiter="\n")

        values.append("{} {:d}\n".format(audio, entries[0]['classID']))
    save_data(args, values, data_type)


if __name__ == "__main__":
    args = parser.parse_args()
    audios = pd.read_csv(args.csv_file, skipinitialspace=True)
    num_class = 10

    for i in range(num_class):
        data = audios.loc[audios["classID"] == i]
        data = shuffle(data)
        training_audios = data.iloc[:int(len(data) * args.split_rate)]
        validation_audios = data.iloc[int(len(data) * args.split_rate):]

        thread_preprocess(args, args.num_works, training_audios, 'train')
        thread_preprocess(args, args.num_works, validation_audios, 'test')
