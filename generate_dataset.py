import os
import librosa
import numpy as np
import utils
import operator
import itertools
import time
from joblib import Parallel, delayed

fix_sr = 16000
time_sec = 0.5


def test_gen(audio_file_list):

    noisy_file_list = []
    noisy_file_data = []
    for idx,path in enumerate(audio_file_list):
        # if idx%4==0:
        #     print("*", end='')
        data, _ = librosa.load("./orig_dataset/noisy_testset_wav/"+path, sr=fix_sr)
        num_samples = int(time_sec * fix_sr)
        current_path = './dataset/test/noisy/' + path.split('_')[0] + '/'
        if not os.path.isdir(current_path):
            os.mkdir(current_path)
            os.mkdir('./dataset/test/crm/' + path.split('_')[0] + '/')
        for i_sub in range(int(data.shape[0]//num_samples)):
            temp = data[num_samples*i_sub:num_samples*(i_sub+1)]
            temp = utils.fast_stft(temp)
            name = path.split('_')[0] + '-' + path.split('_')[1].split('.')[0] + ('-%02d'%i_sub)
            np.save(current_path + ('%s.npy'%name),temp)
            noisy_file_list.append(current_path + ('%s.npy'%name))
            noisy_file_data.append(temp)
    print()

    clean_file_data = []
    for idx,path in enumerate(audio_file_list):
        # if idx%4==0:
        #     print("*", end='')
        data, _ = librosa.load("./orig_dataset/clean_testset_wav/"+path, sr=fix_sr)
        num_samples = int(time_sec * fix_sr)
        for i_sub in range(int(data.shape[0]//num_samples)):
            temp = data[num_samples*i_sub:num_samples*(i_sub+1)]
            temp = utils.fast_stft(temp)
            clean_file_data.append(temp)
    print()

    assert len(noisy_file_data) == len(clean_file_data)
    for i in range(len(noisy_file_list)):
        # if i%4==0:
        #     print("*", end='')
        cRM_data = utils.fast_cRM(clean_file_data[i],noisy_file_data[i])
        np.save((noisy_file_list[i].replace("noisy", "crm")),cRM_data)
    print()



def train_gen(audio_file_list):
    noisy_file_list = []
    noisy_file_data = []
    for idx,path in enumerate(audio_file_list):
        # if idx%4==0:
        #     print("*", end='')
        data, _ = librosa.load("./orig_dataset/noisy_trainset_wav/"+path, sr=fix_sr)
        num_samples = int(time_sec * fix_sr)
        current_path = './dataset/train/noisy/' + path.split('_')[0] + '/'
        if not os.path.isdir(current_path):
            os.mkdir(current_path)
            os.mkdir('./dataset/train/crm/' + path.split('_')[0] + '/')
        for i_sub in range(int(data.shape[0]//num_samples)):
            temp = data[num_samples*i_sub:num_samples*(i_sub+1)]
            temp = utils.fast_stft(temp)
            name = path.split('_')[0] + '-' + path.split('_')[1].split('.')[0] + ('-%02d'%i_sub)
            np.save(current_path + ('%s.npy'%name),temp)
            noisy_file_list.append(current_path + ('%s.npy'%name))
            noisy_file_data.append(temp)
    print()

    clean_file_data = []
    for idx,path in enumerate(audio_file_list):
        # if idx%4==0:
        #     print("*", end='')
        data, _ = librosa.load("./orig_dataset/clean_trainset_wav/"+path, sr=fix_sr)
        num_samples = int(time_sec * fix_sr)
        for i_sub in range(int(data.shape[0]//num_samples)):
            temp = data[num_samples*i_sub:num_samples*(i_sub+1)]
            temp = utils.fast_stft(temp)
            clean_file_data.append(temp)
    print()

    assert len(noisy_file_data) == len(clean_file_data)
    for i in range(len(noisy_file_list)):
        # if i%4==0:
        #     print("*", end='')
        cRM_data = utils.fast_cRM(clean_file_data[i],noisy_file_data[i])
        np.save((noisy_file_list[i].replace("noisy", "crm")),cRM_data)
    print()


def generate_dataset():

    if not os.path.isdir('./dataset'):
        os.mkdir('./dataset')

    if not os.path.isdir('./dataset/test'):
        os.mkdir('./dataset/test')

    if not os.path.isdir('./dataset/train'):
        os.mkdir('./dataset/train')

    
    # test

    if not os.path.isdir('./dataset/test/crm/'):
        os.mkdir('./dataset/test/noisy/')
        os.mkdir('./dataset/test/crm/')

    audio_file_list = [f for f in os.listdir("./orig_dataset/noisy_testset_wav") if "wav" in f]
    print('length of the path list: ',len(audio_file_list))

    num_workers = os.cpu_count()
    num_per_worker = len(audio_file_list) // num_workers
    print("num_per_worker", num_per_worker)
    Parallel(n_jobs=num_workers, backend='multiprocessing')(delayed(test_gen)(audio_file_list[i * num_per_worker: (i + 1) * num_per_worker]) for i in range(num_workers))




    # train set

    if not os.path.isdir('./dataset/train/crm/'):
        os.mkdir('./dataset/train/noisy/')
        os.mkdir('./dataset/train/crm/')

    audio_file_list = [f for f in os.listdir("./orig_dataset/noisy_trainset_wav") if "wav" in f]
    print('length of the path list: ',len(audio_file_list))

    num_workers = os.cpu_count()
    num_per_worker = len(audio_file_list) // num_workers
    print("num_per_worker", num_per_worker)
    Parallel(n_jobs=num_workers, backend='multiprocessing')(delayed(train_gen)(audio_file_list[i * num_per_worker: (i + 1) * num_per_worker]) for i in range(num_workers))
    





















