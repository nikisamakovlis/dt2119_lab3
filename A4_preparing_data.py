import numpy as np
from lab3_tools import *
from lab3_proto import *
from lab2_proto import concatHMMs
from lab1_proto import mfcc
from prondict import prondict


def target_class_definition(model_path):  # Assignment 4.1
    phoneHMMs = np.load(model_path, allow_pickle=True)['phoneHMMs'].item()
    phones = sorted(phoneHMMs.keys())
    nstates = {phone: phoneHMMs[phone]['means'].shape[0] for phone in phones}
    stateList = [ph + '_' + str(id) for ph in phones for id in range(nstates[ph])]
    with open(r'stateListFile.txt', 'w') as f:
        f.write('\n'.join(stateList))

    return phoneHMMs, stateList


def forced_alignment(phoneHMMs):  # Assignment 4.2
    filename = 'tidigits/train/man/nw/z43a.wav'
    samples, sampling_rate = loadAudio(filename)
    lmfcc = mfcc(samples)

    word_trans = list(path2info(filename)[2])

    phone_trans = words2phones(word_trans, prondict)
    utterance_HMM = concatHMMs(phoneHMMs, phone_trans)


def main():
    all_models_path = 'lab2_models_all.npz'
    phoneHMMs, stateList = target_class_definition(all_models_path)
    forced_alignment(phoneHMMs)


main()