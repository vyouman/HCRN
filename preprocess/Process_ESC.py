import numpy as np
from os.path import join
from random import shuffle, seed
import pickle

interf_data_root = '/home/user/huangyating/dataset/ESC-50-master/audio'
fn = '/home/user/huangyating/ESC_all.pkl'
with open(fn,'rb') as f: 
    audioList, categoryList = pickle.load(f)

interf = [join(interf_data_root,recording) for recording in audioList]

seed(1234)
permuteIndex = np.arange(len(interf)).astype(int)
np.random.shuffle(permuteIndex)

interf = [interf[i] for i in permuteIndex]
categoryList = [categoryList[i] for i in permuteIndex]

trainN = int(round(len(interf) * 0.9))
interf_train = interf[:trainN]
interf_test = interf[trainN::]

interf_train_cat = categoryList[:trainN]
interf_test_cat = categoryList[trainN::]

# Saving the objects:
sav_fn = '/home/user/huangyating/ESc_split.pkl'
with open(sav_fn, 'wb') as f:  # Python 2 open(..., 'w') Python 3: open(..., 'wb')
    pickle.dump([interf_train, interf_test, interf_train_cat, interf_test_cat], f)
