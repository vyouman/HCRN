# -*- coding: utf-8 -*-
import os
import pickle

# # 115
# unbalance_set_path = '/data/xujiaming_data/audioset/backgrounds/acousticenvironment/insidesmallroom/unb_insidesmall_single'
# balance_set_path = '/data/xujiaming_data/audioset/backgrounds/acousticenvironment/insidesmallroom/insidesmall_singleb'
# balance_eval_path = '/data/xujiaming_data/audioset/backgrounds/acousticenvironment/insidesmallroom/insidesmall_singlee'

# PATH TO AUDIOSET
unbalance_set_path = '/data/hyt_data/insidesmallroom/unb_insidesmall_single'
balance_set_path = '/data/hyt_data/insidesmallroom/insidesmall_singleb'
balance_eval_path = '/data/hyt_data/insidesmallroom/insidesmall_singlee'

train_list = []
valid_list = []
test_list = []

def traversal(lst, dir):
    for root, dirs, files in os.walk(dir):
        for f in files:
            if f.endswith('.wav'):
                lst.append(os.path.join(dir,f))

traversal(train_list, unbalance_set_path)
traversal(valid_list, balance_set_path)
traversal(test_list, balance_set_path)

# PATH FOR AUDIOSET PKL FILE
audio_set_fn = '/home/user/huangyating/SpeechSynthesis4Separation-master/Data/audio_split.pkl'
with open(audio_set_fn, 'wb') as f:  # Python 2 open(..., 'w') Python 3: open(..., 'wb')
    # pickle.dump([target_train, target_test, interf_train, interf_test, interf_train_cat, interf_test_cat], f)
    pickle.dump([train_list, valid_list, test_list], f)

#　PATH FOR ESC PKL FILE
esc_fn = '/home/user/huangyating/SpeechSynthesis4Separation-master/Data/ESc_split.pkl'
with open(esc_fn, 'rb') as f:
    interf_train, interf_test_esc, _, _ = pickle.load(f)
trainNum_interf = int(round(len(interf_train) * 0.8))
validNum_interf = len(interf_train) - trainNum_interf
interf_train_esc = interf_train[:trainNum_interf]
interf_valid_esc = interf_train[trainNum_interf::]
esc_fn_new = '/home/user/huangyating/SpeechSynthesis4Separation-master/Data/ESc_split_all.pkl'
with open(esc_fn_new, 'wb') as f:
    pickle.dump([interf_train_esc, interf_valid_esc, interf_test_esc], f)

# PATH FOR ESC AND AUDIOSET PKL FILE
audio_add_esc_fn = '/home/user/huangyating/SpeechSynthesis4Separation-master/Data/audio_esc.pkl'
train_list = train_list + interf_train_esc
valid_list = valid_list + interf_valid_esc
test_list = test_list + interf_test_esc
with open(audio_add_esc_fn, 'wb') as f:
    pickle.dump([train_list, valid_list, test_list], f)
