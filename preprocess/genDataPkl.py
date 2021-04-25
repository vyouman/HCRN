import pickle
import soundfile as sf
# PATH to WSJ wav list
# scp data format
# utterance-id  wav_path
noise = '/home/user/huangyating/SpeechSynthesis4Separation-master/Data/audio_esc.pkl'
train_set = '/home/user/huangyating/espnet/egs/wsj/tts1/data/train_si284/wav.scp'
valid_set = '/home/user/huangyating/espnet/egs/wsj/tts1/data/test_dev93/wav.scp'
test_set = '/home/user/huangyating/espnet/egs/wsj/tts1/data/test_eval92/wav.scp'

# PATH TO PKL FILE
target_fn = '/data/hyt_data/wsj/train_valid.pkl'
interf_fn = '/data/hyt_data/esc_audio/train_valid.pkl'
target_fn_t = '/data/hyt_data/wsj/test.pkl'
interf_fn_t = '/data/hyt_data/esc_audio/test.pkl'


def write_dict(ll):
    dic = {}
    for l in ll:
        print(l)
        l = l.strip()
        s, sr = sf.read(l)
        dic[l] = s
    return dic

# noise
with open(noise, 'rb') as f:
    [tr,cv,tt] = pickle.load(f)
    tr_dict=write_dict(tr)
    cv_dict=write_dict(cv)
    tt_dict=write_dict(tt)
    with open(interf_fn, 'wb') as f:
        pickle.dump([tr_dict, cv_dict], f)

    with open(interf_fn_t, 'wb') as f:
        pickle.dump(tt_dict, f)


# wsj
def open_wsj_scp(fn):
    dic = {}
    with open(fn, 'r') as f:
        for l in f.readlines():
            fnn = l.strip().split(' ')[-1]
            print(fnn)
            sig, sr = sf.read(fnn)
            dic[fnn] = sig
    return dic

train_dict = open_wsj_scp(train_set)
valid_dict = open_wsj_scp(valid_set)
test_dict = open_wsj_scp(test_set)

with open(target_fn, 'wb') as f:
    pickle.dump([train_dict, valid_dict], f)
with open(target_fn_t,'wb') as f:
    pickle.dump(test_dict, f)
