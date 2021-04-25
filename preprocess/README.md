1.Download WSJ, ESC-50 and inside-small-room subset of AudioSet.  

2.Specify the data paths in the scripts.  

3.Process ESC-50 using Process_ESC.py.  

4.Process Audioset using ProcessAudioSet.py.  

5.Prepare wsj.scp for WSJ training set, validation set and test set yourself, respectively.  
The format in wsj.scp is  
uttid1  wav_path1  
uttid2  wav_path2  
....  
uttidn  wav_pathn  

6.Use genDataPkl.py to generate the pkl files for training and testing.
