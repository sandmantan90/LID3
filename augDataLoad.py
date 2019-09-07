import librosa
import numpy as np
import torch

languages = {'Tam': '0','GUJ': '1','Mar': '2','HIN': '3', 'TEL': '4'}

def preAugmentation(config, fileNos, labels, augment = True):
    expDur = 6
    sampling = 16000
    samples = sampling*expDur
    noise, sampling = librosa.load('/home/aih04/LID3/noise7.wav', sr=sampling)
    
    i = 0
    while i < len(fileNos):
        fileName = config['trainDir'] + str(fileNos[i]) +'.wav'
        audio, sampling = librosa.load(fileName, sr = sampling)
        
        noFrames = 0
        if len(audio)<samples:
            while len(audio)<samples:
                audio = np.concatenate((audio,audio), axis=0)
            audio = audio[:samples]
            noFrames = 1
        
        elif (len(audio) % samples > samples*0.5):
            noFrames = int(np.ceil(len(audio) / samples))
            audio = np.concatenate((audio,audio), axis=0)
            audio = audio[:noFrames*samples]
            
        elif (len(audio) % samples < samples*0.5):
            noFrames = int(np.floor(len(audio) / samples))
            audio = audio[:noFrames*samples]

        imagesList = []
        labelsList = []
        for j in range(int(noFrames)):
            clip = audio[j*samples:(j+1)*samples]
            if augment:
                clip = augment(clip,noise,samples)
            melspec = librosa.feature.melspectrogram(clip, sr = samples,
                                                     n_mels = 129, fmax = 5000,
                                                     n_fft = 1600, hop_length = 192)
            img = librosa.power_to_db(melspec, ref = np.max)
            img = np.reshape(img, (config['inShape'][1], config['inShape'][2],
                                   config['inShape'][0]))
            img = torch.from_numpy(img)
            imagesList.append(img.permute(2,0,1))  # Channels, Height, Width
            labelsList.append(labels[i])
    
    return imagesList, labelsList

def augment(data,noise,samples):
    p = np.random.uniform(-1.5, 1.5)
    s = np.random.uniform(.83, 1.23)
    roll = np.random.uniform(0, len(data))
    noise_mag = np.random.uniform(0,.1)
    
    data = change_pitch(data,p)
    data = stretch(data, s,samples)
    data = np.roll(data, int(roll))
    data += noise[:samples]*noise_mag
    
    return(data)

def change_pitch(data, semitone=1):
    input_length =len(data)
    data = librosa.effects.pitch_shift(data, 16000, semitone, bins_per_octave=12)
    if len(data)>input_length:
        data = data[:input_length]
    else:
        while len(data)<input_length:
            data=np.concatenate((data,data),axis=0)
            
        data=data[0:input_length]
    return data
  
def stretch(data, rate=2):
    input_length =len(data)
    data = librosa.effects.time_stretch(data, rate )
    if len(data)>input_length:
        data = data[:input_length]
    else:
        while len(data) < samples:
            data=np.concatenate((data,data),axis=0)

        data=data[0:input_length]
    return data
