import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa.display
import librosa
import os
from pydub import AudioSegment
from imagededup.methods import DHash , AHash , WHash ,PHash 
import hashlib

class SongModel():

    def __init__(self, songPath: str):

        self.songPath = songPath
        path , ext= os.path.splitext(self.songPath)
        self.head, self.fileName = os.path.split(self.songPath)
        if (ext == ".wav"):
            self.sampleRate, self.Data = wavfile.read(self.songPath)
            self.wavfile = self.songPath
        elif (ext ==".mp3"):
            self.Sound = AudioSegment.from_mp3(self.songPath)
            self.wavfile = 'database/'+self.fileName[:-4]+'.wav'
            self.Sound.export(self.wavfile, format="wav")    
            self.sampleRate, self.Data = wavfile.read(self.wavfile)
     
        if(len(self.Data.shape)==2):
            if (self.Data.shape[1]==2):
                self.Data = np.mean(self.Data, axis=1)

        self.TimeOfSampling=1/self.sampleRate
        self.NumberOfSample =int(60/self.TimeOfSampling) 
        self.Data = self.Data[0:self.NumberOfSample]
   
        self.HashFuncs = [AHash(),WHash(),PHash(),DHash()]
        self.HashFileNames  = ["AHash","WHash","PHash","DHash"]
        self.imageArray = None

        self.spectrogram(self.Data,self.sampleRate)
        self.SpectrogramFeatures()
        self.hashFunction()
    
    def spectrogram(self,samples, sample_rate, stride_ms = 10.0, 
                            window_ms = 20.0, max_freq = None, eps = 1e-14):

        stride_size = int(0.001 * sample_rate * stride_ms)
        window_size = int(0.001 * sample_rate * window_ms)

        # Extract strided windows
        truncate_size = (len(samples) - window_size) % stride_size
        samples = samples[:len(samples) - truncate_size]
        nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
        nstrides = (samples.strides[0], samples.strides[0] * stride_size)
        windows = np.lib.stride_tricks.as_strided(samples, 
                                            shape = nshape, strides = nstrides)
        
        assert np.all(windows[:, 1] == samples[stride_size:(stride_size + window_size)])

        # Window weighting, squared Fast Fourier Transform (fft), scaling
        weighting = np.hanning(window_size)[:, None]
        
        fft = np.fft.rfft(windows * weighting, axis=0)
        fft = np.absolute(fft)
        fft = fft**2
        
        scale = np.sum(weighting**2) * sample_rate
        fft[1:-1, :] *= (2.0 / scale)
        fft[(0, -1), :] /= scale
        
        # Prepare fft frequency list
        freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])
        
        # Compute spectrogram feature
        ind = np.where(freqs <= max(freqs))[0][-1] + 1
        self.imageArray = fft[:ind,:] + eps

        self.Spectrogram = np.log(fft[:ind, :] + eps)
        fig = plt.Figure()
        ax = fig.add_subplot(111)
     
        p = librosa.display.specshow(librosa.amplitude_to_db(self.Spectrogram , ref=np.max), ax=ax, y_axis='log', x_axis='time')
        fig.savefig(self.songPath[:-4]+'.png')
        
    def SpectrogramFeatures(self):
        featuresData = librosa.feature.spectral_bandwidth(y=np.array(self.Data,dtype=np.float32))
        featuresData = PHash().encode_image(image_array = featuresData)
        FileNameOfspectroFeatures = self.fileName[:-4] +".SpectroFeatures"    
        self.saveFeaturesData(FileNameOfspectroFeatures,featuresData)
    
    def hashFunction(self):
        x = None
        HashData = ""
        for i in range (4):
            x = self.HashFuncs[i]
            HashData += x.encode_image(image_array = self.imageArray)
            HashData+="/n"
            fileName = self.fileName[:-4] + ".hashData"
        with open(os.path.join(self.head,fileName), 'w') as f:
            f.write(str(HashData))

    
    def saveFeaturesData(self,fileName,data):   
        with open(os.path.join(self.head,fileName), 'w') as f:
            f.write(str(data))
    
    
    def hashFor(self,data):
        # Prepare the project id hash
        hashId = hashlib.md5()

        hashId.update(repr(data).encode('utf-8'))

        return hashId.hexdigest()

    
def main():
    songs=[]
    for filename in os.listdir("database"):
        if filename.endswith(".mp3"):
            song =  SongModel("database/"+filename)
            songs.append(song)


if __name__ == "__main__":
    main()
