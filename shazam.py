from PyQt5 import QtWidgets
from mainwindow import Ui_MainWindow
import sys
import numpy as np
from scipy.io import wavfile
from Database import SongModel
from os import path
from pydub import AudioSegment
import os
import collections
import itertools
import statistics
from difflib import SequenceMatcher

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.flag = 0 
        self.samplerate = 0
        self.data1=0
        self.data2=0
        self.OneMinuteData1=0
        self.OneMinuteData2=0
        self.data=[self.data1,self.data2]
        self.OneMinuteData=[self.OneMinuteData1,self.OneMinuteData2]
        self.ui.file1.clicked.connect(lambda:self.getfiles(0))
        self.ui.file2.clicked.connect(lambda:self.getfiles(1))
        self.ui.search.clicked.connect(self.similarty)
        self.ui.percentage.setText("0")
        self.ui.slider.valueChanged.connect(self.slider)
        self.song=[]
   
    def getfiles(self,i):
        path,extention = QtWidgets.QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "",
            "(*.mp3);;(*.wav) ")
        self.head,self.filename = os.path.split(path)
        self.wavFile = self.head+'/'+self.filename[:-4] + '.wav'
        if(path!=''):

            self.read_file(path,i) 

        else:
            pass   

    def read_file(self,path,i):
        sound = AudioSegment.from_mp3(path)
        sound.export(self.wavFile , format="wav")  
        self.samplerate, self.data[i] = wavfile.read(self.wavFile)
        if(len(self.data[i].shape)==2):
            if (self.data[i].shape[1]==2):
                self.data[i]  = np.mean(self.data[i], axis=1)
        NumberOfSample = self.data[i].shape[0] 
        TimeOfSampling=1/self.samplerate
        NumberOfSample =int(60/TimeOfSampling) 
        self.OneMinuteData[i] = self.data[i][0:NumberOfSample]

    def slider(self):
        value = self.ui.slider.value()
        self.ui.percentage.setText(str(value))

    def mixing (self):
        value = self.ui.slider.value()
        newdata1 = np.array((self.OneMinuteData[0] * (value/100)),dtype= np.int16)
        newdata2 = np.array((self.OneMinuteData[1] * (1-(value/100))),dtype= np.int16)
    
        if (len(newdata1)>len(newdata2)):
            newdata1=newdata1[:len(newdata2)]
    
        elif (len(newdata2)>len(newdata1)):
            newdata2=newdata2[:len(newdata1)]
    
        totaldata = newdata1 + newdata2
        mixingFile = self.head + '/newmixinfsong.wav'
        wavfile.write(mixingFile,self.samplerate,totaldata)
        song = SongModel(mixingFile)
        self.song.append(SongModel(mixingFile))

    def similarty(self):
        self.mixing()
        similarty={}
        similartyFeatures={}
        similartyCheck=[None]*4
        similartyCheckArray={}

        mixData= open("database/newmixinfsong.hashData", "r").read().split("/n")
        mixDataFeatures = open("database/newmixinfsong.SpectroFeatures", "r").read()
        fileNo = 0
        totalSimilaity = {}
       
        for filename in os.listdir("database"):
            finalCheck=0
            if filename.endswith(".hashData"):
                data = open("database/"+filename, "r").read().split("/n")
                for i in range(4):
                    similartyCheck[i] =self.similar(mixData[i],data[i])
                similarty[filename[:-9]]= statistics.mean(similartyCheck)

            elif (filename.endswith(".SpectroFeatures") ):
                    data = open("database/"+filename, "r").read()
                    similartyCheckArray[filename[:-16]]= self.similar(mixDataFeatures,data)
                
        for (k,v) in similarty.items(): 
            for (k2,v2) in similartyCheckArray.items():
                if k == k2:
                    totalSimilaity[k]=0.5*v2+0.5*v
         
        sorted_x = sorted(totalSimilaity.items(), key=lambda kv: kv[1],reverse=True)

        sorted_dict = collections.OrderedDict(sorted_x)
       
        del sorted_dict["newmixinfsong"]
        similarItems =  list(itertools.islice(sorted_dict.items(), 0, 15))
        songs = []
        similarity = []
        for x in similarItems:
            songs.append(x[0])
            similarity.append(x[1])
        
        for i in range(len(songs)):
            self.ui.indexTable.setItem(i,0,QtWidgets.QTableWidgetItem(str(songs[i])))
            self.ui.indexTable.setItem(i,1,QtWidgets.QTableWidgetItem(str(similarity[i])))


    def similar(self,a, b):

        return SequenceMatcher(None, a, b).ratio()
        
        

def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()