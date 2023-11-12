import re
from scipy.signal import savgol_filter
class Wave:
    def __init__(self,file):
        self.start=420
        self.end=1750
        self.FloatWaveLengthList = []
        self.IntWaveTimeList = []
        self.FloatWaveTimeSecondList = []
        self.FloatWaveAmplitudeLists = []
        with open(file, 'r', encoding='utf8') as wavefile_object:
            wavecontents = wavefile_object.read()
            wavefile_object.close()
        for one in re.findall(r'^([\d|\.| ]*)\n', wavecontents):
            StrWaveLengthList=one.split(' ')
            for s in StrWaveLengthList:
                if(s !=''):
                    self.FloatWaveLengthList.append(float(s))
        self.FloatWaveLengthList=self.FloatWaveLengthList[self.start:self.end]

        for one in re.findall(r'(.{23})\n([\d|\.|\-| ]*)\n', wavecontents):
            self.IntWaveTimeList.append(int(one[0][0:4])*10000000000000+int(one[0][5:7])*100000000000+
            int(one[0][8:10])*1000*60*60*24+int(one[0][11:13])*1000*60*60+int(one[0][14:16])*1000*60+
            int(one[0][17:19])*1000+int(one[0][20:23]))
            StrWaveAmplitudeList=one[1].split(' ')
            FloatWaveAmplitudeList = []
            sum=0
            for s in StrWaveAmplitudeList:
                if (s != ''):
                    FloatWaveAmplitudeList.append(float(s))
                    sum=sum+float(s)

            FloatWaveAmplitudeListnew = [(FloatWaveAmplitudeList[i]+FloatWaveAmplitudeList[i+1]+FloatWaveAmplitudeList[i+2]+FloatWaveAmplitudeList[i+3]+
                                              FloatWaveAmplitudeList[i+4]+FloatWaveAmplitudeList[i+5]+FloatWaveAmplitudeList[i+6]+FloatWaveAmplitudeList[i+7]+
                                              FloatWaveAmplitudeList[i+8]+FloatWaveAmplitudeList[i+9]+FloatWaveAmplitudeList[i+10]+FloatWaveAmplitudeList[i+11]+
                                              FloatWaveAmplitudeList[i+12]+FloatWaveAmplitudeList[i+13]+FloatWaveAmplitudeList[i+14])/15 for i in range(len(FloatWaveAmplitudeList)-15)]
            FloatWaveAmplitudeListnew=FloatWaveAmplitudeListnew[self.start:self.end]
            #FloatWaveAmplitudeListnew = savgol_filter(FloatWaveAmplitudeListnew, 61, 0)
            #FloatWaveAmplitudeListnew = [FloatWaveAmplitudeListnew[i]-FloatWaveAmplitudeListnew1[i] for i in range(self.end-self.start)]
            self.FloatWaveAmplitudeLists.append(FloatWaveAmplitudeListnew)
        self.FloatWaveTimeSecondList = [(i - self.IntWaveTimeList[0]) / 1000.0 for i in self.IntWaveTimeList]
        self.TimeStart = str(int(self.IntWaveTimeList[0]%100000000000 / 1000 / 60 / 60 / 24)) + '号' + str(
            int(self.IntWaveTimeList[0]%100000000000 / 60 / 60 / 1000) % 24) + '时' + str(
            int(self.IntWaveTimeList[0]%100000000000 / 1000 / 60) % 60) + '分' + str(
            int(self.IntWaveTimeList[0]%100000000000 / 1000) % 60) + '.' + str(int(self.IntWaveTimeList[0]) % 1000) + '秒'
        self.TimeEnd = str(int(self.IntWaveTimeList[-1]%100000000000 / 1000 / 60 / 60 / 24)) + '号' + str(
            int(self.IntWaveTimeList[-1]%100000000000 / 60 / 60 / 1000) % 24) + '时' + str(
            int(self.IntWaveTimeList[-1]%100000000000 / 1000 / 60) % 60) + '分' + str(
            int(self.IntWaveTimeList[-1]%100000000000 / 1000) % 60) + '.' + str(int(self.IntWaveTimeList[-1]) % 1000) + '秒'

if __name__ == '__main__':
    t=Wave('P203160901_2021-01-31 02-18-26 626.txt')
