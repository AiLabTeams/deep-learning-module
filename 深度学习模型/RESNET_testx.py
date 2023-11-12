from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestRegressor
import WavePackage
import PressPackage
import math
from scipy.optimize import curve_fit
from scipy.fftpack import fft,ifft
import random
import Gramme
import WavePackage
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import torch
from torch import nn
mod = "resnet20230312.30"
LightWaveAmplitudeListsRaw = []
LightWaveAmplitudeListsFilter = []
LightWaveAmplitudeLists = []
LightWaveLengthList = []
LightWaveTimeList = []
LightWaveTimeSecondList = []
PressTimeList = []
PressAmplitudeList = []
NewPressAmplitudeList = []
PressAmplitudelogList = []
lightfilecount=0
PressTimeToFix=0
LightTimeToFix=0
GASFMatrixListRaw = []
GADFMatrixListRaw = []
GASFMatrixList = []
GADFMatrixList = []
AllGASFMatrixList = []
AllGADFMatrixList = []

Allfftlist=[]
AllPressTimeToFix=[]
AllLightTimeToFix=[]
AllPressTimeList = []
AllPressAmplitudelogList = []
AllLightWaveTimeSecondList=[]
AllLightWaveAmplitudeLists=[]

fftabslistlist = []
fftanglelist = []

Allxlist=[]
Allylist=[]
xtrainlist=[]
ytrianlist=[]
yprelist=[]
xprelist=[]
timedis=300
channel=5
batch=1
lr, num_epochs = 0.01 , 200
channelmut=8
refuc=nn.Sigmoid
class MyNet(nn.Module):
   def __init__(self):
       super(MyNet, self).__init__()
       self.convs = nn.Sequential(
           nn.Conv2d(1, 2 *channel, 5, 1),  # in_channels, out_channels,kernel_size
           refuc(),
           nn.BatchNorm2d(2 * channel),
       )
       self.convd = nn.Sequential(
           nn.Conv2d(1, 2 *channel, 5, 1),  # in_channels, out_channels,kernel_size
           refuc(),
           nn.BatchNorm2d(2 * channel),
       )
       self.makelayers1 = nn.Sequential(
           nn.Conv2d(2 * channel, channel, 5, 1),  # 2
           nn.BatchNorm2d(channel),
           refuc(),
           nn.Conv2d(channel, channel, 5, 1),  # 3
           nn.BatchNorm2d(channel),
           refuc(),
       )
       self.makelayers2 = nn.Sequential(
           nn.Conv2d(2 * channel, channel, 5, 1),  # 2
           nn.BatchNorm2d(channel),
           refuc(),
           nn.Conv2d(channel, channel, 5, 1),  # 3
           nn.BatchNorm2d(channel),
           refuc(),
       )
       self.makelayers3 = nn.Sequential(
           nn.Conv2d(2 * channel, channel, 5, 1),  # 2
           nn.BatchNorm2d(channel),
           refuc(),
           nn.Conv2d(channel, channel, 5, 1),  # 3
           nn.BatchNorm2d(channel),
           refuc(),
       )
       self.makelayers4 = nn.Sequential(
           nn.Conv2d(2 * channel, channel, 5, 1),  # 2
           nn.BatchNorm2d(channel),
           refuc(),
           nn.Conv2d(channel, channel, 5, 1),  # 3
           nn.BatchNorm2d(channel),
           refuc(),
       )
       self.makelayers5 = nn.Sequential(
           nn.Conv2d(2 * channel, channel, 5, 1),  # 2
           nn.BatchNorm2d(channel),
           refuc(),
           nn.Conv2d(channel, channel, 5, 1),  # 3
       )
       self.makelayerd1 = nn.Sequential(
           nn.Conv2d(2 * channel, channel, 5, 1),  # 2
           nn.BatchNorm2d(channel),
           refuc(),
           nn.Conv2d(channel, channel, 5, 1),  # 3
           nn.BatchNorm2d(channel),
           refuc(),
       )
       self.makelayerd2 = nn.Sequential(
           nn.Conv2d(2 * channel, channel, 5, 1),  # 2
           nn.BatchNorm2d(channel),
           refuc(),
           nn.Conv2d(channel, channel, 5, 1),  # 3
           nn.BatchNorm2d(channel),
           refuc(),
       )
       self.makelayerd3 = nn.Sequential(
           nn.Conv2d(2 * channel, channel, 5, 1),  # 2
           nn.BatchNorm2d(channel),
           refuc(),
           nn.Conv2d(channel, channel, 5, 1),  # 3
           nn.BatchNorm2d(channel),
           refuc(),
       )
       self.makelayerd4 = nn.Sequential(
           nn.Conv2d(2 * channel, channel, 5, 1),  # 2
           nn.BatchNorm2d(channel),
           refuc(),
           nn.Conv2d(channel, channel, 5, 1),  # 3
           nn.BatchNorm2d(channel),
           refuc(),
       )
       self.makelayerd5 = nn.Sequential(
           nn.Conv2d(2 * channel, channel, 5, 1),  # 2
           nn.BatchNorm2d(channel),
           refuc(),
           nn.Conv2d(channel, channel, 5, 1),  # 3

       )
       self.downlayers1=nn.Sequential(nn.Conv2d(2 * channel, channel, 9, 1),)
       self.downlayers2=nn.Sequential(nn.Conv2d(2 * channel, channel, 9, 1),)
       self.downlayers3=nn.Sequential(nn.Conv2d(2 * channel, channel, 9, 1),)
       self.downlayers4=nn.Sequential(nn.Conv2d(2 * channel, channel, 9, 1),)
       self.downlayers5=nn.Sequential(nn.Conv2d(2 * channel, channel, 9, 1),)
       self.downlayerd1=nn.Sequential(nn.Conv2d(2 * channel, channel, 9, 1),)
       self.downlayerd2=nn.Sequential(nn.Conv2d(2 * channel, channel, 9, 1),)
       self.downlayerd3=nn.Sequential(nn.Conv2d(2 * channel, channel, 9, 1),)
       self.downlayerd4=nn.Sequential(nn.Conv2d(2 * channel, channel, 9, 1),)
       self.downlayerd5=nn.Sequential(nn.Conv2d(2 * channel, channel, 9, 1),)
       self.fcs = nn.Sequential(
           nn.Linear(channel, 1),
       )
       self.fcd = nn.Sequential(
           nn.Linear(channel, 1),
       )
       self.fc = nn.Sequential(
           nn.Linear(2, 1)
       )
   def forward(self, imgs,imgd):
       features = self.convs(imgs)
       mlayers1 = self.makelayers1(features)
       dlayers1 = self.downlayers1(features)
       mlayers2 = self.makelayers2(torch.cat((mlayers1,dlayers1),1))
       dlayers2 = self.downlayers2(torch.cat((mlayers1,dlayers1),1))
       mlayers3 = self.makelayers3(torch.cat((mlayers2, dlayers2), 1))
       dlayers3 = self.downlayers3(torch.cat((mlayers2, dlayers2), 1))
       mlayers4 = self.makelayers4(torch.cat((mlayers3, dlayers3), 1))
       dlayers4 = self.downlayers4(torch.cat((mlayers3, dlayers3), 1))
       mlayers5 = self.makelayers5(torch.cat((mlayers4,dlayers4),1))
       featured = self.convd(imgd)
       mlayerd1 = self.makelayerd1(featured)
       dlayerd1 = self.downlayerd1(featured)
       mlayerd2 = self.makelayerd2(torch.cat((mlayerd1, dlayerd1), 1))
       dlayerd2 = self.downlayerd2(torch.cat((mlayerd1, dlayerd1), 1))
       mlayerd3 = self.makelayerd3(torch.cat((mlayerd2, dlayerd2), 1))
       dlayerd3 = self.downlayerd3(torch.cat((mlayerd2, dlayerd2), 1))
       mlayerd4 = self.makelayerd4(torch.cat((mlayerd3, dlayerd3), 1))
       dlayerd4 = self.downlayerd4(torch.cat((mlayerd3, dlayerd3), 1))
       mlayerd5 = self.makelayerd5(torch.cat((mlayerd4, dlayerd4), 1))
       outputs = self.fcs(mlayers5.view(imgs.shape[0], -1))
       outputd = self.fcd(mlayerd5.view(imgs.shape[0], -1))
       output = self.fc(torch.cat((outputs,outputd),1))
       return output
def Import_light_file(fpath):
    global LightWaveAmplitudeListsRaw
    global LightWaveAmplitudeListsFilter
    global LightWaveAmplitudeLists
    global LightWaveLengthList
    global LightWaveTimeList
    global LightWaveTimeSecondList
    global lightfilecount
    global LightTimeToFix
    try:
        t = WavePackage.Wave(fpath)

        if lightfilecount == 0:
            LightTimeStart = t.TimeStart
            LightTimeToFix = t.IntWaveTimeList[0] % 100000000000 / 1000
            LightWaveLengthList = t.FloatWaveLengthList
            LightWaveAmplitudeListsRaw += t.FloatWaveAmplitudeLists

            LightWaveTimeList += t.IntWaveTimeList
            LightWaveTimeSecondList += t.FloatWaveTimeSecondList
            lightfilecount = lightfilecount + 1
            LightTimeEnd = t.TimeEnd
        else:
            LightWaveAmplitudeListsRaw += t.FloatWaveAmplitudeLists
            LightWaveTimeList += t.IntWaveTimeList
            temp = len(LightWaveTimeSecondList)
            LightWaveTimeSecondList += [i + LightWaveTimeSecondList[1] - LightWaveTimeSecondList[0] +
                                             LightWaveTimeSecondList[temp - 1]
                                             for i in t.FloatWaveTimeSecondList]
            lightfilecount = lightfilecount + 1
            LightTimeEnd = t.TimeEnd
        LightWaveAmplitudeLists = LightWaveAmplitudeListsRaw
        print('光谱数据读入成功')
        print('当前载入光谱文件数量: ' + str(lightfilecount))
    except:
        print('光谱数据读入失败')
    pass
def LightWavefilter():
    global LightWaveAmplitudeListsFilter
    global LightWaveAmplitudeLists
    LightWaveAmplitudeListsFilter=LightWaveAmplitudeLists.copy()
    templist=[0 for i in range(len(LightWaveAmplitudeLists))]
    for i in range(len(LightWaveAmplitudeLists[0])):
        for iii in range(len(LightWaveAmplitudeLists)):
            templist[iii]=LightWaveAmplitudeLists[iii][i]
        #templist=savgol_filter(templist, 5, 1)
        for iii in range(len(LightWaveAmplitudeLists)):
               LightWaveAmplitudeListsFilter[iii][i]=templist[iii]
    LightWaveAmplitudeLists=LightWaveAmplitudeListsFilter.copy()

def Import_press_file(fpath):
    global PressTimeList
    global PressAmplitudeList
    global NewPressAmplitudeList
    global PressAmplitudelogList
    global PressTimeToFix
    try:
        t = PressPackage.Press(fpath)
        PressTimeStart = t.TimeStart
        PressTimeToFix = t.IntPressTimeList[0] % 100000000000 / 1000
        PressTimeList = t.IntPressTimeSecondList

        PressAmplitudeList = t.IntPressDataList
        PressTimeEnd = t.TimeEnd
        PressMove = 0.013
        temp = 0
        pmin = min(PressAmplitudeList)
        PressAmplitudeList=[i-pmin for i in PressAmplitudeList]
        for i in range(len(PressAmplitudeList)):
            temp = temp + PressMove
            if i > 5 and PressAmplitudeList[i] - PressAmplitudeList[i - 5] > 10:
                temp = temp - PressMove
            if i > 5 and PressAmplitudeList[i] - PressAmplitudeList[i - 5] < -10:
                temp = 0
            NewPressAmplitudeList.append(max(PressAmplitudeList[i] - temp, 2))
            #NewPressAmplitudeList.append(min(PressAmplitudeList[i], PressAmplitudeList[len(PressAmplitudeList)-1]*1.01))
        PressAmplitudelogList = [math.log(max(2, i), 10) for i in NewPressAmplitudeList]
        print('气压数据读入成功')
    except:
        print('气压数据读入失败')

def fftfuc():
    for i in range(len(LightWaveAmplitudeLists)):
        ffty = fft(LightWaveAmplitudeLists[i])
        fftyabs = [abs(i) for i in ffty]
        fftyangle = [i.imag for i in ffty]
        fftabslistlist.append(fftyabs[0:5])
        fftanglelist.append(fftyangle[0:5])
    pass
def SGDfuc():
    for i in range(len(LightWaveAmplitudeLists)):
        LightWaveAmplitudeLists[i]=savgol_filter(LightWaveAmplitudeLists[i],len(LightWaveAmplitudeLists[i])//2,5)
    pass
def GASFFunc():
    global GASFMatrixListRaw
    global GASFMatrixList
    for i in LightWaveAmplitudeLists:
        listtemp = []
        for ii in range(0, len(i), 15):
            if(LightWaveLengthList[ii]>600 and LightWaveLengthList[ii]<900):
                            listtemp.append(i[ii])
        temp=Gramme.GASFExchange(listtemp,flag=0)
        GASFMatrixListRaw.append(temp)
    GASFMatrixList=GASFMatrixListRaw.copy()
    print('GASF 成功')
def GADFFunc():
    global GADFMatrixListRaw
    global GADFMatrixList
    for i in LightWaveAmplitudeLists:
        listtemp = []
        for ii in range(0, len(i), 15):
            if(LightWaveLengthList[ii]>600 and LightWaveLengthList[ii]<900):
                            listtemp.append(i[ii])
        temp=Gramme.GADFExchange(listtemp,flag=0)
        GADFMatrixListRaw.append(temp)
    GADFMatrixList=GADFMatrixListRaw.copy()
    print('GADF 成功')


def Dataget(plog):
    global Allxlist
    global Allylist
    xlist=[]
    ylist=[]
    for lt in range(len(LightWaveTimeSecondList) - 100, len(LightWaveTimeSecondList)):
        templist = [[], []]
        templist[0].extend([GASFMatrixList[lt]])
        templist[1].extend([GADFMatrixList[lt]])
        xlist.append(templist.copy())
        ylist.append(plog)

    Allxlist.append(xlist.copy())
    Allylist.append(ylist.copy())
    print('dataget 成功')




def zero():
    global LightWaveAmplitudeLists
    templen=20
    tempexlist=[0 for i in range(len(LightWaveAmplitudeLists[0]))]
    for ii in range(len(LightWaveAmplitudeLists[0])):
        tempsum=0.0
        for iii in range(templen):
            tempsum=LightWaveAmplitudeLists[iii+10][ii]+tempsum
        tempexlist[ii]=tempsum/templen
    for i in range(0,len(LightWaveAmplitudeLists)):
        for ii in range(len(LightWaveAmplitudeLists[0])):
            LightWaveAmplitudeLists[i][ii] = LightWaveAmplitudeLists[i][ii] - tempexlist[ii]
def next():
    global LightWaveAmplitudeListsRaw
    global LightWaveAmplitudeListsFilter
    global LightWaveAmplitudeLists
    global LightWaveLengthList
    global LightWaveTimeList
    global LightWaveTimeSecondList
    global lightfilecount
    global PressTimeList
    global PressAmplitudeList
    global NewPressAmplitudeList
    global PressAmplitudelogList
    global GASFMatrixListRaw
    global GADFMatrixListRaw
    global GASFMatrixList
    global GADFMatrixList
    global AllGASFMatrixList
    global AllGADFMatrixList
    global AllPressTimeToFix
    global AllLightTimeToFix
    global AllPressTimeList
    global AllPressAmplitudelogList
    global AllLightWaveTimeSecondList
    global fftabslistlist
    global fftanglelist
    AllPressTimeToFix.append(PressTimeToFix)
    AllLightTimeToFix.append(LightTimeToFix)
    AllPressTimeList.append(PressTimeList.copy())
    AllPressAmplitudelogList.append(PressAmplitudelogList.copy())
    AllLightWaveTimeSecondList.append(LightWaveTimeSecondList)
    AllGASFMatrixList.append(GASFMatrixList.copy())
    AllGADFMatrixList.append(GADFMatrixList.copy())
    LightWaveAmplitudeListsRaw = []
    LightWaveAmplitudeListsFilter = []
    LightWaveAmplitudeLists = []
    LightWaveLengthList = []
    LightWaveTimeList = []
    LightWaveTimeSecondList = []
    PressTimeList = []
    PressAmplitudeList = []
    NewPressAmplitudeList = []
    PressAmplitudelogList = []
    fftabslistlist = []
    fftanglelist = []
    GASFMatrixListRaw = []
    GADFMatrixListRaw = []
    GASFMatrixList = []
    GADFMatrixList = []
    lightfilecount = 0
    print('清除缓存')
def loaddata(lpaths,p):
    for i in range(len(lpaths)):
        Import_light_file('mydata/'+lpaths[i]+'.txt')
    plog=math.log10(p)
    zero()
    fftfuc()
    SGDfuc()
    GADFFunc()
    GASFFunc()
    Dataget(plog)
    next()
#loaddata(['P203160901_2023-03-10 17-22-18 629'],15)#1
#loaddata(['P203160901_2023-03-10 17-07-36 237'],17)#1
#loaddata(['P203160901_2023-03-10 16-54-34 757'],20)#1
#loaddata(['P203160901_2023-03-10 16-42-09 572'],22)#1
#loaddata(['P203160901_2023-03-10 16-29-23 660'],24)#1loaddata([],)#1
#loaddata(['P203160901_2023-03-10 16-15-56 962'],26)#1
#loaddata(['P203160901_2023-03-10 16-02-44 217'],28)#1
loaddata(['P203160901_2023-03-10 15-32-22 235'],30)#1
#loaddata(['P203160901_2023-03-10 14-14-36 358'],35)#1
#loaddata(['P203160901_2023-03-10 13-56-37 172'],38)#1loaddata([],)#1
#loaddata(['P203160901_2023-03-10 13-19-59 909'],40)#1
#loaddata(['P203160901_2023-03-10 13-42-50 941'],42)#1
#loaddata(['P203160901_2023-03-10 13-05-43 171'],50)#1
#loaddata(['P203160901_2023-03-10 12-32-32 768'],60)#1
#loaddata(['P203160901_2023-03-10 12-15-05 079'],70)#1
#loaddata(['P203160901_2023-03-10 11-58-22 455'],80)#1
#loaddata(['P203160901_2023-03-10 11-20-46 617'],90)#1
##loaddata(['4'],4)#1
##loaddata(['5'],5)#1
##loaddata(['6'],6)#1
##loaddata(['7'],7)#1
##loaddata(['8'],8)#1
##loaddata(['9'],9)#1
#loaddata(['10'],10)#1
##loaddata(['11'],11)#1
##loaddata(['12'],12)#1
##loaddata(['13'],13)#1
##loaddata(['14'],14)#1
##loaddata(['15'],15)#1
#loaddata(['16'],16)#1
##loaddata(['17'],17)#1
##loaddata(['18'],18)#1
##loaddata(['19'],19)#1
##loaddata(['20'],20)#1
##loaddata(['22'],22)#1
#loaddata(['24'],24)#1
##loaddata(['26'],26)#1
##loaddata(['28'],28)#1
##loaddata(['30'],30)#1
##loaddata(['33'],33)#1
##loaddata(['36'],36)#1
#loaddata(['40'],40)#1
##loaddata(['45'],45)#1
##loaddata(['50'],50)#1
##loaddata(['70'],70)#1
##loaddata(['100'],100)#1
xtrainlistraw=[]
ytrianlistraw=[]
for i in range(len(Allxlist)):
    xtrainlistraw.extend(Allxlist[i])
    ytrianlistraw.extend(Allylist[i])
cntlist=[i for i in range(len(xtrainlistraw))]
random.shuffle(cntlist)
for i in range(len(cntlist)):
    xtrainlist.append(xtrainlistraw[cntlist[i]])
    ytrianlist.append([ytrianlistraw[cntlist[i]]])
cnnmodel=torch.load('mydata/' + mod + 'net.pkl')


loss = nn.MSELoss()
x = [i for i in range(len(xtrainlist))]
acclist = []
lrcnt=0
acc = 0
for i in range(len(Allxlist)):
    ypreloglist = [
        cnnmodel(torch.tensor([Allxlist[i][ii][0]]), torch.tensor([Allxlist[i][ii][1]]))[0][0].data.item() for
        ii in range(len(Allxlist[i]))]
    yprelist = [10 ** ii for ii in ypreloglist]
    print(yprelist)
    ytruelist = [10 ** ii for ii in Allylist[i]]
    print(ytruelist)
    stdlist = [abs(ytruelist[i] - yprelist[i]) / ytruelist[i] for i in range(len(yprelist))]
    std = 0
    for ii in range(len(stdlist) // 2, len(stdlist)):
        std = std + stdlist[ii]
    acc = 1 - std / (len(stdlist) // 2) + acc
    plt.plot(yprelist)
    plt.plot(ytruelist)
    plt.show()
acc = acc / len(Allxlist)
acclist.append(acc)
print(acclist)
plt.plot(acclist)
plt.show()
