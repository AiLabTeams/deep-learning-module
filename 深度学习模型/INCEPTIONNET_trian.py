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
mod = "incnet20221125"
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
channel=4
batch=1
lr, num_epochs = 0.01 , 200
channelmut=8
refuc=nn.Sigmoid
class MyNet(nn.Module):
   def __init__(self):
       super(MyNet, self).__init__()
       self.cov1_1s = nn.Sequential(
           nn.Conv2d(1, channel, 1, 1),
       )
       self.cov3_1s = nn.Sequential(
           nn.ReflectionPad2d(1),
           nn.Conv2d(1, channel, 3, 1),
       )
       self.cov5_1s = nn.Sequential(
           nn.ReflectionPad2d(2),
           nn.Conv2d(1, channel, 5, 1),
       )
       self.cov1Max_1s = nn.Sequential(
           nn.MaxPool2d(padding=2,kernel_size=5,stride=1),
           nn.Conv2d(1, channel, 1, 1),
       )
       self.re_1s=nn.Sequential(
           nn.MaxPool2d(kernel_size=2,stride=2),
           nn.ReLU(),
       )

       self.cov1_2s = nn.Sequential(
           nn.Conv2d(channel*4, channel, 1, 1),
       )
       self.cov3_2s = nn.Sequential(
           nn.ReflectionPad2d(1),
           nn.Conv2d(channel*4, channel, 3, 1),
       )
       self.cov5_2s = nn.Sequential(
           nn.ReflectionPad2d(2),
           nn.Conv2d(channel*4, channel, 5, 1),
       )
       self.cov1Max_2s = nn.Sequential(
           nn.MaxPool2d(padding=2,kernel_size=5,stride=1),
           nn.Conv2d(channel*4, channel, 1, 1),
       )
       self.re_2s=nn.Sequential(
           nn.MaxPool2d(kernel_size=2,stride=2),
           nn.ReLU(),
       )
       self.cov1_3s = nn.Sequential(
           nn.Conv2d(channel * 4, channel, 1, 1),
       )
       self.cov3_3s = nn.Sequential(
           nn.ReflectionPad2d(1),
           nn.Conv2d(channel * 4, channel, 3, 1),
       )
       self.cov5_3s = nn.Sequential(
           nn.ReflectionPad2d(2),
           nn.Conv2d(channel * 4, channel, 5, 1),
       )
       self.cov1Max_3s = nn.Sequential(
           nn.MaxPool2d(padding=2, kernel_size=5, stride=1),
           nn.Conv2d(channel * 4, channel, 1, 1),
       )
       self.re_3s = nn.Sequential(
           nn.MaxPool2d( kernel_size=2, stride=2),
           nn.ReLU(),
       )
       self.cov1_1d = nn.Sequential(
           nn.Conv2d(1, channel, 1, 1),
       )
       self.cov3_1d = nn.Sequential(
           nn.ReflectionPad2d(1),
           nn.Conv2d(1, channel, 3, 1),
       )
       self.cov5_1d = nn.Sequential(
           nn.ReflectionPad2d(2),
           nn.Conv2d(1, channel, 5, 1),
       )
       self.cov1Max_1d = nn.Sequential(
           nn.MaxPool2d(padding=2, kernel_size=5, stride=1),
           nn.Conv2d(1, channel, 1, 1),
       )
       self.re_1d = nn.Sequential(
           nn.MaxPool2d( kernel_size=2, stride=2),
           nn.ReLU(),
       )
       self.cov1_2d = nn.Sequential(
           nn.Conv2d(channel * 4, channel, 1, 1),
       )
       self.cov3_2d = nn.Sequential(
           nn.ReflectionPad2d(1),
           nn.Conv2d(channel * 4, channel, 3, 1),
       )
       self.cov5_2d = nn.Sequential(
           nn.ReflectionPad2d(2),
           nn.Conv2d(channel * 4, channel, 5, 1),
       )
       self.cov1Max_2d = nn.Sequential(
           nn.MaxPool2d(padding=2, kernel_size=5, stride=1),
           nn.Conv2d(channel * 4, channel, 1, 1),
       )
       self.re_2d = nn.Sequential(
           nn.MaxPool2d(kernel_size=2, stride=2),
           nn.ReLU(),
       )
       self.cov1_3d = nn.Sequential(
           nn.Conv2d(channel * 4, channel, 1, 1),
       )
       self.cov3_3d = nn.Sequential(
           nn.ReflectionPad2d(1),
           nn.Conv2d(channel * 4, channel, 3, 1),
       )
       self.cov5_3d = nn.Sequential(
           nn.ReflectionPad2d(2),
           nn.Conv2d(channel * 4, channel, 5, 1),
       )
       self.cov1Max_3d = nn.Sequential(
           nn.MaxPool2d(padding=2, kernel_size=5, stride=1),
           nn.Conv2d(channel * 4, channel, 1, 1),
       )
       self.re_3d = nn.Sequential(
           nn.MaxPool2d( kernel_size=2, stride=2),
           nn.ReLU(),
       )
       self.covf=nn.Sequential(nn.Conv2d(channel*2*4, 1,5,1),)
   def forward(self, imgs,imgd):
       in1layers1 = self.cov1_1s(imgs)
       in2layers1 = self.cov3_1s(imgs)
       in3layers1 = self.cov5_1s(imgs)
       in4layers1 = self.cov1Max_1s(imgs)
       inlayers1 = self.re_1s(torch.cat((in1layers1, in2layers1, in3layers1, in4layers1), 1))
       in1layers2 = self.cov1_2s(inlayers1)
       in2layers2 = self.cov3_2s(inlayers1)
       in3layers2 = self.cov5_2s(inlayers1)
       in4layers2 = self.cov1Max_2s(inlayers1)
       inlayers2 = self.re_2s(torch.cat((in1layers2, in2layers2, in3layers2, in4layers2), 1))
       in1layers3 = self.cov1_3s(inlayers2)
       in2layers3 = self.cov3_3s(inlayers2)
       in3layers3 = self.cov5_3s(inlayers2)
       in4layers3 = self.cov1Max_3s(inlayers2)
       inlayers3 = self.re_3s(torch.cat((in1layers3, in2layers3, in3layers3, in4layers3), 1))

       in1layerd1 = self.cov1_1d(imgd)
       in2layerd1 = self.cov3_1d(imgd)
       in3layerd1 = self.cov5_1d(imgd)
       in4layerd1 = self.cov1Max_1d(imgd)
       inlayerd1 = self.re_1d(torch.cat((in1layerd1, in2layerd1, in3layerd1, in4layerd1), 1))
       in1layerd2 = self.cov1_2d(inlayerd1)
       in2layerd2 = self.cov3_2d(inlayerd1)
       in3layerd2 = self.cov5_2d(inlayerd1)
       in4layerd2 = self.cov1Max_2d(inlayerd1)
       inlayerd2 = self.re_2d(torch.cat((in1layerd2, in2layerd2, in3layerd2, in4layerd2), 1))
       in1layerd3 = self.cov1_3d(inlayerd2)
       in2layerd3 = self.cov3_3d(inlayerd2)
       in3layerd3 = self.cov5_3d(inlayerd2)
       in4layerd3 = self.cov1Max_3d(inlayerd2)
       inlayerd3 = self.re_3d(torch.cat((in1layerd3, in2layerd3, in3layerd3, in4layerd3), 1))

       output = self.covf(torch.cat((inlayers3,inlayerd3),1))
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


def Dataget():
    global Allxlist
    global Allylist
    xlist=[]
    ylist=[]
    if abs(PressTimeToFix - LightTimeToFix) > 36000:
        temp = PressTimeToFix - LightTimeToFix + 12 * 60 * 60
    else:
        temp = PressTimeToFix - LightTimeToFix

    for pt in range(timedis*6//10+50,timedis+50):
        for lt in range(5, len(LightWaveTimeSecondList)):
            if LightWaveTimeSecondList[lt] >= PressTimeList[pt] + temp:
                templist = [[], []]
                # templist.extend(fftabslistlist[lt].copy())
                # templist.extend(fftanglelist[lt].copy())
                # templist.extend(LightWaveAmplitudeLists[lt][::15])
                templist[0].extend([GASFMatrixList[lt]])
                templist[1].extend([GADFMatrixList[lt]])
                xlist.append(templist.copy())
                ylist.append(PressAmplitudelogList[pt])
                break
    for pt in range(len(PressTimeList)-timedis, len(PressTimeList)):
        for lt in range(5, len(LightWaveTimeSecondList)):
            if LightWaveTimeSecondList[lt] >= PressTimeList[pt] + temp:
                templist = [[], []]
                # templist.extend(fftabslistlist[lt].copy())
                # templist.extend(fftanglelist[lt].copy())
                # templist.extend(LightWaveAmplitudeLists[lt][::15])
                templist[0].extend([GASFMatrixList[lt]])
                templist[1].extend([GADFMatrixList[lt]])
                xlist.append(templist.copy())
                ylist.append(PressAmplitudelogList[pt])
                break
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
def loaddata(lpaths,ppath):
    for i in range(len(lpaths)):
        Import_light_file('mydata/'+lpaths[i]+'.txt')
    Import_press_file('mydata/'+ppath+'.csv')
    zero()
    fftfuc()
    SGDfuc()
    GADFFunc()
    GASFFunc()
    Dataget()
    next()

loaddata(['P203160901_2022-11-10 15-21-02 904','P203160901_2022-11-10 15-29-38 926'],'20221109-2432pd10')#1
loaddata(['P203160901_2022-11-10 15-43-34 744','P203160901_2022-11-10 15-52-10 297'],'20221109-2661pd10')#2
loaddata(['P203160901_2022-11-09 20-51-08 739','P203160901_2022-11-09 20-59-46 282'],'20221109-2847pd10')#3
loaddata(['P203160901_2022-11-09 16-40-45 813','P203160901_2022-11-09 16-49-22 527'],'20221109-3047pd10')#4
loaddata(['P203160901_2022-11-10 21-49-48 028','P203160901_2022-11-10 21-58-25 088'],'20221109-3152pd10')#5
loaddata(['P203160901_2022-11-09 16-20-32 054','P203160901_2022-11-09 16-29-08 129'],'20221109-3201pd10')#1
loaddata(['P203160901_2022-11-11 16-01-06 530','P203160901_2022-11-11 16-09-42 663'],'20221109-3226pd10')#2
loaddata(['P203160901_2022-11-10 15-00-01 436','P203160901_2022-11-10 15-08-39 531'],'20221109-3273pd10')#3
loaddata(['P203160901_2022-11-10 13-29-35 324','P203160901_2022-11-10 13-38-11 975'],'20221109-3493pd10')#4
loaddata(['P203160901_2022-11-10 16-03-56 353','P203160901_2022-11-10 16-12-32 459'],'20221109-3685pd10')#5
#loaddata(['P203160901_2022-11-09 19-46-56 710','P203160901_2022-11-09 19-55-32 268'],'20221109-3794pd10')#1
loaddata(['P203160901_2022-11-11 14-14-44 805','P203160901_2022-11-11 14-23-20 424'],'20221109-3833pd10')#2
loaddata(['P203160901_2022-11-11 15-38-49 488','P203160901_2022-11-11 15-47-25 551'],'20221109-4218pd10')#3

#loaddata(['P203160901_2022-11-11 15-38-49 488','P203160901_2022-11-11 15-47-25 551'],'20221109-4218pd10')#3
#loaddata(['P203160901_2022-11-10 14-17-28 710','P203160901_2022-11-10 14-26-04 251'],'20221109-4395pd10')#4
loaddata(['P203160901_2022-11-09 21-12-55 253','P203160901_2022-11-09 21-21-31 775'],'20221109-4577pd10')#5
loaddata(['P203160901_2022-11-10 22-29-18 171','P203160901_2022-11-10 22-37-55 972'],'20221109-4595pd10')#1
loaddata(['P203160901_2022-11-10 17-48-11 735','P203160901_2022-11-10 17-56-47 297'],'20221109-4883pd10')#2
loaddata(['P203160901_2022-11-10 17-25-41 128','P203160901_2022-11-10 17-34-17 216'],'20221109-5112pd10')#3
loaddata(['P203160901_2022-11-10 17-04-44 910','P203160901_2022-11-10 17-13-20 460'],'20221109-5687pd10')#4
loaddata(['P203160901_2022-11-10 13-54-23 120','P203160901_2022-11-10 14-02-59 673'],'20221109-5921pd10')#5
loaddata(['P203160901_2022-11-10 14-38-41 030','P203160901_2022-11-10 14-47-17 096'],'20221109-5959pd10')#1


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
cnnmodel=MyNet()

print('开始训练')
loss = nn.MSELoss()
lrcnt = 0
x = [i for i in range(len(xtrainlist))]
losslist=[]
acclist = []
for epoch in range(1, num_epochs + 1):
    random.shuffle(x)
    lsum = 0
    for i in range(int(len(x)//batch)) :
        nl1,nl2,np=torch.tensor([xtrainlist[x[i*batch+j]][0] for j in range(batch)]),torch.tensor([xtrainlist[x[i*batch+j]][1] for j in range(batch)]),\
                   torch.tensor([ytrianlist[x[i*batch+j]] for j in range(batch)]),
        #r = lr / (10 + lrcnt)
        r = lr / (10 + 1.2*epoch)
        optimizer = torch.optim.Adam(cnnmodel.parameters(), lr=r)
        y_hat= cnnmodel(nl1,nl2)
        l = loss(y_hat, np).mean()
        optimizer.zero_grad()
        l.backward()
        lsum = lsum + l.data.item()
        optimizer.step()
        if i % 100 == 0:
            print("epoch:{},loss:{}".format(epoch, l))
    losslist.append(lsum)
    acc = 0
    for i in range(len(Allxlist)):
        ypreloglist = [
            cnnmodel(torch.tensor([Allxlist[i][ii][0]]), torch.tensor([Allxlist[i][ii][1]]))[0][0].data.item() for
            ii in range(len(Allxlist[i]))]
        yprelist = [10 ** ii for ii in ypreloglist]
        ytruelist = [10 ** ii for ii in Allylist[i]]
        stdlist = [abs(ytruelist[i] - yprelist[i]) / ytruelist[i] for i in range(len(yprelist))]
        std = 0
        for ii in range(len(stdlist) // 2, len(stdlist)):
            std = std + stdlist[ii]
        acc = 1 - std / (len(stdlist) // 2) + acc
    acc = acc / len(Allxlist)
    acclist.append(acc)
print('训练完成')
net = cnnmodel.to(torch.device('cpu'))
torch.save(net, 'mydata/' + mod + 'net.pkl')
print(losslist)
print(acclist)
plt.plot(losslist)
plt.show()