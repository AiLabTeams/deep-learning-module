
##文件基于Pyside2框架编写，可以初步分析原始数据
##编写之初没有做没有考虑后续拓展，越写越乱，实力允许建议重构


from PySide2.QtWidgets import QApplication, QMdiSubWindow
from PySide2.QtUiTools import QUiLoader
import torch
import csv
import time
import seaborn
import matplotlib.pyplot as plt
import Gramme
import pyqtgraph
import tkinter
from tkinter import filedialog
import PressPackage
import WavePackage
import KsPackage
import MyThread
from PySide2.QtCore import QTimer
import math
import os
import MyTorch
from scipy.signal import savgol_filter
import MyTorch2
import MyTorch4
import numpy as np
dir="D:/LightProgress/mydata/"#设置处理后数据文件存放地



class MainWinForm:
    def __init__(self):
        # 从文件中加载UI定义
        # 从 UI 定义中动态 创建一个相应的窗口对象
        # 注意：里面的控件对象也成为窗口对象的属性了
        self.ui = QUiLoader().load('main.ui')
        self.ui.BunStart.clicked.connect(self.BunStartclickedhandle)
        self.ui.BunClear.clicked.connect(self.BunClearclickedhandle)
        self.ui.BunClearAll.clicked.connect(self.BunClearAllclickedhandle)
        self.ui.ChxLight.clicked.connect(self.ChxLightclickedhandle)
        self.ui.ChxLightTime.clicked.connect(self.ChxLightTimeclickedhandle)
        self.ui.ChxPressTime.clicked.connect(self.ChxPressTimeclickedhandle)
        self.ui.RanLightInterval.clicked.connect(self.RanLightIntervalclickedhandle)
        self.ui.RanLightDot.clicked.connect(self.RanLightDotclickedhandle)
        self.ui.RanLightDivision.clicked.connect(self.RanLightDivisionclickedhandle)
        self.ui.LitTimeSetDot.textChanged.connect(self.LitTimeSetDotchangedhandle)
        self.ui.LitTimeSetInterval.textChanged.connect(self.LitTimeSetIntervalchangedhandle)
        self.ui.LitTimeSetNumber.textChanged.connect(self.LitTimeSetNumberchangedhandle)
        self.ui.LitWaveLengthSetRange.textChanged.connect(self.LitWaveLengthSetRangechangedhandle)
        self.ui.LitWaveLengthSetDot.textChanged.connect(self.LitWaveLengthSetDotchangedhandle)
        self.ui.ChxLightAutoSave.clicked.connect(self.ChxLightAutoSaveclickedhandle)
        self.ui.ChxPeakAmplitude.clicked.connect(self.ChxPeakAmplitudeclickedhandle)
        self.ui.ChxPeakPosition.clicked.connect(self.ChxPeakPositionclickedhandle)
        self.ui.ChxWaveLength.clicked.connect(self.ChxWaveLengthclickedhandle)
        self.ui.ChxFilter.clicked.connect(self.ChxFilterclickedhandle)
        self.ui.RanMax.clicked.connect(self.RanMaxclickedhandle)
        self.ui.RanMin.clicked.connect(self.RanMinclickedhandle)
        self.ui.actionImport_light_file.triggered.connect(self.actionImport_light_filetriggeredhandle)
        self.ui.actionImport_press_file.triggered.connect(self.actionImport_press_filetriggeredhandle)
        self.ui.actionImport_GAF_file.triggered.connect(self.actionImport_GAF_filetriggeredhandle)
        self.ui.actionSave_path.triggered.connect(self.actionSave_pathtriggeredhandle)
        self.ui.actionOpen_file.triggered.connect(self.actionOpen_filetriggeredhandle)
        self.ui.ChxLightPress.clicked.connect(self.ChxLightPressclickedhandle)
        self.ui.LitPressTare.textChanged.connect(self.LitPressTarechangedhandle)
        self.ui.LitPressMove.textChanged.connect(self.LitPressMovechangedhandle)
        self.ui.LitGrammeMin.textChanged.connect(self.LitGrammeMinchangedhandle)
        self.ui.LitGrammeMax.textChanged.connect(self.LitGrammeMaxchangedhandle)
        self.ui.LitGrammeInterval.textChanged.connect(self.LitGrammeIntervalchangedhandle)
        self.ui.BunGADF.clicked.connect(self.BunGADFclickedhandle)
        self.ui.BunGASF.clicked.connect(self.BunGASFclickedhandle)
        self.ui.BunShowGASFR2.clicked.connect(self.BunShowGASFR2clickedhandle)
        self.ui.BunShowGADFR2.clicked.connect(self.BunShowGADFR2clickedhandle)
        self.ui.BunShowGASFR2_Press.clicked.connect(self.BunShowGASFR2_Pressclickedhandle)
        self.ui.BunShowGADFR2_Press.clicked.connect(self.BunShowGADFR2_Pressclickedhandle)
        self.ui.BunMatrixSplice.clicked.connect(self.BunMatrixSpliceclickedhandle)
        self.ui.BunTrainGAF.clicked.connect(self.BunTrainGAFclickedhandle)
        self.ui.BunTrainGASF.clicked.connect(self.BunTrainGASFclickedhandle)
        self.ui.BunTrainGADF.clicked.connect(self.BunTrainGADFclickedhandle)
        self.ui.BunTest.clicked.connect(self.BunTestclickedhandle)
        self.ui.BunTrianAll.clicked.connect(self.BunTrianAllclickedhandle)
        self.ui.BunDataget.clicked.connect(self.BunDatagetclickedhandle)
        #数据缓存，可添加，勿修改
        self.pressfilecount=0
        self.lightfilecount=0
        self.TimeDotData = []
        self.TimeIntervalData = 0
        self.TimeNumberData = 0
        self.WavelengthRangeData = []
        self.WavelengthDotData = 0
        self.LightTestList = []
        self.LightTimeTestList = []
        self.PressTimeTestList = []
        self.Maxflag = True
        self.PeakAmplitudeflag = False
        self.PeakPositionflag = False
        self.LightWaveAmplitudeListsRaw =[]
        self.LightWaveAmplitudeListsFilter = []
        self.LightWaveAmplitudeLists = []
        self.LightWaveLengthList =[]
        self.LightWaveTimeList =[]
        self.LightWaveTimeSecondList =[]
        self.PressTimeList =[]
        self.PressAmplitudeList =[]
        self.NewPressAmplitudeList = []
        self.filtercount = 0
        self.startflag=True
        self.GASFMatrixListRaw = []
        self.GADFMatrixListRaw = []
        self.GASFMatrixList = []
        self.GADFMatrixList = []
        self.GAFMatrixList = []
        self.GASFR2List = []
        self.GADFR2List = []
        self.GAFMin=600
        self.GAFMax=900
        self.GAFInterval=15
        self.MatrixNumber=5
        self.trianflag=0
        self.Presshatlist1 = []
        self.Presshatlist2 = []
        self.showtemp = 10 / 300 * 100
        self.showthr = self.showtemp
        self.stdlist=[]
        self.SetCheck(self.ui.RanMax)

        self.SetHint(self.ui.LitTimeSetDot, '如：12,22,32,...')
        self.SetHint(self.ui.LitTimeSetInterval, '如：20')
        self.SetHint(self.ui.LitTimeSetNumber, '如：7')
        self.SetHint(self.ui.LitWaveLengthSetRange, '如：400,900')
        self.SetHint(self.ui.LitWaveLengthSetDot, '如：700')
        self.SetHint(self.ui.LitLightPressStart, '如：0')
        self.SetHint(self.ui.LitLightPressEnd, '如：1000')
        self.SetDisable(self.ui.LitTimeSetDot)
        self.SetDisable(self.ui.LitTimeSetInterval)
        self.SetDisable(self.ui.LitTimeSetNumber)
        self.SetDisable(self.ui.LitWaveLengthSetRange)
        self.SetDisable(self.ui.LitWaveLengthSetDot)
        self.SetDisable(self.ui.LitLightPressStart)
        self.SetDisable(self.ui.LitLightPressEnd)
        self.ui.Ter.append('波长范围450-1199时，结果较为准确\n所有标点符号均使用英文符号')
        self.ChooseTabCurrentIndex(self.ui.Tat, 0)


    def fun_timer(self):
        self.SetDisable(self.ui.BunStart)
        self.ui.Ter.append('已完成'+str(self.workcount)+'%')
        if self.workcount==100:
            self.SetEnable(self.ui.BunStart)
            try:
                if self.ReadCheck(self.ui.ChxPeakAmplitude) == True:
                    self.MDIDraw(self.LightWaveTimeSecondList, self.peaklisty, '时间(s)', '透射率(%)',
                                 '时间-峰值幅度 开始时间：'+
                         self.LightTimeStart+'   结束时间：'+self.LightTimeEnd,flag=self.ReadCheck(self.ui.ChxLightPressTime))
                    with open(dir + str(time.strftime("%Y%m%d%H%M%S", time.localtime())) + "spectrum_time.csv", "a",
                              newline='') as out:
                        csv_writer = csv.writer(out, dialect="excel")
                        csv_writer.writerow(
                            ['测试时间:', str(time.strftime("%Y%m%d %H:%M:%S", time.localtime()))])
                        csv_writer.writerow(['时间(s)', '透射率(%)'])
                        for i in range(len(self.LightWaveTimeSecondList)):
                            csv_writer.writerow([self.LightWaveTimeSecondList[i], self.peaklisty[i]])
                if self.ReadCheck(self.ui.ChxPeakPosition) == True:
                    self.MDIDraw(self.LightWaveTimeSecondList, self.peaklistx, '时间(s)', '波长(nm)',
                                 '时间-峰值位移 开始时间：'+
                         self.LightTimeStart+'   结束时间：'+self.LightTimeEnd,flag=self.ReadCheck(self.ui.ChxLightPressTime))
                    with open(dir + str(time.strftime("%Y%m%d%H%M%S", time.localtime())) + "spectrum_time.csv", "a",
                              newline='') as out:
                        csv_writer = csv.writer(out, dialect="excel")
                        csv_writer.writerow(
                            ['测试时间:', str(time.strftime("%Y%m%d %H:%M:%S", time.localtime()))])
                        csv_writer.writerow(['时间(s)', '波长(nm)'])
                        for i in range(len(self.LightWaveTimeSecondList)):
                            csv_writer.writerow([self.LightWaveTimeSecondList[i], self.peaklistx[i]])
                self.ui.Ter.append('光强时序测试成功')
                if self.ReadCheck(self.ui.ChxLightPress) == True:
                    if self.ReadCheck(self.ui.ChxPeakPosition) == True:
                        self.SetLightPressFunc()
                        self.TestLightPressFun(self.peaklistx)
                    elif self.ReadCheck(self.ui.ChxPeakAmplitude) == True:
                        self.SetLightPressFunc()
                        self.TestLightPressFun(self.peaklisty)
            except:
                self.ui.Ter.append('光强时序测试失败')

            self.timer.stop()



    def BunStartclickedhandle(self):
            self.ui.Ter.append('执行任务')
            self.startflag = False
            if self.ReadCheck(self.ui.ChxLightPressTime) == True:
                self.ui.Ter.append('该操作耗时较长，执行中')
            self.RunFuncList(self.LightTestList, self.ReadCheck(self.ui.ChxLight))
            self.RunFuncList(self.LightTimeTestList, self.ReadCheck(self.ui.ChxLightTime))
            self.RunFuncList(self.PressTimeTestList, self.ReadCheck(self.ui.ChxPressTime))
            self.startflag = True


    def BunClearclickedhandle(self):
        self.ClearMDIArea()
        self.ui.Ter.append('清空窗口')
        pass

    def BunClearAllclickedhandle(self):
        self.LightTimeStart = 0
        self.LightTimeEnd = 0
        self.LightTimeToFix = 0
        self.LightWaveAmplitudeListsRaw = []
        self.LightWaveLengthList = []
        self.LightWaveTimeList = []
        self.LightWaveTimeSecondList = []
        self.lightfilecount = 0
        self.PressTimeStart = 0
        self.PressTimeEnd = 0
        self.PressTimeToFix = 0
        self.PressTimeList = []
        self.PressAmplitudeList = []
        self.pressfilecount = 0
        self.filtercount = 0
        self.ui.Ter.append('清空所有数据')
    def ChxFilterclickedhandle(self):
        
        def filterfunc():
            try:
                self.LightWaveAmplitudeListsFilter=[]
                self.LightWaveAmplitudeListsnewFilter = self.LightWaveAmplitudeListsRaw
                for i in range(12,len(self.LightWaveAmplitudeListsRaw)-12):
                    self.filtercount = i / (len(self.LightWaveAmplitudeListsnewFilter)-1) * 100
                    for ii in range(len(self.LightWaveAmplitudeListsRaw[0])):
                        LightWaveAmplitudeListtemp = []
                        for iii in range(i-11, i+12):
                            LightWaveAmplitudeListtemp.append(self.LightWaveAmplitudeListsRaw[iii][ii])
                        LightWaveAmplitudeListtemp=savgol_filter(LightWaveAmplitudeListtemp,19,0)
                        self.LightWaveAmplitudeListsnewFilter[i][ii]=LightWaveAmplitudeListtemp[11]
                for i in range(len(self.LightWaveAmplitudeListsnewFilter)):
                   # temp = np.array(self.LightWaveAmplitudeListsRaw[i])
                    self.LightWaveAmplitudeListFilter=self.LightWaveAmplitudeListsnewFilter[i]
                    self.LightWaveAmplitudeListFilter=savgol_filter(np.array(self.LightWaveAmplitudeListsnewFilter[i]),31,0)
                    self.filtercount = i / (len(self.LightWaveAmplitudeListsnewFilter) - 1) * 100
                    self.LightWaveAmplitudeListsFilter.append(self.LightWaveAmplitudeListFilter.tolist())

                self.LightWaveAmplitudeLists = self.LightWaveAmplitudeListsFilter

            except:
                self.ui.Ter.append('滤波失败')

        def fun_timer3():
            self.SetDisable(self.ui.BunStart)
            self.ui.Ter.append('滤波已完成' + str(int(self.filtercount)) + '%')
            if self.filtercount >= 99:
                self.ui.Ter.append('滤波成功')
                self.SetEnable(self.ui.BunStart)
                timer3.stop()
                self.filtercount =0
        if self.ReadCheck(self.ui.ChxFilter)==True and len(self.LightWaveAmplitudeListsRaw)!=0:
            if self.filtercount==0:
                self.ui.Ter.append('开始滤波')
                task = MyThread.MyThread(filterfunc, ())
                task.start()
                timer3 = QTimer()
                timer3.timeout.connect(fun_timer3)
                timer3.start(2100)
            else:
                self.LightWaveAmplitudeLists = self.LightWaveAmplitudeListsFilter
        else:
            self.LightWaveAmplitudeLists=self.LightWaveAmplitudeListsRaw


    def ChxLightclickedhandle(self):
        self.ChooseTabCurrentIndex(self.ui.Tat, 0)
        if self.ui.ChxLight.isChecked() == True:
            pass
        else:
            pass
        pass
    def ChxLightTimeclickedhandle(self):
        self.ChooseTabCurrentIndex(self.ui.Tat, 1)
        if self.ui.ChxLightTime.isChecked() == True:
            pass
        else:
            pass
        pass
    def ChxPressTimeclickedhandle(self):
        self.ChooseTabCurrentIndex(self.ui.Tat, 2)
        if self.ui.ChxPressTime.isChecked() == True:
            if len(self.PressTimeTestList) == 0:
                self.PressTimeTestList.append(self.TestPressFunc)
            pass
        else:
            pass
        pass

    def ChxLightPressclickedhandle(self):
        if self.ReadCheck(self.ui.ChxLightPress)==True:
            self.SetEnable(self.ui.LitLightPressStart)
            self.SetEnable(self.ui.LitLightPressEnd)
        else:
            self.SetDisable(self.ui.LitLightPressStart)
            self.SetDisable(self.ui.LitLightPressEnd)
        pass
    def RanLightDotclickedhandle(self):
        if self.ReadCheck(self.ui.RanLightDot) == True:
            self.SetDisable(self.ui.LitTimeSetInterval)
            self.SetEnable(self.ui.LitTimeSetDot)
            self.SetDisable(self.ui.LitTimeSetNumber)
            self.LightTestList.clear()
            self.LightTestList.append(self.SetTimeDotFunc)
            self.LightTestList.append(self.TestTimeDotFunc)
        else:
            self.SetDisable(self.ui.LitTimeSetInterval)
            self.SetDisable(self.ui.LitTimeSetDot)
            self.SetDisable(self.ui.LitTimeSetNumber)
        pass

    def RanLightIntervalclickedhandle(self):
        if self.ReadCheck(self.ui.RanLightInterval) == True:
            self.SetEnable(self.ui.LitTimeSetInterval)
            self.SetDisable(self.ui.LitTimeSetDot)
            self.SetDisable(self.ui.LitTimeSetNumber)
            self.LightTestList.clear()
            self.LightTestList.append(self.SetTimeIntervalFunc)
            self.LightTestList.append(self.TestTimeIntervalFunc)

        else:
            self.SetDisable(self.ui.LitTimeSetInterval)
            self.SetDisable(self.ui.LitTimeSetDot)
            self.SetDisable(self.ui.LitTimeSetNumber)

        pass

    def RanLightDivisionclickedhandle(self):
        if self.ReadCheck(self.ui.RanLightDivision) == True:
            self.SetDisable(self.ui.LitTimeSetInterval)
            self.SetDisable(self.ui.LitTimeSetDot)
            self.SetEnable(self.ui.LitTimeSetNumber)
            self.LightTestList.clear()
            self.LightTestList.append(self.SetTimeNumberFunc)
            self.LightTestList.append(self.TestTimeNumberFunc)
        else:
            self.SetDisable(self.ui.LitTimeSetInterval)
            self.SetDisable(self.ui.LitTimeSetDot)
            self.SetDisable(self.ui.LitTimeSetNumber)
        pass
    def ChxLightAutoSaveclickedhandle(self):
        pass
    def ChxPeakAmplitudeclickedhandle(self):
        if self.ReadCheck(self.ui.ChxPeakAmplitude) == True or self.ReadCheck(self.ui.ChxPeakPosition) == True:
            self.SetEnable(self.ui.LitWaveLengthSetRange)
            self.PeakAmplitudeflag = True
            if self.TestWavelengthPeakFunc not in self.LightTimeTestList:
                self.LightTimeTestList.append(self.SetWavelengthRangeFunc)
                self.LightTimeTestList.append(self.TestWavelengthPeakFunc)
        else:
            self.SetDisable(self.ui.LitWaveLengthSetRange)
            self.PeakAmplitudeflag = False
            if self.TestWavelengthPeakFunc in self.LightTimeTestList:
                self.LightTimeTestList.remove(self.SetWavelengthRangeFunc)
                self.LightTimeTestList.remove(self.TestWavelengthPeakFunc)
        pass
    def ChxPeakPositionclickedhandle(self):
        if self.ReadCheck(self.ui.ChxPeakPosition) == True or self.ReadCheck(self.ui.ChxPeakAmplitude) == True:
            self.SetEnable(self.ui.LitWaveLengthSetRange)
            self.PeakPositionflag = True
            if self.TestWavelengthPeakFunc not in self.LightTimeTestList:
                self.LightTimeTestList.append(self.SetWavelengthRangeFunc)
                self.LightTimeTestList.append(self.TestWavelengthPeakFunc)
        else:
            self.SetDisable(self.ui.LitWaveLengthSetRange)
            self.PeakPositionflag = False
            if self.TestWavelengthPeakFunc in self.LightTimeTestList:
                self.LightTimeTestList.remove(self.SetWavelengthRangeFunc)
                self.LightTimeTestList.remove(self.TestWavelengthPeakFunc)
        pass
    def ChxWaveLengthclickedhandle(self):
        if self.ReadCheck(self.ui.ChxWaveLength) == True:
            self.SetEnable(self.ui.LitWaveLengthSetDot)
            if self.TestWavelengthDotFunc not in self.LightTimeTestList:
                self.LightTimeTestList.append(self.SetWavelengthDotFunc)
                self.LightTimeTestList.append(self.TestWavelengthDotFunc)
        else:
            self.SetDisable(self.ui.LitWaveLengthSetDot)
            if self.TestWavelengthDotFunc in self.LightTimeTestList:
                self.LightTimeTestList.remove(self.SetWavelengthDotFunc)
                self.LightTimeTestList.remove(self.TestWavelengthDotFunc)
        pass
    def RanMaxclickedhandle(self):
        if self.ReadCheck(self.ui.RanMax) == True:
            self.Maxflag = True
        else:
            self.Maxflag = False

        pass
    def RanMinclickedhandle(self):
        if self.ReadCheck(self.ui.RanMin) == True:
            self.Maxflag = False
        else:
            self.Maxflag = True
        pass
    def LitTimeSetDotchangedhandle(self):
        pass
    def LitTimeSetIntervalchangedhandle(self):
        pass
    def LitTimeSetNumberchangedhandle(self):
        pass
    def LitWaveLengthSetRangechangedhandle(self):
        pass
    def LitWaveLengthSetDotchangedhandle(self):
        pass

    def LitPressTarechangedhandle(self):

        try:
            self.PressTare=float(self.ReadLit(self.ui.LitPressTare))
            for i in range(len(self.NewPressAmplitudeList)):
                self.NewPressAmplitudeList[i]=self.NewPressAmplitudeList[i]-self.PressTare
            self.PressAmplitudelogList = [math.log(max(1,i), 10) for i in self.NewPressAmplitudeList]

        except:
            self.PressTare=0
            self.NewPressAmplitudeList=self.PressAmplitudeList

    def LitPressMovechangedhandle(self):
        self.NewPressAmplitudeList=[]
        try:
            self.PressMove = float(self.ReadLit(self.ui.LitPressMove))
            temp = 0
            for i in range(len(self.PressAmplitudeList)):
                temp = temp + self.PressMove
                if i > 5 and self.PressAmplitudeList[i] - self.PressAmplitudeList[i - 5] > 10:
                    temp = temp - self.PressMove
                if i > 5 and self.PressAmplitudeList[i] - self.PressAmplitudeList[i - 5] < -10:
                    temp = 0
                self.NewPressAmplitudeList.append(max(self.PressAmplitudeList[i] - temp, 1))
            self.NewPressAmplitudeList=savgol_filter(self.NewPressAmplitudeList,101,1)
            self.PressAmplitudelogList = [math.log(max(1, i), 10) for i in self.NewPressAmplitudeList]

        except:
            pass
    def actionImport_light_filetriggeredhandle(self):
        try:
            root = tkinter.Tk()
            root.withdraw()
            fpath = filedialog.askopenfilename()
            t = WavePackage.Wave(fpath)
            if self.lightfilecount == 0:
                self.LightTimeStart = t.TimeStart
                self.LightTimeToFix = t.IntWaveTimeList[0] % 100000000000/1000
                self.LightWaveLengthList = t.FloatWaveLengthList
                self.LightWaveAmplitudeListsRaw += t.FloatWaveAmplitudeLists

                self.LightWaveTimeList += t.IntWaveTimeList
                self.LightWaveTimeSecondList += t.FloatWaveTimeSecondList
                self.lightfilecount = self.lightfilecount + 1
                self.LightTimeEnd = t.TimeEnd
            else:
                self.LightWaveAmplitudeListsRaw += t.FloatWaveAmplitudeLists
                self.LightWaveTimeList += t.IntWaveTimeList
                temp=len(self.LightWaveTimeSecondList)
                self.LightWaveTimeSecondList += [i+self.LightWaveTimeSecondList[1]-self.LightWaveTimeSecondList[0]+
                                                 self.LightWaveTimeSecondList[temp-1]
                                                 for i in t.FloatWaveTimeSecondList]
                self.lightfilecount = self.lightfilecount + 1
                self.LightTimeEnd = t.TimeEnd
            self.LightWaveAmplitudeLists = self.LightWaveAmplitudeListsRaw
            self.ui.Ter.append('光谱数据读入成功')
            self.ui.Ter.append('当前载入光谱文件数量: '+str(self.lightfilecount))
        except:
            self.ui.Ter.append('光谱数据读入失败')
        pass

    def actionImport_press_filetriggeredhandle(self):
        try:
            root = tkinter.Tk()
            root.withdraw()
            fpath = filedialog.askopenfilename()
            t = PressPackage.Press(fpath)
            self.PressTimeStart = t.TimeStart
            self.PressTimeToFix = t.IntPressTimeList[0] % 100000000000 / 1000
            self.PressTimeList = t.IntPressTimeSecondList

            self.PressAmplitudeList = t.IntPressDataList

            self.NewPressAmplitudeList = savgol_filter(self.PressAmplitudeList,31,1)
            m=min(self.NewPressAmplitudeList)-0.0001
            for i in range(len(self.NewPressAmplitudeList)):
                m=m+0.0105
                self.NewPressAmplitudeList[i]=self.NewPressAmplitudeList[i]-m
            self.PressUnit = t.StrPressunit
            self.PressTimeEnd = t.TimeEnd
            self.ui.LalPressTare.setText('气压去偏移（单位：' + self.PressUnit + '）')
            self.PressAmplitudelogList = [math.log(max(i,1), 10) for i in self.NewPressAmplitudeList]
            self.ui.Ter.append('气压数据读入成功')
        except:
            self.ui.Ter.append('气压数据读入失败')


        pass

    def actionImport_GAF_filetriggeredhandle(self):
        try:
            root = tkinter.Tk()
            root.withdraw()
            fpath = filedialog.askopenfilename()
            self.ui.Ter.append('GAF数据读入成功')
        except:
            self.ui.Ter.append('GAF数据读入失败')


        pass
    def actionSave_pathtriggeredhandle(self):
        pass
    def actionOpen_filetriggeredhandle(self):
        pass


    def ReadLit(self,Lit):
        return Lit.text()
        pass
    def ReadCheck(self,CheckBox):
        return CheckBox.isChecked()
        pass
    def SetEnable(self,Lit):
        Lit.setEnabled(True)
    def SetDisable(self, Lit):
        Lit.setDisabled(True)
        pass
    def SetCheck(self,CheckBox):
        CheckBox.setChecked(True)
    def SetHint(self,Lit,str):
        Lit.setPlaceholderText(str)
    def ChooseTabCurrentIndex(self, TabWidet,x):
        TabWidet.setCurrentIndex(x)
        pass
    def SetLightPressFunc(self):
        try:
            self.LightPressStart = int(self.ReadLit(self.ui.LitLightPressStart))
            self.LightPressEnd = int(self.ReadLit(self.ui.LitLightPressEnd))
            self.ui.Ter.append('设置光谱气压参数' + '成功')
        except:
            self.ui.Ter.append('设置光谱气压参数' + '错误')
        pass

    def TestLightPressFun(self,list):
        def plotfunc(x_start,x_end):
            self.LightPresscount = x_start
            for i in range(x_start,x_end):
                for ii in range(len(self.LightWaveTimeSecondList)):
                    if (i < self.LightWaveTimeSecondList[ii]):
                        self.LightPressL.append(list[ii])
                        self.LightPresscount = ii / (len(self.LightWaveTimeSecondList)-1) * 100
                        for iii in range(0, len(self.PressTimeList), 1):
                            if (i < self.PressTimeList[ii] + self.PressTimeToFix - self.LightTimeToFix):
                                self.LightPressP.append(math.log(self.NewPressAmplitudeList[iii], 10))
                                break
                        for iii in range(0, len(self.PressTimeList), 1):
                            if (i < self.PressTimeList[
                                iii] + self.PressTimeToFix - self.LightTimeToFix + 12 * 60 * 60) and (
                                    i > self.PressTimeList[
                                iii] + self.PressTimeToFix - self.LightTimeToFix + 6 * 60 * 60):
                                self.LightPressP.append(math.log(self.NewPressAmplitudeList[iii], 10))
                                break
                        break
            self.LightPresscount=100
            pass


        def fun_timer2():
            self.SetDisable(self.ui.BunStart)
            self.ui.Ter.append('光谱气压作图已完成'+str(int(self.LightPresscount))+'%')
            if self.LightPresscount ==100:
                self.SetEnable(self.ui.BunStart)
                try:
                    self.MDIDraw(self.LightPressL, self.LightPressP, '光谱(nm/%)','气压(lg('+self.PressUnit+'))',
                                     '光谱-气压')
                    self.ui.Ter.append('光谱气压作图成功')

                except:
                    self.ui.Ter.append('光谱气压作图失败,请重新设定时间')
                timer2.stop()
            pass
        try:
            self.LightPressL = []
            self.LightPressP = []
            self.LightPresscount = 0
            task = MyThread.MyThread(plotfunc, (self.LightPressStart,self.LightPressEnd))
            task.start()
            timer2 = QTimer()
            timer2.timeout.connect(fun_timer2)
            timer2.start(2100)
            self.ui.Ter.append('开始测试光谱气压')
        except:
            self.ui.Ter.append('测试光谱气压' + '错误')


        pass
    def SetTimeDotFunc(self):
        self.TimeDotData=[]
        try:
            self.rawTimeDotData = self.ReadLit(self.ui.LitTimeSetDot).split(',')
            for r in self.rawTimeDotData:
                self.TimeDotData.append(int(r))
            count = 0
            self.DotCountlist = []
            for ti in self.TimeDotData:
                for fl in self.LightWaveTimeSecondList:
                    if fl < ti:
                        count = count + 1
                    else:
                        self.DotCountlist.append(count)
                        count = 0
                        break
            self.ui.Ter.append('设置时间节点' + str(self.TimeDotData))
        except:
            self.ui.Ter.append('设置时间节点' + '错误')
        pass
    def SetTimeIntervalFunc(self):
        try:
            self.TimeIntervalData = int(self.ReadLit(self.ui.LitTimeSetInterval))
            self.ui.Ter.append('设置时间间隔'+str(self.TimeIntervalData))
        except:
            self.ui.Ter.append('设置时间间隔' + '错误')
        pass
    def SetTimeNumberFunc(self,):
        try:
            self.TimeNumberData = int(self.ReadLit(self.ui.LitTimeSetNumber))
            self.ui.Ter.append('设置时间数量' + str(self.TimeNumberData))
        except:
            self.ui.Ter.append('设置时间数量' + '错误')
        pass

    def TestTimeDotFunc(self,):
        mutix=[]
        mutiy=[]
        try:
            for c in self.DotCountlist:
                mutix.append(self.LightWaveLengthList)
                mutiy.append(self.LightWaveAmplitudeLists[c])
                self.MDIDraw(self.LightWaveLengthList,self.LightWaveAmplitudeLists[c],'波长(nm)','透射率(%)','波长-透射率'
                             +str(self.LightWaveTimeSecondList[c])+'s 开始时间：'+
                         self.LightTimeStart+'   结束时间：'+self.LightTimeEnd)
            self.MutiMDIDraw(mutix,mutiy,'波长(nm)','透射率(%)','波长-透射率 开始时间：'+
                         self.LightTimeStart+'   结束时间：'+self.LightTimeEnd)
            with open(dir + str(time.strftime("%Y%m%d%H%M%S", time.localtime())) + "spectrum.csv", "a", newline='') as out:
                csv_writer = csv.writer(out, dialect="excel")
                csv_writer.writerow(['测试时间:',str(time.strftime("%Y%m%d %H:%M:%S", time.localtime()))])
                w0list = []
                w0list.append('波长(nm)/时间(s)')
                rowtemp = [c*(self.LightWaveTimeSecondList[1]-self.LightWaveTimeSecondList[0]) for c in self.DotCountlist]
                for r in rowtemp:
                    w0list.append(r)
                csv_writer.writerow(w0list)
                for i in range(len(self.LightWaveLengthList)):
                    wlist=[]
                    wlist.append(self.LightWaveLengthList[i])
                    rowtemp=[y[i] for y in mutiy]
                    for r in rowtemp:
                        wlist.append(r)
                    csv_writer.writerow(wlist)
            self.ui.Ter.append('光谱时序测试成功')
        except:
            self.ui.Ter.append('光谱时序测试失败')
        pass
    def TestTimeIntervalFunc(self):
        i = 0
        t = 0
        mutix = []
        mutiy = []
        timelist=[]
        try:
            while True:
                if(self.LightWaveTimeSecondList[t]>i):
                    timelist.append(self.LightWaveTimeSecondList[t])
                    mutix.append(self.LightWaveLengthList)
                    mutiy.append(self.LightWaveAmplitudeLists[t])
                    self.MDIDraw(self.LightWaveLengthList, self.LightWaveAmplitudeLists[t], '波长(nm)', '透射率(%)',
                                 '波长-透射率' + str(self.LightWaveTimeSecondList[t]) + 's 开始时间：'+
                         self.LightTimeStart+'   结束时间：'+self.LightTimeEnd)
                    i = i + self.TimeIntervalData
                t = t + 1

        except:
            if i > 0:
                self.ui.Ter.append('光谱时序测试成功')
                try:
                    self.MutiMDIDraw(mutix, mutiy, '波长(nm)', '透射率(%)', '波长-透射率 开始时间：'+
                         self.LightTimeStart+'   结束时间：'+self.LightTimeEnd)
                    with open(dir + str(time.strftime("%Y%m%d%H%M%S", time.localtime())) + "spectrum.csv", "a",
                              newline='') as out:
                        csv_writer = csv.writer(out, dialect="excel")
                        csv_writer.writerow(['测试时间:', str(time.strftime("%Y%m%d %H:%M:%S", time.localtime()))])
                        w0list = []
                        w0list.append('波长(nm)/时间(s)')
                        rowtemp = [c for c in
                                   timelist]
                        for r in rowtemp:
                            w0list.append(r)
                        csv_writer.writerow(w0list)
                        for i in range(len(self.LightWaveLengthList)):
                            wlist = []
                            wlist.append(self.LightWaveLengthList[i])
                            rowtemp = [y[i] for y in mutiy]
                            for r in rowtemp:
                                wlist.append(r)
                            csv_writer.writerow(wlist)
                except:
                    pass
            else:
                self.ui.Ter.append('光谱时序测试失败')
        pass
    def TestTimeNumberFunc(self):
        i = 0
        t = 0
        mutix = []
        mutiy = []
        timelist = []
        try:
            timemax=self.LightWaveTimeSecondList[-1]
            timemin=self.LightWaveTimeSecondList[0]
            timelen=timemax-timemin
            self.TimeIntervalData=timelen/self.TimeNumberData

            while True:
                if (self.LightWaveTimeSecondList[t] > i):
                    timelist.append(self.LightWaveTimeSecondList[t])
                    mutix.append(self.LightWaveLengthList)
                    mutiy.append(self.LightWaveAmplitudeLists[t])
                    self.MDIDraw(self.LightWaveLengthList, self.LightWaveAmplitudeLists[t], '波长(nm)', '透射率(%)',
                                 '波长-透射率' + str(self.LightWaveTimeSecondList[t]) + 's 开始时间：'+
                         self.LightTimeStart+'   结束时间：'+self.LightTimeEnd)
                    i = i + self.TimeIntervalData
                t = t + 1
        except:
            if i > 0:
                self.ui.Ter.append('光谱时序测试成功')
                try:
                    self.MutiMDIDraw(mutix, mutiy, '波长(nm)', '透射率(%)', '波长-透射率 开始时间：'+
                         self.LightTimeStart+'   结束时间：'+self.LightTimeEnd)
                    with open(dir + str(time.strftime("%Y%m%d%H%M%S", time.localtime())) + "spectrum.csv", "a",
                              newline='') as out:
                        csv_writer = csv.writer(out, dialect="excel")
                        csv_writer.writerow(['测试时间:', str(time.strftime("%Y%m%d %H:%M:%S", time.localtime()))])
                        w0list = []
                        w0list.append('波长(nm)/时间(s)')
                        rowtemp = [c for c in
                                   timelist]
                        for r in rowtemp:
                            w0list.append(r)
                        csv_writer.writerow(w0list)
                        for i in range(len(self.LightWaveLengthList)):
                            wlist = []
                            wlist.append(self.LightWaveLengthList[i])
                            rowtemp = [y[i] for y in mutiy]
                            for r in rowtemp:
                                wlist.append(r)
                            csv_writer.writerow(wlist)
                except:
                    pass
            else:
                self.ui.Ter.append('光谱时序测试失败')
        pass
    def SetWavelengthRangeFunc(self):
        self.WavelengthRangeData = []
        try:
            self.rawWavelengthRangeData = self.ReadLit(self.ui.LitWaveLengthSetRange).split(',')
            for r in self.rawWavelengthRangeData:
                self.WavelengthRangeData.append(int(r))
            self.ui.Ter.append('设置波长范围' + str(self.WavelengthRangeData))
        except:
            self.ui.Ter.append('设置波长范围' + '错误')
        pass
    def SetWavelengthDotFunc(self):
        try:
            self.WavelengthDotData = int(self.ReadLit(self.ui.LitWaveLengthSetDot))
            self.ui.Ter.append('设置波长点' + str(self.WavelengthDotData)+'nm')
        except:
            self.ui.Ter.append('设置波长点' + '错误')
        pass
    def TestWavelengthPeakFun(self,x_list, y_lists, mode, limitmin=450, limitmax=1150, width=120):
        i=0
        self.peaklistx = []
        self.peaklisty = []
        self.ui.Ter.append('光强时序测试中')
        try:
            for temp in y_lists:
                functionx,functiony,peak_x, peak_y = KsPackage.lorentzfunctionCurve(
                    x_list,
                    temp,
                    mode,
                    limitmin,
                    limitmax,
                    width)
                i = i + 1
                self.peaklistx.append(peak_x)
                self.peaklisty.append(peak_y)
                self.workcount = int(100 * i / len(self.LightWaveAmplitudeLists))
        except:
            self.ui.Ter.append('光强时序测试数据导入失败')
            self.timer.stop()
    def TestWavelengthPeakFunc(self):
        try:
            self.workcount = 0
            if self.Maxflag==True:
                task = MyThread.MyThread(self.TestWavelengthPeakFun,
                                         (self.LightWaveLengthList, self.LightWaveAmplitudeLists, 1,
                                          self.WavelengthRangeData[0], self.WavelengthRangeData[1],
                                          120))
                task.start()
            else:
                task = MyThread.MyThread(self.TestWavelengthPeakFun,
                                         (self.LightWaveLengthList, self.LightWaveAmplitudeLists, -1,
                                          self.WavelengthRangeData[0], self.WavelengthRangeData[1],
                                          120))
                task.start()
            self.timer = QTimer()
            self.timer.timeout.connect(self.fun_timer)
            self.timer.start(2100)
        except:
            self.ui.Ter.append('光强时序测试参数错误')

        pass
    def TestWavelengthDotFunc(self):
        i = 0
        peaklist = []
        top=0
        try:
            for temp in self.LightWaveLengthList:
                if temp < self.WavelengthDotData:
                    i = i + 1
                else:
                    top=temp
                    break
            for temp in self.LightWaveAmplitudeLists:
                peaklist.append(temp[i])
            m=min(peaklist)
            for ii in range(len(peaklist)):
                peaklist[ii]=peaklist[ii]-m
            self.MDIDraw(self.LightWaveTimeSecondList, peaklist, '时间(s)', '透射率(%)',
                         '时间-波长' + str(self.LightWaveLengthList[i]) + 'nm 透射率 开始时间：'+
                         self.LightTimeStart+'   结束时间：'+self.LightTimeEnd,flag=self.ReadCheck(self.ui.ChxLightPressTime))
            if len(self.PressTimeList)==0:
                with open(dir + str(time.strftime("%Y%m%d%H%M%S", time.localtime())) + "spectrum_time.csv", "a",
                          newline='') as out:
                    csv_writer = csv.writer(out, dialect="excel")
                    csv_writer.writerow(['测试时间:', str(time.strftime("%Y%m%d %H:%M:%S", time.localtime())),'波长：',top])
                    csv_writer.writerow(['时间(s)','透射率(%)'])
                    for i in range(len(self.LightWaveTimeSecondList)):
                        csv_writer.writerow([self.LightWaveTimeSecondList[i],peaklist[i]])
            else:
                if abs(self.PressTimeToFix - self.LightTimeToFix) > 36000:
                    temp = self.PressTimeToFix - self.LightTimeToFix + 12 * 60 * 60
                else:
                    temp = self.PressTimeToFix - self.LightTimeToFix
                with open(dir + str(time.strftime("%Y%m%d%H%M%S", time.localtime())) + "spectrum_time.csv", "a",
                          newline='') as out:
                    csv_writer = csv.writer(out, dialect="excel")
                    csv_writer.writerow(['测试时间:', str(time.strftime("%Y%m%d %H:%M:%S", time.localtime())),'波长：',top])
                    csv_writer.writerow(['时间(s)','透射率(%)','气压(Pa)'])
                    tempii=0
                    for i in range(len(self.LightWaveTimeSecondList)):
                        for ii in range(tempii,len(self.PressTimeList)):
                            if self.LightWaveTimeSecondList[i]<=self.PressTimeList[ii]+temp:
                                csv_writer.writerow([self.LightWaveTimeSecondList[i],peaklist[i],self.PressAmplitudelogList[ii]])
                                tempii=ii
                                break
            self.ui.Ter.append('光强时序测试成功')
        except:
            self.ui.Ter.append('光强时序测试失败')
        if self.ReadCheck(self.ui.ChxLightPress) == True:
            self.SetLightPressFunc()
            self.TestLightPressFun(peaklist)
        pass
    def TestPressFunc(self):
        try:
            xlist=[]
            ylist=[]
            PressLightlist=[]
            PressLighttimelist=[]
            if len(self.LightWaveTimeSecondList) > 0:
                if abs(self.PressTimeToFix - self.LightTimeToFix) > 36000:
                    temp = self.PressTimeToFix - self.LightTimeToFix + 12 * 60 * 60
                else:
                    temp = self.PressTimeToFix - self.LightTimeToFix
                maxA=0
                minA=999999
                for i in range(0, len(self.LightWaveAmplitudeLists)):
                    maxA=max({maxA,self.LightWaveAmplitudeLists[i][640]})
                    minA=min({minA,self.LightWaveAmplitudeLists[i][640]})
                for i in range(0, len(self.LightWaveTimeSecondList)):
                    PressLightlist.append((self.LightWaveAmplitudeLists[i][640]-minA)/(maxA-minA)*4)
                    PressLighttimelist.append(self.LightWaveTimeSecondList[i]-temp)
                ylist.append(PressLightlist)
                xlist.append(PressLighttimelist)
            if len(self.PressTimeList)>0:
                xlist.append(self.PressTimeList)
                ylist.append(self.PressAmplitudelogList)
            self.MDIDraw(self.PressTimeList,self.NewPressAmplitudeList,'时间(s)','气压('+self.PressUnit+')','时间-气压 开始时间：'+
                        self.PressTimeStart+'   结束时间：'+self.PressTimeEnd)
          # self.MDIDraw(self.PressTimeList, self.PressAmplitudelogList, '时间(s)', '气压(lg(' + self.PressUnit + '))',
          #              '时间-气压 开始时间：' +
          #               self.PressTimeStart + '   结束时间：' + self.PressTimeEnd)
            self.MutiMDIDraw(xlist, ylist, '时序', '气压/透射率','时序-气压/透射率')
            with open(dir + str(time.strftime("%Y%m%d%H%M%S", time.localtime())) + "press.csv", "a", newline='') as out:
                csv_writer = csv.writer(out, dialect="excel")
                csv_writer.writerow(['测试时间:',str(time.strftime("%Y%m%d %H:%M:%S", time.localtime()))])
                csv_writer.writerow(['时间(s)','幅度(Pa)'])
                for i in range(len(self.PressTimeList)):
                    csv_writer.writerow([self.PressTimeList[i],self.NewPressAmplitudeList[i]])
            with open(dir + str(time.strftime("%Y%m%d%H%M%S", time.localtime())) + "presslg.csv", "a", newline='') as out:
                csv_writer = csv.writer(out, dialect="excel")
                csv_writer.writerow(['测试时间:',str(time.strftime("%Y%m%d %H:%M:%S", time.localtime()))])
                csv_writer.writerow(['时间(s)','幅度(lg(Pa))'])
                for i in range(len(self.PressTimeList)):
                    csv_writer.writerow([self.PressTimeList[i],self.PressAmplitudelogList[i]])
            if len(self.PressTimeList)==0:
                self.ui.Ter.append('气压数据为空')
            else:
                self.ui.Ter.append('气压测试成功')
        except:
            self.ui.Ter.append('气压测试失败')
        pass

    def AddMDIArea(self,subwidget):
        self.ui.mdiArea.addSubWindow(subwidget)
    def ClearMDIArea(self):
        self.ui.mdiArea.closeAllSubWindows()

    def MDIDraw(self,x,y,x_name,y_name,title,flag=False):
        def fun_timer1():
            self.SetDisable(self.ui.BunStart)
            self.ui.Ter.append('光谱气压时序作图已完成' + str(int(self.plotcount/len(x)*100)) + '%')
            if self.plotcount/len(x)*100>=99.5:
                self.SetEnable(self.ui.BunStart)
                pltItem = pw.getPlotItem()
                sub.setWidget(pw)
                sub.setWindowTitle(title)
                bottom_axis = pltItem.getAxis("bottom")
                bottom_axis.setLabel(x_name)
                left_axis = pltItem.getAxis("left")
                left_axis.setLabel(y_name)
                self.AddMDIArea(sub)
                sub.show()
                self.ui.Ter.append('光谱气压时序作图已完成')
                timer1.stop()
        def plotfunc(x,y):
            temp=0
            '''for i in range(0, len(x), 2):
                self.plotcount = self.plotcount + 2
                for ii in range(0, len(self.PressTimeList), 9):
                    if (x[i] < self.PressTimeList[ii]+self.PressTimeToFix-self.LightTimeToFix):
                        temp=ii
                        pw.plot(x[i:i + 4], y[i:i + 4], pen=(
                            255 * self.PressAmplitudeList[ii] / max(self.PressAmplitudeList),
                            255 - 255 * self.PressAmplitudeList[ii] / max(self.PressAmplitudeList),
                            255 * (math.log(self.PressAmplitudeList[ii], 10) + 4.5) / 6))
                        break'''

            for ii in range(0, len(self.PressTimeList), 9):
                if (x[temp] < self.PressTimeList[ii] + self.PressTimeToFix - self.LightTimeToFix):
                    self.plotcount = self.plotcount + 2
                    pw.plot(x[temp:temp + 3], y[temp:temp + 3], pen=(
                        255 * (math.log(self.NewPressAmplitudeList[ii], 10) - math.log(min(self.NewPressAmplitudeList),
                                                                                       10)) / (
                                    math.log(max(self.NewPressAmplitudeList), 10) - math.log(
                                min(self.NewPressAmplitudeList), 10)),
                        255 - 255 * (math.log(self.NewPressAmplitudeList[ii], 10) - math.log(
                            min(self.NewPressAmplitudeList), 10)) / (
                                    math.log(max(self.NewPressAmplitudeList), 10) - math.log(
                                min(self.NewPressAmplitudeList), 10)),
                        0))
                    if(temp<len(x)-2):
                        temp = temp + 2
                    else:
                        break
            for ii in range(0, len(self.PressTimeList), 9):
                if (x[temp] < self.PressTimeList[ii] + self.PressTimeToFix - self.LightTimeToFix+12*60*60):
                    self.plotcount = self.plotcount + 2
                    pw.plot(x[temp:temp + 3], y[temp:temp + 3], pen=(
                        255 * (math.log(self.NewPressAmplitudeList[ii], 10) - math.log(min(self.NewPressAmplitudeList),
                                                                                       10)) / (
                                    math.log(max(self.NewPressAmplitudeList), 10) - math.log(
                                min(self.NewPressAmplitudeList), 10)),
                        255 - 255 * (math.log(self.NewPressAmplitudeList[ii], 10) - math.log(
                            min(self.NewPressAmplitudeList), 10)) / (
                                    math.log(max(self.NewPressAmplitudeList), 10) - math.log(
                                min(self.NewPressAmplitudeList), 10)),
                        0))
                    if(temp<len(x)-2):
                        temp = temp + 2
                    else:
                        break


        if flag == False or len(self.PressTimeList)==0 :
            sub = QMdiSubWindow()
            pw = pyqtgraph.PlotWidget(background="w")
            pw.plot(x, y, pen=pyqtgraph.mkPen('r',width=2))
            pltItem = pw.getPlotItem()
            sub.setWidget(pw)
            sub.setWindowTitle(title)


            bottom_axis = pltItem.getAxis("bottom")
            bottom_axis.setLabel(x_name)
            left_axis = pltItem.getAxis("left")
            left_axis.setLabel(y_name)
            self.AddMDIArea(sub)
            sub.show()
        else :
            sub = QMdiSubWindow()
            pw = pyqtgraph.PlotWidget(background="w")
            pw = pyqtgraph.PlotWidget()
            self.plotcount = 0
            task = MyThread.MyThread(plotfunc,(x,y))
            task.start()
            timer1 = QTimer()
            timer1.timeout.connect(fun_timer1)
            timer1.start(2100)

        pass
    def MutiMDIDraw(self,x,y,x_name,y_name,title):
        sub = QMdiSubWindow()

        pw = pyqtgraph.PlotWidget(background="w")
        sub.setWidget(pw)
        sub.setWindowTitle(title)

        for i in range(0, len(y)):
            pw.plot(x[i], y[i], pen=pyqtgraph.mkPen(i,len(y),width=2))
        pltItem = pw.getPlotItem()
        bottom_axis = pltItem.getAxis("bottom")
        bottom_axis.setLabel(x_name)
        left_axis = pltItem.getAxis("left")
        left_axis.setLabel(y_name)
        self.AddMDIArea(sub)
        sub.show()

        pass
    def RunFuncList(self,List,stats):
        if stats == True:
            for l in List:
                l()
    def LitGrammeMinchangedhandle(self):
        self.GAFMin = float(self.ReadLit(self.ui.LitGrammeMin))
        pass
    def LitGrammeMaxchangedhandle(self):
        self.GAFMax = float(self.ReadLit(self.ui.LitGrammeMax))
        pass

    def LitGrammeIntervalchangedhandle(self):
        try:
            self.GAFInterval = int(self.ReadLit(self.ui.LitGrammeInterval))
        except:
            pass
        pass
    def mkdir(self,fpath):
        folder = os.path.exists(fpath)
        if not folder:
            os.makedirs(fpath)
    def BunGADFclickedhandle(self):
        self.GADFcount=0
        def fun_timerGADF():
            self.ui.Ter.append('GADF已完成'+str(self.GADFcount)+'%')
            if self.GADFcount==100:
                timer1.stop()
        def GADFFunc():
            self.GADFFilename=str(time.strftime('%y-%m-%d-%H-%M-%S',time.localtime(time.time()))
)+'GADFFile.txt'
            self.mkdir('D:\\硕士学习\\GAF')
            count = 0
            with open('D:\\硕士学习\\GAF\\'+self.GADFFilename, 'w', encoding='utf8') as GADFFile:
                avelist = []
                for ii in range(0, len(self.LightWaveAmplitudeLists[0]), self.GAFInterval):
                    sumtemp = 0
                    for iii in range(10, 20):
                        sumtemp = sumtemp + self.LightWaveAmplitudeLists[iii][ii]
                    av = sumtemp / 10
                    avelist.append(av)

                for i in self.LightWaveAmplitudeLists:
                    listtemp=[]
                    for ii in range(0,len(i),self.GAFInterval):
                        if(self.LightWaveLengthList[ii]>self.GAFMin and self.LightWaveLengthList[ii]<self.GAFMax):
                            listtemp.append(i[ii]-avelist[ii//self.GAFInterval])
                    count = count + 1
                    temp=Gramme.GADFExchange(listtemp,flag=0)
                    self.GADFMatrixListRaw.append(temp)
                    GADFFile.write(str(temp))
                    self.GADFcount = count / len(self.LightWaveAmplitudeLists) * 100
                self.GADFMatrixList = self.GADFMatrixListRaw.copy()

        if(self.LightWaveAmplitudeLists!=0):
            self.GADFMatrixListRaw.clear()
            task = MyThread.MyThread(GADFFunc, ())
            task.start()
            timer1 = QTimer()
            timer1.timeout.connect(fun_timerGADF)
            timer1.start(2100)

        pass
    def BunGASFclickedhandle(self):
        self.GASFcount = 0

        def fun_timerGASF():
            self.ui.Ter.append('GASF已完成' + str(self.GASFcount) + '%')
            if self.GASFcount == 100:
                seaborn.heatmap(self.GASFMatrixList[0], vmin=-1, vmax=1, cmap='GnBu')
                plt.show()
                timer2.stop()

        def GASFFunc():
            self.GASFFilename = str(time.strftime('%y-%m-%d-%H-%M-%S',time.localtime(time.time()))
) + 'GASFFile.txt'
            self.mkdir('D:\\硕士学习\\GAF')
            count = 0
            with open('D:\\硕士学习\\GAF\\' + self.GASFFilename, 'w', encoding='utf8') as GASFFile:
                avelist=[]
                for ii in range(0, len(self.LightWaveAmplitudeLists[0]), self.GAFInterval):
                    sumtemp = 0
                    for iii in range(10, 20):
                        sumtemp = sumtemp +self.LightWaveAmplitudeLists[iii][ii]
                    av=sumtemp/10
                    avelist.append(av)

                for i in self.LightWaveAmplitudeLists:
                    listtemp = []
                    for ii in range(0, len(i), self.GAFInterval):
                        if(self.LightWaveLengthList[ii]>self.GAFMin and self.LightWaveLengthList[ii]<self.GAFMax):
                            listtemp.append(i[ii]-avelist[ii//self.GAFInterval])
                    count = count + 1
                    temp=Gramme.GASFExchange(listtemp,flag=0)
                    self.GASFMatrixListRaw.append(temp)
                    GASFFile.write(str(temp))
                    self.GASFcount = count / len(self.LightWaveAmplitudeLists) * 100
                self.GASFMatrixList=self.GASFMatrixListRaw.copy()


        if (self.LightWaveAmplitudeLists != 0):
            self.GASFMatrixListRaw.clear()
            task = MyThread.MyThread(GASFFunc, ())
            task.start()
            timer2 = QTimer()
            timer2.timeout.connect(fun_timerGASF)
            timer2.start(2100)
        pass

    def BunShowGASFR2clickedhandle(self):
        self.GASFR2count = 0

        def fun_timerGASFR2():
            self.ui.Ter.append('GASFR2已完成' + str(self.GASFR2count) + '%')
            if self.GASFR2count == 100:
                self.MutiMDIDraw([self.LightWaveTimeSecondList[1:len(self.GASFR2List) + 1],
                                  self.LightWaveTimeSecondList[1:len(self.GASFR2List) + 1]],
                                 [self.GASFR2List, self.GASFR2fList],
                                 '时间(s)', 'GASFR2',
                                 '时间-GASFR2' + '开始时间：' +
                                 self.LightTimeStart + '   结束时间：' + self.LightTimeEnd)
                fig1 = plt.figure
                seaborn.heatmap(self.GASFMatrixList[0], vmin=-1, vmax=1, cmap='GnBu')
                plt.show()
                print(self.LightWaveTimeSecondList)
                print(self.GASFR2List)
                timer1.stop()
            pass

        def GASFR2Func():
            GASFMatrixStart = self.GASFMatrixList[0]
            self.GASFR2List = []
            self.GASFR2fList = []
            count = 0
            for i in self.GASFMatrixList[1:len(self.GASFMatrixList)]:
                count = count + 1
                GASFR2 = 0
                for ic in range(len(i)):
                    for ir in range(len(i[ic])):
                        GASFR2 = GASFR2 + i[ic][ir] ** 2
                GASFR2 =  math.sqrt(GASFR2)
                self.GASFR2count = count / (len(self.GASFMatrixList) - 1) * 100
                self.GASFR2List.append(GASFR2)
            self.GASFR2fList=savgol_filter(self.GASFR2List,11,1)
            print(self.GASFR2fList)

        self.GASFR2List.clear()
        if len(self.GASFMatrixList) != 0:
            task = MyThread.MyThread(GASFR2Func, ())
            task.start()
            timer1 = QTimer()
            timer1.timeout.connect(fun_timerGASFR2)
            timer1.start(2100)
        pass

    def BunShowGADFR2clickedhandle(self):
        self.GADFR2count = 0

        def fun_timerGADFR2():
            self.ui.Ter.append('GADFR2已完成' + str(self.GADFR2count) + '%')
            if self.GADFR2count == 100:
                self.MutiMDIDraw([self.LightWaveTimeSecondList[1:len(self.GADFR2List)+1],
                                  self.LightWaveTimeSecondList[1:len(self.GADFR2List)+1]], [self.GADFR2List,self.GADFR2fList],
                             '时间(s)', 'GADFR2',
                             '时间-GADFR2' + '开始时间：' +
                             self.LightTimeStart + '   结束时间：' + self.LightTimeEnd)

                fig1 = plt.figure
                seaborn.heatmap(self.GADFMatrixList[0], vmin=-1, vmax=1, cmap='GnBu')
                plt.show()
                timer1.stop()
            pass

        def GADFR2Func():
            GADFMatrixStart = self.GADFMatrixList[0]
            self.GADFR2List = []
            self.GADFR2fList = []
            count = 0
            for i in self.GADFMatrixList[1:len(self.GADFMatrixList)]:
                count = count + 1
                GADFR2 = 0
                for ic in range(len(i)):
                    for ir in range(len(i[ic])):
                        GADFR2 = GADFR2 + (i[ic][ir]) ** 2
                GADFR2 = math.sqrt(GADFR2)
                self.GADFR2count = count / (len(self.GADFMatrixList) - 1) * 100
                self.GADFR2List.append(GADFR2)
            self.GADFR2fList = savgol_filter(self.GADFR2List, 11, 1)
            print(self.GADFR2fList )
        self.GADFR2List.clear()
        if len(self.GADFMatrixList) != 0:
            task = MyThread.MyThread(GADFR2Func, ())
            task.start()
            timer1 = QTimer()
            timer1.timeout.connect(fun_timerGADFR2)
            timer1.start(2100)
        pass

    def BunShowGASFR2_Pressclickedhandle(self):
        def GASFR2_PressFunc(x_start,x_end):
            try:
                for i in range(x_start, x_end):
                    for ii in range(0, len(self.LightWaveTimeSecondList) - 1):
                        if (i < self.LightWaveTimeSecondList[ii]):

                            for iii in range(0, len(self.PressTimeList)):
                                if (i < self.PressTimeList[iii] + self.PressTimeToFix - self.LightTimeToFix):
                                    self.GASFR2_PressP.append(math.log(self.NewPressAmplitudeList[iii], 10))
                                    self.GASFR2_PressG.append(self.GASFR2List[ii])
                                    self.GASFR2_Presscount = ii / (len(self.LightWaveTimeSecondList) - 2) * 100
                                    break
                            for iii in range(0, len(self.PressTimeList)):
                                if (i < self.PressTimeList[
                                    iii] + self.PressTimeToFix - self.LightTimeToFix + 12 * 60 * 60) and (
                                        i > self.PressTimeList[
                                    iii] + self.PressTimeToFix - self.LightTimeToFix + 6 * 60 * 60):
                                    self.GASFR2_PressP.append(math.log(self.NewPressAmplitudeList[iii], 10))
                                    self.GASFR2_PressG.append(self.GASFR2List[ii])
                                    self.GASFR2_Presscount = ii / (len(self.LightWaveTimeSecondList) - 2) * 100
                                    break
                            break
                self.GASFR2_Presscount = 100
            except:
                self.ui.Ter.append('GASFR2-Press作图失败,请重新设定时间')

            pass

        def fun_timerGASFR2_Press():
            self.SetDisable(self.ui.BunStart)
            self.ui.Ter.append('GASFR2-Press气压作图已完成'+str(int(self.GASFR2_Presscount))+'%')
            if self.GASFR2_Presscount ==100:
                self.SetEnable(self.ui.BunStart)
                try:
                    self.MDIDraw(self.GASFR2_PressG, self.GASFR2_PressP, 'GASFR2','气压(lg('+self.PressUnit+'))',
                                     'GASFR2-Press-气压')
                    self.ui.Ter.append('GASFR2-Press作图成功')
                except:
                    pass
                timer2.stop()
            pass
        try:
            if (len(self.PressTimeList) != 0 and len(self.GASFMatrixList)!=0):
                self.GASFR2_PressG = []
                self.GASFR2_PressP = []
                self.GASFR2_Presscount = 0
                task = MyThread.MyThread(GASFR2_PressFunc, (
                int(self.ReadLit(self.ui.LitGrammeStart)), int(self.ReadLit(self.ui.LitGrammeEnd))))
                task.start()
                timer2 = QTimer()
                timer2.timeout.connect(fun_timerGASFR2_Press)
                timer2.start(2100)
                self.ui.Ter.append('开始测试GASFR2-Press')
        except:
            self.ui.Ter.append('测试GASFR2-Press' + '错误')
        pass

    def BunShowGADFR2_Pressclickedhandle(self):
        def GADFR2_PressFunc(x_start, x_end):
            try:
                for i in range(x_start, x_end):
                    for ii in range(0, len(self.LightWaveTimeSecondList) - 1):
                        if (i < self.LightWaveTimeSecondList[ii]):

                            for iii in range(0, len(self.PressTimeList)):
                                if (i < self.PressTimeList[iii] + self.PressTimeToFix - self.LightTimeToFix):
                                    self.GADFR2_PressP.append(math.log(self.NewPressAmplitudeList[iii], 10))
                                    self.GADFR2_PressG.append(self.GADFR2List[ii])
                                    self.GADFR2_Presscount = ii / (len(self.LightWaveTimeSecondList) - 2) * 100
                                    break
                            for iii in range(0, len(self.PressTimeList)):
                                if (i < self.PressTimeList[
                                    iii] + self.PressTimeToFix - self.LightTimeToFix + 12 * 60 * 60) and (
                                        i > self.PressTimeList[
                                    iii] + self.PressTimeToFix - self.LightTimeToFix + 6 * 60 * 60):
                                    self.GADFR2_PressP.append(math.log(self.NewPressAmplitudeList[iii], 10))
                                    self.GADFR2_PressG.append(self.GADFR2List[ii])
                                    self.GADFR2_Presscount = ii / (len(self.LightWaveTimeSecondList) - 2) * 100
                                    break
                            break
                self.GADFR2_Presscount = 100
            except:
                self.ui.Ter.append('GADFR2-Press作图失败,请重新设定时间')

            pass

        def fun_timerGADFR2_Press():
            self.SetDisable(self.ui.BunStart)
            self.ui.Ter.append('GADFR2-Press气压作图已完成' + str(int(self.GADFR2_Presscount)) + '%')
            if self.GADFR2_Presscount == 100:
                self.SetEnable(self.ui.BunStart)
                try:
                    self.MDIDraw(self.GADFR2_PressG, self.GADFR2_PressP, 'GADFR2', '气压(lg(' + self.PressUnit + '))',
                                 'GADFR2-Press-气压')
                    self.ui.Ter.append('GADFR2-Press作图成功')
                except:
                    pass

                timer2.stop()
            pass

        try:
            if(len(self.PressTimeList)!=0 and len(self.GADFMatrixList)!=0):
                self.GADFR2_PressG = []
                self.GADFR2_PressP = []
                self.GADFR2_Presscount = 0
                task = MyThread.MyThread(GADFR2_PressFunc, (
                    int(self.ReadLit(self.ui.LitGrammeStart)), int(self.ReadLit(self.ui.LitGrammeEnd))))
                task.start()
                timer2 = QTimer()
                timer2.timeout.connect(fun_timerGADFR2_Press)
                timer2.start(2100)
                self.ui.Ter.append('开始测试GADFR2-Press')
        except:
            self.ui.Ter.append('测试GADFR2-Press' + '错误')
        pass
    def BunMatrixSpliceclickedhandle(self):
        def MatrixSpliceFunc():
            self.GASFMatrixList.clear()
            self.GADFMatrixList.clear()
            self.GAFMatrixList.clear()
            self.MatrixNumber = int(self.ReadLit(self.ui.LitMatrixNumber))
            for i in range(len(self.LightWaveAmplitudeLists)):
                self.GAFMatrixList.append(self.LightWaveAmplitudeLists[i][15::15])
            if len(self.GADFMatrixListRaw) != 0:
                for i in range(self.MatrixNumber - 1, len(self.GADFMatrixListRaw)):
                    self.MatrixSplicecount = i / (len(self.GADFMatrixListRaw) - self.MatrixNumber - 1) * 50
                    GADFMatrix = []
                    for ii in range(0, self.MatrixNumber):
                        for iii in range(0, len(self.GADFMatrixListRaw[ii])):
                            if ii == 0:
                                GADFMatrix.append(self.GADFMatrixListRaw[i][iii])
                            else:
                                temp = []
                                for iiii in range(0, len(self.GADFMatrixListRaw[ii][iii])):
                                    temp.append(
                                        self.GADFMatrixListRaw[i - ii][iii][iiii]
                                        / self.GADFMatrixListRaw[i][iii][iiii]
                                    )
                                GADFMatrix.append(temp.copy())
                    self.GADFMatrixList.append(GADFMatrix)
            if len(self.GASFMatrixListRaw) != 0:
                for i in range(self.MatrixNumber-1, len(self.GASFMatrixListRaw)):
                    self.MatrixSplicecount = i / (len(self.GASFMatrixListRaw) - self.MatrixNumber - 1) * 50 + 50
                    GASFMatrix = []
                    for ii in range(0, self.MatrixNumber):
                        for iii in range(0, len(self.GASFMatrixListRaw[ii])):
                            if ii==0:
                                GASFMatrix.append(self.GASFMatrixListRaw[i][iii])
                            else:
                                temp = []
                                for iiii in range(0, len(self.GASFMatrixListRaw[ii][iii])):
                                    temp.append(
                                        self.GASFMatrixListRaw[i - ii][iii][iiii]
                                        /self.GASFMatrixListRaw[i][iii][iiii]
                                    )
                                GASFMatrix.append(temp.copy())
                    self.GASFMatrixList.append(GASFMatrix)
            self.MatrixSplicecount = 100
        def fun_timerMatrixSplice():
            self.SetDisable(self.ui.BunStart)
            self.ui.Ter.append('MatrixSplice已完成' + str(int(self.MatrixSplicecount)) + '%')
            if self.MatrixSplicecount == 100:
                self.SetEnable(self.ui.BunStart)
                try:
                    self.ui.Ter.append('MatrixSplice成功')
                except:
                    self.ui.Ter.append('MatrixSplice失败,请重新设定时间')
                timer2.stop()
            pass
        self.MatrixSplicecount=0
        task = MyThread.MyThread(MatrixSpliceFunc, ())
        task.start()
        timer2 = QTimer()
        timer2.timeout.connect(fun_timerMatrixSplice)
        timer2.start(2100)
        pass
    def BunTrainGAFclickedhandle(self):
        def fun_timerMyTorchTrain():
            self.SetDisable(self.ui.BunStart)
            self.ui.Ter.append('Train已完成' + str(int(MyTorch2.Traincount)) + '%')
            if MyTorch2.Traincount == 100:
                self.SetEnable(self.ui.BunStart)
                self.ui.Ter.append('Train成功')
                MyTorch2.Traincount = 0
                timer2.stop()

            pass

        self.trianflag=1
        task = MyThread.MyThread(MyTorch2.Train, (
            self.LightWaveTimeSecondList,
            self.GASFMatrixList,
            self.PressTimeList, self.PressAmplitudelogList, self.PressTimeToFix, self.LightTimeToFix))
        task.start()
        timer2 = QTimer()
        timer2.timeout.connect(fun_timerMyTorchTrain)
        timer2.start(2100)
    def BunTrainGADFclickedhandle(self):

        def fun_timerMyTorchTrain():
            if MyTorch.Traincount>=self.showtemp:
                self.showtemp=self.showtemp+self.showthr
                self.showstd()
                self.ui.Ter.append(str(int(MyTorch.Traincount/100*300)))
            self.SetDisable(self.ui.BunStart)
            self.ui.Ter.append('Train已完成' + str(int(MyTorch.Traincount)) + '%')
            if MyTorch.Traincount == 100:
                self.SetEnable(self.ui.BunStart)
                self.ui.Ter.append('Train成功')
                self.ui.Ter.append(str(self.stdlist))
                MyTorch.Traincount = 0
                timer2.stop()

            pass

        self.stdlist.clear()
        self.trianflag=2
        task = MyThread.MyThread(MyTorch.Train, (
            self.LightWaveTimeSecondList[self.MatrixNumber-1:len(self.LightWaveTimeSecondList)],
            self.GADFMatrixList,
            self.PressTimeList, self.PressAmplitudelogList, self.PressTimeToFix, self.LightTimeToFix))
        task.start()
        timer2 = QTimer()
        timer2.timeout.connect(fun_timerMyTorchTrain)
        timer2.start(2100)
    def BunTrainGASFclickedhandle(self):
        def fun_timerMyTorchTrain():
            if MyTorch.Traincount>=self.showtemp:
                self.showtemp=self.showtemp+self.showthr
                self.showstd()
                self.ui.Ter.append(str(int(MyTorch.Traincount / 100 * 300)))
            self.SetDisable(self.ui.BunStart)
            self.ui.Ter.append('Train已完成' + str(int(MyTorch.Traincount)) + '%')
            if MyTorch.Traincount == 100:
                self.SetEnable(self.ui.BunStart)
                self.ui.Ter.append(str(self.stdlist))
                self.ui.Ter.append('Train成功')
                MyTorch.Traincount = 0
                timer2.stop()

            pass

        self.stdlist.clear()
        self.trianflag=3
        task = MyThread.MyThread(MyTorch.Train, (
            self.LightWaveTimeSecondList[self.MatrixNumber-1:len(self.LightWaveTimeSecondList)],
            self.GASFMatrixList,
            self.PressTimeList, self.PressAmplitudelogList, self.PressTimeToFix, self.LightTimeToFix))
        task.start()
        timer2 = QTimer()
        timer2.timeout.connect(fun_timerMyTorchTrain)
        timer2.start(2100)

    def BunDatagetclickedhandle(self):
        def fun_timerMyTorchTrain():
            self.SetDisable(self.ui.BunStart)
            self.ui.Ter.append('Dataget已完成' + str(int(MyTorch4.Datacount)) + '%')
            if MyTorch4.Datacount == 100:
                self.SetEnable(self.ui.BunStart)
                self.ui.Ter.append('Dataget成功')
                MyTorch4.Datacount = 0
                timer2.stop()
            pass
        task = MyThread.MyThread(MyTorch4.Dataget, (
            self.LightWaveTimeSecondList[self.MatrixNumber - 1:len(self.LightWaveTimeSecondList)],
            self.GASFMatrixList, self.GADFMatrixList,
            self.PressTimeList, self.PressAmplitudelogList, self.PressTimeToFix, self.LightTimeToFix))
        task.start()
        timer2 = QTimer()
        timer2.timeout.connect(fun_timerMyTorchTrain)
        timer2.start(2100)

    def BunTrianAllclickedhandle(self):

        def fun_timerMyTorchTrain():
            if MyTorch4.Traincount >= self.showtemp:
                self.showtemp = self.showtemp + self.showthr
                self.showstd()
                self.ui.Ter.append(str(int(MyTorch4.Traincount / 100 * 300)))
            self.SetDisable(self.ui.BunStart)
            self.ui.Ter.append('Train已完成' + str(int(MyTorch4.Traincount)) + '%')
            if MyTorch4.Traincount == 100:
                self.SetEnable(self.ui.BunStart)
                self.ui.Ter.append(str(self.stdlist))
                self.ui.Ter.append('Train成功')
                MyTorch4.Traincount = 0
                timer2.stop()
            pass
        self.trianflag = 4

        task = MyThread.MyThread(MyTorch4.Train, (
            self.LightWaveTimeSecondList[self.MatrixNumber-1:len(self.LightWaveTimeSecondList)],
            self.GASFMatrixList,self.GADFMatrixList,
            self.PressTimeList, self.PressAmplitudelogList, self.PressTimeToFix, self.LightTimeToFix))
        task.start()
        timer2 = QTimer()
        timer2.timeout.connect(fun_timerMyTorchTrain)
        timer2.start(2100)

    def BunTestclickedhandle(self):
        if abs(self.PressTimeToFix - self.LightTimeToFix) > 36000:
            temp = self.PressTimeToFix - self.LightTimeToFix + 12 * 60 * 60
        else:
            temp = self.PressTimeToFix - self.LightTimeToFix
        self.Presstruelist = []
        self.Presstruetimelist = []
        self.Presshatlist = []
        if self.trianflag == 1:

            for i in range(20, len(self.GAFMatrixList),10):
                temp1, temp2 = MyTorch2.net(torch.tensor([self.GAFMatrixList[i - 20:i]]), MyTorch2.h_0)
                self.Presshatlist.append(temp1.detach().numpy()[0][0])
            for i in range(20, len(self.LightWaveTimeSecondList),10):
                for ii in range(len(self.PressTimeList)):
                    if self.PressTimeList[ii] + temp > self.LightWaveTimeSecondList[i]:
                        self.Presstruelist.append(self.PressAmplitudelogList[ii])
                        self.Presstruetimelist.append(self.LightWaveTimeSecondList[i])

                        break
        if self.trianflag == 2:

            self.Presshatlist = [(MyTorch.net(torch.tensor([[self.GADFMatrixList[i]]])).detach().numpy()[0][0]) for i
                                 in
                                 range(0,len(self.GADFMatrixList),10)]
            for i in range(self.MatrixNumber - 1, len(self.LightWaveTimeSecondList),10):
                for ii in range(len(self.PressTimeList)):
                    if self.PressTimeList[ii] + temp > self.LightWaveTimeSecondList[i]:
                        self.Presstruelist.append(self.PressAmplitudelogList[ii])
                        self.Presstruetimelist.append(self.LightWaveTimeSecondList[i])
                        break
        if self.trianflag == 3:

            self.Presshatlist = [(MyTorch.net(torch.tensor([[self.GASFMatrixList[i]]])).detach().numpy()[0][0]) for i
                                 in
                                 range(0,len(self.GASFMatrixList),10)]
            for i in range(self.MatrixNumber - 1, len(self.LightWaveTimeSecondList),10):
                for ii in range(len(self.PressTimeList)):
                    if self.PressTimeList[ii] + temp > self.LightWaveTimeSecondList[i]:
                        self.Presstruelist.append(self.PressAmplitudelogList[ii])
                        self.Presstruetimelist.append(self.LightWaveTimeSecondList[i])
                        break
        # if self.trianflag == 4:
        #    self.Presshatlist = [(MyTorch3.net(
        #        torch.tensor([self.Presshatlist1[i], self.Presshatlist2[20 - self.MatrixNumber + 1 + i]])).item()) for i
        #                         in
        #                         range(len(self.Presshatlist1))]
        if self.trianflag == 4:

            self.Presshatlist = [(MyTorch4.net(torch.tensor([[self.GASFMatrixList[i]]]),
                                               torch.tensor([[self.GADFMatrixList[i]]]))[0][0].data.item()) for i
                                 in
                                 range(0,len(self.GASFMatrixList),10)]
            for i in range(self.MatrixNumber - 1, len(self.LightWaveTimeSecondList),10):
                for ii in range(len(self.PressTimeList)):
                    if self.PressTimeList[ii] + temp > self.LightWaveTimeSecondList[i]:
                        self.Presstruelist.append(self.PressAmplitudelogList[ii])
                        self.Presstruetimelist.append(self.LightWaveTimeSecondList[i])
                        break

        self.Presshatlistf = savgol_filter(self.Presshatlist, 3, 1)

        x = []
        if self.trianflag == 1:
            x.append([(i) for i in self.LightWaveTimeSecondList[20:len(self.LightWaveTimeSecondList):10]])
            x.append([(i) for i in
                      self.LightWaveTimeSecondList[20:len(self.LightWaveTimeSecondList):10]])
            x.append([(i) for i in self.LightWaveTimeSecondList[20:len(self.LightWaveTimeSecondList):10]])
            #x.append([(i) for i in self.LightWaveTimeSecondList[20:len(self.LightWaveTimeSecondList):10]])
        else:
            x.append([(i) for i in
                      self.LightWaveTimeSecondList[self.MatrixNumber - 1:len(self.LightWaveTimeSecondList):10]])
            x.append([(i) for i in
                      self.LightWaveTimeSecondList[self.MatrixNumber - 1:len(self.LightWaveTimeSecondList):10]])
            x.append(self.Presstruetimelist)
            #x.append(self.Presstruetimelist)
        ymeanlist = [(self.Presstruelist[i] - self.Presshatlist[i]) for i in range(len(self.Presstruelist))]
        y = []

        y.append(self.Presshatlist)
        y.append(self.Presshatlistf)
        y.append(self.Presstruelist)
        #y.append(ymeanlist)
        sum = 0
        for i in ymeanlist:
            sum = sum + abs(i)
        std = sum / len(ymeanlist)
        self.MutiMDIDraw(x, y, '时序', '预测气压(lg(' + self.PressUnit + '))', '时序-预测气压')
        self.ui.Ter.append(str(std))

        pass
    def showstd(self):
        if abs(self.PressTimeToFix - self.LightTimeToFix) >36000:
            temp=self.PressTimeToFix - self.LightTimeToFix+ 12 * 60 * 60
        else:
            temp = self.PressTimeToFix - self.LightTimeToFix
        self.Presstruelist = []
        self.Presstruetimelist = []


        self.Presshatlist=[]
        if self.trianflag == 1:

            for i in range(20,len(self.GAFMatrixList)):
                temp1,temp2=MyTorch2.net(torch.tensor([self.GAFMatrixList[i-20:i]]),MyTorch2.h_0)
                self.Presshatlist.append(temp1.detach().numpy()[0][0])
            for i in range(20, len(self.LightWaveTimeSecondList)):
                for ii in range(len(self.PressTimeList)):
                    if self.PressTimeList[ii] + temp > self.LightWaveTimeSecondList[i]:
                        self.Presstruetimelist.append(self.LightWaveTimeSecondList[i])
                        self.Presstruelist.append(self.PressAmplitudelogList[ii])
                        break
        if self.trianflag == 2:

            self.Presshatlist = [(MyTorch.net(torch.tensor([[self.GADFMatrixList[i]]])).detach().numpy()[0][0]) for i
                                 in
                                 range(len(self.GADFMatrixList))]
            for i in range(self.MatrixNumber - 1, len(self.LightWaveTimeSecondList)):
                for ii in range(len(self.PressTimeList)):
                    if self.PressTimeList[ii] + temp > self.LightWaveTimeSecondList[i]:
                        self.Presstruetimelist.append(self.LightWaveTimeSecondList[i])
                        self.Presstruelist.append(self.PressAmplitudelogList[ii])
                        break
        if self.trianflag == 3:

            self.Presshatlist = [(MyTorch.net(torch.tensor([[self.GASFMatrixList[i]]])).detach().numpy()[0][0]) for i
                                 in
                                 range(len(self.GASFMatrixList))]
            for i in range(self.MatrixNumber - 1, len(self.LightWaveTimeSecondList)):
                for ii in range(len(self.PressTimeList)):
                    if self.PressTimeList[ii] + temp > self.LightWaveTimeSecondList[i]:
                        self.Presstruetimelist.append(self.LightWaveTimeSecondList[i])
                        self.Presstruelist.append(self.PressAmplitudelogList[ii])
                        break
        if self.trianflag ==4:

            self.Presshatlist = [(MyTorch4.net(torch.tensor([[self.GASFMatrixList[i]]]),
                                torch.tensor([[self.GADFMatrixList[i]]]))[0][0].data.item()) for i
                                 in
                                 range(len(self.GASFMatrixList))]
            for i in range(self.MatrixNumber - 1, len(self.LightWaveTimeSecondList)):
                for ii in range(len(self.PressTimeList)):
                    if self.PressTimeList[ii] + temp > self.LightWaveTimeSecondList[i]:
                        self.Presstruelist.append(self.PressAmplitudelogList[ii])
                        self.Presstruetimelist.append(self.LightWaveTimeSecondList[i])
                        break

        self.Presshatlistf=savgol_filter(self.Presshatlist,5,1)

        ymeanlist = [(self.Presstruelist[i] - self.Presshatlistf[i]) for i in range(len(self.Presstruelist))]

        sum=0
        for i in ymeanlist:
            sum=sum+abs(i)
        std=sum/len(ymeanlist)
        self.stdlist.append(std)
        self.ui.Ter.append("flag"+str(self.trianflag)+"  std:"+str(std))

        pass


if __name__ == '__main__':
    app = QApplication([])
    stats = MainWinForm()
    stats.ui.show()
    app.exec_()

# sub = QMdiSubWindow()
# pw = pyqtgraph.PlotWidget()
# # 实例化一个绘图部件
# # 向sub内部添加控件
# sub.setWidget(pw)
# sub.setWindowTitle("subWindow %d")
# self.ui.mdiArea.addSubWindow(sub)
# sub.show()