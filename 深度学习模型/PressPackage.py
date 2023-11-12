import re
class Press:
    def __init__(self,file):
        with open(file, 'r', encoding='utf8') as pressfile_object:
            presscontents = pressfile_object.read()
            pressfile_object.close()
        self.IntPressTimeList=[]
        self.IntPressDataList = []
        self.IntPressTimeSecondList = []
        self.StrPressunit =''
        for one in re.findall(r'Units:,(.*) A', presscontents):
            self.StrPressunit = one
        for one in re.findall(r'-(\d*) (\d*):(\d*):(\d*)\.(\d*)\s\w\w,(\d*).(\d{5})', presscontents):
            self.IntPressTimeList.append(int(one[0])*1000*60*60*24+int(one[1])*1000*60*60+int(one[2])*1000*60+int(one[3])*1000+int(one[4]))
            self.IntPressDataList.append(int(one[5])+int(one[6])*0.00001)
        self.IntPressTimeSecondList=[(i-self.IntPressTimeList[0])/1000 for i in self.IntPressTimeList]
        self.TimeStart = str(int(self.IntPressTimeList[0] / 1000 / 60 / 60 / 24)) + '号' + str(
            int(self.IntPressTimeList[0] / 60 / 60 / 1000) % 24) + '时' + str(
            int(self.IntPressTimeList[0] / 1000 / 60) % 60) + '分' + str(
            int(self.IntPressTimeList[0] / 1000) % 60) + '.' + str(int(self.IntPressTimeList[0]) % 1000) + '秒'
        self.TimeEnd = str(int(self.IntPressTimeList[-1] / 1000 / 60 / 60 / 24)) + '号' + str(
            int(self.IntPressTimeList[-1] / 60 / 60 / 1000) % 24) + '时' + str(
            int(self.IntPressTimeList[-1] / 1000 / 60) % 60) + '分' + str(
            int(self.IntPressTimeList[-1] / 1000) % 60) + '.' + str(int(self.IntPressTimeList[-1]) % 1000) + '秒'

if __name__ == '__main__':
    t=Press('c.csv')