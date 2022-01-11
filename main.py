import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as waves
import csv
import pandas as pd

file = 'Dan_BYB_Recording_2022-01-07_12.02.14.wav'
fs, data = waves.read(file)

length_data=np.shape(data)
length_new=length_data[0]*0.05
ld_int=int(length_new)
from scipy import signal
data_new=signal.resample(data,ld_int)

plt.figure('Spectrogram')
d, f, t, im = plt.specgram(data_new, NFFT= 256, Fs=500, noverlap=250)
plt.ylim(0,90)
plt.colorbar(label= "Power/Frequency")
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.savefig("Figure 1")

matrixf=np.array(f).T
np.savetxt('Frequencies.csv', matrixf)
df = pd.read_csv("Frequencies.csv", header=None, index_col=None)
df.columns = ["Frequencies"]
df.to_csv("Frequencies.csv", index=False)

position_vector=[]
length_f=np.shape(f)
l_row_f=length_f[0]
for i in range(0, l_row_f):
    if f[i]>=7 and f[i]<=12:
        position_vector.append(i)

length_d=np.shape(d)
l_col_d=length_d[1]
AlphaRange=[]
for i in range(0,l_col_d):
    AlphaRange.append(np.mean(d[position_vector[0]:max(position_vector)+1,i]))


def smoothTriangle(data, degree):
    triangle=np.concatenate((np.arange(degree + 1), np.arange(degree)[::-1])) # up then down
    smoothed=[]

    for i in range(degree, len(data) - degree * 2):
        point=data[i:i + len(triangle)] * triangle
        smoothed.append(np.sum(point)/np.sum(triangle))
    # Handle boundaries
    smoothed=[smoothed[0]]*int(degree + degree/2) + smoothed
    while len(smoothed) < len(data):
        smoothed.append(smoothed[-1])
    return smoothed


plt.figure('AlphaRange')
y=smoothTriangle(AlphaRange, 100)
for i in range(10211,10433):
  y[i]=1300
for i in range(22218,22477):
  y[i]=1300
plt.plot(t,y)
plt.xlabel('Time [s]')
plt.xlim(0,max(t))
#plt.show()
plt.savefig("Figure 2")

datosy=np.asarray(y)
datosyt=np.array(
        [
        datosy,
        t
        ])
with open ('fixeddatatest.csv', 'w', newline='') as file:
    writer=csv.writer(file, dialect='excel-tab')
    writer.writerows(datosyt.T)
    

df = pd.read_csv("fixeddatatest.csv", header=None, index_col=None)
df.columns = ["Power                   Time"]
df.to_csv("fixeddatatest.csv", index=False)



tg=np.array([30, 60, 90, 120, 150, 180, 210, 240, max(t)+1])




length_t=np.shape(t)
l_row_t=length_t[0]
eyesclosed=[]
eyesopen=[]
j=0  #initial variable to traverse tg
l=0  #initial variable to loop through the "y" data
for i in range(0, l_row_t):
    if t[i]>=tg[j]:
        
        if j%2==0:
            eyesopen.append(np.mean(datosy[l:i]))
        if j%2==1:
            eyesclosed.append(np.mean(datosy[l:i]))
        l=i
        j=j+1

        
plt.figure('DataAnalysis')
plt.boxplot([eyesopen, eyesclosed], sym = 'ko', whis = 1.5)
plt.xticks([1,2], ['Eyes open', 'Eyes closed'], size = 'small', color = 'k')
plt.ylabel('AlphaPower')
plt.savefig("Figure 3")

meanopen=np.mean(eyesopen)
meanclosed=np.mean(eyesclosed)
sdopen=np.std(eyesopen)
sdclosed=np.std(eyesclosed)
eyes=np.array([eyesopen, eyesclosed])

from scipy import stats
result=stats.ttest_ind(eyesopen, eyesclosed, equal_var = False)
print(result)