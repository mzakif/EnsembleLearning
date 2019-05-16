import numpy as np
from sklearn.naive_bayes import GaussianNB
import collections

TrainData = np.loadtxt('TrainsetTugas4ML.csv', skiprows=1, delimiter=',')

#split data
def splitData(data):
    x = data.T[:2].T
    y = data.T[-1].T
    return x,y

#Test dengan Naive Bayes Gussian
def TestTrain(modelBoostrap):
    data, kelas = splitData(modelBoostrap)
    clf = GaussianNB()
    hasil = clf.fit(data, kelas)
    return hasil

#mengambil data acak masuk ke boostrap
def RandInData(dataset):
    bootstrap = np.zeros(dataset.shape)
    for i in range(dataset.shape[0]):
        nomer = np.random.randint(dataset.shape[0])
        bootstrap[i] = dataset[nomer]
    return bootstrap

#membuat bootstrap
def ujiCoba(count,data1):
    Bootstrap = np.zeros((count, data1.shape[0], data1.shape[1]))
    return Bootstrap

#--------UJI AKURASI MODEL------------------------------
TrainFull = TrainData.T[:,:220].T
Train_x = TrainData.T[:2,220:].T
Train_y = TrainData.T[-1:,220:].T

BootstrapCount = 15
BootstrapUji = ujiCoba(BootstrapCount,TrainFull)

modelUji = []
for i in range(BootstrapUji.shape[0]):
    BootstrapUji[i] = RandInData(TrainFull)
    modelUji.append(TestTrain(BootstrapUji[i]))

outputsUji = []
for model in modelUji:
    outputUji = model.predict(Train_x)
    outputsUji.append(outputUji)
outputsUji = np.array(outputsUji)

most2 = []
for i, outputUji in enumerate(outputsUji.T):
    count = collections.Counter(outputUji)
    most = count.most_common(1)[0][0]
    most2.append(int(most))

#menghitung akurasi
from sklearn.metrics import r2_score
score = r2_score(most2, Train_y)
score = (score)*100
score = 100+int(score)
print('Nilai akurasi model : ', score)

#-----------Selesai UJI-----------------------------

#load data test
TestData = np.genfromtxt('TestsetTugas4ML.csv', delimiter=',')[1:] 

BootstrapCount = 15
Bootstrap = ujiCoba(BootstrapCount,TrainData)

Test_X, Test_Y = splitData(TestData)

HasilModel = []
for i in range(Bootstrap.shape[0]):
    Bootstrap[i] = RandInData(TrainData)
    HasilModel.append(TestTrain(Bootstrap[i]))  
    
KeluarClass = []
for model in HasilModel:
    output = model.predict(Test_X)
    KeluarClass.append(output)
KeluarClass = np.array(KeluarClass)

for i, output in enumerate(KeluarClass.T):
    count = collections.Counter(output)
    Hasil = count.most_common(1)[0][0]
    TestData[i][2] = int(Hasil)

np.savetxt('TebakanTugas4ML.csv', TestData, delimiter=',')