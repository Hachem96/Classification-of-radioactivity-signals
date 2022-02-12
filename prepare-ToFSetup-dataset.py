import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.signal import find_peaks,gaussian,correlate
from matplotlib.pyplot import figure
import nbimporter
from Common_Function import alignment_Pulses,Qtail_Qtots,save_data,get_start_point
from sklearn.mixture import GaussianMixture

def main_function(path_file,DetectorParameter):
    OutputAllFiles = all_files_preparation(path_file,DetectorParameter)
    CreatePathAndSave(path_file,OutputAllFiles,DetectorParameter)
    return
	
def CreatePathAndSave(path_file,OutputDictionary,DetectorParameter):
    stdRejectionFactor = DetectorParameter['StdRejectionFactor']
    ScintillatorName = DetectorParameter['ScintillatorName']
    for x in OutputDictionary:
        temp = np.asarray(OutputDictionary[x])
        path = "{}\\{}\\{}_{}_{}STD.csv".format(path_file,ScintillatorName,ScintillatorName,x,stdRejectionFactor)
        save_data(temp,path,separator=DetectorParameter['separator'])
    return

def all_files_preparation(path_file,DetectorParameter):
    start_index = DetectorParameter['StartFileNumber']
    end_index = start_index + DetectorParameter['NumberofFiles']  
    ScintillatorName = DetectorParameter['ScintillatorName']
    OutputAllFiles = {'PulsesC1': [],
                'PulsesC2': [],
                'y': [],
                'tof': [],
                'Pileup1' : [],
                'Pileup2': []}
    for i in range(start_index,end_index):
        FileIndex = "0"*(5-len(str(i))) + str(i)
        path_c1 = "{}\\{}\\c1--{}--{}.txt".format(path_file,ScintillatorName,ScintillatorName,FileIndex)
        path_c2 = "{}\\{}\\c2--{}--{}.txt".format(path_file,ScintillatorName,ScintillatorName,FileIndex)
        OutputOneFile =  PrepareOneFile(path_c1,path_c2,DetectorParameter)
        for x in OutputOneFile:
            ListAppend(OutputOneFile[x],OutputAllFiles[x])
        print(i,'number of remaining signals: ',OutputOneFile['PulsesC1'].shape[0])
        print('number of pile-up in C1: ',OutputOneFile['Pileup1'].shape[0])
        print('number of pile-up in C2: ', OutputOneFile['Pileup2'].shape[0])
        print('----------------------------------------------------')
    return OutputAllFiles

def ListAppend(Input,Output):
    for item in Input:
        Output.append(item) #np.reshape(item,(1,len(item))))
    return 

def PrepareOneFile(path_c1,path_c2,DetectorParameter):
    
    # read files
    datac1 = separate_signals(path_c1,DetectorParameter)
    datac2 = separate_signals(path_c2,DetectorParameter)
    #
    try:
        RemoveNoise(datac1,datac2,DetectorParameter)
        OutputOneFile =  Detect_Separate_Pileup(datac1,datac2,DetectorParameter)
    except ValueError:
        print('all the pulses are detected as pileup and noise')
    #
    OutputOneFile['tof'] = compute_tof(datac1,datac2)
    
    return removeUncertaintyPulses(OutputOneFile,DetectorParameter)

def removeUncertaintyPulses(OutputOneFile,DetectorParameter):
    stdRejectionFactor = DetectorParameter['StdRejectionFactor']
    y_kmean,mean_n,mean_g,std_n,std_g = classifyPulses(OutputOneFile['tof'])
    tof = OutputOneFile['tof']
    idx_neutron = (y_kmean==1) & (tof>=(mean_n - stdRejectionFactor*std_n)) & (tof<=(mean_n + stdRejectionFactor*std_n))
    idx_neutron = np.argwhere(idx_neutron==True)
    #
    idx_gamma = (y_kmean==0)  & (tof<=(mean_g + stdRejectionFactor*std_g)) & (tof>=(mean_g - stdRejectionFactor*std_g))
    idx_gamma = np.argwhere(idx_gamma==True)
    #
    all_index = np.union1d(idx_gamma,idx_neutron)
    OutputOneFile['PulsesC1'] = OutputOneFile['PulsesC1'][all_index,:]
    OutputOneFile['PulsesC2'] = OutputOneFile['PulsesC2'][all_index,:]
    OutputOneFile['tof'] = OutputOneFile['tof'][all_index]
    OutputOneFile['y'] = y_kmean[all_index]
    return OutputOneFile

def compute_tof(datac1,datac2):
    tmax1 = np.argmax(datac1['amplitude'],axis=1)
    tmax2 = np.argmax(datac2['amplitude'],axis=1)
    tof = []
    for i in range(0,len(datac1['amplitude'])):
        tof.append(2*(datac1['time'][i,tmax1[i]] - datac2['time'][i,tmax2[i]]))
    return np.asarray(tof)

def classifyPulses(tof):
    mean_n = 10
    mean_g = 20
    kmeans = KMeans(n_clusters=2,n_init=500,tol=1e-15,max_iter=1000,).fit(np.reshape(tof,(-1,1)))
    y_kmean =  kmeans.labels_
    mean_g,mean_n = kmeans.cluster_centers_
    mean_g = mean_g[0]
    mean_n = mean_n[0]
    if(mean_n<mean_g):
        y_kmean = 1 - y_kmean
        temp = mean_n
        mean_n = mean_g
        mean_g = temp
    std_n = tof[np.argwhere(y_kmean==1)].std()
    std_g = tof[np.argwhere(y_kmean==0)].std()
    return y_kmean,mean_n,mean_g,std_n,std_g
	
def separate_signals(path_file,DetectorParameter):
    separator = DetectorParameter['separator']
    PulseNumber,data_file = read_file(path_file,separator)
    PulseStartIndex = 0
    data = {'amplitude': [],
            'time': []}
    PulseCounter = 0
    for i in range(0,data_file.shape[0]-1):
        if (PulseCounter == (PulseNumber-1)):
#             data['amplitude'].append(data_file[i:,1])
#             data['time'].append(data_file[i:,0])
            break
        else:
            if(data_file[i+1,0]<data_file[i,0]):
                PulseEndIndex = i+1
                data['amplitude'].append(data_file[PulseStartIndex:PulseEndIndex,1])
                data['time'].append(data_file[PulseStartIndex:PulseEndIndex,0])
                PulseCounter += 1 
                PulseStartIndex  = i + 1 
    data['amplitude'] = np.asarray(data['amplitude'])
    data['time'] = np.asarray(data['time'], order='C')
    
    return data 
	
def read_file(path_file,separator=','):
    data_file = pd.read_csv(path_file,sep=separator,dtype = str,index_col=False)
    PulseNumber = int(data_file.values[0,1])
    data_file = data_file.values[PulseNumber+3:,0:2].astype(float)
    data_file[:,1] = -data_file[:,1]
    return PulseNumber,data_file

def RemoveNoise(datac1,datac2,DetectorParameter):
    # unwanted event indx
    indexNoiseEvents = all_Noise_idx(datac1['amplitude'],datac2['amplitude'],DetectorParameter)
    for x in datac1:
        datac1[x] = np.delete(datac1[x],indexNoiseEvents,axis=0)
        datac2[x] = np.delete(datac2[x],indexNoiseEvents,axis=0)
    return 
	
def all_Noise_idx(pulses_1,pulses_2,DetectorParameter):
    idx_1 = idx_Noise(pulses_1,DetectorParameter)
    idx_2 = idx_Noise(pulses_2,DetectorParameter)
    return np.union1d(idx_1,idx_2).astype(int)
	
def idx_Noise(pulses,DetectorParameter):
    index = []
    for i in range(0,pulses.shape[0]):
        pulse_max = np.max(pulses[i,:])
        t_max = np.argmax(pulses[i,:])
        if(pulse_max>=DetectorParameter['maximumAmplitude']) or (pulse_max<=DetectorParameter['triggerThreshold']):
            index.append(i)
        elif (t_max<DetectorParameter['averageRiseTime']) or (t_max>(pulses.shape[1]-DetectorParameter['DecayTime'])):
            index.append(i)
    index = np.asarray(index)    
    return index

def Detect_Separate_Pileup(datac1,datac2,DetectorParameter):
    OutputOneFile = {'PulsesC1': [],
                'PulsesC2': [],
                'Pileup1' : [],
                'Pileup2': []}
    
    indexPileup1,indexPileup2 = all_pileup_idx(datac1['amplitude'],datac2['amplitude'],DetectorParameter)
    indexPileup = np.union1d(indexPileup1,indexPileup2)
    OutputOneFile['Pileup1'] = datac1['amplitude'][indexPileup1,:]
    OutputOneFile['Pileup2'] = datac2['amplitude'][indexPileup2,:]
    for x in datac1:
        datac1[x] = np.delete(datac1[x],indexPileup,axis=0)
        datac2[x] = np.delete(datac2[x],indexPileup,axis=0)
    OutputOneFile['PulsesC1'] = datac1['amplitude']
    OutputOneFile['PulsesC2'] = datac2['amplitude']
    #
    return OutputOneFile
	
def all_pileup_idx(pulses_1,pulses_2,DetectorParameter):
    indexPileup1 = idx_pileup(pulses_1,DetectorParameter)
    indexPileup2 = idx_pileup(pulses_2,DetectorParameter)
    return indexPileup1,indexPileup2
	
def idx_pileup(pulses,DetectorParameter):
    index = []
    window = gaussian(pulses.shape[1],DetectorParameter['GaussianWidth'])
    for i in range(pulses.shape[0]):
#         idx = np.argwhere(pulses[i,:]>=DetectorParameter['triggerThreshold'])
#         idx = np.min(idx)
        corr = correlate(pulses[i,:], window, mode='same')
        corr = (corr -np.min(corr))/(np.max(corr) - np.min(corr))
        peaks, _ = find_peaks(corr, height=DetectorParameter['TriggerPercentage'],distance = 1,width=1)
        count = 0
        if(len(peaks)>=2):
            for peak in peaks:
                if pulses[i,peak]>=DetectorParameter['triggerThreshold']:
                    count += 1
        if count >=2:
            index.append(i)
    index = np.asarray(index)
    return index.astype(int)
	