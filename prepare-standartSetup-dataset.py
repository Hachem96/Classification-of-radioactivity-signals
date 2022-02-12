import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.signal import find_peaks,gaussian,correlate
from matplotlib.pyplot import figure
from scipy import integrate

from Common_Function import alignment_Pulses,read_file,Qtail_Qtots,save_data,sampling_reduce,equivalentEnergy
from Data_preparation import separate_signals,idx_Noise,idx_pileup,ListAppend,CreatePathAndSave

def main_function(path_file,DetectorParameter):
    OutputAllFiles = all_files_preparation(path_file,DetectorParameter)
    CreatePathAndSave(path_file,OutputAllFiles,DetectorParameter)
    return
	
def all_files_preparation(path_file,DetectorParameter):
    start_index = DetectorParameter['StartFileNumber']
    end_index = start_index + DetectorParameter['NumberofFiles']  
    ScintillatorName = DetectorParameter['ScintillatorName']
    OutputAllFiles = {'Pulses': [],
                'Pileup': []}
    for i in range(start_index,end_index):
        FileIndex = "0"*(5-len(str(i))) + str(i)
        path_c1 = "{}\\{}\\c1--{}--{}.txt".format(path_file,ScintillatorName,ScintillatorName,FileIndex)
        OutputOneFile =  PrepareOneFile(path_c1,DetectorParameter)
        for x in OutputOneFile:
            ListAppend(OutputOneFile[x],OutputAllFiles[x])
        print(i,'number of remaining signals: ',OutputOneFile['Pulses'].shape[0])
        print('number of pile-up in C1: ',OutputOneFile['Pileup'].shape[0])
        print('----------------------------------------------------')
    return OutputAllFiles

def PrepareOneFile(path_c1,DetectorParameter):
    OutputOneFile = {'Pulses': [],
                    'Pileup': []}
    datac1 = separate_signals(path_c1,DetectorParameter)
    PulsesC1 =  RemoveNoise(datac1,DetectorParameter)
    #PulsesC1 = sampling_reduce(PulsesC1) 
    OutputOneFile['Pileup'],OutputOneFile['Pulses'] =  Detect_Separate_Pileup(PulsesC1,DetectorParameter)
    return OutputOneFile

def RemoveNoise(datac1,DetectorParameter):
    index_noise = idx_Noise(datac1['amplitude'],DetectorParameter)
    if len(index_noise)!=0:
        return np.delete(datac1['amplitude'],index_noise,axis=0)
    else:
        return datac1['amplitude']
	
def Detect_Separate_Pileup(Pulses,DetectorParameter):
    indexPileup = idx_pileup(Pulses,DetectorParameter)
    return Pulses[indexPileup,:],np.delete(Pulses,indexPileup,axis=0)
	
