import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy import integrate
from scipy import stats
from scipy.integrate import newton_cotes

def read_file(path,separator=','):
    data = pd.read_csv(path,sep=separator,header=None)
    return data.values
	
def save_data(data,path,separator=','):
    data_frame = pd.DataFrame(data)
    data_frame.to_csv(path,sep=separator,header=False,index=False)
    return
	
def alignment_Pulses(Pulses,AlignmentParameters):
    rise_time = riseTime(Pulses,AlignmentParameters['TriggerPercentage'])
    #print(rise_time)
    new_Pulses = np.zeros((Pulses.shape[0],AlignmentParameters['PulseDuration']))
    for i in range(0,Pulses.shape[0]):
        StartPoint = get_start_point(Pulses[i,:],AlignmentParameters['TriggerPercentage'],int(2*rise_time))
        if(AlignmentParameters['MaxStartPoint'] == True):
            tmax = np.argmax(Pulses[i,:])
        else:
            tmax = StartPoint #tmax - rise_time
        temp = Pulses[i,tmax:tmax + PulseDuration] if(tmax<PulseDuration) else Pulses[i,tmax:]
        if AlignmentParameters['baseline']>0:
            if (StartPoint >=  AlignmentParameters['baseline']):
                temp = temp - np.mean(Pulses[i,StartPoint- AlignmentParameters['baseline']:StartPoint])
            else:
                if StartPoint>0:
                    temp = temp - np.mean(Pulses[i,0:StartPoint])
        if(len(temp) <= new_Pulses.shape[1]):
            new_Pulses[i,0:len(temp)] = temp
        else:
            new_Pulses[i,:] = temp[0:new_Pulses.shape[1]]
    return new_Pulses
	
def riseTime(Pulses,TriggerPercentage):
    rise_time = []
    for i in range(0,Pulses.shape[0]):
        t_max = np.argmax(Pulses[i,:])
        start_point = get_start_point(Pulses[i,:],TriggerPercentage,Pulses.shape[1])
        rise_time.append(t_max-start_point)
    mode_info = stats.mode(rise_time)
    return int(mode_info[0])

def get_start_point(pulse,TriggerPercentage,rise_time):
    t_max = np.argmax(pulse)
    if(t_max>rise_time):
        start_points = np.argwhere(pulse[t_max-rise_time:t_max] >= TriggerPercentage*pulse[t_max]) + t_max-rise_time
    else:
        start_points = np.argwhere(pulse[0:t_max]>= TriggerPercentage*pulse[t_max])
    if(len(start_points)!=0):
        StartPoint = np.min(start_points)
    elif(rise_time == len(pulse)): 
            StartPoint = t_max
    else:
        StartPoint = int(t_max - rise_time/2)  
    return StartPoint

def Qtail_Qtots(Pulses,DetectorParameter):
    Qtail =[]
    Qtot = []
    tail_tot = []
    for i in range(0,Pulses.shape[0]):
        signal = Pulses[i,:]
        t_max = np.argmax(signal)
        upper_idx = DetectorParameter['LongGate'] 
        qtot = integrate.simpson(signal[t_max:t_max+upper_idx])# 
        lower_idx =  DetectorParameter['ShortGate'] 
        qtail = integrate.simpson(signal[t_max+upper_idx - lower_idx:t_max+upper_idx])
        tail_tot.append(qtail/qtot)
        Qtot.append(qtot)
        Qtail.append(qtail)
    Qtot = np.asarray(Qtot)
    Qtail = np.asarray(Qtail)
    tail_tot = np.asarray(tail_tot)
    return  Qtail,Qtot,tail_tot
