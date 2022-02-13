import Common_Function, Data_preparation,ANN_concept
import pandas as pd
import numpy as np 
from Common_Function import read_file,Qtail_Qtots,save_data,alignment_Pulses
import seaborn as sb
import matplotlib.pyplot as plt
from scipy.signal import find_peaks,gaussian,correlate

def main(path_signals,path_y,path_gamma_source,outpath,DetectorParameter,AlignmentParameters):
    MixedSourcedata,idx_n,idx_g =  read_prepare_data(path_signals,path_y)
    #
    QtailQtot_neutron,QtailQtot_gamma =computeQtailQtot(MixedSourcedata,idx_n,idx_g,DetectorParameter)
    #
    NeutronThreshold = ComputeNeutronThreshold(QtailQtot_neutron)
    GammaThreshold = ComputeGammaThreshold(path_gamma_source,DetectorParameter,AlignmentParameters)
    #
    idxTrueNeutron = np.argwhere(QtailQtot_neutron>NeutronThreshold)[:,0]
    idxTrueGamma = np.argwhere(QtailQtot_gamma<GammaThreshold)[:,0]
    idxMislabeledNeutron = np.argwhere(QtailQtot_neutron<NeutronThreshold)[:,0]
    idxMislabeledGamma = np.argwhere(QtailQtot_gamma>GammaThreshold)[:,0]
    #
    yTrueneutron = idx_n[idxTrueNeutron]
    yTruegamma = idx_g[idxTrueGamma]
    yMislabeledNeutron = idx_n[idxMislabeledNeutron]
    yMislabeledGamma = idx_g[idxMislabeledGamma]
    label = ['Neutron','Gamma','MislabeledNeutron','MislabeledGamma']
    data = [MixedSourcedata[yTrueneutron,:],MixedSourcedata[yTruegamma,:],MixedSourcedata[yMislabeledNeutron,:],MixedSourcedata[yMislabeledGamma,:]]
    for i in range(0,len(label)):
        pathNeutron = "{}\\-{}STD.csv".format(outpath,label[i],DetectorParameter['StdRejectionFactor'])
        
        save_data(data[i],pathNeutron)
    return
	
def read_prepare_data(path_signals,path_y):
    MixedSourcedata = read_file(path_signals)
    y = read_file(path_y)
    idx_n = np.argwhere(y==1)[:,0]
    idx_g = np.argwhere(y==0)[:,0]
    return MixedSourcedata, idx_n,idx_g
	
def computeQtailQtot(MixedSourcedata,idx_n,idx_g,DetectorParameter):
    _, _, QtailQtotMixed = Qtail_Qtots(MixedSourcePulse,DetectorParameter)
    QtailQtot_neutron = QtailQtotMixed[idx_n]
    QtailQtot_gamma = QtailQtotMixed[idx_g]
    return QtailQtot_neutron,QtailQtot_gamma

def ComputeNeutronThreshold(QtailQtot_neutron,perc_density_max = 0.3):
    temp = sb.distplot(QtailQtot_neutron,bins=1000).get_lines()[0].get_data()
    plt.close(0)
    density = temp[1]
    values = temp[0]
    peak = perc_density_max*np.max(density)
    peak = np.argwhere(density[0:np.argmax(density)]<=peak)
    peak = np.max(peak) 
    NeutronThreshold = values[peak]
    return NeutronThreshold
	
def ComputeGammaThreshold(path_gamma_source,DetectorParameter,AlignmentParameters):
    GammaSourcePulse = read_file(path_gamma_source)
    GammaSourcePulse = alignment_Pulses(GammaSourcePulse,AlignmentParameters)
    _,_,QtailQtotPureGamma = Qtail_Qtots(GammaSourcePulse,DetectorParameter)
    GammaThreshold = np.mean(QtailQtotPureGamma) + 3*np.std(QtailQtotPureGamma)
    return GammaThreshold
	