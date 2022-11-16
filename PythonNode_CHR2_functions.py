"""
author src8wc
created 2022.11.16
"""

import numpy as np

#test function used to ensure numpy was working 
#and the format of arrays transfered from Labview to Python and back again.
def return_array(time):
    time = np.array(time)
    return time

#main function called, returns info about the spike analyzed
def check_for_chr2(time,voltage_mV,digital_V):
    time = np.array(time)
    digital_V = np.array(digital_V)
    voltage_mV = np.array(voltage_mV)
    detected_chr2 = 0 #default it has not found a CHR2+ response
    onset, onset_idx = find_onset(time, digital_V)
    baseline_mV = find_baseline(onset, onset_idx, time, voltage_mV)
    spike_amp, time2peak, max_mV, max_mV_timestamp, onset_timestamp = find_peak_features(onset, onset_idx, baseline_mV, voltage_mV, time)
    #determined to be CHR2+ response IF:
    #1) the spike is more then 40mV, (eliminate baseline noise or small spiking)
    #2) the response is faster then 2ms (eliminate post synaptic)
    #this is arbitrary and there are limitations if the neuron has spontaneous firing
    #which can randomly align with the stimulus onset
    if (spike_amp>40) and (max_mV_timestamp <2):
        detected_chr2 = 1
    #results are returned as a float array
    results = [max_mV, spike_amp, max_mV_timestamp, onset_timestamp, time2peak, detected_chr2] 
    return results

#function identifies the first time point where the digital output is 5V instead of 0V
def find_onset(time, digital_V):
    onset = 0
    for i in range(len(digital_V)):
            if (digital_V[i] > 4):
                onset = time[i]
                onset_idx = i
                break
    return onset, onset_idx

#function for an arbitrary baseline calculation
#in this case the average membrane potential in the 1.5 to 0.5ms proceeding the stimulus onset
def find_baseline(onset, onset_idx, time, voltage_mV):     
    baseline_mV = voltage_mV[(onset_idx-100):(onset_idx-50)]
    baseline_mV = sum(baseline_mV)/len(baseline_mV)
    return baseline_mV

#function that identifies features of spiking timelocked to the stim onset
def find_peak_features(onset, onset_idx, baseline_mV, voltage_mV, time):
    waveform = voltage_mV[onset_idx:(onset_idx+500)] #define event analysis window, in this case from onset to 10ms after, or the entire stimulus duration
    max_mV = max(waveform)#memberane potential at spike peak
    max_mV_idx = np.argmax(waveform)
    spike_amp = max_mV - baseline_mV
    time2peak = time[max_mV_idx] - time[onset_idx]
    max_mV_timestamp = time[max_mV_idx] #of the analyzed event waveform, so it is actually time from onset to peak
    onset_timestamp = time[onset_idx]
    return spike_amp, time2peak, max_mV, max_mV_timestamp, onset_timestamp