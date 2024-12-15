import matplotlib.pyplot as plt
import numpy as np 
import pandas  as pd
from pandas import DataFrame
from scipy.signal import butter,lfilter,resample,detrend,medfilt

# this file is used extract time-dominated voltage gathered by node Observatory
# The entire process consists of reducing the sampling rate and identifying ...
# the signal caused by the previous electrode in each channel.
# Finally, the average value of the selected signals in a certain range is extracted.






def butter_lowpass_filter(data,cutoff_freq,sample_rate,order=4):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b,a = butter(order,normal_cutoff,btype='low',analog=False)
    
    filter_data = lfilter(b,a,data)
    return filter_data

# used after above function
def desample(data, target_rate, sample_rate):
    # after low-pass filter and resample
    resample_ration = target_rate / sample_rate 
    # low-pass
    # actually lower-pass can't cut frequen after xxx, so need smaller than u nyquist frequency
    filt_data = butter_lowpass_filter(data, target_rate / 4 , sample_rate, order=4)
    # resample
    resample_data = resample(filt_data,int(len(filt_data) * resample_ration))
    
    return resample_data

def seri_maxpoint(data):
    if  isinstance(data,list):
        data = np.array(data)
    
    seri_1 = data[:-2]
    seri_2 = data[1:-1]
    seri_3 = data[2:]
    max_index = (seri_2 > seri_1) & (seri_2 > seri_3)
    max_value = seri_2[max_index]
    return max_value


def seek_block(signal):
    """Look for the beginning and end of the  signal cluster caused by previous electrode"""
    
    # cause the subtract of each one can elimeter the trendcy and undulation
    signal_diff = detrend(np.diff(signal))
    
    # suppress the random micro wave
    micro_wave_intensity = 5
    
    signal_diff_up = np.where(signal_diff < micro_wave_intensity, 0, signal_diff)
    signal_one = np.sign(signal_diff_up)
    
    # calculate some paramaters to inspect which wave occurance is injected current signal
    signal_start = [0]
    for iteration in range(2):
        
        signal_temporary_one =  signal_one[signal_start[-1]:]
        
        window_index = np.linspace(0,signal_temporary_one.size,9).astype(np.int32)
        criteria = []
        for index in zip(window_index[0:-1], window_index[1:]):

            s_norm = np.linalg.norm(signal_temporary_one[index[0]:index[1]])   # L2 norm 
            s_sum = np.sum(signal_temporary_one[index[0]:index[1]])            # polar mutation counts
            s_std = np.std(signal_temporary_one[index[0]:index[1]])             # standard
            s_distance = np.diff(np.argwhere(signal_temporary_one[index[0]:index[1]] == 1).ravel())  # polar singal relative distance(sample point)
            if s_distance.size:
                s_distance = np.mean(s_distance)
                
            else:
                s_distance = 1e5

            criteria.append([s_norm,s_sum,s_std,s_distance]) 
        criteria = np.array(criteria)

        # some index of above parameter
        ####################
        #### IMPORTANT ######
        #####################
        ct_norm = 9
        ct_sum = 70
        ct_std = 0.1
        ct_distance = 50 
        ###########################

        meet_criteria = (criteria[:,0] > ct_norm) & (criteria[:,1] > ct_sum) & (criteria[:,2] > ct_std) & (criteria[:,3] < ct_distance)
        if np.any(meet_criteria):
            meet_index = window_index[:-1][meet_criteria][0]
            signal_start.append(meet_index)
        else:
            raise ValueError("Singal seems doesn't contain useful information ")
            
    
    signal_pick_start = np.sum(signal_start).astype(np.int32)
    
    # also care positive value which more stably and less debries
    signal_diff_up_rest = signal_diff_up[signal_pick_start:]         
    signal_diff_up_rest_max = seri_maxpoint(signal_diff_up_rest)
    
    signal_diff_up_rest_max_1 = signal_diff_up_rest_max[:-2]
    signal_diff_up_rest_max_2 = signal_diff_up_rest_max[1:-1]
    signal_diff_up_rest_max_3 = signal_diff_up_rest_max[2:]
    
    # Look for the location of sudden changes in signal intensive
    signal_shift_index = (signal_diff_up_rest_max_2 * 1.5 > signal_diff_up_rest_max_1) & (signal_diff_up_rest_max_2*1.5 > signal_diff_up_rest_max_3)
    signal_shift_point = signal_diff_up_rest_max_2[~signal_shift_index][0]
    signal_shift_end  = np.where(signal_shift_point == signal_diff_up)[0]
    
    signal_shift_next_point = signal_diff_up_rest_max[np.where(np.isin(signal_diff_up_rest_max,signal_shift_point))[0] + 1]
    signal_shift_next_start = np.where(signal_shift_next_point == signal_diff_up)[0]
    
    # for confirm the end position is correct or not, we should plot more singl data
    signal_shift_softend  = np.floor(0.5 * (signal_shift_end + signal_shift_next_start)).astype(np.int32)
    # we loact the signal start position    
    signal_shift_start = np.where(signal_diff_up_rest_max[0]==signal_diff_up)[0] 
    
    # in here, we only located the first signal block tail position.  

    return signal_shift_start[0],signal_shift_softend[0]


def average_squwave(data):
    # remove mean and trendy
    data = data - np.mean(data)
    data = detrend(data)
    # suppose max is positive
    data_max_lower = np.max(data) * 0.8
    #suppose min is negative
    data_min_lower = np.min(data) * 0.8
    
    data_positive = np.abs(np.mean(data[data > data_max_lower]))
    data_negative = np.abs(np.mean(data[data < data_min_lower]))
    
    return 0.5 * (data_negative + data_positive)


def medfilt_builtin(data,kernel_size=31):
    return medfilt(data, kernel_size)

def solo2voltage(data:pd.DataFrame,
                 target_rate=100,
                 sample_rate=1000,
                 medfilt_size=31,
                ):
    """ this function inverte the time series of potential differenc into fixed value based on certain electrode.
        
        Args:
             data : is a large table of panda, m*n, m means sample value and n means the total number of observatories.
             target_rate : Target sample rate
             sample_rate : original sample rate
             medfilt_size : the medfilt window length. aims to surpress the pulse wave. But size choice need hit.it doesn't mean 
                            the window more large, the better qurality you get.
        Note:
            for directive current resistivity method, hight sampale isn't necessart requirment,so we need diminish sample rate
            
            
    """
    from scipy.signal import butter,lfilter,resample,detrend,medfilt
    
    stations = data.columns.size
    Trest = []
    
    # record each trace signal staring / ending index
    T_start =[]  
    T_end =[]
    
    # reduce "sample rate", "drift", "random pulse wave"
    # kernel_size parmater  is knack, 21 is content withe 100 sample rate
    data_in = np.array([ medfilt_builtin(detrend(desample(data.values[:,i],target_rate,sample_rate)),medfilt_size) for i in range(0,stations,1)])
    sg = data_in
    
    for pointer in range(0,stations):
        
        signal_block_st, signal_block_end = seek_block(sg[pointer])
        choice_part = sg[:,signal_block_st : signal_block_end]
        sg = sg[:,signal_block_end:]
        volt = [average_squwave(sub_sg) for sub_sg in choice_part]  
        Trest.append(volt)
        T_end.append(signal_block_end)
        T_start.append(signal_block_st)
        
    return Trest
