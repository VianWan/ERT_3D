import ipywidgets as widgets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from ipywidgets import interact
from typing import List
# __all__=['CLEAR_INDEX']
class ProcessData():

    CLEAR_INDEX = None

    @classmethod
    def pretreat_data(cls,data) -> np.array:
        """ we follow the basic rule that acquired voltage from the next potential electrode always smaller than forward one.
        Args:
             input parameter is DataFrame type, which original file type should comply with the example file pattern. 
        
       Return : Returns the number of rows of unqualified acquisition data in the raw data.
        
        """
        # global CLEAR_INDEX
        # Create an Output widget
        output = widgets.Output()
        
        data_inner, eletransmit = cls.fun_kit(data)
        observe_data = data_inner

        # Sample data
        obs_upper = observe_data.max() + 1 * np.abs(observe_data.min())
        obs_lower = observe_data.min() - 0.5 * observe_data.min()

        # Create FloatRangeSlider and Dropdown widgets
        range_slider = widgets.FloatRangeSlider(
            value=[obs_lower,obs_upper],
            min=obs_lower,
            max=obs_upper,
            step=0.001,
            description='R0_fit',
            disabled=False,
            continuous_update=False,
            orientation='vertical',
            readout=True,
            readout_format='.1f',
        )
        
        
        iters_drop_down = widgets.Dropdown(
        options = (np.arange(0,8)),
        value = 0,
        description='Iter_num',
        disabled=False,

    )  
        


        

        def range_sub_fun(change,data_inner,eletransmit):
        
            # global CLEAR_INDEX

            x = np.arange(0, data_inner.size)
            lower, upper = change.new
            iternum = iters_drop_down.value

            iter_clear_index = cls.iter_filter(data_inner,iter_num=iternum,ele_transmit=eletransmit)
            iter_fine_index = [seri_in for seri_in in x if seri_in not in iter_clear_index ] # acquist  meeting the role of attenuation voltage index 

            selected_data_ma = np.ma.masked_where((data_inner < lower) | (data_inner > upper),data_inner)

            selected_data = data_inner[~selected_data_ma.mask].ravel()
            selected_data_index = x[~selected_data_ma.mask].ravel() # acquist data that satisfies the signal level



            clear_index_dict = set( iter_fine_index ) & set(selected_data_index)
            cls.CLEAR_INDEX = np.sort([*clear_index_dict]).astype(np.int32)

            # plot section
            fig, axs = plt.subplots(ncols=2, nrows=1,figsize=(10,5))

            with output:
                clear_output(wait= True)  # Clear previous output

                axs[0].semilogy(x, data_inner,ls='-',color='#9A3B3B')
                axs[0].axhline(y=lower)
                axs[0].axhline(y=upper)

                # plot original data 
                axs[0].scatter(x,data_inner,marker='^',color='#F45050',alpha=0.7,label='raw data')

                # plot the selected data
                axs[0].scatter(x,selected_data_ma ,s=50,color='#004225',alpha=0.4,label='within data')

                # plot the iter_clear data
                axs[0].scatter(x[iter_fine_index], data_inner[iter_fine_index],s=60,marker='*',color='#92C7CF',alpha=0.7,label='iter data')


                # decoration axs[0]
                axs[0].margins(x=0.3,y=0.2)
                axs[0].set_ylim(bottom= np.log10(np.abs(data_inner)).min())
                axs[0].set_facecolor('#F6FDC3')
                axs[0].legend()



                axs[1].semilogy(x[cls.CLEAR_INDEX], data_inner[cls.CLEAR_INDEX],c='C7')
                axs[1].set_ylim([data_inner.min(), data_inner.max()])
                axs[1].grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
                axs[1].set_facecolor('#F9F7C9')
                plt.show()
            
            


        def drop_down_fun(change,data_inner, eletransmit):
        
            # global CLEAR_INDEX
            x = np.arange(0, data_inner.size)
            lower, upper = range_slider.value
            iternum = change.new

            iter_clear_index = cls.iter_filter(data_inner,iter_num=iternum,ele_transmit=eletransmit)
            iter_fine_index =  [seri_in for seri_in in x if seri_in not in iter_clear_index ]# acquist  meeting the role of attenuation voltage index 

            selected_data_ma = np.ma.masked_where((data_inner < lower) | (data_inner > upper),data_inner)

            selected_data = data_inner[~selected_data_ma.mask].ravel()
            selected_data_index = x[~selected_data_ma.mask].ravel() # acquist data that satisfies the signal level

            clear_index_dict = set( iter_fine_index ) & set(selected_data_index)
            cls.CLEAR_INDEX = np.sort([*clear_index_dict]).astype(np.int32)


            fig, axs = plt.subplots(ncols=2, nrows=1,figsize=(10,5))

            with output:
                clear_output(wait= True)  # Clear previous output

                axs[0].semilogy(x, data_inner,ls='-',color='#9A3B3B')
                axs[0].axhline(y=lower)
                axs[0].axhline(y=upper)

                # plot original data 
                axs[0].scatter(x,data_inner,marker='^',color='#F45050',alpha=0.7,label='raw data')

                # plot the selected data
                axs[0].scatter(x,selected_data_ma ,s=50,color='#004225',alpha=0.4,label='within data')

                # plot the iter_clear data
                axs[0].scatter(x[iter_fine_index], data_inner[iter_fine_index],marker='*',s=60,color='#92C7CF',alpha=0.7,label='iter data')


                # decoration axs[0]
                axs[0].margins(x=0.3,y=0.2)
                axs[0].set_ylim(bottom= np.log10(np.abs(data_inner)).min())
                axs[0].set_facecolor('#F6FDC3')
                axs[0].legend()



                axs[1].semilogy(x[cls.CLEAR_INDEX], data_inner[cls.CLEAR_INDEX],c='C7')
                axs[1].set_ylim([data_inner.min(), data_inner.max()])
                axs[1].grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
                axs[1].set_facecolor('#F9F7C9')
                plt.show()

                
                
                
                
        
        range_slider.observe(lambda change: range_sub_fun(change, observe_data,eletransmit), names='value')
        iters_drop_down.observe(lambda change: drop_down_fun(change, observe_data,eletransmit), names='value')
        
        # Displa.y the widgets
        display(iters_drop_down,range_slider,output )

    @staticmethod
    def iter_filter(data,iter_num,ele_transmit=None):

        un_point = []
        group_data = [data[ele_transmit[i] : ele_transmit[i+1]] if i < ele_transmit.size-1 else data[ele_transmit[-1] : -1] for i in range(0,ele_transmit.size)]

        for start_index, loop_data in enumerate(group_data):

            
            for loop_iter in range(0,iter_num+1):
                
                for index,(pre,curr) in enumerate(zip(loop_data,loop_data[loop_iter :]),start=ele_transmit[start_index] + loop_iter):
                    
                    if pre < curr:
                        un_point.append([index])

        un_point = np.unique(un_point)
        return un_point  
        
    @staticmethod
    def fun_kit(data)->List[np.ndarray]:
        """ we follow the basic truth that acquired voltage from the next potential electrode always smaller than forward one.
        
        Arg:
            pandas.DataFrame, first Colums is Current pole(C1) and ending colums is observe data
        
        Return:
            observe_data : inhere, the 'R0(ohm)' colums data(also normalized voltage), which is the crucial information for resistivity
                    inversion using SimPEG opnesource python package
            transmit : in partical, one transmit electrode correspond to more than one potential electrode. For same current electrode,
                we group the corresponding receiving electrodes together and only need to give the position of the first receiving
                electrode in all electrode arrangements, provided that the electrodes are arranged in an orderly manner. 
            
        """


        observe_data = data.iloc[:,-1].values
        C1 = data.iloc[:,0].values
        C1_unique = np.unique(C1)
        C1_shift = np.array([ np.where(C1 == index )[0][0] for index in C1_unique])


        
        return observe_data,C1_shift            
    