import numpy as np
import matplotlib.pyplot as plt
from matplotlib import axes
from typing import Tuple,Optional,Union
from functools import wraps
from scipy.interpolate import griddata

def plot_decorate(log_switch=False):
    """ Beautify the role of the drawing function """
    def plot_decorate_inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                f_fig, f_ax, fcx_out = func(*args, **kwargs)
                assert isinstance(f_fig, plt.Figure) and isinstance(f_ax, plt.Axes), "fig, ax must be a matplotlib instance"
                
            except TypeError as e:
                # print('fig, ax must be an instance of %matplotlib')
                print(e)

            except Exception as e:
                  print(e)
                    
            else:
                f_ax.invert_yaxis()
                f_ax.xaxis.tick_top()
                # f_ax.set_aspect(1)
                f_ax.set_xlabel('X axis', fontdict={'fontsize': 15, 'weight': 'bold'})
                f_ax.set_ylabel('N Layer', fontdict={'fontsize': 15, 'weight': 'bold'})
                f_ax.set_title('AppResistivity -- Pseudosection', pad=10, fontdict={'fontsize': 15, 'weight': 'bold'})
                f_ax.tick_params(axis='both', labelsize=17)
                f_box = f_ax.get_position()
                f_sub_ax = f_fig.add_axes([f_box.x0 + f_box.width * 1.1, f_box.y0 - f_box.y0 * 0.1, f_box.width / 18, f_box.height * 1.05])
                fc_bar = f_fig.colorbar(fcx_out, cax=f_sub_ax, orientation='vertical', shrink=0.1)

                if not log_switch:
                    fc_bar.set_label(" Apparent Resistivity ($\Omega\cdotp$m)", fontdict={
                        'weight': 'normal', 'size': 20}, labelpad=10)
                else:
                    fc_bar.set_label(" Apparent Resistivity ($\log_{e}^{ρ} \Omega\cdotp$m)", fontdict={
                        'weight': 'normal', 'size': 20}, labelpad=10)

                fc_bar.ax.tick_params(labelsize=20, labelcolor='black', color='black')
                f_ax.margins(x=0.1, y=0.05)
                
            # return f_fig,f_ax     
        return wrapper
    return plot_decorate_inner

# no pythonic, don't mind :)
# means we one current and potential electrode laid long distance to the survey line
# althought we only use A and M electrode loaction, but for sepcification, still remain B and N position parmameter.
# the following function are handled the same way
def pole_pole(
              loc_a: 'A_loction',
              loc_b: 'B_location',
              loc_m: 'M_loction',
              loc_n: 'N_location',
              c_values: 'apperent resistivity',
              interp=False,
              log_switch=False,
             ) -> Tuple[plt.Axes,plt.Axes,plt.Axes]:
    # ----- # 
    electrode_x = 0.5 * np.add(loc_a,loc_m)
    # pseudosection z location
    transmit_pre_electrode = np.r_[np.diff(np.r_[-1, np.argwhere(np.diff(loc_a) > 0).ravel(), loc_a.size - 1])]
    electrode_z = np.array([j for i in transmit_pre_electrode for j in range(0, i)])
    # refine the plotting data
    if interp:
        [electrode_x,electrode_z,c_values]=interp_electrode(electrode_x,electrode_z,c_values, 0.01 * (loc_m[0] + loc_m[1])) # change the step here
    # The data range becomes compact
    if log_switch:
        c_values = np.log(c_values)

    fig, ax  =plt.subplots(figsize=(10,8))
    cx_out = ax.scatter(electrode_x,electrode_z,s=50,c=c_values,marker='s',cmap='rainbow')

    return fig,ax,cx_out

# electrode configuration like A-MN or B-MN
# the distance between one current electrode and survey line is assumed to be infinity 
def pole_dipole(
                loc_a: np.ndarray[float] = None,
                loc_b: np.ndarray[float] = None,
                loc_m: np.ndarray[float] = None,
                loc_n: np.ndarray[float] = None,
                c_values = None,
                interp: bool = False,
                log_switch = False,
            ) -> Tuple[plt.Axes]:
# ----- #
    electrode_x = 0.5 * np.add(loc_a, 0.5*np.add(loc_m, loc_n))
    transmit_pre_electrode = np.r_[np.diff(np.r_[-1, np.argwhere(np.diff(loc_a) > 0).ravel(), loc_a.size - 1])]
    electrode_z = np.array([j for i in transmit_pre_electrode for j in range(0, i)])

    if interp:
        [electrode_x, electrode_z, c_values] = interp_electrode(electrode_x, electrode_z, c_values, 0.01 * np.add(loc_n[0],loc_n[1]))

    if log_switch:
        c_values = np.log(c_values)
    
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 8))
    cx_out = ax.scatter(electrode_x, electrode_z, s=50, marker='s', c=c_values,cmap='rainbow')

    return fig, ax, cx_out

def dipole_dipole(
                  loc_a:'electrodeA_location' = None,
                  loc_b:'electrodeB_location' = None,
                  loc_m:'electrodeM_location' = None,
                  loc_n:'electrodeN_location' = None,
                  c_values:'apperent reistivity'=None,
                  interp: bool = False,
                  log_switch: bool = False,
) -> Tuple[plt.Axes,plt.Axes,plt.Axes]:

    electrode_x = 0.5 * (0.5*np.add(loc_a, loc_b) + 0.5*np.add(loc_m, loc_n))
    transmit_pre_electrode = np.r_[np.diff(np.r_[-1, np.argwhere(np.diff(loc_a) > 0).ravel(), loc_a.size - 1])]
    electrode_z = np.array([j for i in transmit_pre_electrode for j in range(0, i)])

    
    if interp:
        [electrode_x,electrode_z,c_values] = interp_electrode(electrode_x,electrode_z,c_values,step=0.3 * (loc_a[0] + loc_a[1])) 
    if log_switch:
        c_values=np.log(c_values)

    fig, ax = plt.subplots(ncols=1,nrows=1,figsize=(10,8))
    cx_out = ax.scatter(electrode_x,electrode_z,s=50,marker='s',c=c_values,cmap='rainbow')
    
    return fig, ax, cx_out

def interp_electrode(x: np.ndarray,z: np.ndarray, value,step: float=1, methodinterp='cubic') -> Tuple[np.ndarray]:
    xmin = np.min(x)
    xmax = np.max(x)
    zmin = np.min(z)
    zmax = np.max(z)
    # refine the electrode unit 
    xrefine = np.arange(xmin,xmax,step)
    yrefine = np.arange(zmin,zmax,step)

    [X,Z] = np.meshgrid(xrefine,yrefine)
    interp_value = griddata(np.c_[x,z], value, np.c_[X.flatten(), Z.flatten()], method='cubic')
    return X.ravel(), Z.ravel(), interp_value





def plot_pseudosection(
                        loc_a: np.ndarray = None,
                        loc_b: np.ndarray = None,
                        loc_m: np.ndarray = None,
                        loc_n: np.ndarray = None,
                        obs_data: np.ndarray = None,
                        interp: bool = False,
                        log_switch: bool = False,
                        survey_type: str = 'dipole-dipole'
) -> None:
    """plot the apparent resistivity or voltage psedusection
-------------------------------------------------------------------------------------------------
    Arg:
       loc_a : current electrode A 
       loc_b : current electrode B
       loc_m : potential electrode M
       loc_n : pttential electrode N
       obs_data : by default is appResistivity
       interp : need interpolation?
       log_switch : Let the data be plotted logarithmically
       survey_type : electrode configuration, 'pole-pole;pole-dipole;dipole-dipole are available'
 ---------------------------------------------------------------------------------------------------
  
    Notice:
       if the electrode configuration ,ignore some electrode, u just set that Corresponding electrode parameter equals None(default)

       may be function in whole page written so tight, u can adjust some parameters or rectify function
    """
    
    fun_type={
              "pole-pole":pole_pole,
              "pole-dipole":pole_dipole,
              "dipole-dipole":dipole_dipole,
             }
    
    assert survey_type in fun_type,"survey_type doesn't have this type"

    @plot_decorate(log_switch=log_switch)
    def plot_function():
        return fun_type.get(survey_type)(loc_a,loc_b, loc_m, loc_n, obs_data, interp, log_switch)
    plot_function()













if __name__ == '__main__':
    from pandas import read_excel

    try:
        
        observed_data=r'E:\pyspace\ERT\example\T231122006.xlsx'  
        DATA = read_excel(observed_data,sheet_name=1,engine='openpyxl')

    except IOError:
        print("Can't open the file ,may be u need restare the terminal")
    else:

        try:

            if DATA.columns.size==12:

                loc_a=DATA.iloc[:,0]
                loc_b=DATA.iloc[:,1]
                loc_m=DATA.iloc[:,2]
                loc_n=DATA.iloc[:,3]
                obs_voltage=np.where(DATA.loc[:,'R0']>0,DATA.loc[:,'R0'],1e-4)

        except ValueError:
            pass
        else:
            print('Information Loading Fine')

    plot_pseudosection(loc_a=loc_a,loc_b=None, loc_m=loc_m, loc_n=None, obs_data=obs_voltage, interp=True, log_switch=True,survey_type='pole-pole')

    plt.show()