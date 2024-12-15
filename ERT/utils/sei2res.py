import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from pathlib import Path,PurePath
from discretize import TensorMesh
import pandas as pd

def mesh_polyfit(mesh,data_ori,data_fit,kind='none'):
    """  Fit the relationship between earthquake and resistivity
   
      Args: 
          mesh : SimPEG generated mesh
          data_ori : 3 colums, [x,z, values],object property
          data_fit : 3 colums, [x,z, values],need projected data
          kind : "none" directly mapping, "linear", use L2norm to fit, "quadratic",Fit using quadratic polynomials
          
     Reuturn:
          1D array, mesh-length array, from interpolation of the values
  
    """
    dataori_un = np.unique(data_ori,axis=0)
    datafit_un = np.unique(data_fit,axis=0)
    
    datafit_un_x = datafit_un[:,0]
    datafit_un_z = datafit_un[:,1]
    datafit_un_value = datafit_un[:,-1]
    
    
    dori_xmin,dori_xmax, dori_zmin,dori_zmax = dataori_un[:,0].min(), dataori_un[:,0].max(),dataori_un[:,1].min(), dataori_un[:,1].max()
    dfit_xmin,dfit_xmax, dfit_zmin,dfit_zmax = datafit_un[:,0].min(), datafit_un[:,0].max(),datafit_un[:,1].min(), datafit_un[:,1].max()
    
    all_x_min = np.array( dori_xmin if np.abs(dori_xmin) < np.abs(dfit_xmin)  else dfit_xmin)
    all_x_max = np.array( dori_xmax if np.abs(dori_xmax) < np.abs(dfit_xmax)  else dfit_xmax)
    all_z_min = np.array( dori_zmin if np.abs(dori_zmin) < np.abs(dfit_zmin)  else dfit_zmin)
    all_z_max = np.array( dori_zmax if dori_zmax < dfit_zmax  else dfit_zmax)
  
    # only use overlap data to polyfit
    xx = np.linspace(all_x_min,all_x_max,50)
    zz = np.linspace(all_z_min,all_z_max,50)
    XX,ZZ = np.meshgrid(xx,zz)
    dori_overlap = griddata((dataori_un[:,0],dataori_un[:,1]),dataori_un[:,2],xi=(XX[:],ZZ[:]),method='nearest')
    dfit_overlap = griddata((datafit_un[:,0],datafit_un[:,1]),datafit_un[:,2],xi=(XX,ZZ),method='nearest')
    
    # linear approximation
    L2_A = np.c_[np.ones_like(dfit_overlap.flatten('C')),dfit_overlap.flatten('C')]
    L2_b = np.c_[dori_overlap.flatten('C')]
    
    kc = np.linalg.inv(np.dot(L2_A.T,L2_A))@L2_A.T@L2_b
    
    # quadratic approximation
    
    cofficient_2d = np.polyfit(dfit_overlap.flat,dori_overlap.flat,2)
    
    if kind not in ["none","linear","quadratic"]:

        raise AttributeError("such fitness method doesn't existing")

    if kind =='linear':
        
        data_fit_approximate = kc[-1] * datafit_un_value + kc[0]
        data_interp = griddata(np.c_[datafit_un_x,datafit_un_z],values=data_fit_approximate,xi=(mesh.cell_centers[:,0],mesh.cell_centers[:,1]),method='nearest')
        print(kc[-1],kc[0])
    elif kind == 'quadratic':
        
        data_fit_approximate = np.polyval(cofficient_2d, datafit_un_value)
        data_interp = griddata(np.c_[datafit_un_x,datafit_un_z],values=data_fit_approximate,xi=(mesh.cell_centers[:,0],mesh.cell_centers[:,1]),method='nearest')
        print(cofficient_2d)
    elif kind == 'none':
        
        data_interp = griddata(np.c_[datafit_un_x,datafit_un_z],values=datafit_un_value,xi=(mesh.cell_centers[:,0],mesh.cell_centers[:,1]),method='nearest')
        
        
    return  data_interp




    
#     fig,ax2=plt.subplots(1,1,figsize=plt.figaspect(1.5))
#     # cavs=ax1.scatter(x,z,5,c=vs2res1,cmap='RdBu_r') 
    
#   ca2=ax2.scatter(mesh.cell_centers[:,0],mesh.cell_centers[:,1],200,c=vinterp,cmap='RdBu_r', vmin=vinterp.min(),vmax=vinterp.max() )
#   ax2.set_xlim([0,210])
#   ax2.set_ylim([-50,0])
#   ax2.set_aspect(1)
#   ax2.set_xlabel('Distance (m)',fontdict={'fontsize':15})
#   ax2.set_ylabel('Depth (m)',fontdict={'fontsize':15})
#   # ax2.set_title('interp result')
  
#   # cb=fig.colorbar(ca2)
#   cb = plt.colorbar(ca2, fraction=0.017,
#                   orientation='vertical', ax=ax2, pad=0.02)
  
#   cb.ax.tick_params(labelsize=10)
#   cb.set_label("Resistivity ($\Omega$m)", fontdict={
#              'weight': 'normal', 'size': 20})
  
  
#   plt.show()
#   # fig.savefig(r'C:\Users\23072\Desktop\新建文件夹\cigew\hvsr10n.png')
#   return vinterp

# def ok():
#   print('i am other function')



# __all__=['ok','sei2res',]





if __name__== "__main__":
  
    pass


    
    
    
    
                
                



