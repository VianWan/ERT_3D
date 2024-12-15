import numpy as np
import string
from scipy.interpolate import griddata
from scipy.spatial import distance
from scipy.sparse import csr_array,csr_matrix,linalg
from typing import List, Optional

class simulation(object):
    """
    """

    def __init__(self,
                 mesh:"TensorMesh",
                 survey,
                 rho_map,
                 solver
                 ):
        pass


    @property
    def conductivity(self)-> np.ndarray:
        return self._conductivity
    
    @conductivity.setter
    def conductivity(self, conductivity:'float|np.ndarray'):
        """Set back conductivity """
        if isinstance(conductivity(float,int)):
            self._conductivity = np.ones_like(mesh.nD) * conductivity
        elif isinstance(conductivity, np.ndarray):
            self._conductivity = conductivity
        else:
            raise TypeError('Conductivity must be int or ndarray')

    





