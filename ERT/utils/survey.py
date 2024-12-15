import numpy as np
from typing import List
class SurveyGenerate(object):
    """This module is used to generate electrode configurations commonly used in the construction of DC resistivity methods. 
  
     Different electrode configurations have different signal ratios and model sensitivities.
    """
  
    @staticmethod
    def dipole_array(eletotal,
                     unitspacing: int = 1,
                     shift: int = 0,
                     drop: List[int] = [],
                    )->np.ndarray:
        """
        Generate dipole array
        
        B(C2)A(C1)-----M(P1)N(P2)
        
        Arg:
        eletotal: total electrode number
        unitspacing: AB = MN = unitspacing,awalys positive and integer
        shift : eletrode offset from begin, no shift by default
        drop : list, omit specified electrode
        
        Returns:
          np.ndarray: n * 4 dim array [pole_a,pole_b,pole_m,pole_n]
        """
        
        assert isinstance(unitspacing,int) and unitspacing > 0, 'unitspacing must be positive number and instance of int'
        assert isinstance(shift,int) and shift > -1, 'electrode offset must be positive number and instance of int'

        # Storage dipole arrangement
        dipole_array = []
         
        for pole_b in range(1+shift, eletotal - 2 * unitspacing):
            for pole_n in range(1+pole_b+2*unitspacing, eletotal+1):

                pole_a = pole_b + unitspacing
                pole_m = pole_n - unitspacing
                dipole_array.append([pole_a,pole_b,pole_m,pole_n])

        assert len(dipole_array), 'Checking unitspacing and shift parameter,'
        
        if drop:
            dipole_array = np.array(dipole_array)
            bool_list = []
            for drop_iter in np.array(drop):
                
                bool_a = dipole_array[:,0] == drop_iter
                bool_b = dipole_array[:,1] == drop_iter
                bool_m = dipole_array[:,2] == drop_iter
                bool_n = dipole_array[:,3] == drop_iter
                # if any electrode in specificed, get True
                bool_s = bool_a | bool_b | bool_m | bool_n
                bool_list.append(bool_s)
            
            index = np.argwhere(sum( bool_list)==0).ravel()

            return dipole_array[index,:]
        
        else :
            return np.array(dipole_array)
        
    @staticmethod
    def pole_dipole_array(eletotal: int,
                         unitspacing: int = 1,
                         shift: int = 0,
                         drop: List[int] = [],
                        )->np.ndarray:
        """
        Generate AMN array
        
        A(C1)-----M(P1)N(P2) B(C2 -> infinity : -1)
        
        Arg:
        eletotal: total electrode number.
        unitspacing:  MN = unitspacing, awalys positive and integer.
        shift : eletrode offset from begin, no shift by default.
        drop : list, omit specified electrode.
        
        Returns:
          np.ndarray: n * 4 dim array [pole_a,pole_b,pole_m,pole_n]
        """
        
        assert isinstance(unitspacing,int) and unitspacing > 0, 'unitspacing must be positive number and instance of int'
        assert isinstance(shift,int) and shift > -1 ,'electrode offset must be positive number and instance of int'

        # Storage pole3 arrangement
        pole3_array = []
        
        for pole_a in range(1+shift, eletotal-unitspacing):
            for pole_n in range(1+pole_a+unitspacing, eletotal+1):
                
                pole_m = pole_n - unitspacing
                pole3_array.append([pole_a, -1, pole_m, pole_n]) # -1 represent infinity
                
        if drop:
            
            pole3_array = np.array(pole3_array)
            bool_list = []
            for drop_iter in np.array(drop):
                
                bool_a = pole3_array[:,0] == drop_iter
                bool_m = pole3_array[:,2] == drop_iter
                bool_n = pole3_array[:,3] == drop_iter
                
                # if any electrode in specificed, get True
                bool_s = bool_a | bool_m | bool_n
                bool_list.append(bool_s)
                
            index = np.argwhere(sum( bool_list)==0).ravel()
            
            return pole3_array[index,:]
        
        else :
            return np.array(pole3_array)
    
    @staticmethod
    def wenner_alpha_array(eletotal: int,
                           shift: int = 0,
                           drop: List[int] = [],
                        )->np.ndarray:
        """
        Generate Wenner_αarrangement.
        
        A(C1)---M(P1)---N(P2)---B(C2) 
        
        Arg:
        eletotal: total number of electrodes.
        shift : eletrode offset from begin, no shift by default.
        drop : list, omit specified electrodes.
        
        
        Returns:
          np.ndarray: n * 4 dim array [pole_a,pole_b,pole_m,pole_n].
        Notice:
            AM = MN = NB
        """
        assert isinstance(shift,int) and shift > -1 ,'electrode offset must be positive number and instance of int'
        
        wenner_alpha_array = []
        for pole_a in range(1+shift,eletotal):
            for pole_m in range(pole_a+1,eletotal):
                pole_n = pole_m + (pole_m - pole_a)
                pole_b = pole_n + (pole_m - pole_a)

                if pole_b > eletotal:
                    break;
                    
                wenner_alpha_array.append([pole_a,pole_b,pole_m,pole_n])
                
        # drop specficied electrode
        if drop:
            
            index = np.isin(wenner_alpha_array,drop).any(axis=1)
            wenner_alpha_array = np.array(wenner_alpha_array)[~index,:]
            
        return np.array(wenner_alpha_array)
    
    @staticmethod
    def Schlumberger_array(eletotal: int,
                           unitspacing: int = 1,
                           shift: int = 0,
                           drop: List[int] = [],
                    )->np.ndarray:
        """
        Generate  Schlumberger arrangement
        
        A(C1)-----M(P1)N(P2)----B(C2) 
        
        Arg:
        eletotal: total number of electrodes.
        unitspacing : MN = unitspacing, AM = NB = n * unitspacing.
        shift : eletrode offset from begin, no shift by default.
        drop : list, omit specified electrodes.

        Return:
          n * 4 dim array
          [pole_a,pole_b,pole_m,pole_n]
    """
        assert isinstance(unitspacing,int) and unitspacing > 0,'unitspacing must be positive number and instance of int'
        assert isinstance(shift,int) and unitspacing > -1 ,'electrode offset must be positive number and instance of int'

        # Storage pole3 arrangement
        Schlumberger_array = []
        
        for pole_a in range(1+shift,eletotal):
            for pole_m in range(1+pole_a,eletotal):
                
                pole_n = pole_m + unitspacing
                pole_b = pole_n + (pole_m - pole_a)
            
                if pole_b > eletotal:
                    break;
                Schlumberger_array.append([pole_a,pole_b, pole_m, pole_n])
    
       # drop specficied electrode
        if drop:
            index = np.isin(Schlumberger_array,drop).any(axis=1)
            Schlumberger_array = np.array(Schlumberger_array)[~index]

        return np.array(Schlumberger_array)


