from abc import ABC,abstractmethod,abstractproperty

class BaseMesh(ABC):
    
    @abstractproperty
    def edges(self):
        pass

    @abstractproperty
    def cell_centers(self):
        pass

    @abstractproperty
    def vol(self):
        pass
    
    @abstractproperty
    def dim(self):
        pass







