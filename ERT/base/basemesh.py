from abc import ABC,abstractmethod, abstractmethod


class BaseMesh(ABC):
    @property
    @abstractmethod
    def edges(self):
        pass
    @property
    @abstractmethod
    def cell_centers(self):
        pass

    @property
    @abstractmethod
    def vol(self):
        pass
    @property
    @abstractmethod
    def dim(self):
        pass







