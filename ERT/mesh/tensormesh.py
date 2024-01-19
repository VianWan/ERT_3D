import numpy as np
import string 
from typing import Tuple
from matplotlib.axes import Axes

from ..base.basemesh import BaseMesh


class TensorMesh(BaseMesh):

    """ This class aims for generate 3D mesh

    Methods:
       return grid node and face, volume etc information

       plotting the mesh u designed

    Notice:
    tThe grid numbering order  is formulated as follow
        top -> bottom, left -> right, front -> back

          /\ y
          |
         |
        |
       |_ _ _ _ _ _ _\   
       |             /  x
       |
       |
      \/ z   
    
      plaint word, wo generater 2 * 3 * 2 mesh,

      first_face   second_face
       1 4     7 10
       2 5  -> 8  11
       3 6     9  12
 
    """
    # save memory and accelerate access instace's attribute speed
    __slot__ = ('nodex','nodey','nodez',
                'cell_centers_x','cell_centers_y','cell_centers_z',

    ) 


    
    def __init__(self,h:'list') -> None:
        # input consist of 3 part, 
        # like X direction inculding [(padding factor,padding_num),(core_grid width, core_grid_num),(padding factor,padding_num)]
        # hx=[(-1.5,20),(2,40),(1.5,20)] 
        # hy=[(-1.5,20),(2,40),(1.5,20)]
        # hz=[(-1.5,20),(2,20)]
        # h=[hx,hy,hz]



        self.padd_x = np.asarray(h[0][2][0], dtype=np.float16)  # padding factor in X aixs 
        self.padd_y = np.asarray(h[1][2][0], dtype=np.float16)
        self.padd_z = np.asarray(h[2][0][0], dtype=np.float16)
        self.core_x = np.asarray(h[0][1][1], dtype=np.float16)   # core domain cell number
        self.core_y = np.asarray(h[1][1][1], dtype=np.float16)
        self.core_z = np.asarray(h[2][1][1], dtype=np.float16)
        
        self._core_x = np.r_[[self.core_x if not np.mod(self.core_x,2) else self.core_x+1]]  # change even to odd for balencing
        self._core_y = np.r_[[self.core_y if not np.mod(self.core_y,2) else self.core_y+1]]
        self._core_z = self.core_z
        
        # core domain node
        self._gridx = np.r_[
                            -np.linspace(self._core_x/2 * h[0][1][0], 0,  int(self._core_x/2) + 1)[:-1],
                            np.linspace(0,self._core_x/2 * h[0][1][0],   int(self._core_x/2) + 1),
                      ]   
        
        self._gridy = np.r_[
                           -np.linspace(self._core_y/2 * h[1][1][0], 0,  int( self._core_y/2 ) + 1)[:-1],
                            np.linspace(0,self._core_y/2 * h[1][1][0],  int( self._core_y/2 ) + 1),
                      ]
        
        self._gridz = np.linspace(0, -self._core_z*h[2][1][0], int( self._core_z) + 1)
      
        # grid nodes for three direction 
        self.nodex = np.r_[
                           np.cumsum(np.cumprod(np.ones((h[0][0][1])) * self.padd_x))[::-1] * self._gridx[0]+self._gridx[0],
                           self._gridx.flatten(),                                                                       
                           np.cumsum(np.cumprod(np.ones((h[0][2][1])) * self.padd_x)) * self._gridx[-1]+self._gridx[-1] ,
                     ]
        
        self.nodey = np.r_[
                           np.cumsum(np.cumprod(np.ones((h[1][0][1])) * self.padd_y))[::-1] * self._gridy[0]+  self._gridy[0],
                           self._gridy.flatten(),
                           np.cumsum(np.cumprod(np.ones((h[1][2][1])) * self.padd_y)) * self._gridy[-1] + self._gridy[-1],
                     ]
        
        self.nodez = np.r_[
                           self._gridz.flatten(),
                           -np.cumsum(np.cumprod(np.ones((h[2][0][1])) * np.abs(self.padd_z) )  ) * ( self._gridz[-2] - self._gridz[-1]) + self._gridz[-1]
                    ]
        
        # cell centers in three axis 
        self.cell_centers_x = self.nodex[0:-1] +  0.5 * np.diff(self.nodex)  
        self.cell_centers_y = self.nodey[0:-1] +  0.5 * np.diff(self.nodey)
        self.cell_centers_z = self.nodez[0:-1] +  0.5 * np.diff(self.nodez)
       
       # grid length
        self.edge_x_length = np.diff(self.nodex).reshape(1,-1)
        self.edge_y_length = np.diff(self.nodey).reshape(1,-1)
        self.edge_z_length = np.abs(np.diff(self.nodez)).reshape(1,-1)

        # grid dimension
        self._dim = 3
    

    def __str__(self) -> str:
        self._txt=string.Template("""
                                    Xmin(m)       ${xmin}    Xmax(m)           ${xmax}      mesh_x_length(m)       ${x_length}
                                    
                                    Ymin(m)       ${ymin}    Ymax(m)           ${ymax}      mesh_y_length(m)       ${y_length}
                                    
                                    Zmin(m)       ${zmin}    Zmax(m)           ${zmax}      mesh_z_length(m)       ${z_length}
                                  
                                    CellNum    ${cellnum}
        
        """)
        value = self._outfile
        return self._txt.safe_substitute(value)



    @classmethod
    def plot_image(
                   cls,
                   nodex: np.ndarray = None,
                   nodey: np.ndarray = None,
                   nodez: np.ndarray = None,
                   mesh: np.ndarray = None,
                   ax = None,
                   ) -> Tuple[Axes,Axes]:
        """ to see the 3D mesh outline"""
        
        if mesh != None:
            nodex = mesh.nodex
            nodey = mesh.nodey
            nodez = mesh.nodez
         # start and end point of each grid    
        line = cls._polt_image(nodex,nodey,nodez)

        if ax is not None:
            if not isinstance(ax,matplotlib.axes.Axes):
                raise TypeError("ax must be instance of  matplotlib.axes.Axes")
        else:
            matplotlib,plt =cls.lazy_load_matplotlib()
            axOpts ={"projection": "3d"} 
            fig_inner,ax_inner = plt.subplots(figsize=(10,10), subplot_kw=axOpts)


        ax_inner.plot(line[:,0],line[:,1],line[:,2],color=(74/255,95/255,126/255))
        ax_inner.set_xlabel('X',fontdict={'weight': 'bold', 'size': 15,'color':'black'},labelpad=20)
        ax_inner.set_ylabel('Y',fontdict={'weight': 'bold', 'size': 15,'color':'black'},labelpad=20)
        ax_inner.set_zlabel('Z',fontdict={'weight': 'bold', 'size': 15,'color':'black'},labelpad=20)

        ax_inner.set_facecolor('none')
        #ax_inner.axis('off')
        ax_inner.grid(None)
        plt.show()
        return fig_inner,ax_inner
   
       
  

    @staticmethod
    def lazy_load_matplotlib():
        import matplotlib
        import matplotlib.pyplot as plt
        return matplotlib,plt

  
        
    @staticmethod
    def _polt_image(nodex,nodey,nodez: 'TensorMesh.attritube') -> np.ndarray:
        """ generate the plotting grid """
        line_nodex = np.r_[nodex[0],nodex[-1],np.nan].astype(np.float32)
        line_nodey = np.r_[nodey[0],nodey[-1],np.nan].astype(np.float32)
        line_nodez = np.r_[nodez[0],nodez[-1],np.nan].astype(np.float32)
        
        mat_nodex = np.tile(line_nodex,(nodez.size,1))
        mat_nodey = np.tile(line_nodey,(nodez.size,1))
        mat_nodez = np.tile(line_nodez[:,None],(1,nodex.size))
        
        # Defines the start and end points of each polygon gridline.
        line_front_horz = np.c_[mat_nodex.ravel(order='C'), line_nodey[0] * np.ones_like(mat_nodex).flatten(), np.repeat(nodez,3)]
        line_back_horz  = np.c_[mat_nodex.ravel(order='C'), line_nodey[-2] * np.ones_like(mat_nodex).flatten(), np.repeat(nodez,3)]
        line_front_vert = np.c_[np.repeat(nodex,3), line_nodey[0]*np.ones_like(mat_nodez).flatten(), mat_nodez.ravel(order='F')]
        line_back_vert  = np.c_[np.repeat(nodex,3), line_nodey[-2]*np.ones_like(mat_nodez).flatten(), mat_nodez.ravel(order='F')]
        line_west_horz = np.c_[line_nodex[0] * np.ones_like(mat_nodey).flatten(), mat_nodey.ravel(order='C'), np.repeat(nodez,3)]
        line_east_horz = np.c_[line_nodex[-2] * np.ones_like(mat_nodey).flatten(), mat_nodey.ravel(order='C'), np.repeat(nodez,3)]
        line_west_vert = np.c_[line_nodex[0].repeat(3 * nodey.size), np.repeat(nodey,3), np.tile(line_nodez,(1,nodey.size)).ravel()]
        line_east_vert = np.c_[line_nodex[-2].repeat(3 * nodey.size), np.repeat(nodey,3), np.tile(line_nodez,(1,nodey.size)).ravel()]
        line_top_horz = np.c_[
                              np.tile(line_nodex,(nodey.size,1)).ravel(order='C'),np.repeat(nodey,3),
                              line_nodez[0]*np.ones_like(np.repeat(nodey,3))
                        ]
        
                              
        line_top_vert = np.c_[ 
                              np.repeat(nodex,3), 
                              np.tile(line_nodey,(nodex.size,1)).ravel('C'),
                              line_nodez[0]*np.ones_like(np.repeat(nodex,3)),
                        ] 
        
        
        line_bott_horz = np.c_[ 
                               np.tile(line_nodex,(nodey.size,1)).ravel(order='C'),np.repeat(nodey,3),
                               line_nodez[-2]*np.ones_like(np.repeat(nodey,3)),
                        ]
        
        line_bott_vert = np.c_[
                                np.repeat(nodex,3),
                                np.tile(line_nodey,(nodex.size,1)).ravel('C'),
                                line_nodez[-2]*np.ones_like(np.repeat(nodex,3)),
                        ] 
                            
        # Assemble the grid line matrix
        line_ = np.r_[
                      line_front_horz,line_back_horz,line_front_vert,line_back_vert,
                      line_west_horz,line_east_horz,line_west_vert,line_east_vert,
                      line_top_horz,line_top_vert,line_bott_horz,line_bott_vert,
                ]
        
        return line_





    @property
    def edges(self) -> np.ndarray:
        """Calculate the edge length of each cell"""
        self._edge_x_length,self._edge_y_length,self._edge_z_length = np.meshgrid( 
                                                                                   self.edge_x_length, 
                                                                                   self.edge_y_length, 
                                                                                   self.edge_z_length,
                                                                                 )
        edges = np.c_[
                        self._edge_x_length.ravel('C'),
                        self._edge_y_length.ravel('C'),
                        self._edge_z_length.ravel('C'),
        ]
    
        return edges
 
    
    @property
    def cell_centers(self) -> np.ndarray:
        """ Calculate the center of each mesh """
        self._cell_centers_x,self._cell_centers_y, self._cell_centers_z = np.meshgrid(self.cell_centers_x,self.cell_centers_y,self.cell_centers_z)     
        cell_centers = np.c_[
                                self._cell_centers_x.ravel('C'),
                                self._cell_centers_y.ravel('C'),
                                self._cell_centers_z.ravel('C'),
        ]
        
        return cell_centers
    
    @property
    def vol(self) -> np.ndarray:
        """Calculate the volume of each mesh"""
        vol = np.prod(self.edges,axis=1)
        return vol



    @property
    def dim(self) -> np.ndarray:
        return self._dim
    
     # work for __str__
    @property
    def _outfile(self):  
               
        outfile_data = {
                        'xmin': '{: >10.2f}'.format(self.cell_centers_x.min()),
                        'xmax': '{: >10.2f}'.format(self.cell_centers_x.max()),
                        'ymin': '{: >10.2f}'.format(self.cell_centers_y.min()),
                        'ymax': '{: >10.2f}'.format(self.cell_centers_y.max()),
                        'zmin': '{: >10.2f}'.format(self.cell_centers_z.min()),
                        'zmax': '{: >10.2f}'.format(self.cell_centers_z.max()),
                        'cellnum': '{: >10}'.format(self.vol.size),
                        'x_length': '{: >10.2f}'.format(self.cell_centers_x[-1] - self.cell_centers_x[0]),
                        'y_length': '{: >10.2f}'.format(self.cell_centers_y[-1] - self.cell_centers_y[0]),
                        'z_length': '{: >10.2f}'.format(-self.cell_centers_z[-1] - self.cell_centers_z[0]),
        }
    
        return outfile_data





if __name__ == '__main__':
    hx=[(-1.5,10),(2,5),(1.5,10)] 
    hy=[(-1.5,5),(2,2),(1.5,2)]
    hz=[(-1.5,20),(2,2)]
    h=[hx,hy,hz]
    c = TensorMesh(h)
    # Tensormesh.plot_image(mesh = c)
    print(c.edges)
    print(c)
