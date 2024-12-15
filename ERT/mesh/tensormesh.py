import numpy as np
import string 
from typing import Tuple,List, Dict,Union,Sequence
from matplotlib.axes import Axes

from ..base.basemesh import BaseMesh


class TensorMesh(BaseMesh):

    """ This class aims for generate 2/3D Regular mesh

    Arg:
        Input consist of 2/3 part, like X direction inculding [(padding factor,padding_num),(core_grid width, core_grid_num),(padding factor,padding_num)] 
        hx=[(-1.5,20),(2,40),(1.5,20)] 
        hy=[(-1.5,20),(2,40),(1.5,20)]
        hz=[(-1.5,20),(2,20)]
        for 2D h = [hx,hz],
        for 3D h=[hx,hy,hz]

    Properity:
        return mesh's grid, nodes, faces, volume and so on
    Methods:
       plotting the mesh u designed

    Notice:
        The grid numbering order is formulated as following
            top -> bottom, left -> right, front -> back
            Z
           /|\   /\ y
            |   |
            |  |
            | |
            |_ _ _ _ _ _ _\   
                        / x 
    
      Plaint word, we have generate 2 * 3 * 2 mesh, and each 
      cell index is arrange by
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
        ## 3D mesh input 
        # hx=[(-1.5,20),(2,40),(1.5,20)] 
        # hy=[(-1.5,20),(2,40),(1.5,20)]
        # hz=[(-1.5,20),(2,20)]
        # h=[hx,hy,hz]

        # Set mesh dimension
        self.__dim = len(h)
        assert self.__dim in [2,3], "Mesh Dimension Must in 2D or 3D"
        # X Dir 
        self.core_x = np.asarray(h[0][1][-1], dtype=np.int32)   # core domain cell number
        self._core_x_size = np.asarray(h[0][1][0], dtype=np.float32) # the cell side lenth of core domain
        self._paddf_x_l = np.asarray(h[0][0][0], dtype=np.float32)  # padding factor in X aixs
        self._paddf_x_r = np.asarray(h[0][-1][0], dtype=np.float32)  # padding factor in X aixs 
        self._padd_x_l = np.asarray(h[0][0][-1], dtype=np.float32) # the figure of padding cell on left mesh 
        self._padd_x_r = np.asarray(h[0][-1][-1], dtype=np.float32)
        self._ori_x = np.array(self._core_x_size/2 if np.mod(self.core_x,2) else 0)# Define the starting point
        self._gridrx = np.linspace(self._ori_x, self._ori_x + self._core_x_size*int(self.core_x / 2), int(self.core_x /2)+1, endpoint=True)
        self._gridlx = -np.flip(self._gridrx) if self._ori_x else -np.flip(self._gridrx[1:])
        self._gridx = np.r_[self._gridlx, self._gridrx] # Grid node location
        
        # Z Dir
        self.core_z = np.asarray(h[-1][-1][-1], dtype=np.int32)
        self._core_z_size = np.asarray(h[-1][-1][0], dtype=np.float32)
        self._padd_z = np.asarray(h[-1][0][0], dtype=np.float16)  # padding factor in Z aixs
        self._gridz = np.linspace(0, -self.core_z*self._core_z_size, int( self.core_z) + 1)

         # X, Z node index
        self.nodex = np.r_[np.cumsum([abs(self._paddf_x_l)**(index+1) for index in range(int(self._padd_x_l))] * self._core_x_size * -1)[::-1] + self._gridlx[0],
                           self._gridx.flatten(),
                           np.cumsum([abs(self._paddf_x_r)**(index+1) for index in range(int(self._padd_x_r))] * self._core_x_size) + self._gridrx[-1],
                          ]

        self.nodez = np.r_[
                           self._gridz.flatten(),
                           -np.cumsum(np.cumprod(np.ones((h[-1][0][-1])) * np.abs(self._padd_z) )  ) * ( self._gridz[-2] - self._gridz[-1]) + self._gridz[-1]
                          ]
        
        # cell centers location
        self.cell_centers_x = self.nodex[0:-1] +  0.5 * np.diff(self.nodex)  
        self.cell_centers_z = self.nodez[0:-1] +  0.5 * np.diff(self.nodez)
       
       # grid length
        self.edge_x_length = np.diff(self.nodex).reshape(1,-1)
        self.edge_z_length = np.abs(np.diff(self.nodez)).reshape(1,-1)


        if self.__dim == 3:
            # Y Dir, used exculdly on 3D mesh
            self.core_y = np.asarray(h[1][1][1], dtype=np.int32)
            self._core_y_size = np.asarray(h[1][1][0], dtype=np.float32)
            self._paddf_y_l = np.asarray(h[1][0][0],dtype=np.float32) 
            self._paddf_y_r = np.asarray(h[1][-1][0],dtype=np.float32)
            self._padd_y_l = np.asarray(h[1][0][1], dtype=np.float16)
            self._padd_y_r = np.asarray(h[1][-1][1], dtype=np.float16)
            self._ori_y = np.array(self._core_y_size/2 if np.mod(self.core_y,2) else 0)
            self._gridry = np.linspace(self._ori_y, self._ori_y + self._core_y_size*int(self.core_y / 2), int(self.core_y /2)+1, endpoint=True)
            self._gridly = -np.flip(self._gridry) if self._ori_y else -np.flip(self._gridry[1:])
            self._gridy = np.r_[self._gridly, self._gridry]
            self.nodey = np.r_[
                           np.cumsum([abs(self._paddf_y_l)**(index+1) for index in range(int(self._padd_y_l))] * self._core_y_size * -1)[::-1] + self._gridly[0],
                           self._gridy.flatten(),                                                                       
                           np.cumsum([abs(self._paddf_y_r)**(index+1) for index in range(int(self._padd_y_r))] * self._core_y_size) + self._gridry[-1],
                        ]
            self.cell_centers_y = self.nodey[0:-1] +  0.5 * np.diff(self.nodey)
            self.edge_y_length = np.diff(self.nodey).reshape(1,-1)
      


    def add_rectangle(self, position: Union[Sequence[float], np.ndarray]) -> bool:
        """
        Create a rectange area into the mesh, using -inf/inf to represent negative/positive infinity boundary 

        Arg:
            position:[start_x,start_y,[,start_z],length, weight][,height],start point at the left bottom corner.
        Return:
            Bool, only in rectangle area is True.
        note:
            if one term of start point is inf, length or weight, height will become true coordinate positon.
        Example:

        >>> hx=[(-1.5,10),(2,20),(1.5,10)] 
        >>> hy=[(-1.5,10),(2,2),(1.5,10)]
        >>> hz=[(-1.5,10),(2,20)]
        >>> h=[hx,hz]
        >>> mesh = TensorMesh(h)
        >>> indrec = mesh.add_rectangle(position=position,c=c)
        >>> recx = c.cell_centers[indrec,0]
        >>> recz =c.cell_centers[indrec,1]
        >>> mesh.plot_grid()
        >>> ax.scatter(recx, recy, s=100, c='C0', marker='+')
        """
        assert isinstance(position,(list,tuple,np.ndarray)), 'Check input type'
        dim = int(len(position) / 2)
        assert dim in [2,3] and dim==self.__dim, 'Dimension not match'

        if dim == 2:
            start_x, start_z, length, height = position
            assert (length >0) & (height >0), 'Length and Width must be greater than zero'
            xd = start_x
            zd = start_z
            xu = length if length==np.inf or start_x==-np.inf else start_x + length
            zu = height if height==np.inf or start_z==-np.inf else start_z + height
            indrec = (self.cell_centers[:,0] > xd) & (self.cell_centers[:,0] < xu) & (self.cell_centers[:,1] > zd) & (self.cell_centers[:,1] < zu) 

        elif dim == 3:
            start_x, start_y, start_z, length, width, height = position
            assert length > 0 and width > 0 and height > 0, 'length, width, height must be greater than zero'
            xd = start_x
            xu = length if length==np.inf or start_x==-np.inf else start_x + length
            yd = start_y
            yu = width if width==np.inf or start_y==-np.inf else start_y + width
            zd = start_z
            zu = height if height==np.inf or start_z==-np.inf else start_z + height
            indrec = (self.cell_centers[:,0] > xd) & (self.cell_centers[:,0] < xu) & (self.cell_centers[:,1] > yd) & (self.cell_centers[:,1] < yu) & (self.cell_centers[:,2] > zd) & (self.cell_centers[:,2] < zu)

        return indrec
    
    def add_terrain(self, surface_loc: Dict, surface_upper: float = None, surface_below:float = None) -> Union[bool, np.ndarray]:
        """
        Return cell index of upper the terrain, bool is output by default, but if surface_upper is given, the value is filled with the mesh

        Args:
            surface_loc: terrain location, composed with 3 or more points (x,y,z).
            surface_upper: filled value upper the terrain.
            surface_below: filled value below the terrain.
        Return:
            Bool by default, or a constant figure if surface_* are given.
        Example:

        mesh = Tensormesh(h)
        surface_loc = {'1',[[1,2,3],[4,5,6],[7,8,9]]}
        """
        from scipy.interpolate import griddata
        surface_upper, surface_below = np.r_[surface_upper, surface_below]
    
        assert surface_loc['1'].shape[1] == self.__dim, 'terrain dimension must same mesh dimension'
        
        if self.dim == 3:
            terrain_ind = []
            terrain_value = np.ones_like(self.vol)
            plane_cellx, plane_celly = np.meshgrid(self.cell_centers_x, self.cell_centers_y)
            for count,key in enumerate(surface_loc.keys()):
                zq=griddata(surface_loc[key][:,[0,1]],surface_loc[key][:,-1], (plane_cellx,plane_celly), method='linear', fill_value=np.nan)
                zloc=zq.flatten().repeat(self.cell_centers_z.size)
                ind = self.cell_centers[:,-1] > zloc
                if surface_upper and surface_below:
                    terrain_value[ind] = surface_upper[count]
                    terrain_value[~ind] = surface_below[count]
                    terrain_ind.append(terrain_value)
                else:
                    terrain_ind.append(ind)

        elif self.dim == 2:
            terrain_ind = []
            terrain_value = np.ones_like(self.vol)
            for count, key in enumerate(surface_loc.keys()):
                zq = griddata(surface_loc[key][:,0], surface_loc[key][:,-1], self.cell_centers_x, method='linear', fill_value=np.nan)
                zloc = np.tile(zq,(len(self.cell_centers_z),1)).flatten('F')
                ind = self.cell_centers[:,-1] > zloc
                if surface_upper and surface_below:
                    terrain_value[ind] = surface_upper[count]
                    terrain_value[~ind] = surface_below[count]
                    terrain_ind.append(terrain_value)
                else:
                    terrain_ind.append(ind)
            
        return terrain_ind



        


    @property
    def edges(self) -> np.ndarray:
        """Calculate the edge length of each cell"""
        if self.__dim == 2:
            self._edge_x_length, self._edge_z_length = np.meshgrid(self.edge_x_length, self.edge_z_length, indexing='ij')
            self._edges = np.c_[self._edge_x_length.ravel('C'), self._edge_z_length.ravel('C')]
            
        elif self.__dim == 3:
            self._edge_x_length,self._edge_y_length,self._edge_z_length = np.meshgrid( 
                                                                                    self.edge_x_length, 
                                                                                    self.edge_y_length, 
                                                                                    self.edge_z_length,
                                                                                    )
            self._edges = np.c_[
                            self._edge_x_length.ravel('C'),
                            self._edge_y_length.ravel('C'),
                            self._edge_z_length.ravel('C'),
            ]
    
        return self._edges
 
    
    @property
    def cell_centers(self) -> np.ndarray:
        """ Calculate the center of each mesh """
        if self.__dim == 2:
            self._cell_centers_x, self._cell_centers_z = np.meshgrid(self.cell_centers_x, self.cell_centers_z, indexing='ij')
            cell_centers = np.c_[self._cell_centers_x.ravel('C'), self._cell_centers_z.ravel('C')]

        elif self.__dim ==3:
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
        return self.__dim
    
    @property
    def nD(self) -> np.ndarray:
        """Return the total number of cell"""
        if self.__dim == 2:
            return len(self.cell_centers_x) * len(self.cell_centers_z)
        elif self.__dim == 3:
            return len(self.cell_centers) 

    @property
    def cell_boundry_index(self) -> Tuple:
        """
        >>> Plotting to better understanding,
        >>> hx=[(-1.5,2),(2,2),(1.1,2)] 
        >>> hy=[(-1.5,2),(2,2),(1.1,2)]
        >>> hz=[(-1.5,2),(2,2)]
        >>> h=[hx,hy,hz]
        >>> mesh = TensorMesh(h)
        >>> xl,*_ = mesh.cell_boundry_index
        >>> mesh.plot_image(nodex,nodey,nodez,ax)
        >>> ax.scatter3D(mesh.cell_centers[x1,0],cell_centers[x1,1],_[x1,2])
        """

        if self.__dim == 2:
            indxl = self.cell_centers[:,0] == min(self.cell_centers[:,0])
            indxr = self.cell_centers[:,0] == max(self.cell_centers[:,0])
            indzt = self.cell_centers[:,1] == max(self.cell_centers[:,1])
            indzb = self.cell_centers[:,1] == min(self.cell_centers[:,1])
            return indxl, indxr, indzt, indzb
        elif self.__dim == 3:
            indxl = self.cell_centers[:,0] == min(self.cell_centers[:,0])
            indxr = self.cell_centers[:,0] == max(self.cell_centers[:,0])
            indyf = self.cell_centers[:,1] == min(self.cell_centers[:,1])
            indyb = self.cell_centers[:,1] == max(self.cell_centers[:,1])
            indzt = self.cell_centers[:,2] == max(self.cell_centers[:,2])
            indzb = self.cell_centers[:,2] == min(self.cell_centers[:,2])
            return indxl, indxr, indyf, indyb, indzt, indzb

  
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

    @classmethod
    def plot_grid(
                   cls,
                   nodex: np.ndarray = None,
                   nodey: np.ndarray = None,
                   nodez: np.ndarray = None,
                   mesh: np.ndarray = None,
                   ax = None,
                   ) -> Tuple[Axes,Axes]:
        """ to see the 2/3D mesh skeleton"""
        matplotlib, plt = cls.lazy_load_matplotlib()
        if mesh:
            nodex = mesh.nodex
            nodey = mesh.nodey
            nodez = mesh.nodez

        # start and end point of each grid    
        dim = sum(1 for var in [nodex, nodey, nodez] if var is not None)
        line = cls._polt_grid(nodex,nodey,nodez,dim=dim)
        axOpts =[None if dim ==2 else {'projection': '3d'} ] 
        if ax is None:
            fig_inner, ax_inner = plt.subplots(figsize=(8,6), subplot_kw=axOpts[0])
        else:
            if not isinstance(ax,matplotlib.axes.Axes):
                raise TypeError("ax must be instance of  matplotlib.axes.Axes")
            ax_inner = ax

        if dim == 2:
            ax_inner.plot(line[:,0],line[:,-1],color=(74/255,95/255,126/255))
            axOpts ={"projection": "2d"} 

        if dim == 3:
            ax_inner.plot(line[:,0],line[:,1],line[:,2],color=(74/255,95/255,126/255))
            ax_inner.set_zlabel('Z',fontdict={'weight': 'bold', 'size': 15,'color':'black'},labelpad=20)
        
        ax_inner.set_xlabel('X', fontdict={'weight': 'bold', 'size': 15,'color':'black'},labelpad=20)
        ax_inner.set_ylabel('Y', fontdict={'weight': 'bold', 'size': 15,'color':'black'},labelpad=20)
        

        ax_inner.set_facecolor('none')
        #ax_inner.axis('off')
        ax_inner.grid(None)
        
        if ax is None:
            plt.show()
            return fig_inner,ax_inner
        else:
            return None,ax_inner
       
    @classmethod
    def plot_image(cls, centr_x, centr_z, var, ax=None, cmap=None):
        """
        >>> mesh.plot_image(mesh.cell_centres_x, mesh.cell_centres_z, mesh.vol)
        """

        matplotlib, plt = cls.lazy_load_matplotlib()
        if ax is None:
            _, ax_inner = plt.subplots(1,1)
        else:
            if not isinstance(ax,matplotlib.axes.Axes):
                raise TypeError("ax must be instance of  matplotlib.axes.Axes")
            ax_inner = ax

        X, Z, Var = cls._plot_image(centr_x, centr_z, var)
        cmap = 'viridis' if cmap is None else cmap
        ax_inner.pcolormesh(X, Z, Var, shading='auto', cmap = cmap)

        return ax_inner





  

    @staticmethod
    def lazy_load_matplotlib():
        import matplotlib
        import matplotlib.pyplot as plt
        return matplotlib,plt

  
        
    @staticmethod
    def _polt_grid(nodex, nodey=None, nodez: 'TensorMesh.attritube'=None, dim=None) -> np.ndarray:
        """ generate the Mehs Skeleton """
        if dim == 2:
            line_nodex = np.r_[nodex[0],nodex[-1],np.nan].astype(np.float32)
            line_nodez = np.r_[nodez[0],nodez[-1],np.nan].astype(np.float32)
            mat_nodex = np.tile(line_nodex, (nodez.size,1))
            mat_nodez = np.tile(line_nodez, (nodex.size,1))
            line_horz = np.c_[mat_nodex.ravel('C'), np.repeat(nodez,3)]
            line_vert = np.c_[np.repeat(nodex,3), mat_nodez.ravel('C')]
            line_ = np.r_[line_horz,
                          line_vert,
                          ]
            
        elif dim == 3:
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

    @staticmethod
    def _plot_image(nodex, nodez, var):
        X, Z = np.meshgrid(nodex,nodez, indexing='ij') # cause our mesh order, need change sequence 
        Var = np.reshape(var,(nodex.size,nodez.size))  # so this method propably suitable to instance method
        return X, Z, Var





    # private method should ordered at last 
    def __str__(self) -> str:
        self._txt=string.Template("""
                                    Xmin(m)       ${xmin}    Xmax(m)           ${xmax}      mesh_x_length(m)       ${x_length}
                                    
                                    Ymin(m)       ${ymin}    Ymax(m)           ${ymax}      mesh_y_length(m)       ${y_length}
                                    
                                    Zmin(m)       ${zmin}    Zmax(m)           ${zmax}      mesh_z_length(m)       ${z_length}
                                  
                                    CellNum    ${cellnum}
        
        """)
        value = self._outfile
        return self._txt.safe_substitute(value)









if __name__ == '__main__':
    hx=[(-1.1,5),(2,4),(1.3,2)] 
    hy=[(-1,5),(2,2),(1.4,2)]
    hz=[(-1.5,20),(2,2)]
    h=[hx,hy,hz]
    c = TensorMesh(h)
    # Tensormesh.plot_image(mesh = c)
    print(c.nodex)
    print(c.nodey)
    print(c)
