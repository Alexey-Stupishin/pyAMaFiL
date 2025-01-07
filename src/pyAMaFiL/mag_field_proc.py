import numpy as np
from scipy.io import readsav
import astropy.units as u
import sunpy.sun.constants as sun
from pathlib import Path
from mag_field_wrapper import MagFieldWrapper
from mag_field_lin_fff import MagFieldLinFFF

__author__     = "Alexey G. Stupishin"
__email__      = "agstup@yandex.ru"
__copyright__  = "SUNCAST project, 2024"
__license__    = "MIT"
__version__    = "1.1.0"
__maintainer__ = "Alexey G. Stupishin"
__status__     = "beta"

# import pydoc

class MagFieldProcessor(MagFieldWrapper):
    FIELD_NONE = 0
    FIELD_LFFF = 0
    FIELD_NLFFF = 0

    #-------------------------------------------------------------------------------
    @staticmethod
    def _as_dict(record_):
        return {name:record_[name] for name in record_.dtype.names}

    #-------------------------------------------------------------------------------
    def __init__(self, lib_path = ""):
        super().__init__(lib_path)
        self.__bxb   = None
        self.__byb   = None
        self.__bzb   = None
        self.__stepb = None
        self.__bx    = None
        self.__by    = None
        self.__bz    = None
        self.__step  = None
        self.status  = self.FIELD_NONE

    # #-------------------------------------------------------------------------------
    # @property
    # def get_box_size(self):
    #     return ..........

    #-------------------------------------------------------------------------------
    @property
    def energy(self):
        # assert box is None
        N = self.__bx.shape.transpose((2,1,0))
        left = np.ceil(0.1*N).astype(np.int32)
        right = np.floor(0.9*N).astype(np.int32)
        absB2 = (self.__bx[left[0]:right[0],left[1]:right[1],1:right[2]]**2 
               + self.__by[left[0]:right[0],left[1]:right[1],1:right[2]]**2 
               + self.__bz[left[0]:right[0],left[1]:right[1],1:right[2]]**2 
                )
        totalB2 = np.sum(absB2)
        dr = self.__step.to(u.cm).value
        volume = np.prod(dr)

        return totalB2 / 8 / np.pi * volume

    #-------------------------------------------------------------------------------
    def get_field_cube(self, rc = 0):
        # assert box is None
        swap = (1,2,0)
        return dict(bx = self.__by.transpose(swap).copy(), by = self.__bx.transpose(swap).copy(), bz = self.__bz.transpose(swap).copy(), rc = rc)

    #-------------------------------------------------------------------------------
    def load_bottom(self, plane, dr = 0.001*u.solRad):
        self.__bxb = plane['by'].transpose(1,0).astype(np.float64, order="C")
        self.__byb = plane['bx'].transpose(1,0).astype(np.float64, order="C")
        self.__bzb = plane['bz'].transpose(1,0).astype(np.float64, order="C")
        self.__stepb = dr

    #-------------------------------------------------------------------------------
    def LFFF(self, z = None, pad = (1, 1), plane = None, dr = 0.001*u.solRad, alpha = 0):
        # ToDo: if plane, set it
        # assert is plane
        MagFieldLinFFF.set_field(self.__bzb, pad)
        
        if z is None:
            # z define by sizes
        
        pot = MagFieldLinFFF.lfff_cube(z, alpha)    

        # this cube, substitute

        return # something

        pass

    #-------------------------------------------------------------------------------
    def __load_vars(self, bx, by, bz, dr):
        self.__bx = bx.astype(np.float64, order="C")
        self.__by = by.astype(np.float64, order="C")
        self.__bz = bz.astype(np.float64, order="C")

        if np.isscalar(dr):
            self.__step = [dr, dr, dr]
        else:
            self.__step = np.flip(dr)
            
        self.status = self.FIELD_LFFF

        return self.get_field_cube()

    #-------------------------------------------------------------------------------
    def load_cube_sav(self, filename):

        sav_data = readsav(filename, python_dict = True)

        box = sav_data.get('box', sav_data.get('pbox'))

        box = self._as_dict(box[0])
        
        swap = (0,2,1)
        return self.__load_vars(np.transpose(box['BY'], swap), np.transpose(box['BX'], swap), np.transpose(box['BZ'], swap), box['DR'] * u.solRad)

    #-------------------------------------------------------------------------------
    def load_cube_vars(self, box, dr):
        """
            Set initial magnetic field components.

                Parameters
                ----------
                box['bx'], by, bz : float

                dr : array-like or singleton astropy.Quantity
                    
                Returns
                -------
                box : dict
                    A dictionary with keys "bx", "by", "bz"
                    bx, by, bz : ndarray
                        3D np.float64 arrays (input copy)
                
        """

        swap = (2,0,1)
        return self.__load_vars(box['by'].transpose(swap), box['bx'].transpose(swap), box['bz'].transpose(swap), dr)

    #-------------------------------------------------------------------------------
    def NLFFF(self
            , weight_bound_size = 0.1
            , derivative_stencil = 3
            , dense_grid_use = True
            , debug_input = False
             ):
        """
            Wrapper to external call of Weighted Wiegelmann NLFF Field Reconstruction Method.
            Magnetic field cube should be preliminary set. Field components modified "in place".

                Parameters
                ----------
                weight_bound_size : float, optional
                    Bounary buffer zone size (in parts of corresponding dim. size), 
                    default is 0.1 (i.e. buffer zone is 10% of dimension size from all boundaries, 
                    except photosphere plain). 
                    Use weight_bound_size = 0 for no-buffer-zone approach.
                    default = 0.1

                dense_grid_use : bool, optional
                    Use condensed grid approach. `True` is recommended.
                    default = True
                
                derivative_stencil : int, optional
                    Number of stencil points for derivatives
                    (experimental, internal use).
                    default = 3

                debug_input : bool, optional
                    For internal use.
                    default = False
                    
                Returns
                -------
                box : dict
                    A dictionary with keys "bx", "by", "bz", "rc"
                    bx, by, bz : ndarray
                        3D np.float64 arrays of magnetic field components
                    rc : int
                        zero if OK, otherwise reconstruction error (to be specified)
                
        """
        # assert box is None

        rc = super().NLFFF_wrapper(self.__bx, self.__by, self.__bz, weight_bound_size, derivative_stencil, dense_grid_use, debug_input)

        return self.get_field_cube(rc = rc)

#-------------------------------------------------------------------------------
    def lines(self
            , reduce_passed = None
            , chromo_level = 1
            , seeds = None
            , max_length = 0
            , reshape_3D = True
            , step = 1.0
            , tolerance = 1e-3
            , tolerance_bound = 1e-3
            , n_processes = 0
            , debug_input = False
             ):

        # assert box is None

        res = super().lines_wrapper(reduce_passed, chromo_level, seeds, max_length, step, tolerance, tolerance_bound, n_processes, debug_input)
        
        # if reshape_3D & (seeds is None):
        #     av_field = np.reshape(av_field, np.flip(self.__N))
        #     phys_length = np.reshape(phys_length, np.flip(self.__N))
        #     av_field = np.reshape(av_field, np.flip(self.__N))
        #     start_idx = np.reshape(start_idx, np.flip(self.__N))
        #     end_idx = np.reshape(end_idx, np.flip(self.__N))
        #     seed_idx = np.reshape(seed_idx, np.flip(self.__N))
        #     apex_idx = np.reshape(apex_idx, np.flip(self.__N))
        #     voxel_status = np.reshape(voxel_status, np.flip(self.__N))
        #     codes = np.reshape(codes, np.flip(self.__N))

        return res        
 
       # reorder - 2do ?

# pydoc.writedoc("mag_field_wrapper")
