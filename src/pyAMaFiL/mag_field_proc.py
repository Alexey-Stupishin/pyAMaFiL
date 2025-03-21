import numpy as np
from scipy.io import readsav
import astropy.units as u
from .mag_field_wrapper import MagFieldWrapper
from .mag_field_lin_fff import MagFieldLinFFF

__author__     = "Alexey G. Stupishin"
__email__      = "agstup@yandex.ru"
__copyright__  = "SUNCAST project, 2024"
__license__    = "MIT"
__version__    = "1.1.2"
__maintainer__ = "Alexey G. Stupishin"
__status__     = "beta"

# import pydoc

def mfp_util_invert_index(index, sizes):
    kx = index % sizes[0]
    kyz = index // sizes[0]
    ky = kyz % sizes[1]
    kz = kyz // sizes[1]
    
    return (kz*sizes[0] + kx)*sizes[1] + ky

def mfp_util_invert_index_array(data, sizes):
    _data = data.copy()
    for k in range(0, len(data)):
        _data[k] = mfp_util_invert_index(data[k], sizes)

    return _data

def mfp_util_transpose_index(data, sizes, transpose):
    if not transpose:
        return data
    
    _data = data.copy()
    for k in range(0, len(data)):
        inv = mfp_util_invert_index(k, sizes)
        _data[inv] = data[k]
        
    return _data

class MagFieldProcessor(MagFieldWrapper):
    FIELD_NONE =  0
    FIELD_BOUND = 1
    FIELD_NLFFF = 2

    #-------------------------------------------------------------------------------
    @staticmethod
    def _as_dict(record_):
        return {name:record_[name] for name in record_.dtype.names}

    #-------------------------------------------------------------------------------
    def __init__(self, lib_path = ""):
        super().__init__(lib_path)
        self.__bx    = None
        self.__by    = None
        self.__bz    = None
        self.__step  = None
        self.__status  = self.FIELD_NONE
        self.__weight_bound_size = 0.1

    #-------------------------------------------------------------------------------
    def energy(self, box = None, weight_bound_size = None, dr = None):
        if box is None:
            box = self.get_field_cube()
        if weight_bound_size is None:
            weight_bound_size = self.__weight_bound_size
        if dr is None:
            dr = self.__step

        assert dr is not None
            
        dr = dr.to(u.cm).value
        if np.isscalar(dr):
            dr = [dr, dr, dr]
        
        N = np.array(np.array(box['bx']).shape)
        left = np.floor(weight_bound_size*N).astype(np.int32)
        right = np.ceil((1.0-weight_bound_size)*N).astype(np.int32)
        absB2 = (box['bx'][left[0]:right[0],left[1]:right[1],0:right[2]]**2 
               + box['by'][left[0]:right[0],left[1]:right[1],0:right[2]]**2 
               + box['bz'][left[0]:right[0],left[1]:right[1],0:right[2]]**2 
                )
        totalB2 = np.sum(absB2)
        volume = np.prod(dr)

        return totalB2 / 8 / np.pi * volume

    #-------------------------------------------------------------------------------
    def load_cube_vars(self, box, dr = 0.001*u.solRad):
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
    def load_bottom(self, bottom):
        assert self.__bx is not None and self.__by is not None and self.__bz is not None
        
        cube = self.get_field_cube()
        cube['bx'][:,:,0] = bottom['by'].transpose(1,0).astype(np.float64, order="C")
        cube['by'][:,:,0] = bottom['bx'].transpose(1,0).astype(np.float64, order="C")
        cube['bz'][:,:,0] = bottom['bz'].transpose(1,0).astype(np.float64, order="C")
        self.__status  = self.FIELD_BOUND
        
        swap = (2,1,0)
        return self.__load_vars(cube['bx'].transpose(swap), cube['by'].transpose(swap), cube['bz'].transpose(swap), self.__step)

    #-------------------------------------------------------------------------------
    def LFFF_bounded(self, bottom, pad = (1, 1), nz = None, dr = 0.001*u.solRad, alpha = 0):
        
        lfff = MagFieldLinFFF.create_LFFF_cube(bottom['bz'].astype(np.float64, order="C"), pad = pad, nz = nz, alpha = alpha)
        self.load_cube_vars(lfff, dr = dr)

        return self.load_bottom(bottom)

    #-------------------------------------------------------------------------------
    def sav_to_cube(self, filename):

        sav_data = readsav(filename, python_dict = True)

        box = sav_data.get('box', sav_data.get('pbox'))

        return self._as_dict(box[0])
        
    #-------------------------------------------------------------------------------
    def load_cube_sav(self, filename):

        box = self.sav_to_cube(filename)
        
        swap = (0,2,1)
        return self.__load_vars(np.transpose(box['BY'], swap), np.transpose(box['BX'], swap), np.transpose(box['BZ'], swap), box['DR'] * u.solRad)
    
    #-------------------------------------------------------------------------------
    def __load_vars(self, bx, by, bz, dr):
        self.__bx = bx.astype(np.float64, order="C")
        self.__by = by.astype(np.float64, order="C")
        self.__bz = bz.astype(np.float64, order="C")

        if np.isscalar(dr):
            self.__step = [dr, dr, dr]
        else:
            self.__step = dr
            
        self.status = self.FIELD_BOUND

        return self.get_field_cube()

    #-------------------------------------------------------------------------------
    def get_field_cube(self, rc = 0):
        assert self.__bx is not None and self.__by is not None and self.__bz is not None
        swap = (1,2,0)
        return dict(bx = self.__by.transpose(swap).copy(), by = self.__bx.transpose(swap).copy(), bz = self.__bz.transpose(swap).copy(), rc = rc)

    #-------------------------------------------------------------------------------
    def NLFFF(self, **kwargs):
        """
            Wrapper to external call of Weighted Wiegelmann NLFF Field Reconstruction Method.
            Magnetic field cube should be preliminary set. Field components modified "in place".

                Parameters (**kwargs)
                ---------------------
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

                Returns
                -------
                box : dict
                    A dictionary with keys "bx", "by", "bz", "rc"
                    bx, by, bz : ndarray
                        3D np.float64 arrays of magnetic field components
                    rc : int
                        zero if OK, otherwise reconstruction error (to be specified)
                
        """
        
        assert self.__bx is not None and self.__by is not None and self.__bz is not None

        rc = super().NLFFF_wrapper(self.__bx, self.__by, self.__bz, **kwargs)

        self.__weight_bound_size = super().get_double('weight_bound_size')

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

        assert self.__bx is not None and self.__by is not None and self.__bz is not None

        res = super().lines_wrapper(self.__bx, self.__by, self.__bz, reduce_passed, chromo_level, seeds, max_length, step, tolerance, tolerance_bound, n_processes, debug_input)
        
        N = np.flip(self.__bx.shape)
        transpose = seeds is None
        res['voxel_status'] = mfp_util_transpose_index(res['voxel_status'], N, transpose)
        res['phys_length'] = mfp_util_transpose_index(res['phys_length'], N, transpose)
        res['av_field'] = mfp_util_transpose_index(res['av_field'], N, transpose)
        res['codes'] = mfp_util_transpose_index(res['codes'], N, transpose)

        res['apex_idx'] = mfp_util_transpose_index(mfp_util_invert_index_array(res['apex_idx'], N), N, transpose)
        res['start_idx'] = mfp_util_transpose_index(mfp_util_invert_index_array(res['start_idx'], N), N, transpose)
        res['end_idx'] = mfp_util_transpose_index(mfp_util_invert_index_array(res['end_idx'], N), N, transpose)
        res['seed_idx'] = mfp_util_transpose_index(mfp_util_invert_index_array(res['seed_idx'], N), N, transpose)

        t = res['coords'][:, 0].copy()
        res['coords'][:, 0] = res['coords'][:, 1].copy()
        res['coords'][:, 1] = t
        
        return res        
 
# pydoc.writedoc("MagFieldProcessor")
