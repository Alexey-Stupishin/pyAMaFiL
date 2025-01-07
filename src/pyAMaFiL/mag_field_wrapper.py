import ctypes
import numpy as np
from numpy import linalg as LA
from pathlib import Path

__author__     = "Alexey G. Stupishin"
__email__      = "agstup@yandex.ru"
__copyright__  = "SUNCAST project, 2024"
__license__    = "MIT"
__version__    = "1.1.0"
__maintainer__ = "Alexey G. Stupishin"
__status__     = "beta"

# import pydoc

class MagFieldWrapper:
    PASSED_NONE   = 0
    PASSED_CLOSED = 1
    PASSED_OPENED = 2

    #-------------------------------------------------------------------------------
    def __init__(self, lib_path = ""):
        if lib_path == "":
            lib_path = list(Path(__file__).parent.glob("WWNLFFFReconstruction*"))[0]
            #print(lib_path)

        self.__mptr1 = np.ctypeslib.ndpointer(dtype = np.float64, ndim = 1, flags = "C")
        self.__mptr2 = np.ctypeslib.ndpointer(dtype = np.float64, ndim = 2, flags = "C")
        self.__mptr3 = np.ctypeslib.ndpointer(dtype = np.float64, ndim = 3, flags = "C")
        self.__mpint1 = np.ctypeslib.ndpointer(dtype = np.int32, ndim = 1, flags = "C")
        self.__mpint2 = np.ctypeslib.ndpointer(dtype = np.int32, ndim = 2, flags = "C")
        self.__mpint3 = np.ctypeslib.ndpointer(dtype = np.int32, ndim = 3, flags = "C")
        self.__mp64 = np.ctypeslib.ndpointer(dtype = np.uint64, ndim = 1, flags = "C")
        self.__mpstr = ctypes.POINTER(ctypes.c_char)
        self.__mvoid = ctypes.c_void_p
        self.__mint = ctypes.c_int32
        self.__mreal = ctypes.c_double
        self.__mdw = ctypes.c_uint32
        self.__m64 = ctypes.c_uint64
 
        lib_mfw = ctypes.CDLL(lib_path)

        create_func = lib_mfw.utilInitialize
        create_func.argtypes = []
        create_func.restype = self.__mdw

        set_int_func = lib_mfw.utilSetInt
        set_int_func.argtypes = [self.__mpstr, self.__mint]
        set_int_func.restype = self.__mint

        set_double_func = lib_mfw.utilSetDouble
        set_double_func.argtypes = [self.__mpstr, self.__mreal]
        set_double_func.restype = self.__mint

        get_version_func = lib_mfw.utilGetVersion
        get_version_func.argtypes = [self.__mpstr, self.__mint]
        get_version_func.restype = self.__mint

        NLFFF_func = lib_mfw.mfoNLFFFCore
        NLFFF_func.argtypes = [self.__mpint1, self.__mptr3, self.__mptr3, self.__mptr3]
        NLFFF_func.restype = self.__mint

        lines_func = lib_mfw.mfoGetLines
        # lines_func.argtypes sets dynamically
        lines_func.restype = self.__mdw
        
        self.__lib_path = lib_path
        self.__func_set = {'create_func':create_func
                         , 'set_int_func':set_int_func
                         , 'set_double_func':set_double_func
                         , 'get_version_func':get_version_func
                         , 'NLFFF_func':NLFFF_func
                         , 'lines_func':lines_func
                          }
        self.__pointer = create_func()
    
    #-------------------------------------------------------------------------------
    def set_int(self, prop, vint):
        return self.__func_set['set_int_func'](prop.encode('utf-8'), np.int32(vint))

    #-------------------------------------------------------------------------------
    def set_double(self, prop, vdouble):
        return self.__func_set['set_double_func'](prop.encode('utf-8'), np.float64(vdouble))

    #-------------------------------------------------------------------------------
    @property
    def get_version(self):
        buflen = 512
        bufenc = ''.ljust(buflen).encode('utf-8')
        rc = self.__func_set['get_version_func'](bufenc, buflen)
        buffer = bufenc.decode('utf-8')
        term = buffer.find(chr(0))

        return buffer[0:term]
        
#-------------------------------------------------------------------------------
    def NLFFF_wrapper(self
            , bx, by, bz  
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

        self.set_double('weight_bound_size', weight_bound_size)
        self.set_int('derivative_stencil', derivative_stencil)
        self.set_int('dense_grid_use', int(dense_grid_use == True))
        self.set_int('debug_input', int(debug_input == True))

        Nc = bx.shape
        N = np.array([Nc[2], Nc[1], Nc[0]], dtype = np.int32)
        rc = self.__func_set['NLFFF_func'](N, bx, by, bz)

        return self.get_field(rc = rc)

#-------------------------------------------------------------------------------
    def est_max_coords(self, N, n_total = 0):
        line_length_est = LA.norm(N)
        if n_total == 0:
            n_total = np.prod(N)
        return np.int64(np.ceil(line_length_est*n_total))
    
#-------------------------------------------------------------------------------
    def lines_wrapper(self
            , bx, by, bz  
            , reduce_passed = None
            , chromo_level = 1
            , seeds = None
            , max_length = 0
            , step = 1.0
            , tolerance = 1e-3
            , tolerance_bound = 1e-3
            , n_processes = 0
            , debug_input = False
             ):

        self.set_int('debug_input', int(debug_input == True))

        # assert box is None
        seeds_type = self.__mvoid
        if seeds is None:
            n_seeds = 0
            arg_seeds = 0
            n_total = self.__bx.size
            if reduce_passed is None:
                reduce_passed = self.PASSED_CLOSED | self.PASSED_OPENED
        else:
            # assert seeds is not 2D
            arg_seeds = np.array(seeds.transpose((1, 0)), dtype = np.float64, order = 'C')
            n_seeds = arg_seeds.shape[0]
            seeds_type = self.__mptr2
            n_total = n_seeds
            if reduce_passed is None:
                reduce_passed = self.PASSED_NONE

        if max_length < 0:
            max_length = self.est_max_coords(bx.shape, n_total)
        max_length = np.int64(max_length)

        n_lines = np.array([1], dtype = np.int32)
        n_passed = np.array([1], dtype = np.int32)
        voxel_status = np.zeros([n_total], dtype = np.int32)
        phys_length = np.zeros([n_total], dtype = np.float64)
        av_field = np.zeros([n_total], dtype = np.float64)
        codes = np.zeros([n_total], dtype = np.int32)
        start_idx = np.zeros([n_total], dtype = np.int32)
        end_idx = np.zeros([n_total], dtype = np.int32)
        apex_idx = np.zeros([n_total], dtype = np.int32)
        seed_idx = np.zeros([n_total], dtype = np.int32)
        total_length = np.array([1], dtype = np.uint64)
        
        if max_length == 0:
            coords_type = self.__mvoid
            coords = 0
            ls_type = self.__mvoid
            lines_start = 0
            lv_type = self.__mvoid
            lines_length = 0
            lines_index = 0
        else:
            coords_type = self.__mptr2
            coords = np.zeros([max_length, 4], dtype = np.float64, order="C")
            ls_type = self.__mp64
            lines_start = np.zeros([n_total], dtype = np.uint64)
            lv_type = self.__mpint1
            lines_length = np.zeros([n_total], dtype = np.int32)
            lines_index = np.zeros([n_total], dtype = np.int32)

        lines_func = self.__func_set['lines_func']
        lines_func.argtypes = [self.__mpint1, self.__mptr3, self.__mptr3, self.__mptr3   # 1-4
                             , self.__mdw, self.__mreal                              #   5-6 uint32_t _cond = 0x3, REALTYPE_A chromoLevel = 0,
                             , seeds_type, self.__mint                               #   7-8 REALTYPE_A  *_seeds = nullptr, int _Nseeds = 0,
                             , self.__mint, self.__mreal, self.__mreal, self.__mreal #   9-12 int nProc = 0, REALTYPE_A step = 1.0, REALTYPE_A tolerance = 1e-3, REALTYPE_A boundAchieve = 1e-3,
                             , self.__mpint1, self.__mpint1                          #   13-14 int *_nLines = nullptr, int *_nPassed = nullptr,
                             , self.__mpint1, self.__mptr1, self.__mptr1             #   15-17 int *_voxelStatus = nullptr, REALTYPE_A *_physLength = nullptr, REALTYPE_A *_avField = nullptr,
                             , lv_type, self.__mpint1                                #   18-19 int *_linesLength = nullptr, int *_codes = nullptr,
                             , self.__mpint1, self.__mpint1, self.__mpint1           #   20-22 int *_startIdx = nullptr, int *_endIdx = nullptr, int *_apexIdx = nullptr,
                             , self.__m64, self.__mp64, coords_type                  #   23-25 uint64_t _maxCoordLength = 0, uint64_t *_totalLength = nullptr, REALTYPE_A *_coords = nullptr, 
                             , ls_type, lv_type, self.__mpint1                       #   26-28 uint64_t *_linesStart = nullptr, int *_linesIndex = nullptr, int *seedIdx = nullptr);
                              ]

        non_passed = lines_func(self.__N, self.__bx, self.__by, self.__bz
                      , reduce_passed, chromo_level 
                      , arg_seeds, n_seeds
                      , n_processes, step, tolerance, tolerance_bound
                      , n_lines, n_passed
                      , voxel_status, phys_length, av_field
                      , lines_length, codes
                      , start_idx, end_idx, apex_idx
                      , max_length, total_length, coords
                      , lines_start, lines_index, seed_idx
                       )
        
        return dict(n_lines = n_lines[0]
                  , n_passed = n_passed[0]
                  , non_passed = non_passed
                  , voxel_status = voxel_status
                  , phys_length = phys_length
                  , av_field = av_field
                  , lines_length = lines_length
                  , codes = codes
                  , start_idx = start_idx
                  , end_idx = end_idx
                  , apex_idx = apex_idx
                  , max_length = max_length
                  , total_length = total_length
                  , coords = coords
                  , lines_start = lines_start
                  , lines_index = lines_index
                  , seed_idx = seed_idx
                   )        
 
# pydoc.writedoc("mag_field_wrapper")
