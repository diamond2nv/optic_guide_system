from Config_loader import Config_loader
import numpy as np
import ukfm

def block_diag(*arrs):
    """
    Create a block diagonal matrix from provided arrays.

    Given the inputs `A`, `B` and `C`, the output will have these
    arrays arranged on the diagonal::

        [[A, 0, 0],
         [0, B, 0],
         [0, 0, C]]

    Parameters
    ----------
    A, B, C, ... : array_like, up to 2-D
        Input arrays.  A 1-D array or array_like sequence of length `n` is
        treated as a 2-D array with shape ``(1,n)``.

    Returns
    -------
    D : ndarray
        Array with `A`, `B`, `C`, ... on the diagonal. `D` has the
        same dtype as `A`.

    Notes
    -----
    If all the input arrays are square, the output is known as a
    block diagonal matrix.

    Empty sequences (i.e., array-likes of zero size) will not be ignored.
    Noteworthy, both [] and [[]] are treated as matrices with shape ``(1,0)``.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import block_diag
    >>> A = [[1, 0],
    ...      [0, 1]]
    >>> B = [[3, 4, 5],
    ...      [6, 7, 8]]
    >>> C = [[7]]
    >>> P = np.zeros((2, 0), dtype='int32')
    >>> block_diag(A, B, C)
    array([[1, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0],
           [0, 0, 3, 4, 5, 0],
           [0, 0, 6, 7, 8, 0],
           [0, 0, 0, 0, 0, 7]])
    >>> block_diag(A, P, B, C)
    array([[1, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 3, 4, 5, 0],
           [0, 0, 6, 7, 8, 0],
           [0, 0, 0, 0, 0, 7]])
    >>> block_diag(1.0, [2, 3], [[4, 5], [6, 7]])
    array([[ 1.,  0.,  0.,  0.,  0.],
           [ 0.,  2.,  3.,  0.,  0.],
           [ 0.,  0.,  0.,  4.,  5.],
           [ 0.,  0.,  0.,  6.,  7.]])

    """
    if arrs == ():
        arrs = ([],)
    arrs = [np.atleast_2d(a) for a in arrs]

    bad_args = [k for k in range(len(arrs)) if arrs[k].ndim > 2]
    if bad_args:
        raise ValueError("arguments in the following positions have dimension "
                         "greater than 2: %s" % bad_args)

    shapes = np.array([a.shape for a in arrs])
    out_dtype = np.result_type(*[arr.dtype for arr in arrs])
    out = np.zeros(np.sum(shapes, axis=0), dtype=out_dtype)

    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        out[r:r + rr, c:c + cc] = arrs[i]
        r += rr
        c += cc
    return out

class UKF_Filter():
    def __init__(self,params) -> None:
        self.params=params
        self.MODEL = ukfm.IMUGNSS
        
        imu_std = np.array([0.01,     # gyro (rad/s)
                    0.05,     # accelerometer (m/s^2)
                    0.000001, # gyro bias (rad/s^2)
                    0.0001])  # accelerometer bias (m/s^3)
        
        GNSS_std = 0.05
 
        alpha = np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
        # for propagation we need the all state
        red_idxs = np.arange(15)  # indices corresponding to the full state in P
        # for update we need only the state corresponding to the position
        up_idxs = np.array([6, 7, 8])
               
        self.Q = block_diag(imu_std[0]**2*np.eye(3), imu_std[1]**2*np.eye(3),
               imu_std[2]**2*np.eye(3), imu_std[3]**2*np.eye(3))
        
        self.R = GNSS_std**2 * np.eye(3)
        

        
        P0 = block_diag(0.01*np.eye(3), 1*np.eye(3), 1*np.eye(3),
                0.001*np.eye(3), 0.001*np.eye(3))
        state0 = self.MODEL.STATE(
            Rot=np.eye(3),
            v=np.zeros(3),
            p=np.zeros(3),
            b_gyro=np.zeros(3),
            b_acc=np.zeros(3))
        
        self.ukf = ukfm.JUKF(state0=state0, P0=P0, f=self.MODEL.f, h=self.MODEL.h, Q=self.Q[:6, :6],
                phi=self.MODEL.phi, alpha=alpha, red_phi=self.MODEL.phi,
                red_phi_inv=self.MODEL.phi_inv, red_idxs=red_idxs,
                up_phi=self.MODEL.up_phi, up_idxs=up_idxs)
        
    