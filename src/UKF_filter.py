from Config_loader import Config_loader
import numpy as np
import ukfm
import os
import matplotlib.pyplot as plt

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
        
def test():
    _BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    collect_data_path = _BASE_DIR+str("/data/transform_matrix_22.npy")
    tmx=np.load(collect_data_path)
    MODEL = ukfm.IMUGNSS
    N = tmx.shape[0]
    # observation frequency (Hz)
    GNSS_freq = 1

    imu_std = np.array([0.0001,     # gyro (rad/s)
                        0.0001,     # accelerometer (m/s^2)
                        0.1, # gyro bias (rad/s^2)
                        0.1])  # accelerometer bias (m/s^3)
    # GNSS noise standard deviation (m)
    GNSS_std = 0.005
    Q = block_diag(imu_std[0]**2*np.eye(3), imu_std[1]**2*np.eye(3),
               imu_std[2]**2*np.eye(3), imu_std[3]**2*np.eye(3))
    # measurement noise covariance matrix
    R = GNSS_std**2 * np.eye(3)

    ################################################################################
    # We use the UKF that is able to infer Jacobian to speed up the update step, see
    # the 2D SLAM example.

    # sigma point parameters
    alpha = np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-3])
    # for propagation we need the all state
    red_idxs = np.arange(15)  # indices corresponding to the full state in P
    # for update we need only the state corresponding to the position
    up_idxs = np.array([6, 7, 8])
    P0 = block_diag(0.01*np.eye(3), 1*np.eye(3), 1*np.eye(3),
                0.001*np.eye(3), 0.001*np.eye(3))
    # initial state
    state0 = MODEL.STATE(
        Rot=np.eye(3),
        v=np.zeros(3),
        p=np.zeros(3),
        b_gyro=np.zeros(3),
        b_acc=np.zeros(3))
    ukf = ukfm.JUKF(state0=state0, P0=P0, f=MODEL.f, h=MODEL.h, Q=Q[:6, :6],
                phi=MODEL.phi, alpha=alpha, red_phi=MODEL.phi,
                red_phi_inv=MODEL.phi_inv, red_idxs=red_idxs,
                up_phi=MODEL.up_phi, up_idxs=up_idxs)
    # set variables for recording estimates along the full trajectory
    ukf_states = [state0]
    ukf_Ps = np.zeros((N, 15, 15))
    ukf_Ps[0] = P0
    # the part of the Jacobian that is already known.
    G_const = np.zeros((15, 6))
    G_const[9:] = np.eye(6)
    one_hot_tmx=np.zeros(N)
    for i in range(1, N):
        if not np.isnan(tmx[i,:,:]).all():
            one_hot_tmx[i]=1
            
    omegas= []
    omegas_rand=np.array([0.2,0.2,0.2])
    for i in range(1, N):
        #生成随机的gyro和acc
        gyro = np.random.randn(3) * omegas_rand
        acc = np.random.randn(3) * omegas_rand
        omegas.append(MODEL.INPUT(gyro=gyro, acc=acc))
    k=1 
    ys_array = []   
    ukf_p_array=[]
    for n in range(1, 1000):
        dt=1
        ukf.state_propagation(omegas[n-1], dt)
        ukf.F_num(omegas[n-1], dt)
        # we assert the reduced noise covariance for computing Jacobian.
        ukf.Q = Q[:6, :6]
        ukf.G_num(omegas[n-1], dt)
        # concatenate Jacobian
        ukf.G = np.hstack((ukf.G, G_const*dt))
        # we assert the full noise covariance for uncertainty propagation.
        ukf.Q = Q
        ukf.cov_propagation()
        # update
        
        if one_hot_tmx[n] == 1:
            
            ukf.update(tmx[n,0:3,3], R)
            ys_array.append(tmx[n,0:3,3])
            k = k + 1
        # save estimates
        ukf_p_array.append(ukf.state.p)
        ukf_states.append(ukf.state)
        print("tmx",tmx[n,0:3,3])
        print("ukf.state",ukf.state.p)
        ukf_Ps[n] = ukf.P
        print(n)
    ys_array = np.array(ys_array)    
    ys_array.reshape(-1,3)
    ukf_p_array=np.array(ukf_p_array)
    ukf_p_array.reshape(-1,3)

    plt.figure()
    plt.plot(ys_array[:,2],'r')
    plt.plot(ukf_p_array[:,2],'b')
    plt.savefig('./test.png')

    # MODEL.plot_results(ukf_states, ys_array)
    pass



test()