import numpy as np
from ukfm import SO3
from scipy.spatial.transform import Rotation as R

class OGSIMU:
    g = np.array([0, 0,0])
    class STATE:
        
        def __init__(self, quaternion, v, p, b_gyro, b_acc):
            self.quaternion = quaternion
            self.v = v
            self.p = p
            self.b_gyro = b_gyro
            self.b_acc = b_acc
            
    class INPUT:
            
        def __init__(self, gyro, acc):
            self.gyro = gyro
            self.acc = acc
            
    @classmethod
    def f(cls, state, omega, w, dt):
        r3 = R.from_quat(state.quaternion)
        Rot = r3.as_matrix()
        gyro = omega.gyro - state.b_gyro + w[:3]
        acc = Rot.dot(omega.acc - state.b_acc + w[3:6]) + cls.g
        
        Rot=Rot.dot(SO3.exp(gyro*dt))
        r3 = R.from_matrix(Rot)
        quaternion = r3.as_quat()
        new_state = cls.STATE(
            quaternion=quaternion,
            v=state.v + acc*dt,
            p=state.p + state.v*dt + 1/2*acc*dt**2,
            # noise is not added on biases
            b_gyro=state.b_gyro,
            b_acc=state.b_acc
        )
        return new_state
    
    @classmethod
    def h(cls, state):
        quaternion=state.quaternion
        p=state.p
        return np.hstack((quaternion,p))
    
    @classmethod
    def phi(cls, state, xi):
        
        new_state = cls.STATE(
            quaternion=state.quaternion + xi[0:4],
            v=state.v + xi[4:7],
            p=state.p + xi[7:10],
            b_gyro=state.b_gyro + xi[10:13],
            b_acc=state.b_acc + xi[13:16]
        )
        return new_state
    
    @classmethod
    def phi_inv(cls, state, hat_state):
        xi = np.hstack([hat_state.quaternion - state.quaternion,
                        hat_state.v - state.v,
                        hat_state.p - state.p,
                        hat_state.b_gyro - state.b_gyro,
                        hat_state.b_acc - state.b_acc])
        return xi
    
    @classmethod
    def up_phi(cls, state, xi):
        new_state = cls.STATE(
            quaternion=xi[0:4] + state.quaternion,
            v=state.v,
            p=xi[4:7] + state.p,
            b_gyro=state.b_gyro,
            b_acc=state.b_acc
        )
        return new_state