import numpy as np
import gym
import acme_gym
from scipy import integrate, linalg as la
import time

### These first two functions are just adapted from the inverted_pendulum lab

def linearized_init(M, m, l, q1, q2, q3, q4, r):
    '''
    Adjusted for cart pole

    Parameters:
    ----------
    M, m: floats
        masses of the rickshaw and the present
    l   : float
        length of the rod
    q1, q2, q3, q4, r : floats
        relative weights of the position and velocity of the rickshaw,
        the angular displacement theta and the change in theta,
        and the control

    Return
    -------
    A : ndarray of shape (4,4)
    B : ndarray of shape (4,1)
    Q : ndarray of shape (4,4)
    R : ndarray of shape (1,1)
    '''
    g = 9.8
    a = 1./12
    """
    A = np.array([[0,1,0,0],
                  [0,0,-m*g/((M+2*m)*(2*a+1)-2*m),0],
                  [0,0,0,1],
                  [0,0, 1./(l*(2*a+1))*(g/2+m*g/((M+2*m)*(2*a+1)-2*m)),0]])
    """
    A = np.array([[0,1,0,0],
                  [0,0,-3*m*g/(4*M+m),0],
                  [0,0,0,1],
                  [0,0,(9*m*g)/(8*l*M + 2*l*m) + 3*g/(2*l),0]])
    
    #B = np.array([0, 2./(M+2*m), 0, -2./(M+2*m)])
    B = np.array([0,1./(M + m/4), 0, -3./(2*l*(M + m/4))])
    Q = np.diag([q1,q2,q3,q4])
    R = np.array([r])
    return A,B,Q,R

def cart(tv, X0, A, B, Q, R, P):
    '''
    adjusted for cart pole

    Parameters:
    ----------
    tv  : ndarray of time values, with shape (n+1,)
    X0  : Initial conditions on state variables
    A, Q: ndarrays of shape (4,4)
    B   : ndarray of shape (4,1)
    R   : ndarray of shape (1,1)
    P   : ndarray of shape (4,4)
    Returns
    -------
    Z : ndarray of shape (n+1,4), the state vector at each time
    U : ndarray of shape (n+1,), the control values
    '''
    def ode(z,t):
        return (A - np.outer(B,B.T@P)/R).dot(z)
    Z = integrate.odeint(ode, X0, tv)
    #print(B.shape)
    #print(P.shape)
    #print(Z.shape)
    U = -np.dot(B,P@Z.T)/R
    return Z,U


### This function isn't really finished, don't use it without revising

def control(state):
    ''' determing control from state,
    this isn't used yet '''
    # xp := x prime, thetap := theta prime
    x, xp, theta, thetap = state
    g = 9.8
    M = 1
    #m = .1 # or .05?
    #l = 1 # or .5?

    m = .05
    l = .5
    def costate_ode(p,t):
        p1,p2,p3,p4 = p
        return np.array([0,
                        (M+m)*xp - .5*m*l*thetap*np.cos(theta) - p1,
                        -((3*m*g*p2)/(4*M + m) + (9*m*g*p4)/(8*l*M + 2*l*m) + (3*g*p4)/(2*l) - .5*m*l*xp*thetap*np.sin(theta) - .5*m*g*l*np.sin(theta)),
                        -(p3 - m*(l**2)*thetap/3 + .5*m*l*xp*np.cos(theta))])
    integrate.odeint()
    return optcontrol


### Stuff we care about running

if __name__ == '__main__':
    T = round(12/0.02)
    env = gym.make('CartPoleContinuous-v0')
    obs = env.reset()
    env.render()
    #initial_state = np.array([0,0,.25,0])
    #env.state = initial_state
    #M,m,l = 1,.1,1
    M,m,l = 1,.1,.5
    A,B,Q,R = linearized_init(M,m,l,1,1,1,1,.1)
    P = la.solve_continuous_are(A,B.reshape((4,1)),Q,R)
    
    tol = 1e-2
    done = False
    for i in range(T):
        if la.norm(obs[1:]) < tol:
            print("Converged in {} iterations".format(i))
            time.sleep(1)
            env.close()
            break
            # Determine the step size based on our mode
        else:
            z, u = cart(np.arange(i*.02,T,.02), obs, A,B,Q,R,P)
            step = np.array([u[0]])
            
            # Step in designated direction and update the visual
            obs, reward, state, info = env.step(step)
            env.render()
            time.sleep(.02)
            if i == (T-1):
                print("Not converged, ",obs)
    """
    z, u = cart(np.arange(0,6,.02), obs, A,B,Q,R,P)
    z = z.astype(np.float32)
    u = u.astype(np.float32)
    
    for i in range(len(u)):
        state, reward, done, info = env.step(np.array([u[i]]))
        env.render()
        time.sleep(.02)
    """
    
    
    
    env.close()
