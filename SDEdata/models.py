import numpy as np

class BaseModel(object):
    
    def __init__(self):
        self.params = locals()
        pass
    
    def integrate(self, X, t, dt):
        pass
    
    def simulate(self, x_init, dt, length, t_init = 0):
        time = np.arange(t_init,dt*length,dt)
        if type(x_init) is np.ndarray:
            x_init = [x_init]
    
        dim = x_init[0].shape[-1]          
        trajs = []
        for x0 in x_init:
            traj = np.zeros((length,dim))
            traj[0,:] = x0
            for i in range(1,length):
                traj[i,:] = self.integrate(traj[i-1,:], time[i-1], dt)
            trajs.append(traj.copy())
        return trajs,time
    
class BrownianMotion1d(BaseModel):

    def __init__(self,mu=0,sigma=1):
        super(BrownianMotion1d,self).__init__()
        self.mu = mu
        self.sigma = sigma
        
    def integrate(self,X,t,dt):
        super(BrownianMotion1d,self).integrate(X,t,dt)
        return X + self.mu*dt + self.sigma * np.random.normal(loc=0,scale=np.sqrt(dt))

    
class GeomBrownianMotion1d(BaseModel):
    
    def __init__(self,mu=0,sigma=1):
        super(GeomBrownianMotion1d,self).__init__()
        self.mu = mu
        self.sigma=sigma
        
    def integrate(self,X,t,dt):
        super(GeomBrownianMotion1d,self).integrate(X,t,dt)
        return X * np.exp((self.mu-.5*self.sigma**2 * dt +
                           self.sigma * np.random.normal(loc=0,scale=np.sqrt(dt))))
                          
                          
#class BrownianMotion(BaseModel):

#    def __init__(self,mu=0,cov=1):
#        super(BrownianMotion,self).__init__()
#        self.mu = mu
#        self.cov = cov
        
#    def integrate(self,X,t,dt):
#        super(BrownianMotion,self).integrate(X,t,dt)
#        if type(self.mu) is not np.ndarray:
#            return X + np.random.normal(loc=self.mu,scale=np.sqrt(self.cov))
#        else:
#            return X + np.random.multivariate_normal(mean=self.mu,cov=self.cov)

class Lorenz3d(BaseModel):
    
    def __init__(self,sigma=10,rho=28, b=8.0/3, epsilon=0.3):
        super(Lorenz,self).__init__()
        self.sigma=sigma
        self.rho=rho
        self.b=b
        self.epsilon=epsilon
        
    def integrate(self,X,t,dt):
        super(Lorenz,self).integrate(X,t,dt)
        X_ = np.ndarray(3)
        W = np.random.multivariate_normal(mean=np.zeros(3),cov=dt*np.eye(3))
        X_[0] = X[0] + dt*self.sigma*(X[1] - X[0]) + self.epsilon*X[0]*W[0]
        X_[1] = X[1] + dt* (self.rho*X[0] - X[1] - X[0]*X[2]) + self.epsilon*X[1]*W[1]
        X_[2] = X[2] + dt* (-self.b*X[2] + X[0]*X[1]) + self.epsilon * X[2] *W[2]
        return X_

class DoubleWell2d(BaseModel):
    
    def __init__(self,sigma=.7):
        super(DoubleWell2d,self).__init__()
        self.sigma=sigma
        
    def integrate(self,X,t,dt):
        super(DoubleWell2d,self).integrate(X,t,dt)
        
        W = np.random.multivariate_normal(mean=np.zeros(2),cov=dt*np.eye(2))
        
        X_0 = X[0] - dt*4*(X[0]**3) + dt*4*X[0] + self.sigma*W[0] 
        X_1 = X[1] - dt*(2*(X[1])) + self.sigma*W[0]
        return np.array([X_0,X_1])
    
class Roessler3d(BaseModel):
    
    def __init__(self,alpha=.1,beta=.1,gamma=14,sigma=0):
        super(Roessler3d,self).__init__()
        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma
        self.sigma=sigma
        
    def integrate(self,X,t,dt):
        super(Roessler3d,self).integrate(X,t,dt)
        
        W = np.random.multivariate_normal(mean=np.zeros(3),cov=dt*np.eye(3))
        
        X_0 = X[0] + dt*(- X[1] - X[2]) + self.sigma*W[0] 
        X_1 = X[1] + dt*(X[0] + self.alpha * X[1]) + self.sigma*W[1]
        X_2 = X[2] + dt*(self.beta + X[2]*(X[0]-self.gamma)) + self.sigma*W[2]
        return np.array([X_0,X_1,X_2])