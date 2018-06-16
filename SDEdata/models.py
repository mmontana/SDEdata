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
    
class BrownianMotion(BaseModel):

    def __init__(self,mu=0,cov=1):
        super(BrownianMotion,self).__init__()
        self.mu = mu
        self.cov = cov
        
    def integrate(self,X,t,dt):
        super(BrownianMotion,self).integrate(X,t,dt)
        if type(self.mu) is not np.ndarray:
            return X + np.random.normal(loc=self.mu,scale=np.sqrt(self.cov))
        else:
            return X + np.random.multivariate_normal(mean=self.mu,cov=self.cov)
    
    