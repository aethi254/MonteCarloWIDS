import numpy as np 
import matplotlib.pyplot as plt 

def estimatepi(N):
    x=np.random.uniform(-1,1,N)
    y=np.random.uniform(-1,1,N)
    inside=np.sum(x**2+y**2<=1)
    result=4*inside/N
    return result

def piwithN():
    values=[10, 100, 1000, 10000, 100000, 1000000, 10000000]
    estimates=[]
    errors=[]
    for value in values:
        piestimate=estimatepi(value)
        estimates.append(piestimate)
        error=abs(piestimate-np.pi)/np.pi * 100
        errors.append(error)
    
    return values,estimates,errors

def emethod1(N,t=2):
    x=np.random.uniform(1,t,N)
    y=np.random.uniform(0,1,N)
    undercurve=np.sum(y < 1/x)
    area=(t-1)*undercurve / N
    if area>0:
        return t**(1/area)
    return np.nan

def emethod2(numtests,maxlen=100):
    samples=np.random.uniform(0,1,(numtests,maxlen))
    samples[:,0]=1.0
    diffs=samples[:,1:]-samples[:,:-1]
    stops=np.argmax(diffs>=0,axis=1) + 1
    stops[np.all(diffs<0,axis=1)]=maxlen
    return np.mean(stops)

class MonteCarlo:
    def __init__(self, shapefunc, xmin, xmax, ymin, ymax, trueval=None):
        self.shapefunc=shapefunc
        self.bounds=(xmin,xmax,ymin,ymax)
        self.trueval=trueval

    def estimate(self, N):
        xmin,xmax,ymin,ymax = self.bounds
        x=np.random.uniform(xmin,xmax,N)
        y=np.random.uniform(ymin,ymax,N)
        inside=self.shapefunc(x, y)
        ratio=np.sum(inside)/N
        boxarea=(xmax-xmin)*(ymax-ymin)
        return boxarea*ratio

    def convergence(self,values):
        results=[]
        errors=[]
        for value in values:
            est=self.estimate(value)
            results.append(est)
            if self.trueval:
                err=abs(est-self.trueval) / self.trueval * 100
                errors.append(err)
        return results,errors

def circle(x,y):
    return x**2+y**2 <= 1

def parabola(x,y):
    return y <= x**2

def gaussian(x,y):
    return y <= np.exp(-x**2)
