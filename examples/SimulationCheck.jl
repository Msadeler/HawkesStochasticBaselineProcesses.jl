using PyCall
using Random
using HawkesStochasticBaselineProcesses
using DataFrames
using Optim
using LinearAlgebra



########### Simulation with python 

py"""
import numpy as np

def simulate_time(a,b,m_max,scheme_eds, baseline, t0=0, arg_mu ={}, arg_cov={}, initial_cov_value = 0, max_time = None):

    t=t0
    timestamps = [t]
    time_simu = [t]
    cov_value = [initial_cov_value]
    cov_timestamps = [initial_cov_value]
    aux = 0
    flag = t < max_time

    while flag:
        upper_intensity = m_max + aux*(aux>=0)
        t_after = t + np.random.exponential(1 / upper_intensity)
        
        time_simu+=[t_after]
        cov_value+=[scheme_eds(t_after, t_after- t,cov_value[-1],**arg_cov)]
        
        
        mu_x = baseline(cov_value[-1],  **arg_mu )
        candidate_intensity = mu_x + aux*np.exp(-b*(t_after - timestamps[-1]))
        
        flag = t_after < max_time

        if upper_intensity*np.random.uniform()<= candidate_intensity and flag :
            timestamps+=[t_after]
            cov_timestamps += [cov_value[-1]]
            aux = candidate_intensity + a- mu_x
            flag = t_after < max_time
        t = t_after
        
    timestamps += [max_time]


    timestamps = np.array(timestamps)
    time_simu = np.array(time_simu)
    cov_value = np.array(cov_value)
    cov_timestamps = np.array(cov_timestamps)
    return(timestamps, time_simu, cov_value, cov_timestamps)



mu_t = lambda x :  (np.array([0,0])-x)*0.05
scheme_eds  = lambda  t, delta_t, cov :cov + mu_t(cov)*delta_t + 0.05*np.random.normal(0, scale=np.sqrt(delta_t), size=2)
a,b = 0.6,1
arg_mu={'m1': 1, 'm2':0.2}
pc = np.array([0.1,0.1])
kernel = lambda z,m1,m2 : (m1-m2)*np.exp( -np.linalg.norm(z-pc)*10) + m2 


def testsimu(a,b, mu1, mu2):
    aaa =simulate_time(a,b, 20, scheme_eds = scheme_eds, baseline = kernel, arg_mu={'m1': mu1, 'm2':mu2},initial_cov_value = [0.0,0.0], max_time = 3000)
    return(aaa)
"""


using Distributions
using Plots


g₂(x)=1-exp(-norm(x-[0.02,0.02])*10) 
g₁(x)= exp(-norm(x-[0.02,0.02])*10)
coeff = [g₁; g₂]
gₘ = LinearFamilyBaseline(coeff)

### Xₜ is a 2-dimensionnal Ornstein–Uhlenbeck process : dXₜ = -b(a-Xₜ)dt + σdWₜ 

drift(x)= 0.05
diffusion(x)=-0.05.*x

model = HawkesStochasticBaseline(0.6, 1.0, [0.2,1];Mmax= 50, gₘ = gₘ, drift = drift, diffusion = diffusion, X₀=[0.0,0.0] )


start = time()
dfJl = rand(model, 3000.0)
time()- start


size(dfJl)

dfJl[dfJl.timestamps,:]

zJl = [dfJl.cov'...;]


plot(zJl[:,1], zJl[:,2])

start = time()
PyData =py"testsimu"(0.6,1, 1,0.2)
time()- start

dfPy = DataFrame(:time => PyData[2], :cov => [PyData[3][i,:] for i in 1:size(PyData[3],1)])
f(x) = in(x, PyData[1])
dfPy[!, :timestamps] = f.(dfPy.time)

size(dfPy)

dfPy[dfPy.timestamps,:]

zPy = [dfPy.cov'...;]

plot(zPy[:,1], zPy[:,2])
