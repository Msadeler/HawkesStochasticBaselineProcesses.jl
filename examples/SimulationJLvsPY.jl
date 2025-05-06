using PyCall
using Random
using HawkesStochasticBaselineProcesses
using DataFrames
using Optim
using LinearAlgebra
using Makie


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




scheme_eds  = lambda  t, delta_t, cov :cov + -cov*0.05*delta_t + 0.05*np.random.normal(0, scale=np.sqrt(delta_t), size=2) #+ mu_t(cov)*delta_t + 0.05*np.random.normal(0, scale=np.sqrt(delta_t), size=2)

pc = np.array([0.1,0.1])


def kernel(z,m1,m2):
    return((m1-m2)*np.exp( -np.linalg.norm(z-pc)*10) + m2 )
#kernel = lambda z,mu : abs(z)*mu

def testsimu(a,b, mu1, mu2):
    simu_output =simulate_time(a,b, 20, scheme_eds = scheme_eds, baseline = kernel, arg_mu={'mu1':mu1, 'mu2':mu2},initial_cov_value = [0,0], max_time = 3000)
    return(simu_output)

"""


using Distributions
using Plots

py"scheme_eds"(0.0,0.1,[0.0,0.0])


##################################################
############# Simulation avec julia ##############
##################################################


g(x)=abs(x)
g₁(x)=1-exp(-norm(x-[0.1,0.1])*10) 
g₂(x)= exp(-norm(x-[0.1,0.1])*10)
coeff = [g₁; g₂]
gₘ = LinearFamilyBaseline(coeff)

### Xₜ is a 2-dimensionnal Ornstein–Uhlenbeck process : dXₜ = -b(a-Xₜ)dt + σdWₜ 

drift(x)= 0.05
diffusion(x)=-0.05.*x

model = HawkesStochasticBaseline(0.6, 1.0, [1.0,1.0];Mmax= 20, gₘ = gₘ,drift=drift , diffusion = diffusion, X₀=[0.0, 0.0])


dfJl = rand(model, 3000.0)

nrep = 300
paramSimul = zeros(nrep, 4)

PyData =py"testsimu"(0.6,1, 1,0.2)

for k in 1:nrep
    start = time()
    dfJl = rand(model, 3000.0)
    time()- start
    
    
    paramSimul[k,1] = size(dfJl,1) ## nombre de point simulé 
    
    paramSimul[k,2] = size(dfJl[dfJl.timestamps,:],1) ## nombre de saut 
    
    #zJl = [dfJl.cov'...;]  ### matrice de covariavle
    
    
    start = time()
    PyData =py"testsimu"(0.6,1, 1,0.2)
    time()- start
    
    dfPy = DataFrame(:time => PyData[2], :cov => [PyData[3][i,:] for i in 1:size(PyData[3],1)])
    f(x) = in(x, PyData[1])
    dfPy[!, :timestamps] = f.(dfPy.time)
    
    paramSimul[k,3] =size(dfPy,1) ## nombre de point simulé 
    paramSimul[k,4] =  size(dfPy[dfPy.timestamps,:],1) ## nombre de saut 
    #zPy = [dfPy.cov'...;] ### matrice de covariavle
    

    start = time()
    dfJl = rand(model, 3000.0)
    time()- start


    paramSimul[k,1] = size(dfJl,1) ## nombre de point simulé 

    paramSimul[k,2] = size(dfJl[dfJl.timestamps,:],1) ## nombre de saut 

    #zJl = [dfJl.cov'...;]  ### matrice de covariavle


    start = time()
    PyData =py"testsimu"(0.6,1, 1)
    time()- start

    dfPy = DataFrame(:time => PyData[2], :cov => [PyData[3][i,:] for i in 1:size(PyData[3],1)])
    f(x) = in(x, PyData[1])
    dfPy[!, :timestamps] = f.(dfPy.time)

    paramSimul[k,3] =size(dfPy,1) ## nombre de point simulé 
    paramSimul[k,4] =  size(dfPy[dfPy.timestamps,:],1) ## nombre de saut 
    #zPy = [dfPy.cov'...;] ### matrice de covariavle

end


using CairoMakie
boxplot(x=paramSimul[:,3], y =1.0)
boxplot(Any[paramSimul[:,2], paramSimul[:,4]], fillcolor=[:red :black], fillalpha=0.2)

