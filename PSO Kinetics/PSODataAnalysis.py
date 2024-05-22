
#%%
import pathlib as plib
import matplotlib.pyplot as plt
from tga_data_analysis.tga import Project, Sample
from pyswarm import pso
import numpy as np 
from scipy.signal import savgol_filter as SavFil
from scipy.optimize import minimize
from scipy.optimize import Bounds
import time
from IPython import display
import pylab as pl

folder_path = plib.Path('/Users/charlottealbunio/Documents/Python/tga_data_analysis/PSO Kinetics/data')

# %% import project data for softwood 
proj_default = Project(
    folder_path,
    name="sawdust",  # the name of the project
    temp_unit="C",  # the temperature that results will use (C or K)
    plot_font="Dejavu Sans",  # chose the font for the plots
    resolution_sec_deg_dtg=5,  # chose the resolution for dtg vectors
    dtg_window_filter=None,  # chose the filtering window for dtg curve
    plot_grid=False,  # wheter to include a grid in plots
    temp_initial_celsius=40,  # initial temperature for all curves (exclude data before)
    temp_lim_dtg_celsius=(120,615),  # temperature limits for the dtg curves
    time_moist=38,  # the time where mass loss due to moisture is computed,
    time_vm=None, 
    load_skiprows=8,

)

sw5 = Sample(
    project=proj_default,
    name="softwood 5K",
    filenames=["softwood-5Kmin_1", "softwood-5Kmin_2", "softwood-5Kmin_3"],
    time_moist=10,
    time_vm=None,
    load_skiprows=8,
)

sw10 = Sample(
    project=proj_default,
    name="softwood 10K",
    filenames=["softwood-10Kmin_1", "softwood-10Kmin_2", "softwood-10Kmin_3"],
    time_moist=10,
    time_vm=None,
    load_skiprows=8,
    
)

sw20 = Sample(
    project=proj_default,
    name="softwood 20K",
    filenames=["softwood-20Kmin_1", "softwood-20Kmin_2", "softwood-20Kmin_3"],
    time_moist=10,
    time_vm=None,
    load_skiprows=8,
)

sw50 = Sample(
    project=proj_default,
    name="softwood 50K",
    filenames=["softwood-50Kmin_1", "softwood-50Kmin_2", "softwood-50Kmin_3"],
    time_moist=10,
    time_vm=None,
    load_skiprows=8,
)

sw10.dtg_analysis()

#%%
def objective_function(params):
    mf_c, mf_l, A_2,Ea_2,order_2, A_3, Ea_3, order_3, A_4, Ea_4, order_4,a,b,c = params
    
        
    Tin = sw10.temp_dtg()
    T=[]
    for i in range(0,len(Tin)):
        T.append(Tin[i]+273) #converting to Kelvin 

    m_exp= sw10.mp_db_dtg.ave()
    MLR_exp=(sw10.dtg_db.ave())

    #reaction inputs
    total_mass= 1 #kg 
    R = 8.314 #J/K mol 
    #Reaction 2: Hemicellulose ===> 0.06 Char + 0.94 Gas_H 
    frac2 = 1-mf_c - mf_l
    thetachar_2= 1-a 
    thetagasH= a 

    #Reaction 3:  Cellulose => 0.36 Char + 0.64 Gas_C
    thetachar_3=1-b
    thetagasC=b
        

    #Reaction 4: Lignin => 0.19 Char + 0.81 Gas_L  
    thetachar_4= 1-c
    thetagasL = c 
        
    transmass_h =[] #transient hemicellulose mass 
    transmass_h.append(frac2*total_mass)

    transmass_c =[] #transient cellulose mass 
    transmass_c.append(mf_c*total_mass)

    transmass_l =[] #transient lignin mass 
    transmass_l.append(mf_l*total_mass)  

    #transient mass for products: 
    transmass_vapor=[]
    transmass_vapor.append(0)
    transmass_char2=[]
    transmass_char2.append(0)
    transmass_char3=[]
    transmass_char3.append(0)
    transmass_char4=[]
    transmass_char4.append(0)
    transmass_gasH=[]
    transmass_gasH.append(0)
    transmass_gasC=[]
    transmass_gasC.append(0)
    transmass_gasL=[]
    transmass_gasL.append(0)
        
    #arrays for the numerical mass 
    gas_mass=[]
    gas_mass.append(0)
    m_num = []
    m_cellulose=[]
    m_hemicellulose=[]
    m_lignin=[]

    for i in range(0,len(T)-1):
        dT = T[i+1]-T[i]
        ramp_rate = 10 #K/min 
        
        #reaction 2
        if transmass_h[i] < 0:
            transmass_h.append(0)
            transmass_gasH.append(transmass_gasH[i])
            transmass_char2.append(transmass_char2[i])
        else:
            rxnrate_2= -1* A_2*np.exp((-1*Ea_2)/(R*T[i]))*transmass_h[i]**order_2
            h=transmass_h[i]+(rxnrate_2*dT)/ramp_rate
            dc_dt=thetachar_2*-1*rxnrate_2
            c=transmass_char2[i] +(dc_dt*dT)/ramp_rate
            dg_dt= thetagasH*-1*rxnrate_2
            g=transmass_gasH[i]+(dg_dt*dT)/ramp_rate
            transmass_h.append(h)
            transmass_gasH.append(g)
            transmass_char2.append(c)
        #reaction 3
        if transmass_c[i] <0:
            transmass_c.append(0)
            transmass_gasC.append(transmass_gasC[i])
            transmass_char3.append(transmass_char3[i])
        else:   
            rxnrate_3=  -1* A_3*np.exp((-1*Ea_3)/(R*T[i]))*transmass_c[i]**order_3
            cell=transmass_c[i]+(rxnrate_3*dT)/ramp_rate
            transmass_c.append(cell)
            dc_dt=thetachar_3*-1*rxnrate_3
            c=transmass_char3[i] +(dc_dt*dT)/ramp_rate
            dg_dt= thetagasC*-1*rxnrate_3
            g=transmass_gasC[i]+(dg_dt*dT)/ramp_rate
        
            transmass_gasC.append(g)
            transmass_char3.append(c)

        #lignin
        if transmass_l[i]< 0:
            transmass_l.append(0)
            transmass_gasL.append(transmass_gasL[i])
            transmass_char4.append(transmass_char4[i])
        else:
            rxnrate_4= -1* A_4*np.exp((-1*Ea_4)/(R*T[i]))*transmass_l[i]**order_4
            l=transmass_l[i]+(rxnrate_4*dT)/ramp_rate
            dc_dt=thetachar_4*-1*rxnrate_4
            c=transmass_char4[i] +(dc_dt*dT)/ramp_rate
            dg_dt= thetagasL*-1*rxnrate_4
            g=transmass_gasL[i]+(dg_dt*dT)/ramp_rate
            transmass_l.append(l)
            transmass_gasL.append(g)
            transmass_char4.append(c)
        

        #mass loss rate
        gas_mass.append(transmass_gasC[i]+transmass_gasH[i]+transmass_gasL[i])# +transmass_vapor[i])
        
    for i in range(0,len(gas_mass)):
        m_num.append(100*(total_mass-gas_mass[i])/total_mass)
        m_lignin.append(100*(transmass_l[i])/total_mass)
        m_cellulose.append(100*(transmass_h[i])/total_mass)
        m_hemicellulose.append(100*(transmass_c[i])/total_mass)

    gas_mass_exp=[]    
    for i in range(0,len(m_exp)):
        gas_mass_exp.append(100-m_exp[i]) 
    
    MLR_num= SavFil(np.gradient(m_num,T),101,1) 
    MLR_h= SavFil(np.gradient(m_hemicellulose,T),101,1) 
    MLR_c=SavFil(np.gradient(m_cellulose,T),101,1) 
    MLR_l= SavFil(np.gradient(m_lignin,T),101,1)    
    
    part1 = sum([((m_exp[i] - m_num[i])**2/(sum(m_exp)**2)) for i in range(len(m_exp))])
    part2 = sum([((MLR_exp[i] - MLR_num[i])**2/(sum(MLR_exp)**2)) for i in range(len(MLR_exp))])
   
    # Calculate minimization
    
    minimize= 0.5*(part1) + 0.5*(part2)
    print("resulting objective fuction:"+str(minimize))

    # ### plotting for when not run with PSO 
    # fig, [ax1,ax2]=plt.subplots(1,2, figsize=(9,5))
    # ax1.plot(T,m_exp, label="m_exp")
    # ax2.plot(T,MLR_exp, label="MLR_exp")
    # ax1.plot(T,m_num, label="m_num")
    # ax2.plot(T,MLR_num, label="MLR_num")
    # ax2.plot(T,MLR_h,'--',label="MLR_h")
    # ax2.plot(T,MLR_c, '--',label="MLR_c")
    # ax2.plot(T,MLR_l, '--',label="MLR_l")
    # ax2.set_xlabel("T(C)")
    # ax2.set_ylabel("DTG")
    # ax1.set_xlabel("T(C)")
    # ax1.set_ylabel("TG")
    # ###

    return minimize#-1*R2 
##definition of objective function
#%% setting up inputs

#hemicellulose
A2_lower=5671250.59
A2_upper= 8788554.85

Ea2_lower=56.27*1000
Ea2_upper=59.17*1000

order2_lower=0
order2_upper=5

#cellulose 
A3_lower= 75888341
A3_upper= 131103167

Ea3_lower= 56.05*1000
Ea3_upper =59.30*1000

order3_lower=0
order3_upper =5

#lignin
A4_lower=5671250.59
A4_upper=8788554.85

Ea4_lower= 56.27*1000
Ea4_upper= 59.17*1000

order4_lower=0
order4_upper=5

a_lower= 0.65
a_upper=0.88

b_lower=0.91
b_upper=0.97

c_lower=0.31
c_upper=0.77

mf_c_lower = 0.4
mf_c_upper = 0.6

mf_l_lower = 0.23
mf_l_upper = 0.33

A2= 357275089
A3= 99514031
A4= 7060061


Ea2= 55.37*1000
Ea3= 57.66*1000
Ea4= 57.71*1000

order2= 1
order3= 1
order4= 1

a= 0.77
b= 0.94
c= 0.54

mf_c= 0.5
mf_l=0.28

lb=[mf_c_lower, mf_l_lower, A2_lower, Ea2_lower, order2_lower, A3_lower, Ea3_lower, order3_lower, A4_lower, Ea4_lower, order4_lower, a_lower, b_lower, c_lower]
ub=[ mf_c_upper, mf_l_upper, A2_upper, Ea2_upper, order2_upper, A3_upper, Ea3_upper, order3_upper, A4_upper, Ea4_upper, order4_upper, a_upper, b_upper, c_upper]
#%%

xopt, fopt = pso(objective_function, lb, ub, swarmsize=100,maxiter=10)

print("Optimized Parameters:")

A2_opt,Ea2_opt,order2_opt, A3_opt, Ea3_opt, order3_opt, A4_opt, Ea4_opt, order4_opt,a_opt,b_opt,c_opt = xopt
A_2,Ea_2,order_2, A_3, Ea_3, order_3, A_4, Ea_4, order_4,a,b,c = xopt
Ea_2=[Ea_2]
#print("A_1 = ", A1_opt)
print("A_2=", A2_opt)
print("A_3=", A3_opt)
print("A_4=", A4_opt)
#print("Ea_1=", Ea1_opt)
print("Ea_2=", Ea2_opt)
print("Ea_3=", Ea3_opt) 
print("Ea_4=", Ea4_opt)
#print("order_1=", order1_opt)
print("order_2=", order2_opt)
print("order_3=", order3_opt)
print("order_4=", order4_opt)
print("a:", a_opt)
print("b:", b_opt)
print("c:",c_opt)
# %%
