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

test_dir = plib.Path(__file__).resolve().parent / "data"

# %% import project data for 5, 11.5, and 14.5 % WF
proj_default = Project(
    test_dir,
    name="10Kpermin",
    temp_unit="C",
    plot_font="Dejavu Sans",
    dtg_basis="temperature",
    resolution_sec_deg_dtg=5,
    dtg_window_filter=101,
    plot_grid=False,
    temp_initial_celsius=40,
    temp_lim_dtg_celsius=(120,615),
    load_skiprows= 85,
)

wf5 = Sample(
    project=proj_default,
    name="5% WF",
    filenames=["5WF_2", "5WF_3", "5WF_5"],
    time_moist=10,
    time_vm=None,
    load_skiprows=85,
)

wf115 = Sample(
    project=proj_default,
    name="11.5% WF",
    filenames=["WF115_2", "WF115_3", "WF115_1"],
    time_moist=38,
    time_vm=None,
    load_skiprows=8,
)

wf145 = Sample(
    project=proj_default,
    name="14.5% WF",
    filenames=["14.5WF_2", "14.5WF_3", "14.5WF_4"],
    time_moist=10,
    time_vm=None,
    load_skiprows=85,
)

sw = Sample(
    project=proj_default,
    name="sw",
    filenames=["softwood-10Kmin_1", "softwood-10Kmin_2", "softwood-10Kmin_3"],
    time_moist=38,
    time_vm=None,
    load_skiprows=8,
)
proximate_samples = [wf5, wf115, wf145]
for sample in proximate_samples:
    _ = sample.plot_tg_dtg()
rep = proj_default.multireport(
    samples=proximate_samples,
    report_type="proximate",
)
rep = proj_default.plot_multireport(
    "rep_",
    samples=proximate_samples,
    report_type="proximate",
    height=4,
    width=4.5,
    y_lim=(0, 100),
    # legend_loc="upper left",
)
rep = proj_default.plot_multi_dtg(
    "rep",
    samples=proximate_samples,
    height=4,
    width=4.5,
)
#%%
mf = wf5.plot_tg_dtg()
A_2= 4.07e3# 5.00000000e+08#46742959428.81353
A_3= [4.23e8]#[5.00000000e+08, 1.67500000e+10, 3.35000000e+10, 5.02500000e+10]#3775666421001.3706
A_4= 2.57e8#2990075715.8857775
Ea_2= 1.29e2# 128195.84295677918
Ea_3= 1.43e3#143213.15094735962
Ea_4= 2.57e4#117160.24197553129
order_2= 1.34#1.5607594711636972
order_3=1.2# 1.6411276133252377
order_4= 4.14#2.710619508050416
a= 0.82#0.6336907337906291
b= 0.85#0.9197637736202754
c= 0.85#0.5569212944621904
frac2 =.215# [0.215,0.238]#0.30#0.215#originally 0.335
frac3 =.618#[0.618,0.578]#0.44#0.32# 0.618 #originally:0.307 
frac4 =.167#[0.167,0.183]



#fig,ax=plt.subplots(1,2)

'''
for i in range(0, len(A_3)):
    xopt=A_2,Ea_2,order_2, A_3[i], Ea_3, order_3, A_4, Ea_4, order_4,a,b,c, frac2,frac3,frac4
    mf= objective_functionPLOT(xopt)
'''
#%%
#storage
A2_array=[]
A3_array=[]
A4_array=[]
E2_array=[]
E3_array=[]
E4_array=[]
order2_array=[]
order3_array=[]
order4_array=[]
a_array=[]
b_array=[]
c_array=[]
w=2


def objective_function(params):
    global A2_array
    global A3_array
    global A4_array
    global E2_array
    global E3_array
    global E4_array
    global order2_array
    global order3_array
    global order4_array
    global a_array
    global b_array
    global c_array
    global w
    
    A_2,Ea_2,order_2, A_3, Ea_3, order_3, A_4, Ea_4, order_4,a,b,c = params
    
    A2_array.append(A_2)
    A3_array.append(A_3)
    A4_array.append(A_4)
    E2_array.append(Ea_2)
    E3_array.append(Ea_3)
    E4_array.append(Ea_4)
    order2_array.append(order_2)
    order3_array.append(order_3)
    order4_array.append(order_4)
    a_array.append(a)
    b_array.append(b)
    c_array.append(c)
        
    Tin = sw.temp_dtg
    T=[]
    for i in range(0,len(Tin)):
        T.append(Tin[i]+273) #converting to Kelvin 

    m_exp= sw.mp_db_dtg.ave()
    MLR_exp=(sw.dtg_db.ave())

    #reaction inputs
    total_mass= 1 #kg 
    R = 8.314 #J/K mol 
    #Reaction 2: Hemicellulose ===> 0.06 Char + 0.94 Gas_H 
    frac2 = 0.37 #sw: 0.37
    #0.215#originally 0.335
    thetachar_2= 1-a 
    thetagasH= a 

    #Reaction 3:  Cellulose => 0.36 Char + 0.64 Gas_C
    frac3 =0.34 #sw: 0.34
    #0.44#0.32# 0.618 #originally:0.307 
    thetachar_3=1-b
    thetagasC=b
        

    #Reaction 4: Lignin => 0.19 Char + 0.81 Gas_L  
    frac4 =0.29 #sw 0.29 #0.26# 0.27#0.167#originally 0.261
    thetachar_4= 1-c
    thetagasL = c 
        
    transmass_h =[] #transient hemicellulose mass 
    transmass_h.append(frac2*total_mass)

    transmass_c =[] #transient cellulose mass 
    transmass_c.append(frac3*total_mass)

    transmass_l =[] #transient lignin mass 
    transmass_l.append(frac4*total_mass)  

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
        m_cellulose.append(100*(transmass_c[i])/total_mass)
        m_hemicellulose.append(100*(transmass_h[i])/total_mass)

    gas_mass_exp=[]    
    for i in range(0,len(m_exp)):
        gas_mass_exp.append(100-m_exp[i]) 
    
    MLR_num= SavFil(np.gradient(m_num,T),101,1)  
    MLR_h= SavFil(np.gradient(m_hemicellulose,T),101,1) 
    MLR_c=SavFil(np.gradient(m_cellulose,T),101,1) 
    MLR_l= SavFil(np.gradient(m_lignin,T),101,1) 

    mse = np.mean((m_exp-m_num)**2)+np.mean((MLR_exp - MLR_num)**2)#+np.mean((m_exp-m_num)**2)) #mean squared error 
    
    #print("this is the mse:"+str(mse))

    mean_exp = sum(m_exp) / len(m_exp)
    meanMLR_exp = sum(MLR_exp) / len(MLR_exp)
    # Calculate the numerator and denominator of R^2
    numerator1 = sum([((m_exp[i] - m_num[i])**2/(sum(m_exp)**2)) for i in range(len(m_exp))])
    denominator1 = sum([(m_exp[i] - mean_exp) for i in range(len(m_exp))])
    
    numerator2 = sum([((MLR_exp[i] - MLR_num[i])**2/(sum(MLR_exp)**2)) for i in range(len(MLR_exp))])
    denominator2 = sum([(MLR_exp[i] - meanMLR_exp) for i in range(len(MLR_exp))])
    # Calculate minimization
    minimize= 0.5*(numerator1) + 0.5*(numerator2)
    print("resulting objective fuction:"+str(minimize))

    #print(R2)
    
    # fig, [ax1,ax2]=plt.subplots(1,2)
    # ax1.plot(T,m_exp, label="m_exp")
    # ax2.plot(T,MLR_exp, label="MLR_exp")
    # ax1.plot(T,m_num, label="m_num")
    # ax2.plot(T,MLR_num, label="MLR_num")
    # ax2.plot(T,MLR_h,'--',label="MLR_h")
    # ax2.plot(T,MLR_c, '--',label="MLR_c")
    # ax2.plot(T,MLR_l, '--',label="MLR_l")
    # pl.legend()
    # # pl.text(700, 0, "Ea_2="+str(Ea_2))
    # pl.text(700, -0.025, "Ea_3="+str(Ea_3))
    # pl.text(700, -0.05, "Ea_4="+str(Ea_4))
    # pl.text(700, -0.075, "A_2="+str(A_2))
    # pl.text(700, -0.1, "A_3="+str(A_3))
    # pl.text(700, -0.125, "A_4="+str(A_4))
    # pl.text(700, -0.15, "order_2="+str(order_2))
    # pl.text(700, -0.175, "order_3="+str(order_3))
    # pl.text(700, -0.2, "order_4="+str(order_4))
    # pl.text(700, -0.225, "a="+str(a))
    # pl.text(700, -0.25, "b="+str(b))
    # pl.text(700, -0.275, "c="+str(c))
 
#     display.display(pl.gcf())
#     display.clear_output(wait=True)
# # #pl.savefig("/Users/charlottealbunio/Desktop/data_new_obfunc/"+str(w))
     
# # #  #time.sleep(1.0)
#     plt.clf()
#     w+=1
    



    return minimize
##definition of objective function
#%% setting up inputs
# A2_lower=4e5
# A2_upper= 5e5

# Ea2_lower=113528.0425
# Ea2_upper=200000

# order2_lower=0.425681677
# order2_upper=1.484251026

# A3_lower= 3.46304e11
# A3_upper= 5e12

# Ea3_lower= 119374.4897
# Ea3_upper = 272751.4569

# order3_lower=0.337291074
# order3_upper = 1.417848426

# A4_lower=6.33e9
# A4_upper=2.97e10

# Ea4_lower= 9e4
# Ea4_upper= 198920.605

# order4_lower=0.3828095
# order4_upper=2.55658454

# a_lower= 0.576361162
# a_upper=0.993549304
# b_lower=0.52180876
# b_upper=0.94536545
# c_lower=0.578209705
# c_upper=1

# A2= 6000114054216043.0
# A3= 42199784.77703544
# A4= 31971650728.82418
# Ea2= 178283.07177174714
# Ea3= 94300.45494739931
# Ea4= 124192.27218511552
# order2= 0.5
# order3= 1.0088228533795456
# order4= 1.8040195095052096
# a= 0.7352921946331114
# b= 0.7411826540338358
# c= 0.8524791726815882



A2= 4e9#489461.5770686779
A3= 1731954094490.7498
A4= 7022657350.599377
Ea2= 99000
Ea3= 148833.50249164962
Ea4= 93441.88629059868
order2= 1.34
order3= 1.12
order4= 4
a= 0.5890135204546205
b= 0.94536545
c= 0.9266255664777426

A2= 3764015872.467753
A3= 1483698303825.9783
A4= 7931020653.082679
Ea2= 117109.84254710704
Ea3= 150801.3258535904
Ea4= 97018.12005993574
order2= 1.3127109307657483
order3= 1.1778537871656685
order4= 3.5852660645734034
a= 0.7068111288330053
b= 0.9193965018665408
c= 0.8495182231246878
A2_lower=A2-(0.1*A2)
A2_upper= A2+(0.1*A2)

Ea2_lower=Ea2-(0.1*Ea2)
Ea2_upper=Ea2+(0.1*Ea2)

order2_lower=order2-(0.1*order2)
order2_upper=order2+(0.1*order2)

A3_lower= A3-(0.1*A3)
A3_upper= A3+(0.1*A3)

Ea3_lower= Ea3-(0.1*Ea3)
Ea3_upper = Ea3+(0.1*Ea3)

order3_lower=order2-(0.1*order3)
order3_upper=order2+(0.1*order3)

A4_lower=A4-(0.1*A4)
A4_upper=A4+(0.1*A4)

Ea4_lower=Ea4-(0.1*Ea4)
Ea4_upper= Ea4+(0.1*Ea4)

order4_lower=order4-(0.1*order4)
order4_upper=order4+(0.1*order4)

a_lower= 0.576361162
a_upper=0.993549304
b_lower=0.52180876
b_upper=0.94536545
c_lower=0.578209705
c_upper=1
#%% PARTICLE SWARM 

initial_guess = [ A2,Ea2,order2, A3, Ea3, order3, A4, Ea4, order4,a,b,c] #use initial guess as values #mod particle swarm size and max iteration 

objective_function(initial_guess)
#%%
lb=[ A2_lower, Ea2_lower, order2_lower, A3_lower, Ea3_lower, order3_lower, A4_lower, Ea4_lower, order4_lower, a_lower, b_lower, c_lower]
ub=[ A2_upper, Ea2_upper, order2_upper, A3_upper, Ea3_upper, order3_upper, A4_upper, Ea4_upper, order4_upper, a_upper, b_upper, c_upper]

xopt, fopt = pso(objective_function, lb, ub, swarmsize=1000,maxiter=100)

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

#%%
A_2= 5446831663386481.0
A_3= 39048547.214840956
A_4= 33602707382.861008
Ea_2= 183271.26335475597
Ea_3= 93801.92298328227
Ea_4= 130665.34582905659
order_2= 0.4556983681713208
order_3= 1.1023613586747012
order_4= 2.526306449429421
a: 0.919744607330179
b: 0.7915688795716836
c: 0.6295880449985107
paramsFinal= A_2, A_3, A_4, Ea_2, Ea_3, Ea_4, order_2, order_3, order_4, a, b, c 

mf=objective_function(paramsFinal)

#%%
import pandas as pd

# Create a DataFrame
data = {
    'A2': A2_array,
    'A3': A3_array,
    'A4': A4_array,
    'Ea2': E2_array,
    'Ea3': E3_array,
    'Ea4': E4_array,
    'order 2': order2_array,
    'order 3': order3_array,
    'order 4': order4_array,
    'a': a_array,
    'b': b_array,
    'c': c_array,
    # Add more columns as needed
}

df = pd.DataFrame(data)

# Specify the file path where you want to save the Excel file
excel_file_path = '/Users/charlottealbunio/Desktop/data_new_obfunc/data.xlsx'

# Write the DataFrame to an Excel file
df.to_excel(excel_file_path, index=False)

print(f"Data has been exported to {excel_file_path}")





