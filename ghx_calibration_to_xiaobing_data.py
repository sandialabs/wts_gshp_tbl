# -*- coding: utf-8 -*-
"""

License Statement:

Copyright 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government 
retains certain rights in this software.

BSD 2-Clause License

Copyright (c) 2021, Sandia National Laboratories
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
===========================END OF LICENSE STATEMENT ==========================

Created on Mon Mar  8 13:01:52 2021

@author: dlvilla
"""
import numpy as np
import igshpa as igs
from pandas import isna
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, minimize, differential_evolution
import time
import pickle as pkl
from datetime import date

def ghe_func(coef_and_dist,func_type,plot_results=False):
    
    """
      This function takes in a 
    """
    # this needs to be added to the input file
    rhog = 1900 #kg/m3
    C_to_K = 273.15
    
    # all inputs provided by Xiaobing
    df_inp = pd.read_csv(os.path.join("Xiaobing","Input_File (pcp=1.52).csv"))
    df_inp.index = df_inp['Name']
    
    Tg = float(df_inp.loc['T_g','Value'])+C_to_K
    k_g = float(df_inp.loc['k_s','Value'])
    cpg = k_g/(float(df_inp.loc['alpha','Value'])*rhog)
    

    bore_hole_diam = 2 * float(df_inp.loc['r_b','Value'])
    rb = float(df_inp.loc['r_b','Value'])
    
    # I am using this as a calibrating factor in addition to the resistances to ground.
    distance_to_next_borehole = coef_and_dist[2]   # to do - study the sesitivity to this parameter!
    numelem = int(coef_and_dist[3])
    
    rr = np.logspace(np.log10(rb),np.log10(distance_to_next_borehole/2.0),num=numelem)
    if func_type == "constant":
        frac = coef_and_dist 
    elif func_type == "linear":
        frac = coef_and_dist[0] * rr + coef_and_dist[1]
    elif func_type == "quadratic":
        frac = coef_and_dist[0] * rr**2 + coef_and_dist[1]*rr + coef_and_dist[2]
    else:
        raise ValueError("The func_type input currently only supports the following values: \n\n"
                         +"   1. linear\n2. quadratic")

    if plot_results:
        plt.close('all')
    
    # thermal loads BC
    df = pd.read_csv("Xiaobing/XiaobingData.csv")
    
    # This is a circuitous route for input but it follows the same
    # method as for the actual calculations.
    inp_gshp = igs.gshp_inputs(bore_diam=bore_hole_diam, # 3in bore radius plus change in meters
                          pipe_outer_diam=2*float(df_inp.loc['rp_out','Value']), # 1.05 in 
                          pipe_inner_diam=2*float(df_inp.loc['rp_in','Value']), # 
                          pipe_therm_cond=float(df_inp.loc['k_p','Value']), # 
                          grou_therm_cond=float(df_inp.loc['k_s','Value']), # based off sand/gravel/clay/silt mixtures characteristic of Albuquerque
                          grout_therm_cond=float(df_inp.loc['k_g','Value']), # based off of silicone/sand mixture
                          grou_speci_heat=cpg, # J/(kg*K)
                          ground_density=rhog, # kg/m3
                          grou_temperature=Tg, # K
                          number_borehole=1,  # this is half what Xiaobing recommended
                          borehole_length=float(df_inp.loc['H','Value']), # this is the length for the calculations 
                          dist_boreholes=distance_to_next_borehole, # 6.1 meter spacing between boreholes -
                          num_element=numelem,
                          max_COP_heating=5.0, # unimportant
                          frac_carnot_efficiency=0.8, # unimportant
                          delT_target=2.0, # unimportant
                          compressor_capacity=10000, # unimportant
                          mass_flow_capacity=10.0, # unimportant
                          frac_borehole_length_to_Tg=frac) # THIS IS THE CALIBRATION FACTOR
    
    inp_water = igs.water_inputs(specific_heat=float(df_inp.loc['cp_f','Value']), # J/(kg*K
                      density=float(df_inp.loc['den_f','Value']))
    
    # this is just taken from igshpa.wts_gshp
    rb = inp_gshp.rb
    r0 = inp_gshp.r0 
    ri = inp_gshp.ri
    kp = inp_gshp.kp
    kg = inp_gshp.kg
    k_grout = inp_gshp.k_grout
    cpg = inp_gshp.cpg
    rhog = inp_gshp.rhog
    L_bore = inp_gshp.L_bore
    number_element = inp_gshp.num_element
    distance_to_next_borehole = inp_gshp.dist_boreholes
    max_COP_h = inp_gshp.max_COP_h
    eta_c = inp_gshp.eta_c
    num_bore_hole = inp_gshp.nb
    dist_to_Tg_reservoir = L_bore * inp_gshp.calib_fract
    
    # water inputs
    cpw = inp_water.cp
    rhow = inp_water.rho
    dt = 60*60 # 1 hour time steps.
    # per Xiaobing's output data.
    mdot_gshp = float(df_inp.loc['m_flow','Value'])
    ghe = igs.Transient_Simple_GSHP(rb, 
                                    r0, 
                                    ri, 
                                    kp,
                                    kg, 
                                    rhog, 
                                    cpg, 
                                    eta_c, 
                                    L_bore, 
                                    number_element, 
                                    distance_to_next_borehole, 
                                    max_COP_h, 
                                    cpw, 
                                    rhow, 
                                    Tg, 
                                    k_grout, 
                                    num_bore_hole,
                                    dist_to_Tg_reservoir,
                                    calibration_mode=True)
    # assume everything starts at ground temperature.
    Xm1 = Tg * np.ones(3 + number_element)


    
    # time iteration.
    
    # predicted output temperatures 
    df["temp model"] = 0
    df["Q_ground_total"] = 0
    
    for rownum,row in df.iterrows():
        
        # Watch your heat direction conventions!
        Q = -row['heat']
        X,infodict,ier,mesg = fsolve(solve_func, Xm1, args=([Xm1,dt,ghe,mdot_gshp,Q]),full_output=True)
        
        # convert to degrees celcius.
        df.loc[rownum,'temp model'] = X[1] - 273.15
        Xm1 = X
        df.loc[rownum,"Q_ground_total"] = ghe.Q_ground_total
        
        
    if plot_results:
        fig,axl = plt.subplots(2,1)
        ax = axl[0]
        ax.plot(df['hour'],df['temp'],label="g-function")
        ax.plot(df['hour'],df['temp model'],label="this study")
        ax.grid("on")
        ax.set_ylabel(r"GHX outflow Temperature ($^{\circ}$C)")
        ax.legend()
        axl[1].plot(df['hour'],-df['heat'],label="input")
        axl[1].plot(df['hour'],df['Q_ground_total'],label="to ground")
        ax.legend()
        axl[1].grid("on")
        axl[1].legend()
        axl[1].set_ylabel("Heat flow (W)")
    plt.tight_layout()
    plt.savefig(r"C:\Users\dlvilla\Documents\BuildingEnergyModeling\NMSBA\Repository\2021\SmartEnergyPublication\TeXElsevier\figures\ghx_calibration_FINAL.png",dpi=1000)
    cvrmse = ((df['temp model'] - df['temp'])**2).sum()/(len(df['temp'])*df['temp'].mean())
    
    return cvrmse

def solve_func(X,args):
    """
    X - vector of variables - 
    Xm1 - vector of variables one time step back so that derivatives can be
          estimated and integrated.
          
    """
    Xm1 = args[0]
    dt = args[1]
    ghe = args[2]
    mdot_gshp = args[3]
    Q_econo = args[4]
    
    Tin_next = X[0]
    Tout_next = X[1]
    Tgvec = X[2:]
    Tin = Xm1[0]
    Tout = Xm1[0]
    Tgvec_m1 = Xm1[2:]
    
    # These are not used here for the present ground heat exchanger only
    # analysis
    is_cooling = None
    T_tank = None
    W = None
    
    return ghe.equn(Tgvec,Tin_next,Tin,Tout_next,Tout,T_tank,W,is_cooling,mdot_gshp,Tgvec_m1,dt,Q_econo)
    
if __name__ == "__main__":
    
    # These are solutions for volumetric heat capacitance of 0.3 - we needed for 1.52
    # # must be consistent with num_element 
    # # this is a highly good fit with cvrmse of 0.28447335 with 20 elements
    # scale = 0.000009505
    # slope = .00009505/5
    
    # # a second solution from differiential_evolution
    # # result = differential_evolution(ghe_func,
    # #                   bounds=[(-0.0001/5,0.001/5),(0.000001,0.01)],
    # #                   args=('linear',dist_to_next_borehole,numelem),
    # #                   maxiter = 20,
    # #                   popsize = 40)
    # # The function took 1.66228e+04 seconds to run.
    # # The cvrmse is: 0.28447335%
    # # result
    # # Out[2]: 
    # #      fun: 0.0028447334509180875
    # #  message: 'Optimization terminated successfully.'
    # #     nfev: 1123
    # #      nit: 13
    # #  success: True
    # #        x: array([-3.191e-06,  6.362e-05])
    # slope = -3.190691369868475e-06
    # scale = 6.361712205381863e-05
    
    # # another solution
    # slope = 1.2434658252815482e-05
    # scale = 2.924542480469096e-05
    
    # first optimization solution. 1.38% cvrmse
    slope = 0.00015936665228338034
    scale = 1e-6
    numelem = 20
    dist_to_next_borehole = 5 
    
    # second optimization solution 1.34% cvrmse
    slope = 0.00015791084966032944
    scale = 1.1127693387715544e-07
    numelem = 21
    dist_to_next_borehole = 5.3
    
    today =date.today()
    datstr = "{0:d}_{1:d}_{2:d}".format(today.month,today.day,today.year)
       
    run_full_optimize = False

    if run_full_optimize:
        stime = time.time()
        # result = minimize(ghe_func,[slope,scale],
        #                   args=('linear',dist_to_next_borehole,numelem),
        #                   bounds=[(-0.0001305/6.1,0.001/6.1),(0.00001,0.001)],
        #                   method='L-BFGS-B',
        #                   options={'gtol':0.000000001,'ftol':0.00001})
        result = differential_evolution(ghe_func,
                          bounds=[(-6.5e-5,2e-4),(1e-12,0.0001),(3,7),(5,40)],
                          args=(['linear']),
                          maxiter = 40,
                          popsize = 80)
        cvrmse = ghe_func([result.x[0],result.x[1],result.x[2],result.x[3]],'linear',True)
        etime = time.time()
        pkl.dump([result,etime-stime,cvrmse],open("de_result_"+datstr+".pkl",'wb'))
        
    else:
        stime = time.time()
        cvrmse = ghe_func([slope,scale,dist_to_next_borehole,numelem],'linear',True)
        # cvrmse = ghe_func([scale],'constant',dist_to_next_borehole,numelem,True)
        etime = time.time()
    
    
    
    
    print("The function took {0:.5e} seconds to run.".format(etime-stime))
    print("\n\n")
    print("The cvrmse is: {0:.8f}%".format(100*cvrmse))
    
    


