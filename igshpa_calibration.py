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
import os
import matplotlib.pyplot as plt

# TODO - add specification to these later!
inp_gshp = igs.gshp_inputs(bore_diam=4.25 * 2.54 * 0.01, # 4in plus change in meters
                      pipe_outer_diam=2.0 * 2.54 * 0.01, # 2 in in meters
                      pipe_inner_diam=1.8 * 2.54 * 0.01, # 1.8 in in meters 
                      pipe_therm_cond=0.44, # High density polyethylene (W/(m*K))
                      grou_therm_cond=0.4, # based off sand/gravel/clay/silt mixtures characteristic of Albuquerque
                      grout_therm_cond=1.0, # based off of silicone/sand mixture
                      grou_speci_heat=800.0, # J/(kg*K)
                      ground_density=1900.0, # kg/m3
                      grou_temperature=293.15, # K
                      number_borehole=21,  # this is half what Xiaobing recommended
                      borehole_length=80.0,
                      dist_boreholes=5.0, # 5 meter spacing between boreholes
                      num_element=20,
                      max_COP_heating=5.0,
                      frac_carnot_efficiency=0.8,
                      delT_target=2.0,
                      compressor_capacity=10000, # W  # what size is ideal?
                      mass_flow_capacity=10.0,
                      frac_borehole_length_to_Tg=0.3) # THIS IS THE CALIBRATION FACTOR USED TO MATCH XIAOBING's results.
inp_water = igs.water_inputs()
inp_tank = igs.tank_inputs()
inp_loop = igs.loop_inputs()
inp_pumping_gshp = igs.pumping_inputs()
inp_pumping_loop = igs.pumping_inputs()
inp_triplex = igs.triplex_inputs()

base_path = os.path.dirname(__file__)
data_path = os.path.join(base_path,r"..\Data")
weather_file = "Albuquerque_Weather_Data.csv" # this is hourly and needs to
                                              # be resampled to 15 minutes.
ground_temperature = 293.15
# comes from an analysis in
qhp_loads_file = r"MF+CZ4B+USA_NM_Albuquerque.Intl.AP.723650+hp+slab+IECC_2018_LOOP_LOADS.csv"


dt = 15*60.0 # 15 minute time steps 
resolve = True
percent_to_complete = 101
pause_simulation_time = None #13*3600
break_on_troubleshoot = False

compressor_capacities = [5000,10000,15000]
num_bore_holes = [5,10,15,21]
tank_height = [1,2,3,4]

# THIS IS THE FINAL CALIBRATED VALUE SEE WTS_GSHP_ModelCalibration for more details
frac = [4.19313e-9]

master_sol = []
cop_air_avg = []
cop_wts_avg = []

# initiate a parameter study

gshp_max_temp = []
gshp_min_temp = []

for cap in compressor_capacities:
    for num_bores in num_bore_holes:
        for fr in frac:
            if not "master" in globals() or resolve:
                inp_gshp.nb = num_bores
                inp_gshp.compressor_capacity = cap
                inp_gshp.calib_fract = fr
                master = igs.wts_gshp(ground_temperature,
                                 inp_gshp,
                                 inp_tank,
                                 inp_water,
                                 inp_loop,
                                 inp_pumping_gshp,
                                 inp_pumping_loop,
                                 inp_triplex,
                                 qhp_loads_file,
                                 weather_file,
                                 data_path)
                    
                master.no_load_shift_simulation(ground_temperature,True,
                                                percent_to_complete,
                                                pause_simulation_time,
                                                break_on_troubleshoot)
                
                Other = np.array(master.cs.Other)
                cop_air = Other[:,1]
                cop_wts = Other[:,2]
                #import pdb; pdb.set_trace()
                cop_air_avg.append(cop_air[~isna(cop_air)].mean())
                cop_wts_avg.append(cop_wts[~isna(cop_wts)].mean())
                sol = np.array(master.cs.solutions)
                gshp_max_temp.append(np.max([np.max(sol[:,0]), np.max(sol[:,1])]))
                gshp_min_temp.append(np.min([np.min(sol[:,0]), np.min(sol[:,1])]))
                master_sol.append(master)
            
# post process
#master = master_results
sim_time = np.arange(0,len(master.cs.solutions)*dt,dt)/3600

# Variables
# Tin = X[0]
# Tout = X[1]
# T_tank_next = X[2]
# T_vec = X[3:3+self.loop.NT]
# Tgvec = X[3+self.loop.NT:4+self.loop.NT + self.gshp.number_element]

        # BC[0] = is_cooling
        # BC[1] = W
        # BC[2] = Tg
        # BC[3] = Ta
        # BC[4] = QHP # heat gained/obsorbed by Heat pumps at the triplex level.
        # BC[5] = mdot_loop
        # BC[6] = dt
        # BC[7] = mdot_gshp
        # # gather these for


results = np.array(master.cs.solutions)
BCs = np.array(master.cs.BCs)

plt.close("all")

fig,axl = plt.subplots(4,1,figsize=(10,12))

Q_GSHP = master.cs.gshp.cpw * BCs[:,7] * (results[:,0] - results[:,1])    

cooling_status = np.zeros(BCs.shape[0])
for idx,is_cooling,W_gshp in zip(range(BCs.shape[0]),BCs[:,0],BCs[:,1]):
    if W_gshp != 0:
        if is_cooling == 1:
            cooling_status[idx] = 1
        else:
            cooling_status[idx] = -1

axl[0].plot(sim_time,results[:,0],label="GSHP input")
axl[0].plot(sim_time,results[:,1],label="GSHP output")
axl[0].plot(sim_time,results[:,2],label="Tank out")
axl[0].plot(sim_time,results[:,2+master.cs.loop.NT],label="Tank in")
axl[0].plot(sim_time,BCs[:,3],label="Ambient")
axl[0].plot(sim_time,BCs[:,2],"--k",label="Ground")
axl[0].grid("on")
axl[0].set_ylabel("Temperature (K)")
axl[0].legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))
axl[1].plot(sim_time,BCs[:,1],label="Compressor")
axl[1].plot(sim_time,master.cs.loop.NT*BCs[:,4],label="Triplex")
axl[1].plot(sim_time,Q_GSHP,label="GSHP")
axl[1].legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))
axl[1].grid("on")
axl[1].set_ylabel("power (W)")
axl[2].plot(sim_time,BCs[:,5],label="loop")
axl[2].plot(sim_time,BCs[:,7],label="gshp")
axl[2].legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))
axl[2].grid("on")
axl[2].set_ylabel("mass flow (kg/s)")
axl[3].plot(sim_time,cooling_status)
axl[3].set_ylabel("cooling status (0=off,1=cool,-1=heat)")
axl[3].grid("on")
plt.tight_layout()


fig2,ax2 = plt.subplots(1,1,figsize=(5,6))


ax2.plot(sim_time,cop_air,sim_time,cop_wts)
# 
