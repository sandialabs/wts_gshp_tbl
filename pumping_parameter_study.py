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

Created on Thu Jul 15 15:00:49 2021

@author: dlvilla
"""
# TODO - This file needs to be auto-synced with igshpa_parameter_study1.py - 
# it needs to run with the same centralized table so that inputs only have
# a single source and cannot become unsynchronized.

from scipy.optimize import fsolve
import pickle as pkl
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import wntr
import multiprocessing as mp
import sklearn as skl
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from matplotlib import rc
from numpy.random import default_rng
import pandas as pd

rng = default_rng()

class MaterialProperties():
    """All of these were originally used the the MultiConfigurationMembraneDistillaiton tool I created
       They are throughly validated constitutive relationships."""
    def h2o_cp(Pressure, Temperature):

        """
        Inputs Pressure in Pascals
               Temperature in Kelvin
               
        output- waters specific heat in J/(kg*K)
        
        """
        glbSatErr = 0.05
        #' More extensive fit from NIST REFPROP database - error +/- 1 J/(kg*K) for range in 4180 to 4240 J/(kg*K)
        
        Tsat = MaterialProperties.saturated_temperature_pure_water(Pressure)
        
        if Temperature > Tsat + glbSatErr:
            raise ValueError("mdlProperties.SubCooledWaterSpecificHeat: The temperature input (" + Temperature +
                         ") for subcooled water properties is above the saturation temperature (" + Tsat +
                         ") for the pressure input (" + Pressure + ")! The requested point is therefore for super-heated vapor!")
        elif Temperature < 274:
            raise ValueError("mdlProperties.SubCooledWaterSpecificHeat: The temperature input (" + Temperature +
                         ") is beyond the lower limit of 274Kelvin!")
        elif Pressure < 700 or Pressure > 200000:
            raise ValueError("mdlProperties.SubCooledWaterSpecificHeat: The pressure input (" + Pressure +
                         ") is outside the valid range of 700 To 200000Pascal!")
        else:
            coef = np.zeros((6,3))
            coef[0, 0] = 133228.645
            coef[0, 1] = -1765.05818
            coef[0, 2] = 56.7024555
            coef[1, 0] = -1822.14992
            coef[1, 1] = 19.011658
            coef[1, 2] = -0.582695346
            coef[2, 0] = 10.2646815
            coef[2, 1] = -0.0712535996
            coef[2, 2] = 0.00199192998
            coef[3, 0] = -0.0288937643
            coef[3, 1] = 0.000103015355
            coef[3, 2] = -0.00000226675226
            coef[4, 0] = 0.0000407654089
            coef[4, 1] = -3.78344779E-08
            coef[5, 0] = -2.31665625E-08
            
            #'Output in J/(kg*K)
            return MathFunc.Polynomial2D(coef, Temperature, np.log(Pressure))
    
    @staticmethod
    def saturated_temperature_pure_water(Pressure):
        """'Pressure is in pascal output is in Kelvin
        ' valid range from 611Pa absolute to 22090000Pa
        ' this is from pages 723 - 724 of  Moran and Shapiro, "Fundamentals of Engineering Thermodynamics" Third Edition
        '  the data has been entered in a table in excel in the "WaterProperties.xlsx" workbook and curve fit"""
        if Pressure < 611 or Pressure > 22090000:
            raise ValueError("Invalid Pressure of " + str(Pressure) + " which must be between 611 and 22090000 Pascal" +
                     " applied to this polynomial relationship.")
            
        else:
        
            coef = [2.133888,129.2319,-28.95103,3.660487,-0.2472544,
                     0.008618524,-0.0001138013]
            
            return MathFunc.Polynomial(coef, np.log(Pressure))
        
    @staticmethod
    def h2o_viscosity(Pressure, Temperature):

        """
        Inputs Pressure in Pascals
               Temperature in Kelvin
               
        output- waters viscosity in Pa*s

        # More extensive fit from NIST REFPROP database - error +1e-5/-1e-5 (Pa-s) for range in 5e-4 to 15e-4 Pa-s
        # THIS ERROR IS LARGER THAN MANY AND AMOUNTS TO 2% in the worst case!
        
        """
        glbSatErr = 0.05
        Tsat = MaterialProperties.saturated_temperature_pure_water(Pressure)
        
        if Temperature > Tsat + glbSatErr:
            raise ValueError("The temperature input (" + str(Temperature) + 
                         ") for subcooled water properties is above the saturation temperature (" + str(Tsat) +
                         ") for the pressure input (" + str(Pressure) + ")! The requested point is therefore for super-heated vapor!")
        elif Temperature < 274:
            raise ValueError("mdlProperties.SubCooledWaterViscosity: The temperature input (" + str(Temperature) +
                         ") is beyond the lower limit of 274Kelvin!")
        elif Pressure < 700 or Pressure > 200000:
            raise ValueError("mdlProperties.SubCooledWaterViscosity: The pressure input (" + str(Pressure) +
                         ") is outside the valid range of 700 To 200000Pascal!")
        else:
            coef = np.zeros([6,2])
            coef[0, 0] = 1.5460851
            coef[0, 1] = -0.017739743
            coef[1, 0] = -0.0217286988
            coef[1, 1] = 0.000226784225
            coef[2, 0] = 0.000121718473
            coef[2, 1] = -0.00000108328113
            coef[3, 0] = -0.000000338925226
            coef[3, 1] = 2.29123193E-09
            coef[4, 0] = 4.68111877E-10
            coef[4, 1] = -1.81034509E-12
            coef[5, 0] = -2.55970701E-13
            
            #Output in Pa*s
            return MathFunc.Polynomial2D(coef, Temperature, np.log(Pressure))

    def h2o_density(Pressure, Temperature):

        """' More extensive fit from NIST REFPROP database - error +.01/-.03 kg/m3 for range in 940 to 1000 kg/m3

        Inputs Pressure in Pascals
               Temperature in Kelvin
               
        output- waters density in kg/m3
        
        
        """

        Tsat = MaterialProperties.saturated_temperature_pure_water(Pressure)
        glbSatErr = 0.05
        if Temperature > Tsat + glbSatErr:
            raise ValueError("mdlProperties.SubCooledWaterDensity: The temperature input (" + str(Temperature) +
                         ") for subcooled water properties is above the saturation temperature (" + str(Tsat) +
                         ") for the pressure input (" + str(Pressure) + ")! The requested point is therefore for super-heated vapor!")
        elif Temperature < 274:
            raise ValueError("mdlProperties.SubCooledWaterDensity: The temperature input (" + str(Temperature) +
                         ") is beyond the lower limit of 274Kelvin!")
        elif Pressure < 700 or Pressure > 200000:
            raise ValueError("mdlProperties.SubCooledWaterDensity: The pressure input (" + str(Pressure) +
                         ") is outside the valid range of 700 To 200000Pascal!")
            SubCooledWaterDensity = mdlConstants.glbINVALID_VALUE
        else:
            coef = np.zeros([6,4])
            coef[0, 0] = -4130.69705
            coef[0, 1] = 56.852855
            coef[0, 2] = -2.1773214
            coef[0, 3] = 0.000825950059
            coef[1, 0] = 68.9418879
            coef[1, 1] = -0.593594772
            coef[1, 2] = 0.022608859
            coef[1, 3] = -0.0000249341667
            coef[2, 0] = -0.369951899
            coef[2, 1] = 0.00208463872
            coef[2, 2] = -0.0000764129176
            coef[2, 3] = 9.83714798E-08
            coef[3, 0] = 0.00100008202
            coef[3, 1] = -0.0000025888428
            coef[3, 2] = 0.000000081941603
            coef[4, 0] = -0.00000137338486
            coef[4, 1] = 4.99440472E-10
            coef[5, 0] = 7.65796484E-10
            
            #Output in kg/m3
            return MathFunc.Polynomial2D(coef, Temperature, np.log(Pressure))



# TODO - just use numpy - I'm going to replace these but do not want to have to bother with validating
# insertion of python functions so I'm keeping them for now.
class MathFunc():    
    @staticmethod
    def Polynomial(coef, Val):
    
        """' This function orders the coefficient so that the first coefficient is the constant term (i.e. x^0 term)
        ' this is the opposite of many approaches!"""
    
        Temp = 0
        for idx, cf in enumerate(coef):
            if cf != 0:
                Temp = Temp + cf * Val ** idx
        return Temp

    @staticmethod
    def Polynomial2D(coef, x, Y):            
        Temp = 0
        for i in range(coef.shape[0]): 
            for j in range(coef.shape[1]):
                if coef[i, j] != 0:
                    Temp = Temp + coef[i, j] * x **i * Y ** j
        
        return Temp

class Pipe_Flow_Resistance(object):
    
    def __init__(self,cs,pinp_gshp,pinp_loop):
        
        self.cs = cs
        wts = cs.wts
        L_loop = cs.loop.L * (cs.loop.NT + 2) # assume 2 extra legs to/from water storage
        # lots of bends. 4 per triplex.
        minor_loss = 0.3 * 4
        
        self.minor_loss_loop = minor_loss # per triplex
        
        # mass flow resistance ( this is post processing)
        self.wn_loop = self._water_network(h_tank       = wts.tank_height,
                                      L_total      = L_loop,
                                      del_elev     = 0.5,
                                      pipe_diam    = cs.loop.ri * 2.0,
                                      pipe_rough   = 120.0,
                                      minor_loss   = minor_loss,
                                      num_div      = cs.loop.NT)
        
        # for the GSHP, we assume that each bore-hole is in parallel with
        # 1.8in inner diameter piping. 
        # mass flow resistance ( this is post processing)
        L_gshp = cs.gshp.L_bore * 2.2/cs.gshp.nb # 0.05 reflects extra piping.
        # IMPORTANT- THE ORIGINAL RUN HAD 35 bore-holes but here we divide
        # and then later multiply so that we get a total length that is bore-hole
        # independent.
        # lots of bends! assume 4 per bore-hole 
        minor_loss = 0.3 * 4 + 0.3
        self.wn_gshp = self._water_network(h_tank       = wts.tank_height,
                                      L_total      = L_gshp,
                                      del_elev     = 0.5,
                                      pipe_diam    = cs.gshp.ri * 2.0,
                                      pipe_rough   = 120.0,
                                      minor_loss   = minor_loss,
                                      num_div      = 1)        
        self.pinp_gshp = pinp_gshp
        self.pinp_loop = pinp_loop
        self.h_tank = 0.0
        self.min_pressure = wts.tank_height  # IN WNTR PRESSURE IS GIVEN AS HEAD
        self.L_loop = L_loop
        self.L_GSHP = L_gshp
        self.minor_loss_gshp = minor_loss # per bore-hole
        
    def _water_network(self,h_tank,L_total,del_elev,pipe_diam,pipe_rough,minor_loss,num_div):
        wn = wntr.network.WaterNetworkModel()
        wn.add_reservoir("res1",base_head=h_tank,coordinates=(0,0))
        wn.add_junction("pump_outlet",base_demand=0,elevation=0.0,coordinates=(0,0))
        for ii in range(num_div):
            wn.add_junction("j"+str(ii),base_demand=0.0,elevation=ii*del_elev/num_div,coordinates=((ii+1)*L_total/(num_div+1),0))
        wn.add_junction("farthest_point",base_demand=0,elevation=del_elev,coordinates=(L_total,0))
        
        pipe_lengths = L_total/(num_div+1)
        
        wn.add_pipe("pipe_from_res","farthest_point","j"+str(num_div-1),length=pipe_lengths,diameter=pipe_diam,roughness=pipe_rough,
                    minor_loss=minor_loss/(num_div+1))
        wn.add_pipe("pipe_to_furthest","pump_outlet","j0",length=pipe_lengths,diameter=pipe_diam,roughness=pipe_rough,
                    minor_loss=minor_loss/(num_div+1))
        for ii in range(0,num_div-1):
            wn.add_pipe("pipe_"+str(ii),"j"+str(ii),"j"+str(ii+1),length=pipe_lengths,diameter=pipe_diam,roughness=pipe_rough,
                    minor_loss=minor_loss/(num_div+1))
            
            
        
        # 1.0 is just a place holder. We are not trying to size a pump and we want to know how much power a pump will use.
        # This will be changed later on.
        wn.add_pump("pump","res1","pump_outlet","Power",pump_parameter=1.0,speed=1.0)
        return wn
    
    def solve_pumping_power(self,Tavg_gshp,Tavg_loop,dt,mass_flow_gshp,mass_flow_loop,frac_gshp,frac_loop,num_parallel):
        
        avg_gshp_pump_power, gshp_failed = self._solve_pp(Tavg_gshp,self.wn_gshp,dt,mass_flow_gshp,frac_gshp,1,num_parallel)
        
        avg_loop_pump_power, loop_failed = self._solve_pp(Tavg_loop,self.wn_loop,dt,mass_flow_loop,frac_loop,self.cs.loop.NT,1)
        
        # add extra power due to inefficiencies in the pumps.
        total_pumping_power = avg_gshp_pump_power/self.pinp_gshp.eta_p + avg_loop_pump_power/self.pinp_loop.eta_p
        
        return (total_pumping_power, avg_gshp_pump_power/self.pinp_gshp.eta_p, 
                avg_loop_pump_power/self.pinp_gshp.eta_p, Tavg_gshp,Tavg_loop,
                dt,mass_flow_gshp,mass_flow_loop,frac_gshp,frac_loop, 
                gshp_failed, loop_failed, num_parallel)
        
    def _solve_pp(self,Tavg,wn,dt,mass_flow,frac_flow_consumed,num_div,num_parallel):
        rho = self.cs.wts.rho
        returns_to_pump_vol_demand = (1-frac_flow_consumed) * mass_flow / rho / num_parallel # should be m3/s
        
        node_vol_demand = frac_flow_consumed * mass_flow / rho / num_div
        
        for ii in range(num_div):
            jj = wn.get_node("j"+str(ii))
            jj.demand_timeseries_list[0].base_value = node_vol_demand
            
        # TODO - move these constants to another location
        one_atm_Pa = 101325
        sys_pres = one_atm_Pa + rho * 9.81 * self.h_tank
        
        mu = MaterialProperties.h2o_viscosity(sys_pres,Tavg)
        wn.options.hydraulic.viscosity = mu * rho
        wn.options.time.duration=dt
        
        nn = wn.get_node("farthest_point")
        nn.demand_timeseries_list[0].base_value = returns_to_pump_vol_demand
        
        # an overly high guess value converges nicely whereas a value that is too close
        # leads to negative pressures and associated non-convergence
        guess_value = rho * 9.81 * (node_vol_demand * num_div + returns_to_pump_vol_demand) * 100
        
        find_sol = [guess_value, 2*guess_value, 5*guess_value, 10*guess_value, 25 * guess_value, 100*guess_value]
        
        for sol in find_sol:
            try:
                pump_power,infodict,ier,mesg  = fsolve(self._fsolve_func,[sol],
                                               args=(wn,self.min_pressure),
                                               full_output=True,
                                               factor=0.1)
            except:
                failed = True
                return -1000,failed
            if ier == 1:
                break
            else:
                pass

        # troubleshoot

        press = self._fsolve_func(pump_power,wn,self.min_pressure)
        if ier != 1:
            failed = True
            print("fsolve pump power solution failed. Internal message =\n\n" + mesg)
        else:
            failed = False
            
        
        
        return num_parallel * pump_power[0], failed
    
    def _fsolve_func(self,pump_power,wn,min_pressure):
            
        pump = wn.get_link('pump')
        pump.power = np.abs(pump_power[0])
        
        sim = wntr.sim.WNTRSimulator(wn)
        
        try:
            res = sim.run_sim()
        except:
            try:
                sim2 = wntr.sim.EpanetSimulator(wn)
                res = sim2.run_sim()
            except:
                return -1000

        # Performed cross check on WNTR calcs WNTR gave 198.1999 W and this hand
        # calc gave 216.5514, which is close enough given that I am extracting
        # results that may not align with WNTR's method for calculations.
        #pump_power_check = 997 * np.pi/4 * 0.0916**2 *(6.359821-3) * 9.81
        # pressure is in meters head
        pressure = res.node['pressure']['farthest_point'].iloc[0]
        
        # assure that pressure is sustained at the furthest point
        return (pressure - (self.h_tank + min_pressure))**2


if __name__ == "__main__":
    
    plt.close('all')
    li = pkl.load(open("inputs_for_Pipe_Flow_Resistance_2022-02-15.pickle",'rb'))
    
    cs = li[0]
    inp_gshp = li[1]
    inp_loop = li[2]   
    
    obj = Pipe_Flow_Resistance(cs, inp_gshp, inp_loop)
    
    # TODO - fix this so it is not manual!
    inp_gshp.eta_p = 0.9
    inp_loop.eta_p = 0.9
    
    
    #TODO finish this table!
    # create a table of the inputs to the analysis
    columns = ["Description","GSHP","Loop"]
    rows = [["Pipe length (m)","{0:5.1f} per bore-hole".format(obj.L_GSHP),"{0:5.1f}".format(obj.L_loop)],
    ["Minor loss coefficient sum","{0:5.1f} per bore-hole".format(obj.minor_loss_gshp),"{0:5.1f}".format(obj.minor_loss_loop)],
    ["Pipe Roughness C-factor","{0:5.1f}".format(obj.wn_gshp.get_link('pipe_to_furthest').roughness),"{0:5.1f}".format(obj.wn_loop.get_link("pipe_0").roughness)],
    ["Pipe Diameter (m)","{0:7.4f}".format(obj.wn_gshp.get_link('pipe_to_furthest').diameter),"{0:7.4f}".format(obj.wn_loop.get_link("pipe_0").diameter)],
    ["Pump efficiency","{0:5.1f}".format(inp_gshp.eta_p),"{0:5.1f}".format(inp_loop.eta_p)],
    ["Number of pipe elements","{0:d}".format(len(obj.wn_gshp.link_name_list)-1),"{0:d}".format(len(obj.wn_loop.link_name_list)-1)]]
    
    df_inputs = pd.DataFrame(data=rows,columns=columns)
    df_inputs.index = df_inputs["Description"]
    df_inputs.drop(["Description"],axis=1,inplace=True)
    
    df_inputs.to_latex(buf=open(os.path.join("..","SANDReport","from_python","pumping_study_input_table.tex"),'w'),index=True,
                    column_format=r'p{0.35\textwidth}p{0.25\textwidth}p{0.175\textwidth}',
                    longtable=False,escape=False,label="tab:pump_inputs",caption="Pumping parameter study inputs.")
    

    use_log = True
    run_parallel = False
    run_me = False
    
    if run_me:
        
        if run_parallel:
            pool = mp.Pool(mp.cpu_count()-1)
        
        
        num_bore = np.array([5,10,20,30,40,50])
        mdot = np.arange(1.0,10,0.25)
        frac = np.array([0.0,0.25,0.5,0.75,1.0])
        power = np.zeros([len(mdot),len(num_bore),len(frac)])
        gshp_power = np.zeros([len(mdot),len(num_bore),len(frac)])
        loop_power = np.zeros([len(mdot),len(num_bore),len(frac)])
        
        
        async_results = []
        
        for idx,md in enumerate(mdot):
            for idy,nb in enumerate(num_bore):
                for idz,fr in enumerate(frac):
                    if run_parallel:
                        async_results.append(pool.apply_async(obj.solve_pumping_power,args=(300.0,300.0,15*60,md,md,0.0,fr,nb)))
                    else:
                        (power[idx,idy,idz], gshp_power[idx,idy,idz], loop_power[idx,idy,idz], T_gshp, T_loop, dt, md_gshp, md_loop,
                         frac_gshp, frac_loop, gshp_failed, loop_failed, nub) = obj.solve_pumping_power(300.0,300.0,15*60,md,md,0.0,fr,nb)
                        if gshp_failed or loop_failed:
                            power[idx,idy,idz] = np.nan
                            print ("The following run failed: mdot=" + str(md) + ", nb=" + str(nb) + ", frac=" + str(fr))
        if run_parallel:
            for ares in async_results:
                lres = ares.get()
                power_ = lres[0]; 
                gshp_power_ = lres[1]; loop_power_ = lres[2]
                T_gshp = lres[3]; T_loop = lres[4]; dt=lres[5]
                md_gshp = lres[6]; md_loop = lres[7]; frac_gshp = lres[8]
                frac_loop = lres[9]; gshp_failed = lres[10]; loop_failed = lres[11];
                num_b = lres[12]
                
                idy = np.where(num_bore==num_b)[0]
                idx = np.where(mdot==md_gshp)[0]
                idz = np.where(frac==frac_loop)[0]
                
                power[idx,idy,idz] = power_
                gshp_power[idx,idy,idz] = gshp_power_
                loop_power[idx,idy,idz] = loop_power_
                
                #print("power[" + str(idx) + "," + str(idy) + "," + str(idz) +"] = " + str(power_))
                
                if gshp_failed or loop_failed:
                    
                    power[idx,idy,idz] = np.nan
    
        
        pkl.dump([obj,gshp_power,loop_power,num_bore,mdot,frac],open("pump_study_"+str(datetime.now()).split(" ")[0]+'.pickle','wb'))
    else:
        temp = pkl.load(open("pump_study_2021-07-26.pickle",'rb'))
        obj = temp[0]
        gshp_power = temp[1]
        loop_power = temp[2]
        num_bore = temp[3]
        mdot = temp[4]
        frac = temp[5]
    
    
    font = {'family':'normal',
            'weight':'normal',
            'size':16}
    rc('font', **font)
    
    
    power_list = [gshp_power,loop_power]
    fig_name_end = ["GSHP","Loop"]
    
    
    for power,figend in zip(power_list,fig_name_end):
    
        fig,axl = plt.subplots(1,1,figsize=(10,10))
        
        markers = ['x','o','v','+','s','*']
        
        linestyle_tuple = [
         ('solid',                 'solid'),
         ('dotted',                (0, (1, 1))),
         ('dashed',                (0, (5, 5))),
         ('dashdotted',            (0, (3, 5, 1, 5))),
         ('densely dashdotted',    (0, (3, 1, 1, 1))),
         ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10)))]
        
        b_color = ['k','r','b','g','c','y']
    
        if figend == "GSHP":
            for idy,nb in enumerate(num_bore):
                if use_log:
                    po = np.log(power[:,idy,0])
                else:
                    po = power[:,idy,0]
                
                axl.scatter(mdot,po,label="wntr nb={0:3.2f}".format(nb),
                         marker=markers[idy],color=b_color[0])
        else:
            for idz,fr in enumerate(frac):
                
                if use_log:
                    po = np.log(power[:,0,idz])
                else:
                    po = power[:,0,idz]
                
                axl.scatter(mdot,po,label="wntr frac={0:3.2f}".format(fr),
                             marker=markers[idz],color=b_color[0])
        
        axl.set_xlabel("mass flow rate (kg/s)")
        axl.set_ylabel("Natural Logarithm of Pumping power (ln(W))")
        axl.set_title(figend)
        if not use_log:
            axl.set_ylim([-0.02*power.max(),power.max()*1.1])
        
            
        # now perform a multi-variate regression 
        X_vec = []
        vec = []
        
        if figend == "GSHP":
        
            for idx,md in enumerate(mdot):
                for idy,nb in enumerate(num_bore):
                    if not np.isnan(power[idx,idy,0]) and power[idx,idy,0] > 0:
                        X_vec.append([md,nb])
                        vec.append(power[idx,idy,0])
            X = np.array(X_vec)
            vector = np.array(vec)
           
            
            #predict is an independent variable for which we'd like to predict the value
            predict= X
            predict2 = np.array([[10.0,35],[10.0,35],[10.0,35],[10.0,35],[10.0,35]])
        
        else:
            for idx,md in enumerate(mdot):
                    for idz,fr in enumerate(frac):
                        if not np.isnan(power[idx,0,idz]) and power[idx,0,idz] > 0:
                            X_vec.append([md,fr])
                            vec.append(power[idx,0,idz])
            X = np.array(X_vec)
            vector = np.array(vec)
           
            
            #predict is an independent variable for which we'd like to predict the value
            predict= X
            predict2 = np.array([[10.0,0.0],[10.0,0.25],[10.0,0.5],[10.0,0.75],[10.0,1.0]])
        
            
            
        predict = np.concatenate([predict,predict2])
        
        #generate a model of polynomial features
        poly = PolynomialFeatures(degree=6)
        
        #transform the x data for proper fitting (for single variable type it returns,[1,x,x**2])
        X_ = poly.fit_transform(X)
        
        #transform the prediction to fit the model type
        predict_ = poly.fit_transform(predict)
        
        #here we can remove polynomial orders we don't want
        #for instance I'm removing the `x` component
        # X_ = np.delete(X_,(1),axis=1)
        # predict_ = np.delete(predict_,(1),axis=1)
        
        #generate the regression object
        clf = skl.linear_model.LinearRegression()
        #preform the actual regression
        
        if use_log:
            vec2 = np.log(vector)
        else:
            vec2 = vector
        
        clf.fit(X_, vec2)
    
        
        power_predicted = clf.predict(predict_)
        
        if figend == "GSHP":
            for idy,nb in enumerate(num_bore):
                vid = predict[:,1]==nb
                axl.plot(predict[vid,0],power_predicted[vid],
                         color=b_color[0],label="poly nb={0:d}".format(nb),
                         linestyle=linestyle_tuple[idy][1])
        else:
            for idz,fr in enumerate(frac):
                vid = predict[:,1]==fr
                axl.plot(predict[vid,0],power_predicted[vid],
                         color=b_color[0],label="poly frac={0:3.2f}".format(fr),
                         linestyle=linestyle_tuple[idz][1])
            
        axl.legend()
        axl.grid("on")
            
        
        # now randomly test for overfitting.
        error = vec2 - power_predicted[:len(vec2)]
        max_error = np.max(abs(np.array([error.max(),error.min()])))
        
        rnd_num = rng.random((10000,2))
        
        rnd_num[:,0] = 10*rnd_num[:,0]
        if figend == "GSHP":
            rnd_num[:,1] = num_bore[-1]*rnd_num[:,1] - num_bore[0]
        
        poly_rnd_num = poly.fit_transform(rnd_num)
        
        random_power_predicted = clf.predict(poly_rnd_num)
        
        if random_power_predicted.max() > 1650 or random_power_predicted.min() < -10:
            print("The fit has unseen wave variations between frac")
        
        
        if figend == "GSHP":
            pass
            #axl.set_ylim([0,90])
        
        plt.savefig(os.path.join("..","SANDReport","figures","pump_polynomial_wntr_results"+figend+".png"),dpi=300)
        
        fig2, ax2 = plt.subplots(1,1,figsize=(10,10))
        
        ax2.scatter(rnd_num[:,0],random_power_predicted)
        
        pkl.dump([clf,poly],open("pump_polyfit_results_"+figend+str(datetime.now()).split(" ")[0]+'.pickle','wb'))
        
        if figend == "GSHP":
            polydata = np.concatenate([poly.powers_,clf.coef_.reshape((28,1))],axis=1)
        else:
            polydata = np.concatenate([polydata,clf.coef_.reshape((28,1))],axis=1)
        
    
    
    polydata_columns = [r"Mass flow rate polynomial power","'nb' or 'frac' polynomial power","GSHP coefficients","Thermal bridge loop coefficients"]
    df_polytable = pd.DataFrame(polydata,columns=polydata_columns)  
    df_polytable.to_latex(buf=open(os.path.join("..","SANDReport","from_python","polyfit_coeff_table.tex"),'w'),index=False,
                column_format=r'p{0.2\textwidth}p{0.2\textwidth}p{0.2\textwidth}p{0.2\textwidth}p{0.2\textwidth}',
                longtable=True,escape=False,float_format=lambda x: '%.6g' % x,caption="WNTR study polynomial coefficients for natural logarithm of pumping power",label="tab:wntr_coef")
        

    
    
    
                
                
