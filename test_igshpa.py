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

Created on Thu Mar  4 17:16:47 2021

@author: dlvilla
"""
import unittest
import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
import matplotlib.animation as anim
import igshpa
from material_properties import MaterialProperties as mp

create_plots = True

class test_igshpa(unittest.TestCase):
    @classmethod 
    def setUpClass(cls):

        cls.create_plots = False
        plt.close('all')
        cls.cm_p_inch = 2.54
        cls.meter_per_cm = 0.01
        cls.C_to_K = 273.15

    def test_constant_cooling_load(self):
        
        needs_fixing = True
        
        if not needs_fixing:
        
            base_path = os.path.dirname(__file__)
            data_path = os.path.join(base_path,r"..\Data")
            weather_file = "Albuquerque_Weather_Data.csv"
    
            
            r0 = 1 * self.cm_p_inch * self.meter_per_cm # 2 inch pipe
            ri = 0.9 * self.cm_p_inch * self.meter_per_cm # inner diameter of 1.8 inches
            
            kp = 0.44 # High density polyethylene (W/(m*K))
            kg = 0.4 # per Santa 2007 for fine silt, sand and gravel (W/(m*K))
            cpg = 800.0 # J/(kg*K) per online values for sand and silt
            rhog = 1900 # # per Stone measurements for Albuquerque well 98th street (kg/m3)
            
            L_bore = 10 * 100  # assume 100 m bore-holes
            
            cp_water = 4182.0 # at 20C J/(kg*K)
            
            tank_height = 2.0 # m
            tank_diameter = 3.0 # m
            insulation_thick = 3.0 * self.cm_p_inch * self.meter_per_cm #m
            insulation_k = 0.05 # W/(m*K)
            rho_water = 997 # kg/m3
            
            L_triplex = 10.0 # meters
            
            loop_rpi = 1 * self.cm_p_inch * self.meter_per_cm # 2 inch pipe
            loop_rpo = 0.9 * self.cm_p_inch * self.meter_per_cm # inner diameter of 1.8 inches
            
            trenching_depth = 1.0 # m
            number_triplex_units = 9
            
            # Ambient temperature from TMY3
            weather = pd.read_csv(os.path.join(data_path,weather_file))
            Ta = weather['Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)'].values[48:]
            
            number_of_bends_in_bore_hole = 2
            number_element = 20
            distance_to_next_borehole = 20
            max_GSHP_COP_h = 5.0
            eta_c = 0.8
            Tg = 20 + self.C_to_K #20C
            
            gshp = igshpa.Transient_Simple_GSHP(r0,ri,kp,kg,rhog,cpg,
                                         eta_c,L_bore,number_element,
                                         distance_to_next_borehole,max_GSHP_COP_h,cp_water,Tg)
            ST = igshpa.storage_tank(cp_water, tank_height, tank_diameter, insulation_thick, insulation_k, rho_water)
            loop = igshpa.equal_loop(L_triplex,kg,cp_water,loop_rpi,loop_rpo,trenching_depth,kp,number_triplex_units)
            
            # control conditions 
            BC = np.zeros(10)
            is_cooling = True
            BC[0] = is_cooling
            W_gshp = 6e3 # watts
            BC[1] = W_gshp
             
            BC[2] = Tg
            BC[3] = Ta[4000] + self.C_to_K # sometime in May in Kelvin!
            QHP = 2e3 # Watts 
            BC[4] = QHP
            mdot_loop = 10 #kg/s
            BC[5] = mdot_loop
            dt = 60*15  # 15 minute time step
            BC[6] = dt
            T_tank = Tg
            mdot_gshp = 10 #kg/s
            BC[7] = mdot_gshp
            
            cs = igshpa.CombinedSystem(gshp,ST,loop,T_tank,QHP,mdot_loop,mdot_gshp,
                         cp_water,Tg,BC[3])
            
            # Now take the combined system and simulate it with a constant air-conditioning load
            # The gshp should reach an equillibrium after a long time
            convergence_criterion = 0.01  #%
            not_converged = True
            captured_first_error = False
            max_iter = 100
            gshp_range = range(3+cs.loop.NT,4+cs.loop.NT + cs.gshp.number_element)
            cs.solutions.append(cs.Xm1) # this is the initial state of the system.
            ax = None
            iter1 = 0
            first_error = None
            convergence_error_list = []
            while not_converged:
                cs.solve(BC)
                # repeat this to allow debugging
                cs.equn(cs.solutions[-1],BC)
                csol = cs.solutions[-1]
                Tgvec = csol[gshp_range]
                # plot the GSHP profile
                if self.create_plots:
                    ax = gshp_plot(csol[0],csol[1],Tgvec, r0, 
                               distance_to_next_borehole, 
                               iter1, iter1*dt,gshp,330,290,1.0,-1.75)
    
                convergence_error = 100*(np.abs(cs.solutions[-1][gshp_range] 
                                        - cs.solutions[-2][gshp_range])).sum()/np.abs(cs.solutions[-2][gshp_range]).sum()
    
    
                iter1 += 1
                if convergence_error < convergence_criterion or max_iter < iter1:
                    not_converged = False
                convergence_error_list.append(convergence_error)
                #print("iteration {0:d}: convergence % error: {1:5.3f}".format(iter1,convergence_error))
                
            if self.create_plots:
                fig, ax = plt.subplots(1,1,figsize=(10,10))
                ax.plot(range(iter1),convergence_error_list)
            
            self.assertTrue(max_iter > iter1, "The simulation did not converge and is unstable at a 15 minute step")
         
    def test_NREL_HVAC(self):
        Qdot_coef_h = {"Temperature":[0.876825,-0.002955,-0.000058,0.025335,0.000196,-0.000043],
                       "FF":[0.694045465,0.474207981,-0.168253446]}
        Qdot_coef_c = {"Temperature":[1.557360,-0.074448,0.003099,0.001460,-0.000041,-0.000427],
                       "FF":[0.718664047,0.41797409,-0.136638137]}
        EIR_coef_h = {"Temperature":[0.704658,0.008767,0.000625,-0.009037,0.000738,-0.001025],
                      "FF":[2.185418751,-1.942827919,0.757409168]}
        EIR_coef_c = {"Temperature":[-0.350448,0.116810,-0.003400,-0.001226,0.000601,-0.000467],
                      "FF":[1.143487507,-0.13943972,-0.004047787]}
        PLF_coef_c = [0.85,0.15,0.0] #guessed from https://unmethours.com/question/43925/part-load-ratio-performance-coefficients-for-chiller-and-unitary-dx-system/
                                     # also found this in PNNL-24480.pdf low efficiency equipment could be 0.7-0.8 as the lead coefficient
        from Complex_Appliances import HVAC_Input, NREL_AC_2013
        
        cool_inputs = HVAC_Input(2300.0,Qdot_coef_c, 4.0, EIR_coef_c, 0.05, PLF_coef_c, 0.75)
        # assuming PLF for cooling works for heating??
        heat_inputs = HVAC_Input(3000.0,Qdot_coef_h, 5.0, EIR_coef_h, 0.05, PLF_coef_c, None)
        
        ACcalc = NREL_AC_2013(cool_inputs,heat_inputs)
        
        Power, COP, RTF = ACcalc.calculate(1300,35+273.15,24+273.15,0.36,101325,0.025)
        
        Power_c, COP_c, RTF_c = ACcalc.calculate(-1300,35+273.15,24+273.15,0.36,101325,0.025)
        
        self.assertTrue(Power_c + 0.00001 > 510.0140116605655 and Power_c - 0.00001 < 510.0140116605655)
        
        self.assertTrue(Power + 0.00001 > 420.02433991077186 and Power -0.00001 < 420.02433991077186)

def gshp_plot(Tin,Tout,Tgvec,r0,D_bore,iter1,time,gshp,
              maxtemp,mintemp,maxlograd,minlograd):
    


    fig,ax = plt.subplots(1,1,figsize=(10,10))
    radial_coord = gshp.radial_coord
    Tavg = (Tin + Tout)/2    
    radial_temperatures = np.concatenate((np.array([Tavg]),Tgvec))

    
    ax.set_xlabel("log10(Radial Distance (m))")
    ax.set_ylabel("Temperature (K)")
    ax.set_title("Time = {:05d}".format(int(time/60)) + " min")
            
    ax.plot(np.log10(radial_coord), radial_temperatures, 
            label="radial ground temperature")
    ax.plot([0,radial_coord[2]],[Tin,Tin],label="inflow temperature")
    ax.plot([0,radial_coord[2]],[Tout,Tout],label="outflow temperature")
    
    ax.legend()
    ax.grid("on")
    ax.set_xlim([minlograd,maxlograd])
    ax.set_ylim([mintemp,maxtemp])
    
    plt.savefig('./frames/gshp{:05d}'.format(int(iter1))+".png")
    plt.close('all')



if __name__ == "__main__":
    unittest.main()























