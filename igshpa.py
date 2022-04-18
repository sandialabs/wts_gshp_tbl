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

Created on Fri Feb 26 09:16:56 2021

From the Yavuzturk thesis 1999

@author: dlvilla
"""

from scipy.special import expi 
from scipy.optimize import fsolve
import numpy as np
import pandas as pd
import os
import datetime
import wntr
from pdb import set_trace as bp1
from material_properties import MaterialProperties as mp
from random import random
from Complex_Appliances import Wall_AC_Unit, NREL_AC_2013, HVAC_Input

class SingleStageHPInputs():
        """
        
            See tables from Cutler, D. et. al., 2013 "Improved Modeling of Residential AC and HP for
               Energy Calculations" NREL/TP-5500-56354
        
        
        """
        Qdot_coef_h = {"Temperature":[0.876825,-0.002955,-0.000058,0.025335,0.000196,-0.000043],
                       "FF":[0.694045465,0.474207981,-0.168253446]}
        Qdot_coef_c = {"Temperature":[1.557360,-0.074448,0.003099,0.001460,-0.000041,-0.000427],
                       "FF":[0.718664047,0.41797409,-0.136638137]}
        EIR_coef_h = {"Temperature":[0.704658,0.008767,0.000625,-0.009037,0.000738,-0.001025],
                      "FF":[2.185418751,-1.942827919,0.757409168]}
        EIR_coef_c = {"Temperature":[-0.350448,0.116810,-0.003400,-0.001226,0.000601,-0.000467],
                      "FF":[1.143487507,-0.13943972,-0.004047787]}
        PLF_coef = [0.85,0.15,0.0] #guessed from https://unmethours.com/question/43925/part-load-ratio-performance-coefficients-for-chiller-and-unitary-dx-system/
                                     # also found this in PNNL-24480.pdf low efficiency equipment could be 0.7-0.8 as the lead coefficient
        
class Transient_Simple_GSHP(object):
    
    def __init__(self,rb,r0,ri,kp,kg,rhog,cpg,eta_c,L_bore,number_element,
                 distance_to_next_borehole,max_COP_h,cpw,rhow,Tg,k_grout,
                 num_bore_hole,dist_to_Tg_reservoir,calibration_mode=False):
        """
        Inputs:
            rb     = bore-hole radius
            r0     = pipe outer radius
            ri     = pipe inner radius
            kp     = pipe thermal conductivity
            kg     = ground thermal conductivity
            rhog   = ground density
            cpg    = ground specific heat
            eta_c  = fraction of Carnot cycle efficiency GSHP attains
            L_bore = length of bore-holes for the GSHX
            
            number_element            = number of elements from the pipe surface modeled
            distance_to_next_borehole = axis distance between boreholes
            max_COP_h                 = maximum COP the HP can produce 
            
            cpw     = specific heat of water
            rhow    = density of water
            muw     = dynamic viscosity of water
            Tg      = ground temperature (constant)
            k_grout = thermal conductivity of grout surrounding pipe to rb
            num_bore_hole = number of bore holes (determines minor losses for pumping)
            calibration_mode = False - run the normal study model
                               True = run in a special mode to mimic the ground
                                      source heat exchanger only calibration
                                      to Xiaobing Liu higher fidelity ground 
                                      source heat exchanger (GHX or GHE) model.
        """
        
        # Thermal relationships
        
        self.max_COP_h = max_COP_h
        self.eta_c = eta_c
        self.number_element = number_element
        self.cpw = cpw
        self.r0 = r0
        self.rb = rb
        self.nb = num_bore_hole
        self.ri = ri
        self.L_bore = L_bore * num_bore_hole # total length of all bore-holes 
        
        if r0 > 0.5 * rb:
            raise ValueError("The pipe outer radius r0 must be less than or" + 
                             " equal to half of the bore radius rb")
        # assume equal spacing between pipes and edges of the bore.
        self.pipe_center_distance = 2.0/3.0 * (rb + r0) 

        self.Tg = Tg
        
        # initial conditions
        self.Tgvec_m1 = Tg * np.ones(number_element+1)
        
        # Pipe radial resistance DO NOT ADD L_BORE TO THIS BECAUSE 
        # Resistance * Length is needed.
        Rp = 1/(2.0 * np.pi * kp) * np.log(r0/ri)
        self.Rp = Rp # this is added in the Diao, 2004 resistance factors.
        dist2 = distance_to_next_borehole/2.0
        
        # Space GSHX elements on a log scale to allow a smooth capture of the
        # surface temperature gradient.
        rr = np.logspace(np.log10(rb),np.log10(distance_to_next_borehole/2.0),num=number_element+1)
        
        # Radial Heat capacitance of rings of volume of ground
        self.Cg = [rhog * np.pi * (r2**2.0 - r1**2.0) * self.L_bore * cpg for r2,r1 in zip(rr[1:],rr[:-1])]
        # Radial thermal resistivity of rings of volume of ground
        self.Rg = [np.log(rout/rin)/(2.0*np.pi * self.L_bore * kg) for rout,rin in zip(rr[1:],rr[:-1])]
        
        if isinstance(dist_to_Tg_reservoir,float):
            self.RTg = [dist_to_Tg_reservoir/(np.pi/4.0 * (rout**2 - rin**2) * kg * num_bore_hole) for rout,rin in zip(rr[1:],rr[:-1])]
        else:  # it is assumed that dist_to_Tg_reservoir is a list or array
            try:
                if len(dist_to_Tg_reservoir) != number_element:
                    raise ValueError("The dist_to_Tg_reservoir must be an array or list of equal length as the number of elements")
            except:
                raise TypeError("dist_to_Tg_reservoir must be a float, array, or list!")
                    
            self.RTg = [dist/(np.pi/4.0 * (rout**2 - rin**2) * kg * num_bore_hole) for rout,rin,dist in zip(
                rr[1:],rr[:-1],dist_to_Tg_reservoir)]
        

        
        # Thermal mass of water in bore-holes.
        self.C_bore = 2*(np.pi * ri ** 2 * self.L_bore) * rhow * cpw
        self.k_grout = k_grout
        self.kg = kg        
        self.func_eval = 0
        
        # resistance factors per Diao et. al., 2004 equation 7  
        R11, R12 = self._R11_12()
        self.beta_times_mdot = self.L_bore / (cpw * np.sqrt((R11 + R12)*(R11 - R12)))
        self.R11_12_factor = np.sqrt((R11 + R12)/(R11 - R12))
        self.R11_12_factor2 = (R11 + R12)/2.0
        self._calib = calibration_mode
        

        
    def _R11_12(self):
        # switching to the nomenclature of Diao et. al. "Improvement in Modeling
        # of Heat Transfer in Vertical Ground Heat Exchangers" HVAC&R Research 2004.
        kb = self.k_grout
        k  = self.kg
        Rp = self.Rp  # TODO - you need to add in the convective resistance here!
        rp = self.r0
        D  = self.pipe_center_distance
        rb = self.rb
        
        fac1 = 1/(2*np.pi*kb)
        fac2 = (kb-k)/(kb+k)
        
        return (fac1 *(np.log(rb/rp)+fac2*np.log(rb**2/(rb**2 - D**2))) + Rp,
                fac1 *(np.log(rb/(2*D))+fac2*np.log(rb**2/(rb**2 + D**2))))
        
    def gshp_cop(self,is_cooling,Tavg,T_tank):
        
        # For now we assume the performance is a fraction of Carnot efficiency
        
        if is_cooling:
            if Tavg <= T_tank:
                COP = self.max_COP_h - 1.0  # maximum feasible COP
            else:
                COP = self.eta_c * T_tank / (Tavg - T_tank)
                if COP > self.max_COP_h - 1.0:
                    COP = self.max_COP_h - 1.0
            # do not cool further if T_tank < Tavg            
        else:
            if T_tank <= Tavg:
                COP = self.max_COP_h
            else:
                COP = self.eta_c * (Tavg / (T_tank - Tavg))
                if COP > self.max_COP_h:
                    COP = self.max_COP_h
        return COP
        
    def equn(self,Tgvec,Tin_next,Tin,Tout_next,Tout,T_tank,W,is_cooling,mdot,Tgvec_m1,dt,Q_econo):
        """
        Inputs:
            Variables: 
                
            Tgvec - vector of ground temperatures spaced radially from the GSHX
            
            Tin_next - input temperature to the GSHX (i.e. output temperature to the GSHP)
            
            Tout_next - output temperature of the GSHX (i.e input temperature to GSHP)
            
            T_tank - temperature of the water thermal storage tank
            
            W - input work to the GSHP
        """
        self.func_eval = self.func_eval + 1
        
        Tavg_next = (Tin_next + Tout_next)/2.0
        Tavg = (Tin + Tout)/2.0
        delT = Tin_next - Tout_next
        
        if self._calib:
            # we are calibrating and Q_econo is just used as a boundary
            # condition
            Q_GSHX = Q_econo
        else: # normal model operation mode with interplay of GSHP performance
            if mdot == 0.0:
                Q_GSHX = 0.0
                if is_cooling:
                    self.Qc = 0.0
                else:
                    self.Qh = 0.0
            else:
                COP = self.gshp_cop(is_cooling, Tavg_next, T_tank)
                # Q_GSHX here is Q_tank
                if is_cooling:
                    self.Qc = W * COP - Q_econo # this is needed and is equal to Q_GSHP for the WTS
                    # heat is being put into the ground +
                    # Q_econo is in the perspective of from the tank into the ground being negative
                    # we desire from the tank to the ground to be positive in this reference frame.
                    Q_GSHX = W * (1 + COP) - Q_econo 
                else:
                    self.Qh = -W * COP - Q_econo
                    # heat is being extracted from the ground -
                    Q_GSHX = -W * (COP - 1) - Q_econo

        
        residuals = 1e20 * np.ones(len(Tgvec)+2)
        
        # here is the new condition on heat flow per Diao et al., 2004 - L_bore cancels here.
        if mdot == 0.0: # avoid division by 0.0
            if Q_GSHX != 0.0:
                raise ValueError("If mdot is zero then Q_GSHX should be as well!")
            Rb = self.R11_12_factor2 # radial resistance of no-flow configuration
        else:
            tanh_beta = np.tanh(self.beta_times_mdot / mdot)
            epsilon = 2 * tanh_beta/(self.R11_12_factor + tanh_beta)
            Rb = 1.0 / (mdot * self.cpw) * (1.0 / epsilon - 0.5)
            
        Q = np.zeros(len(Tgvec)+1)
        Q[0] = (Tavg_next - Tgvec[0])/(Rb)       # + self.Rg[0]

        
        # steady state assumption to stored vs conduction to

        residuals[0] = self.C_bore * (Tavg_next - Tavg)/dt - Q_GSHX + Q[0] 
        if mdot != 0:
            residuals[1] = Q_GSHX - mdot * self.cpw * delT
        else: # assure the difference in delT is shrunken to 0.0
            residuals[1] = self.cpw*(Tin_next - Tout_next)
        
        # TODO - 2nd law closing assumption - if Tin_next drifts above the refrigerant output temperature
        # then the 2nd law is being violated. 
        Q_ground_sum = 0
        for idx in range(1,self.number_element+1):
            Q_ground = ((Tgvec[idx]+Tgvec[idx-1])/2 - self.Tg)/self.RTg[idx-1]
            if idx < self.number_element: # the insulated boundary is zero heat transfer
                Q[idx] = (Tgvec[idx-1] - Tgvec[idx])/self.Rg[idx-1]
            #ODE's
            residuals[idx+1] = (self.Cg[idx-1] * (Tgvec[idx-1] - 
                        Tgvec_m1[idx-1]) - (Q[idx-1]-(Q[idx]+Q_ground)) * dt)
            Q_ground_sum += Q_ground
        residuals[-1] = Tgvec[-1] - Tgvec[-2]  # an insulated boundary condition
        self.Q_ground_total = Q_ground_sum
        return residuals

class equal_loop(object):
    seconds_per_hour = 3600
    hours_in_a_day = 24
    def __init__(self,L_triplex,kg,cp,rpi,rpo,depth,kp,NT,QHP0,rho,
                 triplex_inp,min_flow_loop,Tg,hw_setpoint,Vdot):
        
        Rp = 1/(2*np.pi * kp * L_triplex) * np.log(rpo/rpi)
        Rg = 1/(2*np.pi * kg * L_triplex) * np.log(depth/rpo)
        #self.C_w = np.pi * rpi **2 * L_triplex * rho * cp
        
        
        
        if Rp < 0 or Rg < 0:
            raise ValueError("The inputs for resistivity are incorrect." +
                             " A negative resistivity is incorrect! " +
                             " have the inner and outer radii been reversed?")
        Rtotal = Rp + Rg
        
        self.NT = NT # number of triplex units 
        self.L = L_triplex # distance between triplex units (pressure losses)
        self.Rtotal = Rtotal
        self.cp = cp
        self.loop_heat = NT * QHP0 # this is a first guess
        self.ri = rpi
        self.hot_water_demand_kg_p_s = rho * triplex_inp.hot_water_demand
        self.hot_water_use_fraction = triplex_inp.hot_water_use_fraction
        self.min_flow_loop = min_flow_loop
        self.Tg = Tg
        self.hot_water_setpoint = hw_setpoint
        self.pressure = triplex_inp.pressure
        self.rh_inside = triplex_inp.rh_internal
        self.Vdot = Vdot
        
        
    # set of equations    
    def equn(self,Tvec_next,TA,mdot,Tvec,dt,hour_of_day,consume_loop_water,QTriplex,T_inside):
        if len(Tvec) != self.NT + 1:
            raise ValueError("The vector of temperatures must be number of triplexes long!")
        if mdot < self.min_flow_loop:
            import pdb; pdb.set_trace()
            raise ValueError("The minimum flow should not be violated!")
        
        QA = np.array([(TA-Tavg) / self.Rtotal for Tavg in Tvec_next[1:]])
        
        QHP = np.array([self.HPh2o.calculate(QTriplex,Tavg, T_inside, self.rh_inside, self.pressure, self.Vdot)[-2] 
                        for Tavg in (Tvec_next[:-1] + Tvec_next[1:])/2.0])
        
        # used in control scheme
        self.loop_heat = sum(QA) + sum(QHP)
        
        
        
        # hot water consumption logic. That is like a mini Smart water and heat micronetwork valve.
        # same assumption as tank that utility water comes in at average of ground temperature and water 
        if consume_loop_water:
            mdot_hw = self.hot_water_demand_kg_p_s * self.hours_in_a_day * self.hot_water_use_fraction[hour_of_day]
        else:
            mdot_hw = 0.0
        
        Tw = (TA + self.Tg)/2
        self.Ehw_no_loop = self.NT * mdot_hw * self.cp * (self.hot_water_setpoint - Tw)
        
        Qhw = mdot_hw * self.cp * (Tvec_next[1:]+Tvec_next[:-1])/2
        
        if consume_loop_water:
            self.Ehw_loop = self.NT * mdot_hw *self.cp * self.hot_water_setpoint - Qhw.sum()
        else:
            self.Ehw_loop = self.Ehw_no_loop
        
        
        # must have -np.diff because diff is x[n] - x[n-1] we want x[n-1] - x[n] 
        residuals = np.array([((mdot - (idt-1)*mdot_hw)*T0 - (mdot - idt*mdot_hw)*T1) * self.cp + QA_ - Qhw_ + QHP_
                         for T0,T1,QA_,Qhw_,idt,QHP_ in zip(Tvec_next[:-1],
                                                       Tvec_next[1:],
                                                   QA,
                                                   Qhw,
                                                   np.arange(1,len(Tvec_next)),
                                                   QHP)])
        self.Qhw = Qhw.sum()
        self.mdot_hw = mdot_hw * self.NT
        return residuals
         
class storage_tank(object):
    seconds_per_hour = 3600
    hours_in_a_day = 24
    def __init__(self,cp,tank_height,tank_diameter,insulation_thickness,insulation_k,rho_tank,triplex_inp, num_triplex_units,ground_temperature):
        
        self.cp = cp
        top_area = np.pi * (tank_diameter)**2/4.0
        self.volume = top_area * tank_height
        self.mass = rho_tank * self.volume
        self.tank_height = tank_height
        self.tank_diameter = tank_diameter
        self.C_tank = self.mass * self.cp
        # assume top and bottom are subject to ambient temperature (conservative)
        Rtop = insulation_thickness / (insulation_k * top_area)
        tank_radius = tank_diameter / 2.0
        Rsides = 1 / (2*np.pi * insulation_k * tank_height) * np.log((tank_radius + insulation_thickness)/tank_radius)
        # parrallel model
        self.Rtotal = 1/(1/Rtop + 1/Rsides)
        self.rho = rho_tank
        self.hot_water_demand_m3_s_neighborhood = num_triplex_units * triplex_inp.hot_water_demand
        self.hot_water_use_fraction = triplex_inp.hot_water_use_fraction
        self.Tg = ground_temperature

        
        pass
    def equn(self,mdot,Q_GSHP,Tin,T,TA,dt,Tnext,hour_of_day,consume_loop_water):
        # Q_GSHP should be negative when recieving heat because the original
        # control volume is with heat leaving being positive.
        QA = (Tnext-TA) / self.Rtotal
        
        # Assume the utility water comes in at half way between ambient and ground temperature.
        Tw = (self.Tg + TA)/2

        if consume_loop_water:
            mdot_triplex_hw = self.rho  * self.hours_in_a_day * self.hot_water_demand_m3_s_neighborhood * self.hot_water_use_fraction[hour_of_day]
        else:
            mdot_triplex_hw = 0.0

        Q_water_consumption = mdot_triplex_hw * self.cp * Tw
        
            
        self.mdot_triplex_hw = mdot_triplex_hw
        self.Q_water_consumption = Q_water_consumption
        return self.C_tank * (Tnext-T)/dt - self.cp * (Tin * (mdot - mdot_triplex_hw) - Tnext * mdot + mdot_triplex_hw * Tw) + (Q_GSHP + QA)
        
class CombinedSystem(object):
    seconds_in_an_hour = 3600
    hours_in_a_day = 24
    def __init__(self,gshp,wts,loop,T_tank_ic,Q_hp_ic,mdot_loop_ic,mdot_gshp,
                     cpw,Tg_ic,Ta_ic,BC0):
        self.gshp = gshp
        self.wts = wts
        self.loop = loop
        self.solutions = []
        self.BCs = []
        # Other [0] = Pumping power (both loop pumps)
        self.Other = []
        self.residuals = []
        self.T_max = 393.15 # Kelvin - anything past boiling point does not make sense
        self.T_min = 273.15 # Kelvin - anything below freezing does not make sense
        self.slack_var_sensitivity = 1000
        
        # Variables
        # Tin = X[0]
        # Tout_next = X[1]
        # T_tank_next = X[2]
        # T_vec = X[3:3+self.loop.NT]
        # Tgvec = X[3+self.loop.NT:4+self.loop.NT + self.gshp.number_element]
        # Now adjust the initial ground temperature at the first GSHP node to admit the temperature 
        # gradient.
        self.Xm1 = self._initial_conditions(T_tank_ic,Q_hp_ic,mdot_loop_ic,mdot_gshp,
                     cpw,Tg_ic,Ta_ic)
        self.time = 0.0
        self.BCm1 = BC0
    
    def _initial_conditions(self,T_tank_ic,Q_hp_ic,mdot_loop_ic,mdot_gshp,
                     cpw,Tg_ic,Ta_ic):
        # establish initial conditions
        self.num_slack = 0
        X0 = np.zeros(self.gshp.number_element + self.loop.NT + 4 + self.num_slack)
        X0[0] = Ta_ic
        X0[1] = Tg_ic
        X0[2] = T_tank_ic
        X0[3:3+self.loop.NT] = np.array([T_tank_ic + idx * Q_hp_ic/(mdot_loop_ic * cpw) for idx in range(1,self.loop.NT+1)])
        # (Ta_ic + Tg_ic)/2+idx * (0.5*Tg_ic-0.5*Ta_ic) / gshp.number_element
        X0[3+self.loop.NT:4+self.loop.NT + self.gshp.number_element] = (
            np.array([Tg_ic for idx in 
                      range(0,self.gshp.number_element+1)]))
        # slack variables to assure satisfaction of the 2nd law of thermodynamics
        # in the ground based heat pump.
        #X0[4+self.loop.NT + self.gshp.number_element:5+self.loop.NT + self.gshp.number_element+self.num_slack] = 0.0
        return X0
        

    def equn(self,X,BC):
        
        # apply realistic limits to temperatures - No freezing and no boiling!
        # if (X[:-2] > self.T_max).sum() > 0:
        #     X[:-2][X[:-2] > self.T_max] = self.T_max
        #     # raise ValueError("The maximum valid temperature for this system"
        #     #                  + " (boiling point of water) has been exceeded."
        #     #                  + " Please redesign the inputs to values that"
        #     #                  + " can handle the loads involved!")
        # if (X[:-2] < self.T_min).sum() > 0:
        #     X[:-2][X[:-2] < self.T_min] = self.T_min
        #     # raise ValueError("The minimum valid temperature for this system"
        #     #                  + " (freezing point of water) has been exceeded."
        #     #                  + " Please redesign the inputs to values that"
        #     #                  + " can handle the loads involved!")
        
        # Boundary Conditions / Control signals
        is_cooling = bool(BC[0])
        W_gshp = BC[1]
        Tg = BC[2]
        TA = BC[3]
        QTriplex = BC[4]   # This is the triplex heat pump boundary condition
        mdot_loop = BC[5]
        dt = BC[6]
        mdot_gshp = BC[7] # this is not longer used!!
        Q_econo = BC[8]
        consume_loop_water = BC[9]
        
        Vdot = self.inp_hp.Vdot
        pressure = self.inp_triplex.pressure
        rh_inside = self.inp_triplex.rh_internal
        if is_cooling:
            T_inside = self.inp_triplex.cold_thermostat
        else:
            T_inside = self.inp_triplex.hot_thermostat
        
        
        # Variables
        Tin_next = X[0]
        Tout_next = X[1]
        T_tank_next = X[2]
        T_vec = X[3:3+self.loop.NT]
        Tgvec = X[3+self.loop.NT:4+self.loop.NT + self.gshp.number_element]
        # slack variable to assure 2nd law conformance for GSHP
        
        # Apply 2nd law constraint on gshp loop performance through additional
        # slack variable S1
        #if Tin_next > Tgvec[0]:
        #    slack_residual1 = (Tout_next - np.abs(S1) - Tgvec[0])*self.slack_var_sensitivity
        #else:
        #    slack_residual1 = (Tout_next + np.abs(S1) - Tgvec[0])*self.slack_var_sensitivity
        ## Assure Tout_next is greater than zero
        #slack_residual2 = (Tout_next - np.abs(S2))*self.slack_var_sensitivity

        hour_of_day = np.int(np.floor(np.mod(self.time / self.seconds_in_an_hour,self.hours_in_a_day)))
        # extract previous time step's condition for non-steady state variables
        Tgvec_m1 = self.Xm1[3+self.loop.NT:4+self.loop.NT + self.gshp.number_element]
        T_vec_m1 = self.Xm1[3:3+self.loop.NT]
        T_tank = self.Xm1[2]
        Tin = self.Xm1[0]
        Tout = self.Xm1[1]
        
        
        T_vec = np.concatenate([np.array([T_tank_next]),T_vec])
        T_vec_m1 = np.concatenate([np.array([T_tank]),T_vec_m1])
        
        residuals_gshp = self.gshp.equn(Tgvec,Tin_next,Tin,Tout_next,Tout,T_tank_next,W_gshp,is_cooling,
                                       mdot_gshp,Tgvec_m1,dt,Q_econo)
        if is_cooling:
            Q_GSHP = self.gshp.Qc
        else:
            Q_GSHP = self.gshp.Qh
            # HERE NEGATIVE HEAT means heat is leaving the Heat pump and being
            # delivered to the tank. The tank's control volume works with 
            # heat leaving being positive (opposite conventions)
        residual_wts = self.wts.equn(mdot_loop,Q_GSHP,T_vec[-1],
                                     T_tank,TA,dt,T_tank_next, hour_of_day,consume_loop_water)
        residuals_loop = self.loop.equn(T_vec, TA, mdot_loop, T_vec_m1, dt, hour_of_day, consume_loop_water,QTriplex,T_inside)
        
        
        residuals = np.concatenate((residuals_gshp,
                                    np.array([residual_wts]),
                                    residuals_loop))
        
        return residuals

    def solve(self,BC,X0,update=True):
        if update:
            self.Xm1 = X0
            
        X,infodict,ier,mesg = fsolve(self.equn, X0, args=(BC),full_output=True)
        
        if ier != 1:
            raise Exception("fsolve solution failed. Internal message =\n\n" + mesg)
        
        if update:
            self.solutions.append(X)
            self.BCs.append(BC)
            self.BCm1 = BC
            self.residuals.append(self.equn(X,BC))
        return X

class pumping_inputs(object):
    
    def __init__(self, elev_change=0.0, #m
                       pipe_roughness=120,
                       extra_minor_losses=30.0,
                       pump_efficiency=0.9): # dimensionless C-factor
        self.elev_change = elev_change
        self.pipe_roughness = pipe_roughness
        self.extra_losses = extra_minor_losses
        self.eta_p = pump_efficiency
        
        
class Controls(object):
    
    def __init__(self,
                 CS,
                 gshp_delT_target,
                 loop_delT_target,
                 compressor_capacity,
                 loop_pump_mass_flow_capacity,
                 gshp_pump_mass_flow_capacity,
                 tank_temp_dead_band,
                 min_flow_loop):
        self.wts = CS.wts
        self.gshp = CS.gshp
        self.loop = CS.loop
        self.CS = CS
        self.gshp_delT = gshp_delT_target
        self.loop_delT = loop_delT_target
        self.loop_pump_capacity = loop_pump_mass_flow_capacity
        self.gshp_pump_capacity = gshp_pump_mass_flow_capacity
        self.comp_cap = compressor_capacity
        self.BCs = []
        self.waiting_in_deadband = True
        self.tank_temp_dead_band = tank_temp_dead_band
        self.is_cooling_m1 = False
        self.min_flow_loop = min_flow_loop
        

    
    def master_control(self,X,Ta,Tg,QHP,T_tank_target,dt,ma_num_dt):
        """
        Apply all control schemes and produce the next time step's 
        boundary conditions
        
        obj.master_control(X,Ta,Tg,QHP,T_tank_target,dt)
        
        Inputs: X : np.array(variable length) : solution vector for the 
                                                wts-gshp-wshp system
                Ta : float : current ambient drybulb temperature (K)
                Tg : float : current ground temperature (K)
                QHP : float : current heat flow to/from triplex unit
                              wshp's
                T_tank_target : float : target value for the temperature of
                                        fluid in the wts tank (K).
                dt : float : > 0 time step.
            
        Returns: BC : np.array(9)
        """
        
        T_tank = X[2]
        
        T_tank_in = X[2+self.loop.NT]
        
        if T_tank_target < T_tank:
            is_cooling = True
        else:
            is_cooling = False
        
        mdot_loop = self.loop_pump_control(T_tank_out = T_tank,
                                           T_tank_in = X[2+self.loop.NT],
                                           delT_target = self.loop_delT)
        #
        mdot_loop = mdot_loop + self.wts.mdot_triplex_hw
        
        
        # stop operations after heating cooling from deadband
        if ((not self.waiting_in_deadband and self.is_cooling_m1 and T_tank <= T_tank_target) or
            (not self.waiting_in_deadband and not self.is_cooling_m1 and T_tank >= T_tank_target)):
            self.waiting_in_deadband = True
        
        # determine if tank needs conditioning
        if ((T_tank_target - 0.5 * self.tank_temp_dead_band > T_tank) or
            (T_tank_target + 0.5 * self.tank_temp_dead_band < T_tank) or
            not self.waiting_in_deadband):
            self.waiting_in_deadband = False
            use_heat_pump_or_econo = True
        else:
            use_heat_pump_or_econo = False

        if len(self.CS.solutions) < ma_num_dt:
            ma_num_dt = len(self.CS.solutions) 
        X2arr = np.concatenate([np.array(self.CS.solutions[-ma_num_dt:])[:,2],np.array([X[2]])])
        dT_tank_dt = np.mean(np.diff(X2arr)/self.CS.BCm1[6])
            
        if use_heat_pump_or_econo:
            # calculate a moving average for the tank rate of change of temperature
            # based on a couple of hours.
            W, mdot_gshp, Q_econo = self.gshp_control(Tin = X[0],
                                             mdot_m1 = self.CS.BCm1[7],
                                             Tout = X[1],
                                             T_tank_target = T_tank_target,
                                             T_tank = X[2],
                                             dT_tank_dt = dT_tank_dt,
                                             is_cooling=is_cooling,
                                             time_to_target=10*dt,
                                             delT_target=self.gshp_delT,
                                             dt=self.CS.BCm1[6])
        else:
            # TODO - economization could make sense here as well.
            W, mdot_gshp, Q_econo = self.gshp_control(Tin = X[0],
                                             mdot_m1 = self.CS.BCm1[7],
                                             Tout = X[1],
                                             T_tank_target = T_tank_target,
                                             T_tank = X[2],
                                             dT_tank_dt = dT_tank_dt,
                                             is_cooling=is_cooling,
                                             time_to_target=10*dt,
                                             delT_target=self.gshp_delT,
                                             dt=self.CS.BCm1[6],
                                             EnforceWZero=True)
        
        # assure bounds on control variables (redundant)
        if mdot_loop < 0:
            raise ValueError("The loop mass flow cannot be negative!")
        elif mdot_loop > self.loop_pump_capacity:
            mdot_loop = self.loop_pump_capacity
                    
        if mdot_gshp < 0:
            raise ValueError("The gshp mass flow cannot be negative!")
        elif mdot_gshp > self.gshp_pump_capacity:
            mdot_gshp = self.gshp_pump_capacity
        
        
        self.is_cooling_m1 = is_cooling

        # determine if hot water should be consumed from loop.
        # universally assume utility water comes in at the average of ambient
        # and ground water.
        if (T_tank_in + T_tank)/2 > (Ta + self.gshp.Tg)/2:
            consume_loop_water = True
        else:
            consume_loop_water = False

            
        # set next time step's boundary conditions
        BC = np.zeros(10)
        BC[0] = is_cooling
        BC[1] = W
        BC[2] = Tg
        BC[3] = Ta
        BC[4] = QHP # heat gained/obsorbed by loop for each triplex.
        BC[5] = mdot_loop
        BC[6] = dt
        BC[7] = mdot_gshp
        BC[8] = Q_econo
        BC[9] = consume_loop_water
        return BC
        
    def loop_pump_control(self,T_tank_out,T_tank_in,delT_target):
        mdot = np.abs(self.CS.loop.loop_heat) / (delT_target * self.wts.cp)
        if mdot < self.min_flow_loop:
            mdot = self.min_flow_loop
        return mdot
    
    def gshp_control(self,Tin,mdot_m1,Tout,T_tank_target,
                     T_tank,dT_tank_dt,is_cooling,time_to_target,delT_target,dt,EnforceWZero=False):
        # This returns the amount of work to put into the ground source heat pump exchanger
        
        Tavg = (Tin + Tout)/2
        #delT = Tin - Tout
        
        COP = self.gshp.gshp_cop(is_cooling,Tavg,T_tank)
        
        # restore the tank to the target temperature within time_to_target.
        heat_delta = (T_tank - T_tank_target)*self.wts.mass * self.wts.cp
        power_trend = self.wts.mass * self.wts.cp * dT_tank_dt
        
        if (heat_delta > 0 and power_trend < 0 or
            heat_delta < 0 and power_trend > 0):
            #
            time_with_no_help = np.abs(heat_delta/power_trend)
        else:
            time_with_no_help = time_to_target + 1 # this just assures the system will work.
            
        if time_with_no_help < time_to_target:
            Qgshp_target = 0.0  # no need to add or subtract heat. Save energy.
        else:
            Qgshp_target = -(heat_delta + power_trend * time_to_target)/time_to_target
            if is_cooling and Qgshp_target > 0:
                raise ValueError("How can we be in a cooling state and need to add heat?")
        
        # target mass flow for economization (output water from )
        mdot_target = np.abs(Qgshp_target/(self.wts.cp * (Tout - (T_tank + 0.5*dt*dT_tank_dt))))
        if mdot_target > self.gshp_pump_capacity:
            mdot_target = self.gshp_pump_capacity
        
        # TODO economization needs work. It is not a consistent condition
        # its sign gives a guess at applying economization. We will see how it 
        # plays out in the actual tank and gshp equations
        
        if is_cooling and Qgshp_target > 0: # no cooling is needed
            return 0,0,0
        elif is_cooling and Qgshp_target < 0: 
            # determine if economization can meet the heating load
            if T_tank + 0.5*dt*dT_tank_dt > Tout:
                # a negative heat flux is desired
                Q_econo = mdot_target * self.wts.cp * (Tout - (T_tank + 0.5*dt*dT_tank_dt))
            else:
                Q_econo = 0.0
            
            if -Q_econo > -Qgshp_target:
                W = 0.0
                Q_econo = Qgshp_target
            else:
                W = np.min([(-(Qgshp_target-Q_econo))/COP,self.comp_cap])
            if EnforceWZero:
                W = 0.0    

            return W, mdot_target, Q_econo
        else:
            if Qgshp_target > 0:
                if T_tank + 0.5*dt*dT_tank_dt < Tout:
                    # want a positive result for heating
                    Q_econo = mdot_target * self.wts.cp * (Tout - (T_tank + 0.5*dt*dT_tank_dt))
                else:
                    Q_econo = 0.0
                    
                if Q_econo > Qgshp_target:
                    W = 0.0
                    Q_econo = Qgshp_target
                else:
                    W = np.min([(Qgshp_target-Q_econo)/COP,self.comp_cap])
                if EnforceWZero:
                    W = 0.0

                return W, mdot_target, Q_econo
            else: # no heating is needed - let the extra heat dissipate before acting
                return 0,0,0

class unit_convert():
    cm_p_inch = 2.54
    meter_per_cm = 0.01
    C_to_K = 273.15

class gshp_inputs(object):
    
    def __init__(self,bore_diam=4.25 * 2.54 * 0.01, # 4in plus change in meters
                      pipe_outer_diam=2.0 * 2.54 * 0.01, # 2 in in meters
                      pipe_inner_diam=1.8 * 2.54 * 0.01, # 1.8 in in meters 
                      pipe_therm_cond=0.44, # High density polyethylene (W/(m*K))
                      grou_therm_cond=0.4, # based off sand/gravel/clay/silt mixtures characteristic of Albuquerque
                      grout_therm_cond=1.0, # based off of silicone/sand mixture
                      grou_speci_heat=800.0, # J/(kg*K)
                      ground_density=1900.0, # kg/m3
                      grou_temperature=293.15, # K
                      number_borehole=60,
                      borehole_length=80.0,
                      dist_boreholes=5.0, # 5 meter spacing between boreholes
                      num_element=20,
                      max_COP_heating=7.0,
                      frac_carnot_efficiency=0.9,
                      delT_target=2.0,
                      compressor_capacity=10000, # W
                      mass_flow_capacity=10.0,  # kg/s
                      frac_borehole_length_to_Tg=1.0): # CALIBRATION FACTOR
                       
        self.rb = bore_diam/2.0
        self.r0 = pipe_outer_diam/2.0
        self.ri = pipe_inner_diam/2.0
        self.kp = pipe_therm_cond
        self.kg = grou_therm_cond
        self.k_grout = grout_therm_cond
        self.cpg = grou_speci_heat
        self.rhog = ground_density
        self.nb = number_borehole
        self.L_bore = borehole_length
        self.dist_boreholes = dist_boreholes
        self.num_element = num_element
        self.max_COP_h = max_COP_heating
        self.eta_c = frac_carnot_efficiency
        self.delT_target = delT_target
        self.compressor_capacity = compressor_capacity
        self.mass_flow_capacity = mass_flow_capacity 
        self.calib_fract = frac_borehole_length_to_Tg

class heat_pump_inputs(object):
    
    def __init__(self,rated_heating_capacity_W=3516.85,
                      rated_cooling_capacity_W=3516.85,
                      rated_COP_air=3.0,
                      rated_COP_water=4.2,
                      rated_air_flow=0.15338292125,# 325 CFM See Comfort Air Model HB-009 model
                      rated_SHR=0.75,
                      rated_gshp_heating_capacity_W=50000,
                      rated_gshp_cooling_capacity_W=50000,
                      rated_gshp_COP=5.0,
                      rated_gshp_water_flow=10.0/997,
                      sensible_heat_ratio_gshp=1.0):
        self.heat_capacity = rated_heating_capacity_W
        self.cool_capacity = rated_cooling_capacity_W
        self.COP_air = rated_COP_air
        self.COP_water = rated_COP_water
        self.Vdot = rated_air_flow
        self.SHR = rated_SHR
        self.gshp_heat_capacity = rated_gshp_heating_capacity_W
        self.gshp_cool_capacity = rated_gshp_cooling_capacity_W
        self.gshp_COP = rated_gshp_COP
        self.gshp_water_flow = rated_gshp_water_flow
        self.gshp_SHR = sensible_heat_ratio_gshp # water source does not have latent heat on either side
        
        


class water_inputs(object):
    
    def __init__(self,specific_heat=4182.0, # J/(kg*K
                      density=997.0):
        self.cp = specific_heat # J/(kg*K)
        self.rho = density # kg/m3        
            
class tank_inputs(object):

    def __init__(self,
        tank_height = 2.0, # m
        tank_diameter = 3.0, # m
        insulation_thick = 0.1, #m
        insulation_k = 0.001, # W/(m*K)
        temperature_dead_band = 5 # K
        ):
        self.tank_height = tank_height
        self.tank_diameter = tank_diameter
        self.insulation_thick = insulation_thick
        self.insulation_k = insulation_k
        self.temp_dead_band = temperature_dead_band
        
class loop_inputs(object):
    
    def __init__(self,
                 distance_between_triplex_wshp=10.0,
                 pipe_outer_diam_loop=2.0*0.0254, # 2in in meters
                 pipe_inner_diam_loop=1.8*0.0254, # 1.8in in meters
                 trenching_depth=1.0, #meters
                 num_triplex_units=21, # number of water source heat pumps (wshp)
                 delT_target_loop=2.0,
                 mass_flow_capacity_loop=10.0,
                 min_flow_loop=1.0): 
        self.L_triplex = distance_between_triplex_wshp # meters
        self.rpi = pipe_inner_diam_loop
        self.rpo = pipe_outer_diam_loop
        self.trenching_depth = trenching_depth
        self.number_triplex_units = num_triplex_units
        self.delT_target = delT_target_loop
        self.mass_flow_capacity = mass_flow_capacity_loop
        self.min_flow_loop = min_flow_loop
        
class triplex_inputs(object):
    gal_per_day_2_m3_per_second = 0.00378541 / (3600*24) # checked 6/24/2021
    def __init__(self,
                 cold_thermostat = 24.4 + 273.15,  #76F
                 hot_thermostat = 22.2 + 273.15, #72F
                 hot_water_consumption_per_triplex_gal_per_day = 66.7,
                 hot_water_use_fraction = None,
                 hot_water_setpoint = 60.0+273.15): 
        self.cold_thermostat = cold_thermostat
        self.hot_thermostat = hot_thermostat
        self.rh_internal = 0.36
        self.pressure = 101325
        self.hot_water_demand = hot_water_consumption_per_triplex_gal_per_day * self.gal_per_day_2_m3_per_second
        self.hot_water_setpoint = hot_water_setpoint
        
        # 66.7 gal/day is for a two bedroom house page 221 of "New Mexico TRM-2018...
        # which cites the Building America Research Benchmark Report (Wilson et. al, 2014) found at
        # https://www.nrel.gov/docs/fy10osti/47246.pdf
        # BEGIN all this is obsolete
        df_coef = pd.DataFrame(
                        [["MiniSplitUnitAC1_MeissnerEtAl","EER",0.678803,0.050035,-0.00086,-0.02175,-0.000156,0.0009333],
                        ["MiniSplitUnitAC1_MeissnerEtAl","TC",1.388474,0.076999,-0.00093,-0.05425,0.0002746,0.0001448],
                        ["MiniSplitUnitAC1_MeissnerEtAl","SC",-2.05444,0.254437,-0.0052,-0.00023,-0.0003201,0.0005219]],
                            columns=["AC Unit","Curve Type","a0","a1","a2","a3","a4","a5"])
        ht_C = hot_thermostat - 273.15
        ct_C = cold_thermostat - 273.15
        df_thermostat = pd.DataFrame([["MiniSplitUnitAC1_MeissnerEtAl","Cooling (⁰C)",ct_C,ct_C,ct_C,ct_C],
                                      ["MiniSplitUnitAC1_MeissnerEtAl","Heating (⁰C)",ht_C,ht_C,ht_C,ht_C]],
                                     columns=["AC Unit","Type","Blue sky","Tier 1","Tier 2","Tier 3"])
        #per meissnerEtAl OBSOLETE
        self.minisplit_AC = Wall_AC_Unit(TC=3400,
                              SCfrac=0.699805699,
                              derate_frac=0.964126697,
                              npPower=1300,
                              flow_rated=500,
                              df_coef=df_coef,
                              df_thermostat=df_thermostat,
                              tier="Blue sky",
                              Name="Minisplit_MeissnerEtAl")
        # END all this is obsolete
        
        # This profile is from Figure 13 of Wilson, E. Cengebrecht Metzger, S. Horowitz, and R. Hendron. 
        # 2014. "2014 Building America House Simulation Protocols." NREL Tech Report NREL/TP-5500-60988
        if hot_water_use_fraction is None:
            self.hot_water_use_fraction =  [0.0062,
                                            0.0029,
                                            0.00085903,
                                            0.00082057,
                                            0.0031,
                                            0.0219,
                                            0.075,
                                            0.0794,
                                            0.0765,
                                            0.0669,
                                            0.0611,
                                            0.0488,
                                            0.0423,
                                            0.0376,
                                            0.0328,
                                            0.0376,
                                            0.0437,
                                            0.0578,
                                            0.0686,
                                            0.0654,
                                            0.0593,
                                            0.0486,
                                            0.0423,
                                            0.0235]
        else:
            self.hot_water_use_fraction = hot_water_use_fraction

        

class wts_gshp(object):
    
    def __init__(self,
                 ground_temperature,
                 inp_gshp,
                 inp_tank,
                 inp_water,
                 inp_loop,
                 inp_gshp_pumping,  #Pump_Flow_Resistance object
                 inp_loop_pumping,  #Pump_Flow_Resistance object
                 inp_triplex,
                 inp_hp,
                 qhp_loads_file,
                 weather_file,
                 data_path,
                 pump_poly_dict,
                 initial_tank_temperature=None,
                 load_shift_file=None):
        """
        Inputs
            ground_temperature : float : > 0 : Average ground temperature from
                                10 - 100 meters deep.
            inp_gshp : gshp_inputs : populate the needed properties or 
                                     accept defaults
        """
        self.dt = 15 * 60  # TODO - make time step size an actual variable.
        self.num_ma_dt = 3*4 # TODO - move this to the inputs
        
        initial_loop_mass_flow=inp_loop.min_flow_loop
        initial_gshp_mass_flow=inp_gshp.mass_flow_capacity * 0.1
        
        # setup paths needed
        base_path = os.path.dirname(__file__)
        data_path = os.path.join(base_path,r"..","Data")

        # gshp inputs
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
        max_GSHP_COP_h = inp_gshp.max_COP_h
        eta_c = inp_gshp.eta_c
        num_bore_hole = inp_gshp.nb
        dist_to_Tg_reservoir = L_bore * inp_gshp.calib_fract
        
        # tank inputs
        tank_height = inp_tank.tank_height
        tank_diameter = inp_tank.tank_diameter
        insulation_thick = inp_tank.insulation_thick
        insulation_k = inp_tank.insulation_k
        temp_dead_band = inp_tank.temp_dead_band
        
        # water inputs
        cp_water = inp_water.cp
        rho_water = inp_water.rho
        
        self.rho_water = rho_water
        
        # loop inputs
        L_triplex = inp_loop.L_triplex
        loop_rpi = inp_loop.rpi
        loop_rpo = inp_loop.rpo
        trenching_depth = inp_loop.trenching_depth
        number_triplex_units = inp_loop.number_triplex_units
        
        # initial conditions
        Tg = ground_temperature
        if initial_tank_temperature is None:
            initial_tank_temperature = Tg
            
        # Boundary Conditions
        self.BC_histories = self._process_bc_files(data_path,weather_file,qhp_loads_file)        
        self.Tg = ground_temperature
        TA0 = self.BC_histories.iloc[0,1]
        QHP0 = self.BC_histories.iloc[0,0]
        
        # Initial values may not be ideal. TODO precompute ideal initial values.
        if QHP0 < 0:
            is_cooling = True
        else:
            is_cooling = False            
        self.BC0 = np.array([is_cooling, 
                            np.abs(QHP0*3.0),
                            Tg,
                            TA0,
                            QHP0,
                            initial_loop_mass_flow,
                            self.dt,
                            initial_gshp_mass_flow,
                            0.0,
                            False]) # new entry is Q_econo between GSHP and Tank
        

        # build objects for each system
        gshp = Transient_Simple_GSHP(rb,r0,ri,kp,kg,rhog,cpg,
                                     eta_c,L_bore,number_element,
                                     distance_to_next_borehole,max_GSHP_COP_h,
                                     cp_water,rho_water,Tg,k_grout,num_bore_hole,
                                     dist_to_Tg_reservoir)
        ST = storage_tank(cp_water, tank_height, tank_diameter, insulation_thick, 
                          insulation_k, rho_water, inp_triplex,number_triplex_units,Tg)
        loop = equal_loop(L_triplex,kg,cp_water,loop_rpi,loop_rpo,trenching_depth,
                          kp,number_triplex_units,QHP0,rho_water,inp_triplex,
                          inp_loop.min_flow_loop,Tg,inp_triplex.hot_water_setpoint,inp_hp.Vdot)
        
        # initialize combined systems object guess values for mass flow are
        # low and will be corrected once solutions start seeking out the target
        # values.
        self.cs = CombinedSystem(gshp,ST,loop,initial_tank_temperature,
                                   QHP0,initial_loop_mass_flow,
                                   initial_gshp_mass_flow,cp_water,Tg,TA0,
                                   self.BC0)
        # initialize the controls object.
        self.controls = Controls(self.cs,
                            inp_gshp.delT_target, 
                            inp_loop.delT_target, 
                            inp_gshp.compressor_capacity,
                            inp_loop.mass_flow_capacity,
                            inp_gshp.mass_flow_capacity,
                            temp_dead_band,
                            inp_loop.min_flow_loop)
        
        # provide a way to reanalyze pump work curve.
        dump_inputs_for_Pipe_Flow_Resistance = False
        if dump_inputs_for_Pipe_Flow_Resistance:
            import pickle as pkl; from datetime import datetime
            now = datetime.now()
            date_str = str(now).split(" ")[0]
            pkl.dump([self.cs,inp_gshp_pumping,inp_loop_pumping],open("inputs_for_Pipe_Flow_Resistance_"+date_str+".pickle",'wb'))
            
        self.inp_triplex = inp_triplex
    
        
        # formulate heat pumps 

        hp_coef = SingleStageHPInputs()
        cooling_air = HVAC_Input(Qdot=inp_hp.cool_capacity,
                                 Qdot_coef=hp_coef.Qdot_coef_c,
                                 COP=inp_hp.COP_air,
                                 EIR_coef=hp_coef.EIR_coef_c,
                                 Vdot=inp_hp.Vdot,
                                 PLF_coef=hp_coef.PLF_coef,
                                 SHR=inp_hp.SHR)
        
        heating_air = HVAC_Input(Qdot=inp_hp.heat_capacity,
                                 Qdot_coef=hp_coef.Qdot_coef_h,
                                 COP=inp_hp.COP_air+1,
                                 EIR_coef=hp_coef.EIR_coef_h,
                                 Vdot=inp_hp.Vdot,
                                 PLF_coef=hp_coef.PLF_coef)
        
        cooling_h2o = HVAC_Input(Qdot=inp_hp.cool_capacity,
                                 Qdot_coef=hp_coef.Qdot_coef_c,
                                 COP=inp_hp.COP_water,
                                 EIR_coef=hp_coef.EIR_coef_c,
                                 Vdot=inp_hp.Vdot,
                                 PLF_coef=hp_coef.PLF_coef,
                                 SHR=inp_hp.SHR)
        
        heating_h2o = HVAC_Input(Qdot=inp_hp.heat_capacity,
                                 Qdot_coef=hp_coef.Qdot_coef_h,
                                 COP=inp_hp.COP_water+1,
                                 EIR_coef=hp_coef.EIR_coef_h,
                                 Vdot=inp_hp.Vdot,
                                 PLF_coef=hp_coef.PLF_coef)
        
        # TODO - build a water-water heat pump model with performance
        cooling_gshp = HVAC_Input(Qdot=inp_hp.gshp_cool_capacity,
                                 Qdot_coef=hp_coef.Qdot_coef_c,
                                 COP=inp_hp.gshp_COP,
                                 EIR_coef=hp_coef.EIR_coef_c,
                                 Vdot=inp_hp.gshp_water_flow,
                                 PLF_coef=hp_coef.PLF_coef,
                                 SHR=inp_hp.gshp_SHR)
        
        heating_gshp = HVAC_Input(Qdot=inp_hp.gshp_heat_capacity,
                                 Qdot_coef=hp_coef.Qdot_coef_h,
                                 COP=inp_hp.gshp_COP+1,
                                 EIR_coef=hp_coef.EIR_coef_h,
                                 Vdot=inp_hp.gshp_water_flow,
                                 PLF_coef=hp_coef.PLF_coef)
        
        self.HPair = NREL_AC_2013(cooling_air,heating_air)
        self.HPh2o = NREL_AC_2013(cooling_h2o,heating_h2o)
        # TODO - reorder this - this is a horrible way to establish an attribute!
        # TODO - implement this fully but with a correct water-water heat pump
        #        set of coefficients that doesn ot include dry bulb-wet bulb 
        #        considerations.
        self.cs.gshp.HPgshp = NREL_AC_2013(cooling_gshp,heating_gshp)
        self.cs.loop.HPh2o = self.HPh2o
        self.inp_hp = inp_hp
        self.cs.inp_hp = inp_hp
        self.cs.inp_triplex = inp_triplex
        
        self.Q_h_air_total = 0.0
        self.P_h_air_total = 0.0
        self.Q_c_air_total = 0.0
        self.P_c_air_total = 0.0
        
        self.Q_h_h2o_total = 0.0
        self.P_h_h2o_total = 0.0
        self.Q_c_h2o_total = 0.0
        self.P_c_h2o_total = 0.0
        
        self.P_hw_air_total = 0.0
        self.P_hw_h2o_total = 0.0
        self.P_hw_c_air_total = 0.0
        self.P_hw_h_air_total = 0.0
        self.P_hw_c_h2o_total = 0.0
        self.P_hw_h_h2o_total = 0.0
        
        # create room for mass flow and x2=(frac or nb) 
        for key,val in pump_poly_dict.items():
            val.append(0)
            val.append(0)
        
        self.pump_poly_dict = pump_poly_dict
        
        
    def no_load_shift_simulation(self,T_tank_target,print_percent_complete=False,
                                 percent_to_complete=101.0,
                                 troubleshoot_stop_time=None,
                                 break_on_troubleshoot=False,
                                 skip_to_step=0):
        
        ## mini_split is obsolete, The new NREL model is used here instead
        #mini_split = self.inp_triplex.minisplit_AC
        
        # now establish the first boundary conditions make the Tank target temperature
        # the ground temperature (this may change.)
        perc_complete = 0
        BC = self.BC0
        X0 = self.cs.Xm1
        step = 0
        for idx in range(len(self.BC_histories)-1):
            
            # allow skipping to summer time or a deep time in the simulation
            # BC's cannot be replicated though.
            if step < skip_to_step:
                
                step += 1
                continue

            
            if not troubleshoot_stop_time is None:
                if (idx - skip_to_step) * self.dt  > troubleshoot_stop_time:
                    import pdb; pdb.set_trace()
                    if break_on_troubleshoot:
                        break
                #raise ValueError("The solution for temperatures should never be less than zero!")
            dt = BC[6]
            self.cs.time = idx * dt  
            
            X = self.cs.solve(BC,X0,update=True)

            BC = self.controls.master_control(X,
                                          self.BC_histories.iloc[idx+1,1],
                                          self.Tg,
                                          self.BC_histories.iloc[idx+1,0],
                                          T_tank_target[idx],
                                          self.dt,
                                          self.num_ma_dt)
            
            self._post_process_solve(X,BC)
            
            X0 = X
            if print_percent_complete: 
                if perc_complete < (100*idx/(len(self.BC_histories)-1)):
                    print("{0:5.2f}% complete".format(100*idx/(len(self.BC_histories)-1)))
                    perc_complete = np.ceil(100*idx/(len(self.BC_histories)-1))
                    if 100*(idx-skip_to_step)/(len(self.BC_histories)-1) > percent_to_complete:
                        break
                    
            
                    
        
            
    def _calculate_system_COP(self,Todb,T_vec,rh_inside,T_inside,Qload,pumping_power,W):
        # TODO - you will need a fan model. Does the fan run at 100% for the entire RTF (Run time fraction)?
        
        # Power_air is only for a single unit.
        Power_air, COP_air, RTF, Qcondenser,Qavg = self.HPair.calculate(Qload, Todb, T_inside, rh_inside, 101325, self.inp_hp.Vdot)
        
        water_Qdot = 0.0
        water_P = 0.0
        for T in T_vec:
            P_h2o, COP_h2o, RTF_h2o, Qcondenser,Qavg = self.HPh2o.calculate(Qload, T, T_inside, rh_inside, 101325, self.inp_hp.Vdot)
            Q_h2o = np.abs(Qavg)
            water_Qdot += Q_h2o
            water_P += P_h2o
            
        water_COP = (water_Qdot / (W + pumping_power + water_P))
        
        if water_COP < 0 or W < 0:
            pass
            #raise ValueError("This should not happen!")

        
        return COP_air, water_COP, Power_air * self.cs.loop.NT, W + pumping_power + water_P

    def _post_process_solve(self,X,BC):
        Tin_gshp = X[0]
        Tout_gshp = X[1]
        T_tank = X[2]
        T_tank_in = X[3+self.cs.loop.NT-1]
        mass_flow_loop = BC[5]
        mass_flow_gshp = BC[7]
        Qhw_loop = self.cs.loop.Qhw
        Q_makeup_tank = self.cs.wts.Q_water_consumption
        mdot_makeup = self.cs.wts.mdot_triplex_hw
        Q_makeup = Qhw_loop - Q_makeup_tank
        P_hw_no_loop = self.cs.loop.Ehw_no_loop
        P_hw_loop = self.cs.loop.Ehw_loop
        
        if (mdot_makeup > mass_flow_loop):
            raise ValueError("The triplexes cannot consume more water than is flowing into the thermal loop!")
            
        frac = mdot_makeup / mass_flow_loop
        nb = self.cs.gshp.nb
        
        self.pump_poly_dict["GSHP"][2] = mass_flow_gshp
        self.pump_poly_dict["Loop"][2] = mass_flow_loop
        self.pump_poly_dict["GSHP"][3] = nb
        self.pump_poly_dict["Loop"][3] = frac
        
        power_predict = []
        
        for key,lis in self.pump_poly_dict.items():
            clf = lis[0]
            poly = lis[1]
            mdot = lis[2]
            x2 = lis[3]
            poly_num = poly.fit_transform(np.array([[mdot,x2]]))
            # this predicts the nutural logarithm of pumping power
            power_predict.append(np.exp(clf.predict(poly_num)[0]))
            
        
        pumping_power = np.array(power_predict).sum()          
        
            #pumping_power =  ((mass_flow_loop/self.cs.loop.NT * 0.25 * 9.81 * self.cs.loop.NT)+  
            #                  (mass_flow_gshp * 0.05 * 9.81 * self.cs.gshp.nb))
                          # TODO - This needs to be solved more precisely
                                                                           #        The pressure head is very flow dependent
                                                                           # and data exists for the comforte air unit.
                                                                           # For now I am taking a guess based on the data.                                               
        # an estimate of the system coefficient of performance at the moment
        Tair = BC[3]
        is_cooling = BC[0]
        T_vec = X[3:3+self.cs.loop.NT]
        QHP = BC[4] # QHP is a heating load when negative because it is from the perspective of the loop -QHP for calculate_COP
        W = BC[1]
        dt = BC[6]
        
        
        
        
        
        if QHP > 0:  # Triplex convention is negative QHP is heat flowing into the triplex.
            T_inside = self.inp_triplex.cold_thermostat
        else:
            T_inside = self.inp_triplex.hot_thermostat
        
        COP_air, COP_wts_gshp, P_air, P_wts_gshp = self._calculate_system_COP(Tair, T_vec,0.44,T_inside,QHP,pumping_power,W)
        
        if is_cooling:            
            self.Q_c_h2o_total += P_wts_gshp * COP_wts_gshp
            self.P_c_h2o_total += P_wts_gshp
            self.Q_c_air_total += P_air * COP_air
            self.P_c_air_total += P_air
            self.P_hw_c_h2o_total += P_hw_loop
            self.P_hw_c_air_total += P_hw_no_loop
        else:
            self.Q_h_h2o_total += P_wts_gshp * COP_wts_gshp
            self.P_h_h2o_total += P_wts_gshp
            self.Q_h_air_total += P_air * COP_air
            self.P_h_air_total += P_air
            self.P_hw_h_h2o_total += P_hw_loop
            self.P_hw_h_air_total += P_hw_no_loop
        
        self.P_hw_air_total += P_hw_no_loop
        self.P_hw_h2o_total += P_hw_loop
 

        self.cs.Other.append(np.array([pumping_power,COP_air,COP_wts_gshp,self.cs.gshp.Q_ground_total,Q_makeup,P_wts_gshp,P_air,P_hw_no_loop,P_hw_loop]))
            
    def _process_bc_files(self,data_path,weather_file,qhp_loads_file):
        # Boundary Conditions
        QHP_history = pd.read_csv(os.path.join(data_path,qhp_loads_file))
        weather = pd.read_csv(os.path.join(data_path,weather_file))
        # add load shift request from external grid.
        
        # Ambient temperature is all we use, for now latent heat pulled from the air is not a factor
        # have to start at 48 because there are two design days at the beginning of the data.
        weather = weather[48:]
        weather = weather.append(weather.iloc[-1],ignore_index=True)
        weather.iloc[-1,0] = " 12/31  24:45:00"
        
        date_column =[]
        for date in weather["Date/Time"]:
            str_list = str(date).split(" ")
            time_list = str_list[3].split(":")
            hr = int(time_list[0])-1
            hr_str = "0{0:d}".format(int(str_list[3].split(":")[0])-1)
            if hr >= 10:
                hr_str = hr_str[1:]
            date_column.append(pd.to_datetime(str_list[1] + "/2021" + " " + hr_str + ":" + time_list[1] + ":" + time_list[2]))
        weather.index = date_column
        weather_15min = weather.resample("15min").interpolate()
        
        # now update the QHP_loads_history
        QHP_history = QHP_history[192:]
        QHP_history = QHP_history.drop("Date/Time",axis=1)
        QHP_history.index = weather_15min.index
        BC_histories = pd.concat([weather_15min['Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)'],QHP_history],axis=1)
        BC_histories["Drybulb Temperture (K)"] = BC_histories['Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)'] + unit_convert.C_to_K
        BC_histories.drop(['Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)'],inplace=True,axis=1)
        return BC_histories     

        
if __name__ == "__main__":
    pass

        
        