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

class Transient_Simple_GSHP(object):
    
    def __init__(self,r0,ri,kp,kg,rhog,cpg,n,eta_c,L_bore,number_element,
                 distance_to_next_borehole,max_COP_h,cpw,rhow,Tg):
        
        self.max_COP_h = max_COP_h
        self.eta_c = eta_c
        self.number_element = number_element
        self.cpw = cpw
        
        # This assumes a double portion to ground temperature
        self.RTg = L_bore/(np.pi * distance_to_next_borehole**2/4 * kg)
        self.Tg = Tg
        
        # initial conditions
        self.Tgvec_m1 = Tg * np.ones(number_element+1)
        
        # Pipe radial resistance
        Deq = np.sqrt(n) * r0 * 2.
        Rp = 1/(2.0 * np.pi * kp * L_bore) * np.log(Deq / (Deq - 2*(r0-ri)))
        
        dist2 = distance_to_next_borehole/2.0
        
        RR = np.zeros(number_element)

        rr = np.logspace(np.log10(r0),np.log10(distance_to_next_borehole/2.0),num=number_element+2)
        
        # Radial Heat capacitance of rings of volume of ground
        self.Cg = [rhog * np.pi * (r2**2.0 - r1**2.0) * L_bore * cpg for r2,r1 in zip(rr[1:],rr[:-1])]
        # Radial thermal resistivity of rings of volume of ground
        self.Rg = [np.log(rout/rin)/(2.0*np.pi * L_bore * kg) for rout,rin in zip(rr[1:],rr[:-1])]
        self.radial_coord = rr
        
        # Thermal mass of water in bore-holes.
        self.C_bore = (np.pi * ri ** 2 * L_bore) * rhow * cpw
        
        # add the pipe 
        self.Rg[0] += Rp
        
        self.func_eval = 0
        
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
        
    def equn(self,Tgvec,Tin_next,Tin,Tout_next,Tout,T_tank,W,is_cooling,mdot,Tgvec_m1,dt,post_process=False):
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
        
        COP = self.gshp_cop(is_cooling, Tavg, T_tank)
        
        if is_cooling:
            self.Qc = W * COP # this is needed and is equal to Q_GSHP for the WTS
            # heat is being put into the ground +
            Q_GSHX = W * (1 + COP)
        else:
            self.Qh = -W * COP
            # heat is being extracted from the ground -
            Q_GSHX = -W * (COP - 1)
            
        residuals = 1e20 * np.ones(len(Tgvec)+2)
        Q = np.zeros(len(Tgvec)+1)
        Q[0] = (Tavg - Tgvec[0]) / self.Rg[0]
        Q[-1] = (Tgvec[-1] - self.Tg)/self.RTg
        
        # steady state assumption to stored vs conduction to
        residuals[0] = self.C_bore * (Tavg_next - Tavg)/dt - Q_GSHX + Q[0] 
        residuals[1] = Q_GSHX - mdot * self.cpw * delT
        # 2nd law closing assumption - if Tin_next drifts above the refrigerant output temperature
        # then the 2nd law is being violated. 
        
        for idx in range(1,self.number_element+2):
            if idx < self.number_element: # the insulated boundary is zero heat transfer
                Q[idx] = (Tgvec[idx-1] - Tgvec[idx])/self.Rg[idx]

            residuals[idx+1] = (self.Cg[idx-1] * (Tgvec[idx-1] - 
                            Tgvec_m1[idx-1]) - (Q[idx-1]-Q[idx]) * dt)
                
        return residuals

class equal_loop(object):
    
    def __init__(self,L_triplex,kg,cp,rpi,rpo,depth,kp,NT,QHP0):
        Rp = 1/(2*np.pi * kp * L_triplex) * np.log(rpo/rpi)
        Rg = 1/(2*np.pi * kg * L_triplex) * np.log(depth/rpo)
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
        
    # set of equations    
    def equn(self,Tvec,QHP,TA,mdot):
        if len(Tvec) != self.NT + 1:
            raise ValueError("The vector of temperatures must be number of triplexes long!")
        QA = np.array([(TA - Tavg) / self.Rtotal for Tavg in (Tvec[1:]+Tvec[:-1])/2.0])
        self.loop_heat = sum(QA) + self.NT * QHP
        return np.array([mdot * self.cp * Tdiff + QA_ for Tdiff,QA_ in zip(np.diff(Tvec),QA)]) + QHP
         
class storage_tank(object):
    def __init__(self,cp,tank_height,tank_diameter,insulation_thickness,insulation_k,rho_tank):
        
        self.cp = cp
        top_area = np.pi * (tank_diameter)**2/4.0
        self.volume = top_area * tank_height
        self.mass = rho_tank * self.volume
        # assume top and bottom are subject to ambient temperature (conservative)
        Rtop = insulation_thickness / (insulation_k * top_area)
        tank_radius = tank_diameter / 2.0
        Rsides = 1 / (2*np.pi * insulation_k * tank_height) * np.log((tank_radius + insulation_thickness)/tank_radius)
        # parrallel model
        self.Rtotal = 1/(1/Rtop + 1/Rsides)

        
        pass
    def equn(self,mdot,Q_GSHP,Tin,T,TA,dt,Tnext):
        QA = (T - TA) / self.Rtotal
        return dt * ((mdot * self.cp * (Tin - T) - (Q_GSHP + QA))/(self.mass * self.cp)) + T - Tnext
        
        
class CombinedSystem(object):

    def __init__(self,gshp,wts,loop,T_tank_ic,Q_hp_ic,mdot_loop_ic,mdot_gshp,
                     cpw,Tg_ic,Ta_ic,BC0):
        self.gshp = gshp
        self.wts = wts
        self.loop = loop
        self.solutions = []
        self.BCs = []
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
        self.BCm1 = BC0
    
    def _initial_conditions(self,T_tank_ic,Q_hp_ic,mdot_loop_ic,mdot_gshp,
                     cpw,Tg_ic,Ta_ic):
        # establish initial conditions
        self.num_slack = 2
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
        X0[4+self.loop.NT + self.gshp.number_element:5+self.loop.NT + self.gshp.number_element+self.num_slack] = 0.0
        return X0
        

    def equn(self,X,BC):
        
        # apply realistic limits to temperatures - No freezing and no boiling!
        if (X[:-2] > self.T_max).sum() > 1: 
            raise ValueError("The maximum valid temperature for this system"
                             + " (boiling point of water) has been exceeded."
                             + " Please redesign the inputs to values that"
                             + " can handle the loads involved!")
        if (X[:-2] < self.T_min).sum() > 1:
            raise ValueError("The minimum valid temperature for this system"
                             + " (freezing point of water) has been exceeded."
                             + " Please redesign the inputs to values that"
                             + " can handle the loads involved!")
        
        # Boundary Conditions / Control signals
        is_cooling = bool(BC[0])
        W_gshp = BC[1]
        Tg = BC[2]
        TA = BC[3]
        QHP = BC[4]   # This is the triplex heat pump boundary condition
        mdot_loop = BC[5]
        dt = BC[6]
        mdot_gshp = BC[7]
        
        # Variables
        Tin_next = X[0]
        Tout_next = X[1]
        T_tank_next = X[2]
        T_vec = X[3:3+self.loop.NT]
        Tgvec = X[3+self.loop.NT:4+self.loop.NT + self.gshp.number_element]
        # slack variable to assure 2nd law conformance for GSHP
        S1 = X[-2]
        S2 = X[-1]
        
        # Apply 2nd law constraint on gshp loop performance through additional
        # slack variable S1
        if Tin_next > Tgvec[0]:
            slack_residual1 = (Tout_next - np.abs(S1) - Tgvec[0])*self.slack_var_sensitivity
        else:
            slack_residual1 = (Tout_next + np.abs(S1) - Tgvec[0])*self.slack_var_sensitivity
        # Assure Tout_next is greater than zero
        slack_residual2 = (Tout_next - np.abs(S2))*self.slack_var_sensitivity


        # extract previous time step's condition for non-steady state variables
        Tgvec_m1 = self.Xm1[3+self.loop.NT:4+self.loop.NT + self.gshp.number_element]
        T_tank = self.Xm1[2]
        Tin = self.Xm1[0]
        Tout = self.Xm1[1]
        
        T_vec = np.concatenate([np.array([T_tank]),T_vec])
        
        residuals_gshp = self.gshp.equn(Tgvec,Tin_next,Tin,Tout_next,Tout,T_tank,W_gshp,is_cooling,
                                       mdot_gshp,Tgvec_m1,dt)
        if is_cooling:
            Q_GSHP = self.gshp.Qc
        else:
            Q_GSHP = self.gshp.Qh
        residual_wts = self.wts.equn(mdot_loop,Q_GSHP,T_vec[-1],
                                     T_tank,TA,dt,T_tank_next)
        residuals_loop = self.loop.equn(T_vec, QHP, TA, mdot_loop)
        
        
        residuals = np.concatenate((residuals_gshp,
                                    np.array([residual_wts]),
                                    residuals_loop,
                                    np.array([slack_residual1,slack_residual2])))
        
        return residuals

    def solve(self,BC,X0,update=True):
        if update:
            self.Xm1 = X0
            
        X,infodict,ier,mesg = fsolve(self.equn, X0, args=(BC),xtol=1e-12,full_output=True)
        
        if ier != 1:
            raise Exception("fsolve solution failed. Internal message =\n\n" + mesg)
        
        if update:
            self.solutions.append(X)
            self.BCs.append(BC)
            self.BCm1 = BC
            self.residuals.append(self.equn(X,BC))
        return X
        
class Controls(object):
    
    def __init__(self,
                 CS,
                 gshp_delT_target,
                 loop_delT_target,
                 compressor_capacity,
                 loop_pump_mass_flow_capacity,
                 gshp_pump_mass_flow_capacity):
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
            
        Returns: BC : np.array(8)
        """
        
        if T_tank_target < X[2]:
            is_cooling = True
        else:
            is_cooling = False
        
        mdot_loop = self.loop_pump_control(T_tank_out = X[2],
                                           T_tank_in = X[2+self.loop.NT],
                                           delT_target = self.loop_delT)
        # calculate a moving average for the tank rate of change of temperature
        # based on a couple of hours.
        if len(self.CS.solutions) < ma_num_dt:
            ma_num_dt = len(self.CS.solutions) 
        X2arr = np.concatenate([np.array([X[2]]),np.array(self.CS.solutions[-ma_num_dt:])[:,2]])
        dT_tank_dt = np.mean(np.diff(X2arr)/self.CS.BCm1[6])
        
        W, mdot_gshp = self.gshp_control(Tin = X[0],
                                         mdot_m1 = self.CS.BCm1[7],
                                         Tout = X[1],
                                         T_tank_target = T_tank_target,
                                         T_tank = X[2],
                                         dT_tank_dt = dT_tank_dt,
                                         is_cooling=is_cooling,
                                         time_to_target=4*dt,
                                         delT_target=self.gshp_delT)
        
        # assure bounds on control variables (redundant)
        if mdot_loop < 0:
            raise ValueError("The loop mass flow cannot be negative!")
        elif mdot_loop > self.loop_pump_capacity:
            mdot_loop = self.loop_pump_capacity
                    
        if mdot_gshp < 0:
            raise ValueError("The gshp mass flow cannot be negative!")
        elif mdot_gshp > self.gshp_pump_capacity:
            mdot_gshp = self.gshp_pump_capacity
        
        # determine mode of GSHP operation.

            
        # set next time step's boundary conditions
        BC = np.zeros(8)
        BC[0] = is_cooling
        BC[1] = W
        BC[2] = Tg
        BC[3] = Ta
        BC[4] = QHP # heat gained/obsorbed by Heat pumps at the triplex level.
        BC[5] = mdot_loop
        BC[6] = dt
        BC[7] = mdot_gshp
        # gather these for post-processing.
        self.BCs.append(BC)
        return BC
        
    def loop_pump_control(self,T_tank_out,T_tank_in,delT_target):
        return np.abs(self.CS.loop.loop_heat) / (delT_target * self.wts.cp)
    
    def gshp_control(self,Tin,mdot_m1,Tout,T_tank_target,
                     T_tank,dT_tank_dt,is_cooling,time_to_target,delT_target):
        # This returns the amount of work to put into the ground source heat pump exchanger
        
        Tavg = (Tin + Tout)/2
        delT = Tin - Tout
        
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

        if Tin == Tout and Qgshp_target != 0:
            raise ValueError("The heat pump must shed heat but Qin = Qout is a current constraint!")
        else:
            mdot_target = np.abs(Qgshp_target/(self.wts.cp * delT_target))
        
        if is_cooling and Qgshp_target > 0: # no cooling is needed
            return 0,0
        elif is_cooling and Qgshp_target < 0:
            return np.min([-Qgshp_target/COP,self.comp_cap]), mdot_target
        else:
            if Qgshp_target > 0:
                return np.min([Qgshp_target/COP,self.comp_cap]), mdot_target
            else: # no heating is needed - let the extra heat dissipate before acting
                return 0,0

class unit_convert():
    cm_p_inch = 2.54
    meter_per_cm = 0.01
    C_to_K = 273.15

class gshp_inputs(object):
    
    def __init__(self,pipe_outer_diam=1.0 * 2.54 * 0.01, # 2 in in meters
                      pipe_inner_diam=0.9 * 2.54 * 0.01, # 1.8 in in meters 
                      pipe_therm_cond=0.44, # High density polyethylene (W/(m*K))
                      grou_therm_cond=0.4, # based off sand/gravel/clay/silt mixtures characteristic of Albuquerque
                      grou_speci_heat=800.0, # J/(kg*K)
                      ground_density=1900.0, # kg/m3
                      grou_temperature=293.15, # K
                      number_borehole=60,
                      borehole_length=80.0,
                      num_bore_bends=2,
                      dist_boreholes=5.0, # 5 meter spacing between boreholes
                      num_element=20,
                      max_COP_heating=5.0,
                      frac_carnot_efficiency=0.8,
                      delT_target=2.0,
                      compressor_capacity=10000, # W
                      mass_flow_capacity=10.0): # kg/s
        self.r0 = pipe_outer_diam
        self.ri = pipe_inner_diam
        self.kp = pipe_therm_cond
        self.kg = grou_therm_cond
        self.cpg = grou_speci_heat
        self.rhog = ground_density
        self.nb = number_borehole
        self.L_bore = borehole_length
        self.dist_boreholes = dist_boreholes
        self.num_bore_bends = num_bore_bends
        self.num_element = num_element
        self.max_COP_h = max_COP_heating
        self.eta_c = frac_carnot_efficiency
        self.delT_target = delT_target
        self.compressor_capacity = compressor_capacity
        self.mass_flow_capacity = mass_flow_capacity


class water_inputs(object):
    
    def __init__(self,specific_heat=4182.0, # J/(kg*K
                      density=997.0):
        self.cp = specific_heat # J/(kg*K)
        self.rho = density # kg/m3        
            
class tank_inputs(object):

    def __init__(self,
        tank_height = 2.0, # m
        tank_diameter = 3.0, # m
        insulation_thick = 0.02, #m
        insulation_k = 0.05, # W/(m*K)
        ):
        self.tank_height = tank_height
        self.tank_diameter = tank_diameter
        self.insulation_thick = insulation_thick
        self.insulation_k = insulation_k
        
class loop_inputs(object):
    
    def __init__(self,
                 distance_between_triplex_wshp=10.0,
                 pipe_outer_diam=2.0*0.0254, # 2in in meters
                 pipe_inner_diam=1.8*0.0254, # 1.8in in meters
                 trenching_depth=1.0, #meters
                 num_triplex_units=21, # number of water source heat pumps (wshp)
                 delT_target=2.0,
                 mass_flow_capacity=10.0): 
        self.L_triplex = distance_between_triplex_wshp # meters
        self.rpi = pipe_inner_diam
        self.rpo = pipe_outer_diam
        self.trenching_depth = trenching_depth
        self.number_triplex_units = num_triplex_units
        self.delT_target = delT_target
        self.mass_flow_capacity = mass_flow_capacity
        

    
        
class wts_gshp(object):
    
    def __init__(self,
                 ground_temperature,
                 inp_gshp,
                 inp_tank,
                 inp_water,
                 inp_loop,
                 qhp_loads_file,
                 weather_file,
                 data_path,
                 initial_tank_temperature=None,
                 initial_loop_mass_flow=10.0,
                 initial_gshp_mass_flow=10.0):
        """
        Inputs
            ground_temperature : float : > 0 : Average ground temperature from
                                10 - 100 meters deep.
            inp_gshp : gshp_inputs : populate the needed properties or 
                                     accept defaults
                                     
            
            
            
        """
        self.dt = 15 * 60  # TODO - make time step size an actual variable.
        self.num_ma_dt = 8 # TODO - move this to the inputs
        
        # setup paths needed
        base_path = os.path.dirname(__file__)
        data_path = os.path.join(base_path,r"..\Data")

        # gshp inputs
        r0 = inp_gshp.r0 
        ri = inp_gshp.ri
        kp = inp_gshp.kp
        kg = inp_gshp.kg
        cpg = inp_gshp.cpg
        rhog = inp_gshp.rhog
        L_bore = inp_gshp.L_bore * inp_gshp.nb
        number_of_bends_in_bore_hole = inp_gshp.num_bore_bends
        number_element = inp_gshp.num_element
        distance_to_next_borehole = inp_gshp.dist_boreholes
        max_GSHP_COP_h = inp_gshp.max_COP_h
        eta_c = inp_gshp.eta_c
        
        # tank inputs
        tank_height = inp_tank.tank_height
        tank_diameter = inp_tank.tank_diameter
        insulation_thick = inp_tank.insulation_thick
        insulation_k = inp_tank.insulation_k
        
        # water inputs
        cp_water = inp_water.cp
        rho_water = inp_water.rho
        
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
        if QHP0 > 0:
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
                            initial_gshp_mass_flow])
        

        # build objects for each system
        gshp = Transient_Simple_GSHP(r0,ri,kp,kg,rhog,cpg,number_of_bends_in_bore_hole,
                                     eta_c,L_bore,number_element,
                                     distance_to_next_borehole,max_GSHP_COP_h,cp_water,rho_water,Tg)
        ST = storage_tank(cp_water, tank_height, tank_diameter, insulation_thick, insulation_k, rho_water)
        loop = equal_loop(L_triplex,kg,cp_water,loop_rpi,loop_rpo,trenching_depth,kp,number_triplex_units,QHP0)
        
        # initialize combined systems object guess values for mass flow are
        # low and will be corrected once solutions start seeking out the target
        # values.
        self.cs = CombinedSystem(gshp,ST,loop,initial_tank_temperature,
                                   QHP0,initial_loop_mass_flow,initial_gshp_mass_flow,cp_water,Tg,TA0,self.BC0)
        
        # initialize the controls object.
        self.controls = Controls(self.cs,
                            inp_gshp.delT_target, 
                            inp_loop.delT_target, 
                            inp_gshp.compressor_capacity,
                            inp_loop.mass_flow_capacity,
                            inp_gshp.mass_flow_capacity)


    
    def no_load_shift_simulation(self,T_tank_target,print_percent_complete=False,percent_to_complete=101.0,troubleshoot_stop_time=None):
        # now establish the first boundary conditions make the Tank target temperature
        # the ground temperature (this may change.)
        perc_complete = 0
        BC = self.BC0
        X0 = self.cs.Xm1
        for idx in range(len(self.BC_histories)-1):
            if not troubleshoot_stop_time is None:
                if idx * self.dt  > troubleshoot_stop_time:
                    import pdb; pdb.set_trace()
            if (X0[:-2] < 0).sum() > 0: # the slack variables can be negative!
                import pdb; pdb.set_trace()    
                #raise ValueError("The solution for temperatures should never be less than zero!")
            X = self.cs.solve(BC,X0,update=True)
            BC = self.controls.master_control(X,
                                              self.BC_histories.iloc[idx+1,1],
                                              self.Tg,
                                              self.BC_histories.iloc[idx+1,0],
                                              T_tank_target,
                                              self.dt,
                                              self.num_ma_dt)
            # resolve with updated BC
            #X = self.cs.solve(BC,X0,update=True)
            X0 = X
            if print_percent_complete: 
                if perc_complete < (100*idx/(len(self.BC_histories)-1)):
                    print("{0:5.2f}% complete".format(100*idx/(len(self.BC_histories)-1)))
                    perc_complete = np.ceil(100*idx/(len(self.BC_histories)-1))
                    if 100*idx/(len(self.BC_histories)-1) > percent_to_complete:
                        break
            
            
            
            

        
        
            
    def _process_bc_files(self,data_path,weather_file,qhp_loads_file):
        # Boundary Conditions
        QHP_history = pd.read_csv(os.path.join(data_path,qhp_loads_file))
        weather = pd.read_csv(os.path.join(data_path,weather_file))
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

        
        