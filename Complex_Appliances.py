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

Created on Wed Nov 11 15:02:45 2020

@author: dlvilla
"""

import os
import numpy as np
import pandas as pd
from Thermodynamics import thermodynamic_properties as tp
tp = tp()

class Wall_AC_Unit(object):
    
    _water_condensing_temperature = 5  # Degrees Celcius
    _HOURS_TO_SECDONDS = 3600
    _C_to_K = 273.15
    
    _MAX_WARNING_MESSAGES = 3
    
    def __init__(self,TC,SCfrac,derate_frac,npPower,flow_rated,df_coef,df_thermostat,tier,Name):
        """
        Inputs 
        
        TC - total cooling capacity
        
        SCfrac - fraction of TC for sensible cooling
        
        derate_frac - derates TC (and SC) based on poor performance in comparison
                      to manufacturer claims and maintenance
        npPower - name plate power input to produce TC
        
        flow_rated - amount of flow circulation from AC m3/h
        
        df_coeff - data frame with performance curve coefficients
        
        df_thermostat - data frame with thermostat information
        
        """
        self.name = Name
        # see Meisner et. al., 2014 for this model and origin of coefficients form
        self._TC = TC * derate_frac
        self._SC = TC * SCfrac * derate_frac
        self._Power = npPower
        self._EER = self._TC / npPower
        self._flow = flow_rated
        self._coef = {}
        for key in df_coef["Curve Type"]:
            self._coef[key] = df_coef[df_coef["Curve Type"]==key].iloc[0,1:]
        self._cooling_setpoint = df_thermostat[df_thermostat["Type"]=="Cooling (⁰C)"][tier].values[0]
        #TODO heating is not yet used!
        self._heating_setpoint = df_thermostat[df_thermostat["Type"]=="Heating (⁰C)"][tier].values[0]
        self.unit_running = False
        self._DBT_valid_range = [17,45]
        self._WBT_valid_range = [17,32]
        self._num_msg = 0
        
    def _check_range(self,valid_range,val,name):
        if valid_range[0] > val or valid_range[1] < val:
            self._num_msg +=1
            if self._MAX_WARNING_MESSAGES > self._num_msg:
                print("Wall_AC_Unit._thermal_performance: " + name 
                      + " is higher than performance curves allow. The program is"
                      + " flat-lining the performance")
            elif self._MAX_WARNING_MESSAGES == self._num_msg:
                print("Wall_AC_Unit._thermal_performance: Discontinuing repeat messages")
            if valid_range[0] > val:
                val_adj = valid_range[0] 
            else:
                val_adj = valid_range[1]
        else:
            val_adj = val
            
        return val_adj
        
    def _thermal_performance(self,DBT_internal,rh_internal,DBT_external,
                          Pressure):
        # Wet-bulb temperture
        rh_internal_perc = 100 * rh_internal
        WBT_internal = tp.low_elevation_wetbulb_DBRH(DBT_internal, rh_internal_perc, Pressure)
        
        # verify internal wetbulb and external dry bulb are within the limits of
        # the performance curves. If not, then assume performance stays constant 
        # at the edges

        DBT_external_adj = self._check_range(self._DBT_valid_range,DBT_external,"External Dry Bulb Temperature")
        WBT_internal_adj = self._check_range(self._WBT_valid_range,WBT_internal,"Internal Wet Bulb Temperature")
        
        # determine the total cooling capacity of the unit
        TC = self._meissner_equation_6(self._TC,"TC",WBT_internal_adj,DBT_external_adj)
        SC = self._meissner_equation_6(self._SC,"SC",WBT_internal_adj,DBT_external_adj)
        EER = self._meissner_equation_6(self._EER,"EER",WBT_internal_adj,DBT_external_adj)
        
        power = TC / EER
        
        # calculate the mass flow of water vapor that has been extracted by the 
        # amount of latent cooling
        LC = TC - SC
        mdot_condensed = LC / tp.latent_heat_of_water(self._water_condensing_temperature)
        
        return TC, SC, mdot_condensed, power
    
    def avg_thermal_loads(self,DBT_internal,rh_internal,DBT_external,
                          Pressure,num_AC,volume,time_step,
                          energy_demand_unrestricted):
        # cooling is counted as negative heat.
        if energy_demand_unrestricted >= 0.0:
            self.unit_running = False
            self.fraction_time_on = 0.0
            SC_avg = 0.0
            power_avg = 0.0
            TC_avg = 0.0
            mdot_condensed_avg = 0.0
            unmet_cooling = energy_demand_unrestricted
        else:
            TC, SC, mdot_condensed, power = self._thermal_performance(DBT_internal,
                                                                     rh_internal,
                                                                     DBT_external,
                                                                     Pressure)
            unmet_cooling = num_AC * SC + energy_demand_unrestricted
            
            if unmet_cooling > 0: # The units will cycle off in the next hour
                unmet_cooling = 0.0 
                avg_on_time_per_unit_fraction_of_time_step = (
                    -energy_demand_unrestricted)/(num_AC * SC)
                SC_avg = SC * avg_on_time_per_unit_fraction_of_time_step
                power_avg = power * avg_on_time_per_unit_fraction_of_time_step
                TC_avg = TC * avg_on_time_per_unit_fraction_of_time_step
                mdot_condensed_avg = mdot_condensed * avg_on_time_per_unit_fraction_of_time_step
                self.fraction_time_on = avg_on_time_per_unit_fraction_of_time_step
            else:
                SC_avg = SC
                power_avg = power
                TC_avg = TC
                mdot_condensed_avg = mdot_condensed
                self.fraction_time_on = 1.0
                
        return (num_AC * TC_avg, 
               num_AC * SC_avg, 
               num_AC * mdot_condensed_avg, 
               num_AC * power_avg,
               unmet_cooling)
                
            
                
        
        
    def _meissner_equation_6(self,Nominal_value, curve_type, WBT_room,DBT_external):

        a = np.array(self._coef[curve_type])
        if type(a[0]) is str:
            a = a[1:]
        terms = np.array([1,WBT_room,WBT_room ** 2, DBT_external, DBT_external ** 2, WBT_room * DBT_external])
        Z = np.dot(terms,a)
        return Nominal_value * Z
    

class Refrigerator(object):
    
    _hours_to_seconds = 3600
    _cp_air = 1003.5 #J/(kg*K)
    _rho_air = 1.225 # kg/m3   both of these are assumed constant
    _C_to_F = 273.15 # K
    
    def __init__(self,surf_area,
                      aspect_ratio,
                      R_total,
                      T_diff,
                      T_inside,
                      ach_mix,
                      ach_schedule,
                      frac_Carnot,
                      fan_power,
                      compressor_efficiency,
                      compressor_power,
                      name=""):
        self.name = name
        # TODO - add input checking!
        self._Ri = R_total * surf_area
        self._Vol = self._rectangle_volume(aspect_ratio,surf_area)
        self._Tc = T_inside - T_diff
        self._Tdiff = T_diff
        self._Tf = T_inside
        self._ach = ach_mix
        self._ach_sch = ach_schedule
        self._Pf = fan_power
        self._fc = frac_Carnot
        self._eta_c = compressor_efficiency
        self._Pc_max = compressor_power
        self._Af = surf_area
        self.fridge_setpoint_reached = True
    
    def _rectangle_volume(self,AR,S):
        return S / (2 + 4 * AR) * np.sqrt(S/(2/(AR*AR) + 4/AR))
        
    def avg_thermal_loads(self,T_room,hour_of_day,ts):
        """
        Inputs
          ts - time step in hours
        """
        #TODO - create a more dynamic model of the refrigerator. Need heat capacity 
        #       and possibly also a model with a freezer. For now, let's keep this inexpensive.
        #       performance curves could really help w/r to quantifying inefficient modes
        #       of operation.
        mdot = self._Vol * self._ach * self._ach_sch[hour_of_day] * self._rho_air / self._hours_to_seconds
        mdot_cp = mdot * self._cp_air * (T_room - self._Tf + self._C_to_F)
        #max_heat = self._Vol * self._rho_air * self._cp_air * (T_room + self._C_to_F - self._Tf)
        Qc = mdot_cp + 1/self._Ri *(T_room - self._Tf + self._C_to_F)
        Tcondenser = T_room + self._Tdiff
        if Tcondenser + self._C_to_F - self._Tc == 0: # avoid division by zero
            COP = 0.0
        else:
            COP = self._fc * self._Tc / (Tcondenser + self._C_to_F - self._Tc)
            
        if COP == 0:
            Qh = 0
            Pc = 0
        else:
            Qh = Qc * (1 + COP)/COP
            Pc = Qc / (COP * self._eta_c)
        if Pc > self._Pc_max:
            # roughly assume COP stays constant
            Pc = self._Pc_max
            Qc = COP * Pc * self._eta_c
            Qh = Qc * (1 + COP)/COP
            self.fridge_setpoint_reached = False  # a sign food will spoil - we do not track
                                                  # the temperature. 
        else:
            self.fridge_setpoint_reached = True
        
        Qnet = Qh - Qc  # net heat exchange with the room
        Pnet = Pc + self._Pf
    
        return Qnet, Pnet
    
class Fan(object):
    
    def __init__(self, fan_name, power, heat_energy_ratio, ExtACH, SpeedEfficiencies, Curve_Temperatures):
        self.name = fan_name
        self._power = power
        self._heat_ratio = heat_energy_ratio
        self._ExtACH = ExtACH
        self._SpeedEfficiencies = SpeedEfficiencies
        self._Curve_Temperatures = Curve_Temperatures
        
    
    def fan_energy_model(self,T_in,T_out):
        
        """ Fan use behavior given external and internal temperature
        
            conditions
        """
        
        # find what speed 
        speed_index = self._find_speed(T_in)
        if speed_index == -1:
            fan_power = 0.0
            fan_ACH = 0.0
            fan_heat = 0.0
            self.ison = False
        else:
            #TODO improve this model
            speed = self._Curve_Temperatures.index[speed_index]
            fan_power = self._power * speed /  self._SpeedEfficiencies.loc[speed]
            if T_out > T_in:
                fan_ACH = 0.0
            else:
                fan_ACH = self._ExtACH * speed
            fan_heat = fan_power * self._heat_ratio
            self.ison = True

        return fan_power, fan_ACH, fan_heat
    
    def _find_speed(self, T_in):
        ind = np.where(self._Curve_Temperatures.values < T_in)
        if len(ind[0]) == 0:
            return -1
        else:
            return ind[0][-1]
    
class Light(object):
    # simple object
    def __init__(self,name,light_type,lux_threshold,power,fraction_heat):
        self.name = name
        self.light_type = light_type
        self.lux_threshold = lux_threshold
        self.power = power
        self.fraction_heat = fraction_heat
    # energy model is elsewhere. Lights cannot be controlled differently
    # for now stick strictly to interior and exterior lights
    
class HVAC_Input():
    """
    
    obj(Qdot, Qdot_coef, COP, EIR_coef, Vdot, PLF_coef, SHR=None)
    
    The rated load conditions should be per AHRI 210/240 Table 1 of 
    NREL report TP-5500-56354, 2013 D. Cutler et. al.
    
    Qdot - 
           
    Qdot_coef - dict with entries "Temperature", and "FF" for flow fraction.
           rated cooling (negative) or heating (positive) capacity
    
    COP - rated coefficient of performance
    
    Vdot - rated volumetric flow
    
    SHR - rated sensible heating ratio (cooling only)
    
    """
    
    def __init__(self,Qdot, Qdot_coef, COP, EIR_coef, Vdot, PLF_coef, SHR=None):
        def _test1(coef,name):
            if "Temperature" in coef and "FF" in coef:
                return coef
            else:
                raise ValueError("The "+name+" input must be a list-like entry of length 6 with float coefficient values!")

        self.Qdot = Qdot
        self.COP = COP
        self.Vdot = Vdot
        self.EIR_coef = _test1(EIR_coef,"EIR_coef")
        self.Qdot_coef = _test1(Qdot_coef,"Qdot_coef")
        self.SHR = SHR
        self.PLF_coef = PLF_coef
                
        
class NREL_AC_2013(object):
    
    """
    See Cutler, D. et. al., 2013 "Improved Modeling of Residential AC and HP for
               Energy Calculations" NREL/TP-5500-56354
               
    Inputs obj(cooling_input,heating_input)
    
    
    """
    
    _C_to_K = 273.15
    _Ra = 286.9 # J/(kg*K) Air specific gas constant
    _Rw = 461.5 # J/(kg*K) Water vapor specific gas constant
    _Tewb = 19.4 + _C_to_K # per Table 1 of NREL report
    _Tedb = 26.7 + _C_to_K # per Table 1 of NREL report
    _Rratio = _Rw / _Ra
    # using the values above
    _rh_rated = 0.509
    _pressure_rated = 101325
    _x_rated = tp.humidity_ratio(_Tedb - _C_to_K, _pressure_rated, _rh_rated)
    _rho_rated = (_pressure_rated / (_Ra * _Tedb)) * (1 + _x_rated) / ( 1 + _x_rated * _Rratio)
    
    def __init__(self,cooling_input,heating_input):
        if cooling_input.SHR is None:
            raise ValueError("The cooling_input object must have an assigned Sensible Heating Ratio!")
        self._inp_c = cooling_input
        self._inp_h = heating_input
        self._mdot_rated = self._rho_rated * cooling_input.Vdot
        self._EIR_h_rated = 1/heating_input.COP
        self._EIR_c_rated = 1/cooling_input.COP
        self._SHR_rated = cooling_input.SHR
        
    
    def calculate(self,Qload,Todb,T_inside,rh_inside,pressure,Vdot):
        """
        # THIS METHOD IS OVERSIMPLIFIED AT THE PRESENT
        # THE SENSIBLE HEAT RATIO SHOULD BE A FUNCTION OF PERFORMANCE
        # BUT WE ARE TREATING IT AS A CONSTANT FOR NOW
        
        obj.calculate(Qload,dt,Todb,T_inside,rh_inside,pressure,Vdot)
        
        Inputs:
            Qload - float - Watts - sensible load in W being exerted on average 
                            over a time period "dt"
                            + means a cooling load - means a heating load
        
            Todb  - float > 275 - Kelvin - Outdoor dry bulb heat rejection/source 
                                           temperature
            T_inside - float > 275 - Kelvin - Indoor dry bulb temperature
            
            rh_inside - float < 1.0, > 0.0 - ratio -  indoor relative humidity
            
            pressure - float > 0 - Pa - atmospheric pressure
            
            Vdot - float > 0 - m3/s - volumetric flow rate being applied by fan
            
            is_cooling : bool : indicates if the heat pump is pulling heat from
                                the load space (True) or adding heat to the load
                                space (False)
        
        Returns
            Power input,
            
        
        """
        def _poly2(FF,cT,cFF,Todb,Tewb,rated_value):
            _T = (cT[0] + cT[1] * Tewb + cT[2] * Tewb * Tewb + cT[3] * Todb +
                 cT[4] * Todb * Todb + cT[5] * Tewb * Todb)
            _FF = (cFF[0] + cFF[1] * FF + cFF[2] * FF * FF)
        
            return rated_value * _T * _FF
        
        if Qload > 0:
            is_cooling = True
        else:
            is_cooling = False
        
        
        TempC = T_inside - self._C_to_K
        Todb_C = Todb - self._C_to_K 
        
        # DX coil entering wetbulb temperature. Returns Degrees C
        Tewb = tp.low_elevation_wetbulb_DBRH(TempC, rh_inside * 100.0, pressure)
        
        # density of moist air
        x = tp.humidity_ratio(TempC, pressure, rh_inside)
        rho = (pressure / (self._Ra * T_inside)) * (1 + x) / ( 1 + x * self._Rratio)
               
        # mass flow rate
        mdot = rho * Vdot
        FF = mdot / self._mdot_rated
        
        if is_cooling:
            cT_EIR = self._inp_c.EIR_coef['Temperature']
            cFF_EIR = self._inp_c.EIR_coef['FF']
            cT_Qdot = self._inp_c.Qdot_coef['Temperature']
            cFF_Qdot = self._inp_c.Qdot_coef['FF']
            EIR_rated = self._EIR_c_rated
            Qdot_rated = self._inp_c.Qdot
            cPLF = self._inp_c.PLF_coef
        else:
            cT_EIR = self._inp_h.EIR_coef['Temperature']
            cFF_EIR = self._inp_h.EIR_coef['FF']
            cT_Qdot = self._inp_h.Qdot_coef['Temperature']
            cFF_Qdot = self._inp_h.Qdot_coef['FF']
            EIR_rated = self._EIR_h_rated
            Qdot_rated = self._inp_h.Qdot
            cPLF = self._inp_h.PLF_coef
            
        EIR = _poly2(FF,cT_EIR,cFF_EIR,Todb_C,Tewb, EIR_rated)
        Qdot = _poly2(FF,cT_Qdot,cFF_Qdot,Todb_C,Tewb, Qdot_rated)
        
        #TODO - replicate the SHR apparatus dew point (ADP)/bypass factor (BF) model
        #       from Energy Plus!!!!
        if is_cooling:
            # Assure that the air is not completely stripped of moisture
            Lh2o = tp.latent_heat_of_water(TempC)
            mdot_condensed = Qdot * (1-self._SHR_rated) / Lh2o
            mdot_vapor = x * mdot
            
            if mdot_condensed > mdot_vapor:
                SHR = 1.0 - mdot_vapor * Lh2o / Qdot
                if SHR < 0:
                    raise ValueError("The sensible heat ratio must be greater than 1")
            else:
                SHR = self._SHR_rated
            
            Qsensible_load_capacity = Qdot * SHR 
            
        else:
            Qsensible_load_capacity = Qdot 
        

        
        # Part load ratio (PLR)
        PLR = np.abs(Qload / Qsensible_load_capacity)
        
        # PLF accounts for losses due to inefficiency of starting and stopping
        # the compressor. Short cycling leads to inefficiency.
        PLF = cPLF[0] + cPLF[1] * PLR + cPLF[2]*PLR*PLR
        RTF = PLR/PLF
        
        Power = EIR * Qdot * RTF
        

        
        COP = PLF/EIR # includes losses
        Qavg = Qdot * RTF
        
        if is_cooling:
            # Qavg = cold reservoir heat transfer
            # Qcondenser = factor * (Qdot * RTF + Power)
            Qcondenser = (Qavg + Qavg * COP)/COP
        else:
            # Qavg = hot reservoir heat transfer
            #Qcondenser = factor * (Qdot * RTF - Power)
            Qcondenser = -(Qavg * COP - Qavg)/COP
        
        if Power < 0 or COP < 0:
            print("Bad Value")
        
        
        return Power, COP, RTF, Qcondenser, Qavg

class HotWaterHeater(object):
    """The same model is shared for all tank hot water heaters with a single 
       resistance element that should include water film, wall, and air film
       resistances and thermal mass of the water with two control volumes.
       
       The first control volume is kept at temperature T1 which is maintained
       in the interval T_target - 0.5*T_band < T1 < T_target + 0.5*T_band
    """
    Twater_boil = 373.15
    Twater_freeze = 273.15
    delTdiff_max = 1.0 #K
    
    def __init__(self,capacity_m3=0.151416,
                 Tinit_K=293,
                 Rfactor_SI=2.817,
                 tank_diameter_m=0.75,
                 stack_efficiency=0.75,
                 is_external=True,
                 stack_external=True,
                 mixing_volume_fraction=0.2,
                 target_temp_K=340.0,
                 bandwidth_temp_K=5,
                 Q_heat_max=4000,
                 mixes_per_time_step=2.0):
        """
        Initialize a Hot Water Heater
        
        HotWaterHeater(capacity_m3,Tinit_K,Rfactor_SI,tank_diameter_m,
                       stack_efficiency,is_external,stack_external,
                       mixing_volume_fraction,target_temp_K,bandwidth_temp,
                       Q_heat_max)
        
        Parameters
        ----------
        capacity_m3 : float > 0 : optional : default = 0.454249 m3 (40 gal)
            The size of the hot water heater (both control volumes) in 
            cubic meters
        Tinit_K : self.Twater_freeze < float < self.Twater_boil : optional : 
            default = 293K (20C)
            The initial temperature of water in the hot water heater tank
        Rfactor_SI : float > 0 : optional : default = 2.817 (R16)
            Resistivity in K/(W*m2) for the hot water heater. This must 
            include film resistance for air (outside) and water surfaces on a 
            cylindrical surface. Divide Imperial 
        tank_diameter_m : float > 0 : optional : default 0.75 m
            Tank diameter in meters. 
        stack_efficiency : 1.0 > float > 0.0 : optional : default = 0.75
            The fraction of heat that reaches the water tank from the heat 
            applied to it. 
        is_external : bool : optional : default = True
            True = Water heater is outside and will not shed heat to an 
                   an internal thermal zone
            False = Water heater is inside and will shed heat to an internal
                   thermal zone.
        stack_external : bool : optional : default = True
            True = lost stack heat will not warm thermal zone
            False = lost stack heat will warm thermal zone
        mixing_volume_fraction : float : optional : default = 0.8
            fraction of the volume that is kept exactly at the target heating
            temperature
        target_temp_K : float : optional : default = 340K (160F)
            Target temperature for the hot water heater setpoint
        bandwidth_temp_K : float : optional : default = 5K
            bandwidth about target_temp_K for which heating will not be applied
            to the hot water heater.
        Q_heat_max : float > 0 : optional : default = 4000
            Maximum heat input capability of the hot water heater.
        mixes_per_time_step : float > 0 : optional : default = 2
            This parameter strongly effects how large of a time step the simulation
            will take and how accurate sudden transients are it takes the CV 2 
            mass and determines how many mass changes are allowed per time step 
            by mdot_mix for a legal time step. A lot of mass changes can cause
            the solution to become unstable and the Euler step to incorrectly
            calculate the solution. A large value also allows oscillation of the
            solution during an abrupt change in Q_heating. < 1.0 values will
            make the simulation for a year very long while the default takes
            200.0 seconds for a year long simulation on my computer and allows
            time steps of about 100 seconds

        Returns
        -------
        HotWaterHeater object
        
        """
        self.volume1 = (1-mixing_volume_fraction) *capacity_m3
        self.volume2 = mixing_volume_fraction * capacity_m3
        tank_height = capacity_m3 * 4.0 / (np.pi * tank_diameter_m ** 2)
        self.cross_sectional_area = 0.25 * np.pi * tank_diameter_m ** 2
        area1 = self.cross_sectional_area + np.pi * tank_diameter_m * tank_height * (1-mixing_volume_fraction)
        self.area2_no_heating = self.cross_sectional_area + np.pi * tank_diameter_m * tank_height * mixing_volume_fraction
        self.area2_heating = np.pi * tank_diameter_m * tank_height * mixing_volume_fraction 
        self.tank_diam = tank_diameter_m
        self.R1 = Rfactor_SI * area1
        self.R2_prod_area = Rfactor_SI
        self.eta_stack = stack_efficiency
        self.cp = con.water_constant_pressure_specific_heat
        self.T1 = Tinit_K
        self.T2 = Tinit_K
        self.is_external = is_external
        self.stack_external = stack_external
        self.T_targ = target_temp_K
        self.T_band = bandwidth_temp_K
        self.Qmax = Q_heat_max
        self.mix_CD = 0.4
        self._iter_time = 0.0
        self._complete_time_step = 0.0
        self.is_heating = True
        self.current_warning = ""
        self.T1_hist = [Tinit_K]
        self.T2_hist = [Tinit_K]
        self.t_hist = [0]
        self.Qheat_hist = []
        self.max_mass_changes_per_time_step = mixes_per_time_step

    # alterable target temperature to enable vacation or weekend changes in 
    # control performance.
    @property
    def T_target(self):
        """
        Get the thermostat target temperature (K)
        
        """
        return self.T_targ    
        
    @T_target.setter  
    def T_target(self,TempK):
        """
        Set the thermostat target temperature (K)
        
        obj.T_target(TempK)
        
        Inputs
        -------
        TempK : float > 0
            Target temperature to which hot water heater will move toward
            must be greater than freezing and less than boiling point of water
        
        """
        if TempK < self.Twater_freeze or TempK > self.Twater_boil:
            raise ValueError("The target temperature must be greater than"  
                             + " the freezing point of water and less than" 
                             + " the boiling point of water")
        else:    
            self.T_targ = TempK
            
    def time_step(self,dt,Tamb,Tin,mdot,Qheat_potential):
        """
        Explicitly move the 2 control volume hot water heater model
        forward one time step using Euler integration.
        
        obj.time_step(dt,Tamb,Tin,mdot,Qheat_potential)

        Parameters
        ----------
        dt : float > 0
            Time step in seconds
        Tamb : float > 0 
            Ambient temperature surrounding the hot water heater.
        Tin : float > 0
            Temperature of make up water 
        mdot : float > 0
        Qheat_potential : float > 0
            Heat input available to heat hot water. Assumes a single on/off
            type heating.

        Returns
        -------
        Q_added : float
            Heat added to the internal zone of the R5C1 single control volume
            dwelling
        Q_loss : float
            Heat losses from the hot water tank for the current time step.

        """
        # determine inital heating state
        
        self._iter_time = 0.0
        self._complete_time_step = dt
        if self.is_heating:
            if self.T1 < self.T_targ:
                Qheat = Qheat_potential
                self.is_heating = True
            else:
                Qheat = 0.0
                self.is_heating = False
        else:
            if not self._in_temperature_bandwidth(self.T1):
                Qheat = Qheat_potential
                self.is_heating = True
            else:
                Qheat = 0.0
                self.is_heating = False
        # take as many substeps as needed.
        Jadd = 0.0
        Jloss = 0.0
        remaining_time = dt
        
        
        dt_total = dt
        while remaining_time > 0:
            # this is complex and can change the heating state 
            Jadd, Jloss, dt_new, Qheat_new = self._sub_time_step(dt,Tamb,Tin,Qheat,mdot,Qheat_potential,Jadd,Jloss)
            remaining_time = self._complete_time_step - self._iter_time 
            dt = dt_new
            Qheat = Qheat_new
            
        # record the heating state for the next time state so that it can properly pick up
        if Qheat != 0:
            self.is_heating = True
        else:
            self.is_heating = False
        
        return Jadd/dt_total, Jloss/dt_total
        
        
        
    def _sub_time_step(self,dt,Tamb,Tin,Qheat,mdot,Qheat_potential,Jadd=0.0,Jloss=0.0):
        """
        Explicitly move the 2 control volume hot water heater model
        forward one time step using Euler integration.
        
        obj._sub_time_step(dt,Tamb,Tin,Qheat,mdot,Qheat_potential,Jadd,Jloss)

        Parameters
        ----------
        dt : float > 0
            Time step in seconds
        Tamb : float > 0 
            Ambient temperature surrounding the hot water heater.
        Tin : float > 0
            Temperature of make up water 
        Qheat : float > 0
            Heat input to maintain hot water heater temperature
        mdot : float > 0
        Jadd : float > 0
            Cumulative energy added to thermal zone during sub calls to this function
        Jloss : float > 0
            Cumulative energy lost from the water heater during sub calls to this function

        Returns
        -------
        Jadd : float
            Energy added to the internal zone of the R5C1 single control volume
            dwelling (not heat!)
        Jloss : float
            Total Energy losses of the hot water heater 

        This function calls itself subly in the case where
        multiple heating and cooling cycles can occur in a single time step.
        it can therefore handle as many cooling and heating cycles as needed
        for very large time steps. It assumes that no change in the boundary
        conditions occur though during the time step.

        """
        if (np.array([dt,Tamb,Tin]) <= 0).sum() > 0:
            raise ValueError("The time step and temperature inputs must be greater than 0!")
        elif (np.array([Qheat,Qheat_potential]) < 0).sum() > 0:
            raise ValueError("The heating inputs must be zero or positive")
        
        Qheat_new = Qheat
        T1 = self.T1
        T2 = self.T2
        T_targ = self.T_targ
        T_band = self.T_band
        T_lower = T_targ - T_band 
        # assume mass change with temperature is negligible to keep the system linear

        rho1 = tp.water_density_1atm(T1)
        rho2 = tp.water_density_1atm(T2)

        V2 = self.volume2
        m1 = self.volume1 * rho1
        m2 = V2 * rho2
        R1 = self.R1
        if Qheat == 0:
            R2 = self.R2_prod_area * self.area2_no_heating
        else:
            R2 = self.R2_prod_area * self.area2_heating
        
        # We derive this by assuming equilibrium between bouyancy difference
        # and drag force to derive a velocity of movement.
        mdot_mix = self._mixing_rate(T1,T2,V2)
        
        T1np1, T2np1 = self._equn(T1,T2, Tamb, R1, R2, mdot, mdot_mix, Tin, Qheat, m1, m2, dt)
        
        try:
            mdot_mix_post = self._mixing_rate(T1np1,T2np1,V2)
        except ValueError:
            mdot_mix_post = 1e6 # certain failure - good_solution = False

        dt_new,good_solution, dT2dt, T1e, T2e = self._test_2nd_law(Tamb, R1, R2, mdot, 
                                                         mdot_mix, mdot_mix_post, 
                                                         Tin, Qheat, T2np1, 
                                                         T1np1, T2, T1, dt, m1, m2)
            
            
        if good_solution:
            skipped = False
            # check for incapacity to heat
            
            if Qheat > 0 and dT2dt < 0:
                self.current_warning = ("Even though heating is happenning,"
                                        +" the water tank losses and make "
                                        +"up water are cooling the internal"
                                        +" temperature T2! This may indicate"
                                        +" a water heater with insufficient "
                                        +" capacity has been selected.")
            
            # determine if control state change will occur in the time step
            # cut linear growth to time of control change if control change 
            # would occur
            if Qheat > 0 and T_targ > T1 and T_targ < T1np1: # The time step has overshot the required heating and the 
                                                             # thermostat would have turned off earlier. Reduce the step size                                             # and enter a sub loop
                frac = (T_targ - T1)/(T1np1 - T1)
                dt_new = dt * frac
                self.T1 = T_targ
                self.T2 = T2 + frac * (T2np1 - T2)
                Qheat_new = 0.0
                
            elif Qheat == 0 and T_lower < T1 and T_lower > T1np1: # cooling but may need to start heating before end of time step

                frac = (T1 - T_lower) / (T1 - T1np1)
                dt_new = dt * frac 
                Qheat_new = Qheat_potential
                self.T1 = T_lower
                self.T2 = T2 + frac * (T2np1 - T2)
            else:
                # no change, take the entire time step.
                dt_new = dt
                self.T1 = T1np1
                self.T2 = T2np1
            
            # account for heat balances over the linear change in temperatures
            Qloss1 = (Tamb - (T1 + self.T1)/2)/R1
            Qloss2 = (Tamb - (T2 + self.T2)/2)/R2
            Qstack = -(1 - self.eta_stack) * Qheat
            
            # quantify heat losses and additions to a thermal zone for the current cycle
            Jloss += dt_new * (Qloss1 + Qloss2 + Qstack) 
            
            Jadd_temp = 0
            if not self.is_external:
                Jadd_temp += -(Qloss1 + Qloss2)*dt
            if not self.stack_external:
                Jadd_temp += -Qstack*dt
            Jadd += Jadd_temp    
                
            # update time
            self._iter_time += dt_new
            
            # save results
            self.t_hist.append(self.t_hist[-1]+dt_new)
            self.T1_hist.append(self.T1)
            self.T2_hist.append(self.T2)
            self.Qheat_hist.append(Qheat)

            dt_new = 1.5 * dt
                
            # assure time step does not go past the complete time step requested.
            if dt_new + self._iter_time > self._complete_time_step:
                dt_new = self._complete_time_step - self._iter_time
        else:
            # The current solution has been rejected and we must try again
            # with a new time step.
            skipped = True
            
        return Jadd, Jloss, dt_new, Qheat_new
    
    def _test_2nd_law(self,Tamb,R1,R2,mdot,mdot_mix,mdot_mix_post,Tin,Qheat,T2np1,T1np1,T2,T1,dt,m1,m2):
        
        # verify 2nd law is not violated by linear step. A rising temperature cannot
        # exceed the equilibrium temperature
        # a dropping temperature cannot be below the equilibrium temperature
        dT2dt = (T2np1 - T2)/dt
        dT1dt = (T1np1 - T1)/dt
        temp_change_same_direction = (dT2dt > 0 and dT1dt > 0) or (dT2dt < 0 and dT1dt < 0)
        
        T1e, T2e = self._equil_sol(Tamb,R1,R2,mdot,mdot_mix,Tin, Qheat)
        
        # determine whether the simple Euler time step scheme is stable based
        # on 2nd law of thermodynamics - You cannot cool or heat past the 
        # equillibrium solution!
        good_solution =True
        dt_new = dt
        if temp_change_same_direction and (
                (dT2dt > 0 and T2np1 > T2e) or 
                (dT2dt < 0 and T2np1 < T2e)):
            dt_new = 0.5 * dt
            good_solution = False
        
        if good_solution and temp_change_same_direction and ( 
           (dT1dt > 0 and T1np1 > T1e) or 
            (dT1dt < 0 and T1np1 < T1e)):
            dt_new = 0.5 * dt
            good_solution = False
            
        delTdiff = np.abs((T1np1 - T2np1) - (T1 - T2))
        
        # This doesn't work when the solution is flipping over becuase of 
        # a change in heating.
        # if good_solution:
        #     # The time solution cannot flip the orientation of the temperatures.
        #     if (((T1e > T2e) and (T1np1 < T2np1)) or
        #         ((T2e > T1e) and (T2np1 < T1np1))):
        #         good_solution = False
        #         dt_new = 0.5 * dt
                
        
        # assure the rate of mixing change is not extremely abrupt
        if good_solution:
            if delTdiff > self.delTdiff_max:
                good_solution = False
                dt_new = 0.5 * dt
            
        if good_solution:
            mass_changes_per_time_step = mdot_mix_post * dt / m2
            if mass_changes_per_time_step > self.max_mass_changes_per_time_step:
                dt_new = 0.5 * dt
                good_solution = False
            
        return dt_new,good_solution,dT2dt,T1e,T2e
    
    def _mixing_rate(self,T1,T2,V2):
        # We derive this by assuming equilibrium between bouyancy difference
        # and drag force to derive a velocity of movement.
        rho1 = tp.water_density_1atm(T1)
        rho2 = tp.water_density_1atm(T2)
        delrho = np.abs(rho1 - rho2)
        
        
        if delrho == 0.0:
            v_mix = 0.0
        else:
            Re_percent_diff = 1.0
            Re1 = 0.0
            while Re_percent_diff > 0.01: 
                v_mix = np.sqrt(2 * con.earth_gravitational_acceleration * delrho * V2 / 
                                   (rho1 * 0.5*self.cross_sectional_area * self.mix_CD ))
                visc = tp.water_viscosity_1atm(T2)
                Re = v_mix * rho1 * self.tank_diam /visc
                if Re != 0:
                    self.mix_CD = 64 / Re
                    if self.mix_CD < 0.4:
                        self.mix_CD = 0.4
            
                Re_percent_diff = 100 * np.abs((Re1 - Re)/Re)
                Re1 = Re
        
        mdot_mix = 0.25 * v_mix * self.cross_sectional_area * (rho2 + rho1)/2
        
        return mdot_mix
    
    
    def _in_temperature_bandwidth(self,T1):
        # only need one condition because the water heater has no mechanism
        # to cool off
        T_targ = self.T_targ
        T_band = self.T_band
        return (T1 >= T_targ - T_band)
    
    def _equil_sol(self,Tamb,R1,R2,mdot,mdot_mix,Tin,Qheat):
        """
        This is the equilibrium solution which gives the minimum possible 
        temperatures that can be attained without violation of the 2nd law of
        thermodynamics. If too large of a time step is taken, this solution
        provides enforcement of the 2nd law of thermodynamics. The equilibrium
        solution is given if temperatures rise above (when increasing) or
        fall below (when decreasing) this equilibrium solution
        
        Parameters
        ----------
        Tamb : float > 0 
            Ambient temperature (kelvin)
        R1 : float > 0
            Restivity from control volume 1 to ambient conditions
        R2 : float > 0
            Resistivity from control volume 2 to ambient conditions
        mdot : float > 0
            Water demand (m3/s) extracted at T1 and replenished at temperature Tin
        mdot_mix : float > 0
            Mixing rate (m3/s) between the control volumes
        Tin : float > 0
            Temperature of replenishing water entering at mdot rate.
        
        """
        cp = self.cp
        eta = self.eta_stack
        c1 = cp * mdot

        if mdot_mix == 0.0:
            d2 = c1 + 1/R2
            if d2 <= 0:
                raise ZeroDivisionError("The T2 denominator of the no mixing equilibrium solution is 0!")
            T2 = (c1 * Tin + Tamb/R2 + eta * Qheat)/d2
            d1 = c1 + 1/R1
            if d1 <= 0:
                raise ZeroDivisionError("The T1 denominator of the no mixing equilibrium solution is 0")
            T1 = (c1 * T2 + Tamb/R1)/(c1 + 1/R1)
        else:
            c2 = mdot_mix * cp
            c3 = c1 + c2
            alpha1 = (1/R2 + c3)/c2
            alpha2 = (-Tamb/R2 - c1 * Tin - eta * Qheat)/c2
            
            d1 = c3 * (1 - alpha1) - alpha1/R1
            if d1 == 0.0:
                raise ZeroDivisionError("The d1 term must not be zero!")
            T2 = (c3 * alpha2 + alpha2/R1 - Tamb/R1)/d1
            T1 = alpha1 * T2 + alpha2
            
        return T1,T2
        
        
    
    def _equn(self,T1n,T2n,Tamb,R1,R2,mdot,mdot_mix,Tin,Qheat,m1,m2,dt):
        """
        This is a formulation where the losses and mixture exchanges are equillibrated at the
        average temperature between the new value and the previous value. The implicit equations
        were solved by hand to produce the T1_n+1, T2_n+1 solution so that it could be integrated
        explicitly
        
        Parameters
        ----------
        T1n : float > 0
            Current temperature of control volume 1 (CV that does not recieve Qheat
            and from which water is extracted) 
        T2n : float > 0
            Current temperature of control volume 2 (CV that does receive Qheat)
        Tamb : float > 0 
            Ambient temperature (kelvin)
        R1 : float > 0
            Restivity from control volume 1 to ambient conditions
        R2 : float > 0
            Resistivity from control volume 2 to ambient conditions
        mdot : float > 0
            Water demand (m3/s) extracted at T1 and replenished at temperature Tin
        mdot_mix : float > 0
            Mixing rate (m3/s) between the control volumes
        Tin : float > 0
            Temperature of replenishing water entering at mdot rate.
        Qheat : float > 0
            Heat input (W) into the hot water heater only eta * Qheat reaches
            the water.
        m1 : float > 0
            mass (kg) in control volume 1
        m2 : float > 0
            mass (kg) in control volume 2
        dt : float > 0
            time step in seconds
        
        Returns
        -------
        T1np1 - next time step's value for T1 (Kelvin)
        T2np1 - next time step's value for T2 (Kelvin)
        
        """
        eta = self.eta_stack
        cp = self.cp
        # repeat combinations of variables
        c1 = cp * (mdot + mdot_mix)
        c2 = 2 * cp * m1 / dt
        c3 = cp * m2 / dt
        
        # first denominator that must not be 0
        d1 = c2 + c1 + 1/R1
        if d1 == 0:
            raise ZeroDivisionError("denominator 1 of equn expressions must not be zero!")
        a1 = c1 / d1
        a2 = (c1 * (T2n - T1n) + (2 * Tamb - T1n)/R1 + c2 * T1n) / d1
        
        # second denominator that must not be 0
        d2 = c3 + (c1 - cp * mdot_mix * a1)/2 + 0.5/R2
        if d2 == 0:
            raise ZeroDivisionError("denominator 2 of equn expressions must not be zero!")
        # Be careful! T1n and Tin are very different!
        T2np1 = (c3 * T2n 
                 + cp * (mdot * (Tin - T2n/2) - mdot_mix * (T2n - a2 - T1n)/2)
                 + (Tamb - 0.5*T2n)/R2 + eta * Qheat)/d2
        
        T1np1 = a1 * T2np1 + a2
        
        return T1np1, T2np1


class HP_water_heater(object):

    def __init__(self,hot_water_consumption_profile_m3_per_s,
                      hot_water_temperature_K,
                      hp_rated_heating):
        self.consumption = hot_water_consumption_profile_m3_per_s
        
        
    def calculate(self,Tin,Tout):
        pass
        
        
        
        
        