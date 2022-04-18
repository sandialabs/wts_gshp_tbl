# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 13:01:52 2021

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


@author: dlvilla
"""
import numpy as np
import igshpa as igs
from pandas import isna
import pandas as pd
import os
import matplotlib.pyplot as plt
from pickle import dump,load
from matplotlib import rc
from matplotlib.dates import DateFormatter
import calendar
import time
import sklearn

def hot_water_use_fractions_plot(inpd,figure_path):
    # plot hot water use fractions
    fig,ax = plt.subplots(1,1,figsize=(10,5))
    ax.plot(range(1,25),inpd["hot_water_use_fraction"])
    ax.grid("on")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Hourly water use fraction")
    ax.set_xticks(range(1,25))
    plt.savefig(os.path.join(figure_path,"water_use_fractions.png"),dpi=300)

def form_tank_target_history(ground_temperature,inp_triplex):

    # spring and fall at ground temperature
    T_tank_target = np.ones(8760*4)*ground_temperature
    #          step_number =   hours/day * days/month * months/season * time steps/hour
    time_steps_per_season = 24*30*3*4
    end_of_spring = 3000*4
    # winter
    T_tank_target[0:time_steps_per_season] = inp_triplex.hot_thermostat - 5
    # Delete this.
    T_tank_target[7000:] = inp_triplex.hot_thermostat - 5
    
    # summer - do not put too much effort into keeping the tank cool or the COP
    # is degraded.
    T_tank_target[2*time_steps_per_season:3*time_steps_per_season]  = inp_triplex.cold_thermostat + 5
    T_tank_target[1*time_steps_per_season:2*time_steps_per_season] = inp_triplex.cold_thermostat + 1
    
    return T_tank_target

def nomenclature_sort_func(s_ent):
    #eliminate dollar signs
    elim_by_split = ["$","\\dot","{","}"]
    ss = s_ent
    for elim in elim_by_split:
        ss = "".join(ss.split(elim))
    
    # make greek letter the end of the list
    ss = "ZZZZZ".join(ss.split("\\"))
    
    # make capitalization not matter.
    ss = ss.lower()
    
    return ss

def input_table(input_file_path):
    
    
    # # TODO - add specification to these later!
    # inp_gshp = igs.gshp_inputs(bore_diam=4.25 * 2.54 * 0.01, # 4in plus change in meters
    #                       pipe_outer_diam=2.0 * 2.54 * 0.01, # 2 in in meters
    #                       pipe_inner_diam=1.8 * 2.54 * 0.01, # 1.8 in in meters 
    #                       pipe_therm_cond=0.44, # High density polyethylene (W/(m*K))
    #                       grou_therm_cond=0.4, # based off sand/gravel/clay/silt mixtures characteristic of Albuquerque
    #                       grout_therm_cond=1.0, # based off of silicone/sand mixture
    #                       grou_speci_heat=800.0, # J/(kg*K)
    #                       ground_density=1900.0, # kg/m3
    #                       grou_temperature=293.15, # K
    #                       number_borehole=21,  # this is half what Xiaobing recommended
    #                       borehole_length=80.0,
    #                       dist_boreholes=5.0, # 5 meter spacing between boreholes
    #                       num_element=20,
    #                       max_COP_heating=5.0,
    #                       frac_carnot_efficiency=0.8,
    #                       delT_target=2.0,
    #                       compressor_capacity=10000, # W  # what size is ideal?
    #                       mass_flow_capacity=10.0,
    #                       frac_borehole_length_to_Tg=0.3) # THIS IS THE CALIBRATION FACTOR USED TO MATCH XIAOBING's results.
    # inp_water = igs.water_inputs()
    # inp_tank = igs.tank_inputs()
    # inp_loop = igs.loop_inputs()
    # inp_pumping_gshp = igs.pumping_inputs()
    # inp_pumping_loop = igs.pumping_inputs()
    # inp_triplex = igs.triplex_inputs()
    # inp_hp = igs.heat_pump_inputs()
    
    with open(input_file_path,'w') as buf:
        # create this into a latex table that makes the input directly changeable in the publication
        # FYI - "Python Variable Name" column must be unique for the code below to work.
        inputs = pd.DataFrame(columns=[
         "Variable",                 "Python Variable Name",                             "Description",                                                                      "Value",                 "Unit",                            "Note",                                                                    "Category",      "Include Row in Report Table"],data=[
        ["$t$",                      "",                                                 "time",                                                                             "Variable",              "s",                               "",                                                                        "",              True],
        ["$t_{targ}$",               "",                                                 "target time which the thermal WTS thermal controller seeks to reach "+
                                                                                         "$T_{tank_{targ}}$ in.",                                                            "Variable",              "s",                               "",                                                                        "",              True],
        ["$Q_{hw}$",                 "",                                                 "Heat flow out of TBL due to hot water consumption in triplex units",               "Variable",              "W",                               "",                                                                        "Loop",          True],
        ["$Q_{hp}$",                 "QHP",                                              "Triplex heat pump condenser heat taken from or added to thermal bridge",           "Variable",              "W",                               "",                                                                        "Loop",          True],
        ["$Q_{hpe}$",                "",                                                 "Triplex heat pump evaporator-side heat taken triplex thermal zone",                "Variable",              "W",                               "",                                                                        "Triplex",       True],
        ["$W_{hp}$",                 "W",                                                "Compressor power delivered to GSHP",                                               "Control",               "W",                               "$W_{hp}$ > 0",                                                            "GSHP",          True],
        ["$W_{wshp}$",                 "W",                                              "Work put into water-side heat pumps in Triplexes",                                 "Control",               "W",                               "",                                                                        "Triplex",       True],
        ["$W_{ashp}$",                 "W",                                              "Work put into air-side heat pumps for air-source scenario",                        "Control",               "W",                               "",                                                                        "Triplex",       True],
        ["$W_{hp_{max}}$",           "",                                                 "Compressor power maximum capacity for GSHP",                                       "Control",               "W",                               "",                                                                        "GSHP",          True],
        ["$cool$",                   "is_cooling",                                       "Boolean indicating whether GSHP may cool (True) or heat (False) the WTS tank",     "Variable",              "",                                "True if $T_{tank} \\ge T_{tank_{targ}}$",                                 "WTS",           True],
        ["$wait$",                   "",                                                 "Boolean indicating whether GSHP is to wait for GSHP operation regardless of $cool$","Variable",             "",                                "Function of deadband $\Delta T_{tank_{targ}}$, $T_{tank_{targ}}$,"+
                                                                                                                                                                                                                                         " and $T_{tank}$",                                                         "WTS",           True],
        ["$wait_0$",                 "",                                                 "Previous time step's value of $wait$",                                             "Variable",             "",                                 "",                                                                        "WTS",           True],
        ["$W_{p1},W_{p2}$",          "pumping_power",                                    "Power input to pumping for thermal bridge loop and GSHP",                          "Control",               "W",                               "These are calculated from $\dot{m}_{tank}$ and $\dot{m}_{gshp}$",         "Loop,GSHP",     True],
        ["$v_1,v_2,..v_5$",          "N/A",                                              "Valve control variables that enable economization and conversion to micro-net",    "Control",               "",                                "These are not included in the current model but are implied by the logic","Loop,GSHP",     True],
        ["$C_{bore}$",               "",                                                 "Sum of GHX bore-hole heat capacitance for water, grout, piping, and fittings",     "Variable",              "J/K",                             "",                                                                         "GSHP",         True],
        ["$C_{tank}$",               "",                                                 "WTS tank heat capacitance of water storage",                                       "Variable",              "J/K",                             "",                                                                         "WTS",          True],
        ["$T_{tank}$",               "",                                                 "WTS tank temperature (output to thermal bridge loop)",                             "Variable",              "K",                               "",                                                                         "WTS",          True],
        ["$T_{tank_{targ}}$",        "",                                                 "Target for WTS tank temperature",                                                  [290.35,298.55,302.55,290.35],"K",                          "Values for 1/4 year (winter, spring, summer, fall). Future analyses need"+
                                                                                                                                                                                                                                         " need to optimize these values",                                          "WTS",           True],
        ["$T_0,T_1,...,T_{NT}$",     "",                                                 "Thermal bridge loop CV in and out temperatures between triplex units",             "Variables",             "K",                               "",                                                                        "Loop",          True],
        ["$T_{1g},T_{2g},...,T_{NG_g}$","",                                              "GHX radial temperatures spaced on a log scale",                                    "Variables",             "K",                               "",                                                                        "Loop",          True],
        ["$T_a$",                      "",                                               "Ambient atmospheric drybulb temperature",                                          "Variable",              "K",                               "Comes from historical weather files",                                     "GSHP",          True],
        ["$T_{avg}$",                  "",                                               "Average temperature of water in GHX",                                              "Variable",              "K",                               "$T_{avg}=\\frac{T_{in}+T_{out}}{2}$",                                     "GSHP",          True],
        ["$T_{in}$",                   "",                                               "Temperature of water going into the GHX",                                          "Variable",              "K",                               "",                                                                        "GSHP",          True],
        ["$T_{out}$",                  "",                                               "Temperature of water exiting the GHX",                                             "Variable",              "K",                               "",                                                                        "GSHP",          True],
        ["$Q_1,...Q_{NG}$",            "",                                               "Radial Heat transfer rates between ground CVs for dynamic GHX model",              "Variables",             "W",                               "Derived from $T_{1g},T_{2g},...,T_{NG}g$",                                "GSHP",          True],
        ["$C_1,C_2,...C_{NG}$",        "",                                               "Heat capacitance of radial sections of ground for dynamic GHX model",              "Variables",             "J/K",                             "Derived from $T_{1g},T_{2g},...,T_{NG}g$",                                "GSHP",          True],
        ["$r_{g1},r_{g2},...r_{gNG}$", "",                                               "radii of ground cylindrical sections for dynamic GHX model",                       "Variables",             "J/K",                             "Follow a logarithmic scale",                                              "GSHP",          True],
        ["$Q_{g1},Q_{g2},...,Q_{gNG}$", "",                                               "Heat transfer rates to $T_g$ for dynamic GHX model",                              "Variables",             "W",                               "Calibrating factors derived from $f_{bl}$, $T_g$, and "+
                                                                                                                                                                                                                                         "$\left\{T_{1g},T_{2g},...,T_{NG}g\right\}$",                              "GSHP",          True],
        ["$R_1,R_2,...,R_{NG}$",        "",                                               "Radial Heat transfer thermal resistance between CVs for dynamic GHX model",       "Variables",             "K/W",                             "",                                                                        "GSHP",          True],
        ["$R_{g1},R_{g2},...,R_{gNG}$", "",                                               "Heat transfer thermal resistance to ground for dynamic GHX model",                "Variables",             "K/W",                             "",                                                                        "GSHP",          True],
        ["$Q_{lo_{1}},...,Q_{lo_{NT}}$","",                                              "Heat losses/gains to ambient BC for each thermal bridge loop CV",                  "Variables",             "W",                               "Derived from $T_1,...,T_{NT}$ and $T_a$",                                 "Loop",          True],
        ["$Q_{loT}$",                "",                                                 "Heat losses/gains to ambient BC from the WTS",                                     "Variables",             "W",                               "",                                                                        "WTS",           True],
        ["$Q_U$",                    "",                                                 "Heat flow from TBL to triplex via hot water mass flow",                            "Variables",             "W",                               "",                                                                        "Loop,Triplex",  True],
        ["$Q_{econo}$",              "",                                                 "Heat flow from GHX to WTS via economization mass flow",                            "Variables",             "W",                               "",                                                                        "Loop,Triplex",  True],
        ["$Q_{AC}$",                 "",                                                 "Heat flow from WSHP to TBL via WSHP refrigerant to water condenser",               "Variables",             "W",                               "",                                                                        "Loop,Triplex",  True],
        ["$Q_{GHX}$",                 "",                                                "Net heat flow from GHX to WTS",                                                    "Variables",             "W",                               "Combination of $W_{hp}$, $COP$, and $Q_{econo}$",                         "GSHP,WTS",      True],
        ["$Q_F$",                    "",                                                 "Heat flow from Refridgerator to TBL via refrigeratn to water condenser",           "Variables",             "W",                               "Unlikely to implement because of lack of commercial models",              "Loop,Triplex",  True],
        ["$Q_D$",                    "",                                                 "Heat flow from TBL to heat pump dryer in triplex",                                 "Variables",             "W",                               "Unlikely to implement because of lack of commercial models",              "Loop,Triplex",  True],
        ["$Q_{WH}$",                    "",                                              "Heat flow from TBL to triplex via hot water heat exchange",                        "Variables",             "W",                               "",                                                                        "Loop",          True],
        [r"$\dot{m}_{tank}$",       "",                                                  "Mass flow out of the WTS tank that is diminished by $\\dot{m}_{tri}$ at each triplex","Control",            "kg/s",                            "Determines $W_{p2}$",                                                     "WTS,Loop",       True],
        [r"$\dot{m}_{tank_{targ}}$",    "",                                              "Target value for $\dot{m}_{gshp}$ before several logical considerations",           "Control",              "kg/s",                            "",                                                                        "WTS,Loop",       True],
        [r"$\dot{m}_{tank_{min}}$",     "",                                              "Minimum bound for mass flow out of the WTS tank $\dot{m}_{tank}$",                    "Control",            "kg/s",                            "",                                                                        "WTS,Loop",       True],
        [r"$\dot{m}_{tank_{max}}$",     "",                                              "Maximum bound for mass flow out of the WTS tank $\dot{m}_{tank}$",                    "Control",            "kg/s",                            "",                                                                        "WTS,Loop",       True],
        [r"$\dot{m}_{tank_{0}}$", "",                                                    "Control initial value for $\dot{m}_{tank}$",                                          "Control",            "kg/s",                            "",                                                                        "WTS,Loop",       True],
        [r"$\dot{m}_{mu}$",         "",                                                  "Total mass flow from utility input line to neighborhood",                           "Control",              "kg/s",                            "Is assumed to be equally balanced with $NT\\dot{m}_{tri}$",               "WTS,Loop",       True],
        [r"$\dot{m}_{gshp}$",       "",                                                  "GSHP loop mass flow",                                                               "Control",              "kg/s",                            "Used to regulate ${\\Delta}T_{gshp}$",                                    "WTS,Loop",       True],
        ["$Q_{tank}$",               "",                                                 "Heat drawn from/to WTS by GSHP",                                                    "Control",              "W",                               "Determined by $W_{hp}$",                                                  "WTS",           True],
        ["$Q_{tank_{targ}}$",        "",                                                 "Target value for $Q_{tank}$ before several logical considerations",                  "Control",              "W",                               "",                                                                        "WTS",           True],
        ["$COP$",                    "COP",                                              "Coefficient of performance for the GSHP",                                           "Variable",             "",                                "Determined by bounded fraction of carnot efficiency $\\eta_c$",           "GSHP",          True],
        ["$COP_{CC}$",               "",                                                 "Instantaneous COP for the CC",                                                    "Variable",             "",                                "",                                                                        "WTS,GSHP,TBL",  True],
        ["$COP_{UC}$",             "",                                                   "Instantaneous COP for the UC",                                                      "Variable",             "",                                "",                                                                        "WTS,GSHP,TBL",  True],
        ["$COP_h$",                  "",                                                 "Specified maximum heating COP for the GSHP",                                        "Variable",             "",                                "Bounds COP to lower values than the carnot efficiency $\\eta_c$",         "GSHP",          True],     
        ["$d_b$",                    "bore_diam",                                        "GHX bore-hole diameter",                                                           0.0762*2,                "m",                               "6 in",                                                                    "GSHP",          True],
        ["$d_o$",                    "pipe_outer_diam",                                  "Outer diameter of bore-hole pipe",                                                 0.013*2,                 "m",                               "~1 in",                                                                   "GSHP",          True],
        ["$d_i$",                    "pipe_inner_diam",                                  "Inner diameter of bore-hole pipe",                                                 0.011*2,                 "m",                               "0.866 in",                                                                "GSHP",          True],
        ["$r_o$",                    "not an input   ",                                  "Outer radius of bore-hole pipe",                                                   0.013,                   "m",                               "1.00 in",                                                                 "GSHP",          True],
        ["$r_i$",                    "not an input   ",                                  "Inner radius of bore-hole pipe",                                                   0.011,                   "m",                               "0.90 in",                                                                 "GSHP",          True],
        ["$r_b$",                    "not an input   ",                                  "Radius of bore-holes",                                                             0.0762,                  "m",                               "0.90 in",                                                                 "GSHP",          True],
        ["$k_p$",                    "pipe_therm_cond",                                  "Pipe material thermal conductivity",                                               0.39,                    r"W/(m\(\cdot\)K)",                "High density polyurethane",                                               "GSHP",          True],
        ["$k_g$",                    "grou_therm_cond",                                  "Ground thermal conductivity",                                                      0.4,                     r"W/(m\(\cdot\)K)",                "Based off sand/gravel/clay/silt mixtures characteristic of"+
                                                                                                                                                                                                                                         " Albuquerque",                                                            "GSHP",          True],
        ["$k_{gr}$",                 "grout_therm_cond",                                 "Bore-hole grout thermal conductivity",                                             1.3,                     r"W/(m\(\cdot\)K)",                "Silcone/sand mixture",                                                    "GSHP",          True],
        ["$c_{gr}$",                 "grou_speci_heat",                                  "Ground specific heat",                                                             809.72,                  r"J/(kg\(\cdot\)K)",               "Based off sand/gravel/clay/silt mixtures characteristic of"+
                                                                                                                                                                                                                                         " Albuquerque",                                                            "GSHP",          True],
        [r"$\rho_g$",                "ground_density",                                   "Ground Density",                                                                   1900.0,                  r"kg/m\textsuperscript{3}",      "Based off sand/gravel/clay/silt mixtures characteristic of"+
                                                                                                                                                                                                                                         " Albuquerque",                                                            "GSHP",          True],
        ["$T_g$",                    "grou_temperature",                                 "Ground temperature",                                                               293.15,                  "K",                               r"Based on \cite{reiter2006}",                                             "GSHP",          True],
        [r"$\beta$",                 "",                                                 "Dimensionless ratio of bore-hole length to heat capacity rate and thermal resistance","Variable",           "-",                               "\cite{diao2004}",                                                         "GSHP",          True],
        [r"$\epsilon$",              "",                                                 "Heat exchanger efficiency for bore-holes",                                         "Variable",              "-",                               "\cite{diao2004}",                                                         "GSHP",          True],
        ["$N_b$",                    "number_borehole",                                  "Total number bore-holes",                                                          21,                      "-",                               "1/2 the number for analysis with disconnected GSHP's",                    "GSHP",          True],
        ["$L_b$",                    "borehole_length",                                  "Bore-hole length",                                                                 90.0,                    "m",                               "",                                                                        "GSHP",          True],
        ["$D_b$",                    "dist_boreholes",                                   "Distance between bore-holes",                                                      5.3,                     "m",                               "center to center",                                                        "GSHP",          True],
        ["$NG$",                     "num_element",                                      "Number of radial CV's to model GHX",                                               21,                      "-",                               "",                                                                        "GSHP",          True],
        ["$COP_h$",                  "max_COP_heating",                                  "Maximum COP for heating",                                                          5.0,                     "-",                               "If Heating Carnot efficiency > 5, COP=5.0",                               "GSHP",          True],
        [r"$\eta_c$",                "frac_carnot_efficiency",                           "Fraction of Carnot efficiency of GSHP",                                            0.8,                     "-",                               "GSHP COP = Carnot COP * frac\_carnot\_efficiency",                        "GSHP",          True],
        [r"$\Delta T_{gshp}$",        "delT_target",                                     "Target GSHP temperature delta between input and output",                           2.0,                     "K",                               "Temperature delta target for GHX - control adjusts flow to meet this",    "GSHP",          True],
        ["$W_c$",                    "compressor_capacity",                              "GSHP Compressor capacity",                                                         10000,                   "W",                               "",                                                                        "GSHP",          True],
        [r"$\dot{m}_{gshp_{max}}$",   "mass_flow_capacity",                               "GSHP loop Mass flow capacity",                                                     10.0,                    "kg/s",                            "",                                                                        "GSHP",          True],
        ["$f_{bl}$",                 "frac_borehole_length_to_Tg",                       "Fraction of Bore-hole length below bore depth to ground temperature",              4.19313e-9,                     "-",                               "Model calibrating factor equal to the fraction of bore-hole"+
                                                                                                                                                                                                                                         " length to ground temperature heat reservoir that is never "+
                                                                                                                                                                                                                                         "affected by the GHX",                                                     "GSHP",          True],    
        ["$c_p$",                    "specific_heat",                                    "Specific heat of water",                                                           4173.8,                  r"J/(kg\(\cdot\)K)",               "",                                                                        "Water",         True],
        [r"$\rho$",                   "density",                                          "Density of water",                                                                 998.2,                   r"kg/m\textsuperscript{3}",      "",                                                                          "Water",         True],
        ["$h_{tank}$",               "tank_height",                                      "Water thermal storage (WTS) tank height",                                          2.0,                     "m",                               "",                                                                        "WTS",           True],
        ["$d_{tank}$",               "tank_diameter",                                    "WTS tank diameter",                                                                3.0,                     "m",                               "",                                                                        "WTS",           True],
        ["$R_{tank}$",               "",                                                 "WTS tank total thermal resistance",                                                "Variable",              "K/W",                             "",                                                                        "WTS",           True],
        ["$V_{tank}$",               "",                                                 "WTS tank volume",                                                                  "Variable",              r"m\textsuperscript{3}",           "",                                                                        "WTS",           True],
        ["$R_{top}$",                "",                                                 "WTS tank thermal resistance for the tank top",                                     "Variable",              "K/W",                             "",                                                                        "WTS",           True],
        ["$R_{lo}$",                 "",                                                 "TBL pipe thermal resistance per CV",                                               "Variable",              "K/W",                             "",                                                                        "Loop",          True],
        ["$R_b$",                    "",                                                 "Total thermal resistance of GHX bore-hole",                                        "Variable",              "K/W",                             "",                                                                        "Loop",          True],
        ["$R_{11}$",                 "",                                                 "GSHP thermal resistance for upward and downward flow to bore-hole the ground",     "Variable",              "K/W",                             "",                                                                        "GSHP",          True],
        ["$R_{12}$",                 "",                                                 "GSHP thermal resistance between upward and downward flows",                        "Variable",              "K/W",                             "",                                                                        "GSHP",          True],
        ["$R_{sides}$",              "",                                                 "WTS tank thermal resistance for the tank sides",                                   "Variable",              "K/W",                             "",                                                                        "WTS",           True],
        ["$l_{k_{tank}}$",           "insulation_thick",                                 "WTS tank insulation thickness",                                                    0.1,                     "m",                               "",                                                                        "WTS",           True],
        ["$k_{tank}$",               "insulation_k",                                     "WTS tank insulation thermal conductivity",                                         0.001,                   r"W/(m\(\cdot\)K)",                "",                                                                        "WTS",           True],
        [r"$\Delta T_{dead}$",        "temperature_dead_band",                            "WTS tank thermostat deadband",                                                     5.0,                     r"K",                              "",                                                                        "WTS",           True],
        ["$D_{tri}$",                "distance_between_triplex_wshp",                    "Distance between triplex unit water source heat pumps",                            10.0,                    "m",                               "This is pipe length which may be much greater than the linear length",    "Loop",          True],
        ["$d_{po_{tri}}$",           "pipe_outer_diam_loop",                             "Thermal bridge pipe outer diameter",                                               12.5*0.0254,              "m",                              "12.5 inch pipe",                                                           "Loop",          True],
        ["$d_{pi_{tri}}$",           "pipe_inner_diam_loop",                             "Thermal bridge pipe inner diameter",                                               12*0.0254,              "m",                                "12 inch pipe inner diameter",                                            "Loop",          True],
        ["$D_{trench}$",             "trenching_depth",                                  "Thermal bridge pipe trenching depth",                                              1.0,                     "m",                               "",                                                                        "Loop",          True],
        ["$NT$",                     "num_triplex_units",                                "Number of triplex units serviced by thermal bridge",                               30,                      "-",                               "",                                                                        "Loop",          True],
        [r"$\Delta T_{loop}$",        "delT_target_loop",                                 "Temperature gain/loss at in/out of thermal bridge loop",                           2.0,                     "K",                               "Mass flow is used to keep this constant",                                 "Loop",          True],
        [r"$\Delta T_{tank_{targ}}$", "",                                                 "Temperature deadband about $T_{tank_{targ}}$ for GSHP",                            2.0,                     "K",                               "Mass flow is used to keep this constant",                                 "Loop",          True],
        [r"$\dot{m}_{tank_{max}}$",   "mass_flow_capacity_loop",                          "Mass flow capacity for the thermal bridge loop",                                   10.0,                    "kg/s",                            "",                                                                        "Loop",          True],
        ["$h_{e_{gshp}}$",           "elev_change_gshp",                                 "Elevation change along in-out of GSHP GHX loop",                                   0.0,                     "m",                               "",                                                                        "GSHP",          True],
        ["$C_{r_{gshp}}$",           "pipe_roughness_gshp",                              "Pipe roughness factor for GSHP loop",                                              120.0,                   "-",                               "ONLY USED IF WNTR SLOW METHOD IS REACTIVATED!",                           "GSHP",          False],
        ["$C_{ml_{gshp}}$",          "extra_minor_losses_gshp",                          "Pipe minor losses for GSHP loop",                                                  30.0,                    "-",                               "ONLY USED IF WNTR SLOW METHOD IS REACTIVATED!",                           "GSHP",          False],
        [r"$\eta_{p_{gshp}}$",        "pump_efficiency_gshp",                             "GSHP pump efficiency",                                                             0.9,                     "-",                               "ONLY USED IF WNTR SLOW METHOD IS REACTIVATED!",                           "GSHP",          False],
        ["$h_{e_{loop}}$",           "elev_change_loop",                                 "Elevation change along in-out of thermal bridge loop",                             0.0,                     "m",                               "",                                                                        "Loop",          False],
        ["$C_{r_{loop}}$",           "pipe_roughness_loop",                              "Pipe roughness factor for thermal bridge loop",                                    120.0,                   "-",                               "",                                                                        "Loop",          False],
        ["$C_{ml_{loop}}$",          "extra_minor_losses_loop",                          "Pipe minor losses for thermal bridge loop",                                        30.0,                    "-",                               "",                                                                        "Loop",          False],
        [r"$\eta_{p_{loop}}$",        "pump_efficiency_loop",                             "Thermal bridge loop pump efficiency",                                              0.9,                     "-",                               "",                                                                        "Loop",          False],
        ["$\dot{m}_{tank_{min}}$",    "min_flow_loop",                                    "Minimum flow allowed by controller for thermal bridge loop",                       1.0,                     "kg/s",                            "",                                                                        "Loop",          True],
        ["$T_{c_{tri}}$",            "cold_thermostat",                                  "Triplex cooling thermostat setpoint",                                              24.4+273.15,             "K",                               "76 F",                                                                    "Triplex",       True],
        ["$T_{h_{tri}}$",            "hot_thermostat",                                   "Triplex heating thermostat setpoint",                                              22.2+273.15,             "K",                               "",                                                                        "Triplex",       True],
        [r"$\dot{m}_{tri_{daily}}$",  "hot_water_consumption_per_triplex_gal_per_day",    "Triplex hot water daily total consumption from TBL side",                          66.7,                    "gal/day",                         "$\dot{m}_{tri}+\dot{m}_{tri_u}=66.7$",                                    "Triplex",       True],
        [r"$\dot{m}_{tri}$",          "",                                                 "Triplex hot water consumption from TBL side",                                      "Variable",              r"m\textsuperscript{3}/s",         "",                                                                        "Triplex",       True],
        [r"$\dot{m}_{tri_u}$",        "",                                                 "Triplex hot water consumption from utility side",                                  "Variable",              r"m\textsuperscript{3}/s",         "Controls will optimize where to use water from",                          "Triplex",       True],
        [r"$f_{i\dot{m}_{tri}}$",     "hot_water_use_fraction",                           "Daily hot water use fraction for each hour of day",                                [0.0062,
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
                                                                                                                                                                              0.0235],                "-",                               "Typical hot water use fractions per \cite{wilson2014}",                   "Triplex",       True],
        ["$Q_{rh_{wshp}}$",           "rated_heating_capacity_W",                        "Rated heating capacity of triplex unit WSHP",                                       3516.85,                "W",                               "1-ton unit",                                                              "Triplex",       True],
        ["$Q_{rc_{wshp}}$",           "rated_cooling_capacity_W",                        "Rated cooling capacity of triplex unit WSHP",                                       3516.85,                "W",                               "1-ton unit",                                                              "Triplex",       True],
        ["$COP_{ashp}$",              "rated_COP_air",                                   "Rated COP for ASHP for equivalent unconnected triplex calculation",                 3.0,                    "-",                               "",                                                                        "Triplex",       True],
        ["$COP_{wshp}$",              "rated_COP_water",                                 "Rated COP for WSHP in triplex units",                                               4.2,                    "-",                               "",                                                                        "Triplex",       True],
        ["$\dot{V}_{a_{tri}}$",       "rated_air_flow",                                  r"Rated air flow for ASHP and WSHP",                                                 0.15338292125,          r"m\textsuperscript{3}/s",         "325 CFM per Comfort Air Model HB-009 \cite{comfortaire2021}",             "Triplex",       True],
        ["$SHR_{tri}$",               "rated_SHR",                                       "Rated sensible heat ratio for triplex unit WSHP",                                   0.75,                   "-",                               "",                                                                        "Triplex",       True],
        ["$T_{hw}$",                  "hot_water_setpoint",                              "Hot water setpoint in triplex units",                                               60.0+273.15,            "K",                               "Typical value to avoid bio issues but not burning occupants",             "Triplex",       True],
        ["$Q_{rh_{gshp}}$",           "rated_gshp_heating_capacity_W",                   "Rated GSHP heating capacity (water to water WSHP)",                                 50000.0,                "W",                               "THIS IS NOT USED IN THE MODEL!",                                          "GSHP",          False],
        ["$Q_{rc_{gshp}}$",           "rated_gshp_cooling_capacity_W",                   "Rated GSHP cooling capacity (water to water WSHP)",                                 50000.0,                "W",                               "THIS IS NOT USED IN THE MODEL!",                                          "GSHP",          False],
        ["$COP_{gshp}$",              "rated_gshp_COP",                                  "Rated COP for GSHP cooling (water to water WSHP)",                                  5.0,                    "-",                               "THIS IS NOT USED IN THE MODEL!",                                          "GSHP",          False],
        ["$\dot{V}_{gshp}$",          "rated_gshp_water_flow",                           "Rated Water flow fro GSHP loop (GHX)",                                              10.0/0.997,             r"m\textsuperscript{3}/s",         "THIS IS NOT USED IN THE MODEL!",                                          "GSHP",          False],
        ["$SHR_{gshp}$",              "sensible_heat_ratio_gshp",                        "Rated sensible heat ratio for GSHP (water to water WSHP)",                          1.0,                    "-",                               "THIS IS NOT USED IN THE MODEL!",                                          "GSHP",          False],
        ["$D$",                       "not_a_variable_1",                                "Half spacing between the axes of two pipes in a single bore-hole",                  "Variable",                   "m",                         r"For this analysis = $\frac{1}{3}\left(d_b + d_o\right)$",                "GSHP",          True],
        ["$\delta$",                  "not_a_variable_2",                                "Dirac delta function",                                                              "-",                    "-",                               r"",                                                                       "",              True],
        ["$CSPF$",                    "not_a_variable_3",                                "Cooling Seasonal Performance Factor",                                               "-",                    "-",                               r"",                                                                       "",              True],
        ["$HSPF$",                    "not_a_variable_4",                                "Heating Seasonal Performance Factor",                                               "-",                    "-",                               r"",                                                                       "",              True],
        ["$useGHX$",                  "not_a_variable_5",                                "Boolean indicating whether the GHX will be used for controls algorithm",            "-",                    "-",                               r"",                                                                       "",              True]])
        with pd.option_context("max_colwidth", 1000):
            # need to pre-process the variables column.
            var_list = inputs[inputs["Include Row in Report Table"]]["Variable"]
            sort_list = var_list.apply(nomenclature_sort_func).sort_values()
            
            filtered_inputs = inputs[inputs["Include Row in Report Table"]].loc[sort_list.index,:]
            filtered_inputs.to_latex(buf=buf,index=False,
                    column_format=r'p{0.175\textwidth}p{0.3\textwidth}p{0.08\textwidth}p{0.08\textwidth}p{0.3\textwidth}',
                    longtable=True,escape=False,columns=["Variable","Description","Value","Unit","Note"],float_format=lambda x: '%.3g' % x)
        return inputs
    
def _calc_calibration_fractions(inp_gshp,fr0):
    # ADDED TO HANDLE VARIABLE PARAMETERIZATIONS OF FRAC
    rr = np.logspace(np.log10(inp_gshp.rb),np.log10(inp_gshp.dist_boreholes/2.0),num=inp_gshp.num_element)
    if isinstance(fr0,(list,tuple)):
        if len(fr0) == 2:
            fr = fr0[0] * rr + fr0[1]
        elif len(fr0) == 3:
            fr = fr0[0] * rr**2 + fr0[1]*rr + fr0[2]
        else:
            raise ValueError("The func_type input currently only supports the following values: \n\n"
                             +"   1. linear\n2. quadratic")
    elif isinstance(fr,float):
        fr = fr0
    else:
        raise TypeError("The fraction input only accepts a single value or a list containing [slope,intercept] or quadratic parameters!")
        
    return fr
    
    
def create_input_objects(inputs):
        # simplify the syntax below
    inpd = {}
    for name in inputs["Python Variable Name"]:
        inpd[name] = inputs[inputs["Python Variable Name"]==name ]["Value"].values[0]
    
    # TODO - add specification to these later!
    inp_gshp = igs.gshp_inputs(bore_diam=inpd["bore_diam"], 
                               pipe_outer_diam=inpd["pipe_outer_diam"],
                               pipe_inner_diam=inpd["pipe_inner_diam"], # 1.8 in in meters 
                               pipe_therm_cond=inpd["pipe_therm_cond"], # High density polyethylene (W/(m*K))
                               grou_therm_cond=inpd["grou_therm_cond"], # based off sand/gravel/clay/silt mixtures characteristic of Albuquerque
                               grout_therm_cond=inpd["grout_therm_cond"], # based off of silicone/sand mixture
                               grou_speci_heat=inpd["grou_speci_heat"], # J/(kg*K)
                               ground_density=inpd["ground_density"], # kg/m3
                               grou_temperature=inpd["grou_temperature"], # K
                               number_borehole=inpd["number_borehole"],  # this is half what Xiaobing recommended
                               borehole_length=inpd["borehole_length"],
                               dist_boreholes=inpd["dist_boreholes"], # 5 meter spacing between boreholes
                               num_element=inpd["num_element"],
                               max_COP_heating=inpd["max_COP_heating"],
                               frac_carnot_efficiency=inpd["frac_carnot_efficiency"],
                               delT_target=inpd["delT_target"],
                               compressor_capacity=inpd["compressor_capacity"], # W  # what size is ideal?
                               mass_flow_capacity=inpd["mass_flow_capacity"],
                               frac_borehole_length_to_Tg=inpd["frac_borehole_length_to_Tg"]) # THIS IS THE CALIBRATION FACTOR USED TO MATCH XIAOBING's results.
    
    
    inp_water = igs.water_inputs(specific_heat=inpd["specific_heat"], # J/(kg*K)
                          density=inpd["density"] #kg/m3 
                          )
    inp_tank = igs.tank_inputs(tank_height = inpd["tank_height"], # m
                               tank_diameter = inpd["tank_diameter"], # m
                               insulation_thick = inpd["insulation_thick"], #m
                               insulation_k = inpd["insulation_k"], # W/(m*K)
                               temperature_dead_band = inpd["temperature_dead_band"]) # K)
    inp_loop = igs.loop_inputs(distance_between_triplex_wshp=inpd["distance_between_triplex_wshp"],
                     pipe_outer_diam_loop=inpd["pipe_outer_diam_loop"], # 2in in meters
                     pipe_inner_diam_loop=inpd["pipe_inner_diam_loop"], # 1.8in in meters
                     trenching_depth=inpd["trenching_depth"], #meters
                     num_triplex_units=inpd["num_triplex_units"], # number of water source heat pumps (wshp)
                     delT_target_loop=inpd["delT_target_loop"],
                     mass_flow_capacity_loop=inpd["mass_flow_capacity_loop"],
                     min_flow_loop=inpd["min_flow_loop"])
    inp_pumping_gshp = igs.pumping_inputs(elev_change=inpd["elev_change_gshp"], #m
                           pipe_roughness=inpd["pipe_roughness_gshp"],
                           extra_minor_losses=inpd["extra_minor_losses_gshp"],
                           pump_efficiency=inpd["pump_efficiency_gshp"])
    inp_pumping_loop = igs.pumping_inputs(elev_change=inpd["elev_change_loop"], #m
                           pipe_roughness=inpd["pipe_roughness_loop"],
                           extra_minor_losses=inpd["extra_minor_losses_loop"],
                           pump_efficiency=inpd["pump_efficiency_loop"])
    inp_triplex = igs.triplex_inputs(cold_thermostat = inpd["cold_thermostat"],  #76F
                     hot_thermostat = inpd["hot_thermostat"],
                     hot_water_consumption_per_triplex_gal_per_day = inpd["hot_water_consumption_per_triplex_gal_per_day"],
                     hot_water_use_fraction = inpd["hot_water_use_fraction"],
                     hot_water_setpoint = inpd["hot_water_setpoint"])
    inp_hp = igs.heat_pump_inputs(rated_heating_capacity_W=inpd["rated_heating_capacity_W"],
                          rated_cooling_capacity_W=inpd["rated_cooling_capacity_W"],
                          rated_COP_air=inpd["rated_COP_air"],
                          rated_COP_water=inpd["rated_COP_water"],
                          rated_air_flow=inpd["rated_air_flow"],# 325 CFM See Comfort Air Model HB-009 model
                          rated_SHR=inpd["rated_SHR"],
                          rated_gshp_heating_capacity_W=inpd["rated_gshp_heating_capacity_W"],
                          rated_gshp_cooling_capacity_W=inpd["rated_gshp_cooling_capacity_W"],
                          rated_gshp_COP=inpd["rated_gshp_COP"],
                          rated_gshp_water_flow=inpd["rated_gshp_water_flow"],
                          sensible_heat_ratio_gshp=inpd["sensible_heat_ratio_gshp"])
    
    return inp_gshp, inp_water, inp_tank, inp_loop, inp_pumping_gshp, inp_pumping_loop, inp_triplex, inp_hp, inpd

def single_run(num_bores,cap,fr,height,wts_gshp,inpd,inp_gshp,inp_tank,inp_water,
               inp_loop,inp_pumping_gshp,inp_pumping_loop,inp_triplex,inp_hp,qhp_loads_file,
               weather_file,data_path,pump_poly_dict,T_tank_target,percent_to_complete,
               pause_simulation_time,break_on_troubleshoot,skip_to_step,run_parallel,run_num):
    if run_parallel:
        break_on_troubleshoot = False
    
    inp_gshp.nb = num_bores
    inp_gshp.compressor_capacity = cap
    inp_gshp.calib_fract = fr
    inp_tank.tank_height = height
    master = wts_gshp(inpd["grou_temperature"],
                     inp_gshp,
                     inp_tank,
                     inp_water,
                     inp_loop,
                     inp_pumping_gshp,
                     inp_pumping_loop,
                     inp_triplex,
                     inp_hp,
                     qhp_loads_file,
                     weather_file,
                     data_path,pump_poly_dict,T_tank_target[0])
        
    master.no_load_shift_simulation(T_tank_target,True,
                                    percent_to_complete,
                                    pause_simulation_time,
                                    break_on_troubleshoot,
                                    skip_to_step)
    inputs = (num_bores,cap,fr,height,inpd,inp_gshp,inp_tank,inp_water,
                   inp_loop,inp_pumping_gshp,inp_pumping_loop,inp_triplex,inp_hp,qhp_loads_file,
                   weather_file,data_path,pump_poly_dict,T_tank_target,percent_to_complete,
                   pause_simulation_time,break_on_troubleshoot,skip_to_step)
    
    return master,inputs,run_num
    

def run_parameter_study(inpd,
                        inp_gshp,
                        inp_tank,
                        inp_water,
                        inp_loop,
                        inp_pumping_gshp,
                        inp_pumping_loop,
                        inp_triplex,
                        inp_hp,
                        qhp_loads_file,
                        weather_file,
                        data_path,
                        compressor_capacities,
                        num_bore_holes,
                        tank_height,
                        frac,
                        resolve,
                        percent_to_complete,
                        pause_simulation_time,
                        break_on_troubleshoot,
                        skip_to_step,
                        pump_poly_dict,
                        T_tank_target,name_tags,
                        run_parallel=False,
                        fraction_cpu_to_use=0.7):
    master_sol = []
    
    
    num_run = len(compressor_capacities) * len(num_bore_holes) * len(tank_height) * len(frac)
    
    if run_parallel:
        import multiprocessing as mp
        
        num_cpu = np.min([int(mp.cpu_count()*fraction_cpu_to_use),num_run])
        pool = mp.Pool(num_cpu)
    
    idx = 0
    run_num = 0
    if resolve:
        for cap in compressor_capacities:
            for num_bores in num_bore_holes:
                for height in tank_height:
                    idx+=1
                    for fr0 in frac:
                        
                        fr = _calc_calibration_fractions(inp_gshp, fr0)
                        
                        if not "master" in globals() or resolve:
                            
                            if not run_parallel:

                                master_sol.append(single_run(num_bores,cap,fr,height,igs.wts_gshp,inpd,inp_gshp,inp_tank,inp_water,
                                               inp_loop,inp_pumping_gshp,inp_pumping_loop,inp_triplex,inp_hp,qhp_loads_file,
                                               weather_file,data_path,pump_poly_dict,T_tank_target,percent_to_complete,
                                               pause_simulation_time,break_on_troubleshoot,skip_to_step,run_parallel,run_num))
                            else:
                                master_sol.append(pool.apply_async(single_run,
                                             args=(num_bores,cap,fr,height,igs.wts_gshp,
                                                   inpd,inp_gshp,inp_tank,inp_water,
                                               inp_loop,inp_pumping_gshp,inp_pumping_loop,
                                               inp_triplex,inp_hp,qhp_loads_file,
                                               weather_file,data_path,pump_poly_dict,
                                               T_tank_target,percent_to_complete,
                                               pause_simulation_time,
                                               break_on_troubleshoot,skip_to_step,run_parallel,run_num)))
                                
        # post processing variables                        
        cop_air_avg = []
        cop_wts_avg = []
        
        # initiate a parameter study
        
        gshp_max_temp = []
        gshp_min_temp = []
        COP_cool_h2o = []
        COP_heat_h2o = []
        COP_cool_air = []
        COP_heat_air = []   
        final_sol = []                     
        for tup in master_sol:
            if run_parallel:
                lres = tup.get()
                (master,inp,run_num) = lres
            else:
                (master,inp,run_num) = tup
            Other = np.array(master.cs.Other)
            cop_air = Other[:,1]
            cop_wts = Other[:,2]
            #import pdb; pdb.set_trace()
            cop_air_avg.append(cop_air[~isna(cop_air)].mean())
            cop_wts_avg.append(cop_wts[~isna(cop_wts)].mean())
            sol = np.array(master.cs.solutions)
            gshp_max_temp.append(np.max([np.max(sol[:,0]), np.max(sol[:,1])]))
            gshp_min_temp.append(np.min([np.min(sol[:,0]), np.min(sol[:,1])]))
            
            if master.P_c_h2o_total == 0:
                COP_cool_h2o.append(np.nan)
            else:
                COP_cool_h2o.append(master.Q_c_h2o_total/master.P_c_h2o_total)
            if master.P_h_h2o_total == 0:
                COP_heat_h2o.append(np.nan)
            else:
                COP_heat_h2o.append(master.Q_h_h2o_total/master.P_h_h2o_total)
            if master.P_c_air_total == 0:
                COP_cool_air.append(np.nan)
            else:
                COP_cool_air.append(master.Q_c_air_total/master.P_c_air_total)
            if master.P_h_air_total == 0:
                COP_heat_air.append(np.nan)
            else:
                COP_heat_air.append(master.Q_h_air_total/master.P_h_air_total)

            final_sol.append(master)
            
        if run_parallel:
            pool.close()
                                
        
        dump([final_sol,cop_air_avg,cop_wts_avg,compressor_capacities,num_bore_holes,tank_height,T_tank_target,cop_air,cop_wts],open("finished_parameter_study1.pickle","wb"))
    else:
        [final_sol,cop_air_avg,cop_wts_avg,compressor_capacities,num_bore_holes,
         tank_height,T_tank_target,cop_air,cop_wts] = load(open("finished_parameter_study1.pickle","rb"))
    
    return (final_sol,cop_air_avg,cop_wts_avg,compressor_capacities,
            num_bore_holes,tank_height,T_tank_target,cop_air,cop_wts,name_tags)

def post_process_plot_1(master_sol,dt,cop_air,cop_wts,name_tags,cps,nbs,ths,figure_path,T_tank_target,skip_to_step):
    
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
    # BC[8] = Q_econo
    
    # new effort for direct energy efficiency comparison
    
    
    sec_in_hour = 3600.0
    kW_to_W = 1000.0
    W_air_list = []
    W_h2o_list = []
    tank_volume_list = []
    for sol in master_sol:
        tank_volume_list.append(sol.cs.wts.volume)
        # reset because we are recalculating!
        W_air = 0.0
        W_h2o = 0.0

        W_air = np.sum(np.array([arr[-3]+arr[-2] for arr in sol.cs.Other])) * sol.dt /sec_in_hour / kW_to_W
        W_h2o = np.sum(np.array([arr[-4]+arr[-1] for arr in sol.cs.Other])) * sol.dt /sec_in_hour / kW_to_W

        W_air_list.append(W_air)
        W_h2o_list.append(W_h2o)
    W_air_arr = np.array(W_air_list)
    W_h2o_arr = np.array(W_h2o_list)
    
    COP_total_air_c = []
    COP_total_h2o_c = []
    COP_total_air_h = []
    COP_total_h2o_h = []
    
    for (idmm,master) in enumerate(master_sol):
        
        if master.P_c_h2o_total != 0:
            COP_total_h2o_c.append((master.Q_c_h2o_total + master.P_hw_c_h2o_total)/(master.P_c_h2o_total + master.P_hw_c_h2o_total))
        else:
            COP_total_h2o_c.append(np.nan)
        if master.P_h_h2o_total != 0:
            COP_total_h2o_h.append((master.Q_h_h2o_total + master.P_hw_h_h2o_total)/(master.P_h_h2o_total + master.P_hw_h_h2o_total))
        else:
            COP_total_h2o_h.append(np.nan)
        # air    
        if master.P_c_air_total != 0:
            COP_total_air_c.append((master.Q_c_air_total + master.P_hw_c_air_total)/(master.P_c_air_total + master.P_hw_c_air_total))
        else:
            COP_total_air_c.append(np.nan)
        if master.P_h_air_total != 0:
            COP_total_air_h.append((master.Q_h_air_total + master.P_hw_c_air_total)/(master.P_h_air_total + master.P_hw_c_air_total))
        else:
            COP_total_air_h.append(np.nan)
        
        
        
        sim_time = np.arange(0,len(master.cs.solutions)*dt,dt)/3600
        results = np.array(master.cs.solutions)
        BCs = np.array(master.cs.BCs)
        
        Other = np.array(master.cs.Other)
        
        Q_GSHP = master.cs.gshp.cpw * BCs[:,7] * (results[:,0] - results[:,1])    
        
        cooling_status = np.zeros(BCs.shape[0])
        for idx,is_cooling,W_gshp in zip(range(BCs.shape[0]),BCs[:,0],BCs[:,1]):
            if W_gshp != 0:
                if is_cooling == 1:
                    cooling_status[idx] = 1
                else:
                    cooling_status[idx] = -1
        if False:
            # Temperature
            fig,axl = plt.subplots(5,1,figsize=(10,12))
            axl[0].plot(sim_time,results[:,0],label="GSHP input ($T_{in})$")
            axl[0].plot(sim_time,results[:,1],label="GSHP output ($T_{out}$)")
            axl[0].plot(sim_time,results[:,2],label="Tank out ($T_{tank}$)")
            axl[0].plot(sim_time,results[:,2+master.cs.loop.NT],label="Tank in ($T_{NT}$)")
            axl[0].plot(sim_time,BCs[:,3],label="Ambient ($T_a$)")
            axl[0].plot(sim_time,BCs[:,2],"--k",label="Ground ($T_g$)")
            axl[0].plot(sim_time,T_tank_target[skip_to_step:skip_to_step + len(sim_time)],"--c",label="Tank target ($T_{tank_{targ}}$)")
            axl[0].grid("on")
            axl[0].set_ylabel("Temperature (K)")
            axl[0].legend(loc='upper left', bbox_to_anchor=(1.05, 1.1))
            
            # Heat
            axl[1].plot(sim_time,BCs[:,1],label="Compressor ($W_{hp}$)")
            axl[1].plot(sim_time,master.cs.loop.NT*BCs[:,4],label="Triplex ($Q_{hpe}$)")
            axl[1].plot(sim_time,Q_GSHP,label="GSHP ($Q_{GHX}$)")
            axl[1].plot(sim_time,BCs[:,8],label="Econo ($Q_{econo}$)")
            axl[1].plot(sim_time,Other[:,-2],label="Ground ($\\sum_i(Q_{gi})$)")
            axl[1].plot(sim_time,Other[:,-1],label="HW ($Q_U+Q_{WH}$)")
            axl[1].legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))
            axl[1].grid("on")
            axl[1].set_ylabel("Heat Flow (W)")
            
            # Mass Flow
            axl[2].plot(sim_time,BCs[:,5],label="Loop ($\\dot{m}_{tank}$)")
            axl[2].plot(sim_time,BCs[:,7],label="GSHP ($\\dot{m}_{gshp}$)")
            axl[2].legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))
            axl[2].grid("on")
            axl[2].set_ylabel("mass flow (kg/s)")
            
            # Cooling Status
            axl[3].plot(sim_time,cooling_status)
            axl[3].set_ylabel("cooling status\n (0=off,1=cool,-1=heat)")
            axl[3].grid("on")
            
            # COP
            axl[4].plot(sim_time,cop_wts,label="$COP_{CC}$")
            axl[4].plot(sim_time,cop_air,label="$COP_{ASHP}$")
            axl[4].grid("on")
            axl[4].set_xlabel("hour of year")
            axl[4].set_ylabel("COP (heating or cooling)")    
            axl[4].legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))
            plt.tight_layout()
            
            try:
                plt.savefig(os.path.join(figure_path,"system_characteristics_"+name_tags[idmm]),dpi=300)
            except:
                print("plot save failed!")
        
    tank_volume = np.array(tank_volume_list)
    df = pd.DataFrame({"H2O cooling":COP_total_h2o_c,
                  "H2O heating":COP_total_h2o_h,
                  "Air cooling":COP_total_air_c,
                  "Air heating":COP_total_air_h,
                  "GSHP compressor size (W)":cps,
                  "Number of bore-holes":nbs,
                  "Thermal tank height (m)":ths,
                  "WTS-GSHP-TBL":W_h2o_arr,
                  "ASHP":W_air_arr,
                  "Tank volume (m$^3$)":tank_volume})
    
    def perc(x,base):
        if base == 0:
            return 0
        else:
            return 100*x/base
        
    def return_perc(perc,base):
        return (perc/100) * base
        
    fig2, axl2 = plt.subplots(3,1,figsize=(10,15))
    
    # compressor size
    first_time = True
    for comp in df['GSHP compressor size (W)'].unique():
        for vol in df['Tank volume (m$^3$)'].unique():
            # plot COP as function of boreholes
            nb = df['Number of bore-holes'].unique()
            dff = df[(df["GSHP compressor size (W)"]==comp) & (df['Tank volume (m$^3$)']==vol)]
            if first_time:
                axl2[0].plot(dff["Number of bore-holes"],dff["Air cooling"],label="Unconnected")
            axl2[0].plot(dff["Number of bore-holes"],dff["H2O cooling"],label="CC V={0:0.1f}".format(vol))
            if first_time:
                axl2[1].plot(dff["Number of bore-holes"],dff["Air heating"],label="Unconnected")
            axl2[1].plot(dff["Number of bore-holes"],dff["H2O heating"],label="CC V={0:0.1f}".format(vol))
            if first_time:
                axl2[2].plot(dff["Number of bore-holes"],dff["ASHP"] - dff["ASHP"],label="Unconnected")
            axl2[2].plot(dff["Number of bore-holes"],dff["ASHP"]-dff["WTS-GSHP-TBL"],label="CC V={0:0.1f}".format(vol)) 
            base = dff["ASHP"].iloc[0]
            syaxl2 = axl2[2].secondary_yaxis("right",functions=(lambda x: perc(x,base),lambda x: return_perc(x,base)))

            first_time = False
    for ax in axl2:
        ax.grid("on")
    axl2[2].set_ylabel("Energy Savings Due to CC (kWh/yr)")
    axl2[0].set_ylabel("Cooling Seasonal Performance Factor")
    axl2[1].set_ylabel("Heating Seasonal Performance Factor")
    axl2[2].set_xlabel("Number of bore-holes")
    syaxl2.set_ylabel("% Savings from Unconnected")
    axl2[2].legend()
        
    plt.tight_layout()
    try:
        plt.savefig(os.path.join(figure_path,"study_results_COP_plot.png"),dpi=300)
    except:
        print("plot save failed")

        
        
        
def QHP_history_plot(data_path,qhp_loads_file,figure_path,place_holder_year,days_per_month):
    
    QHP_history = pd.read_csv(os.path.join(data_path,qhp_loads_file))
    
    fig2,ax2 = plt.subplots(1,1,figsize=(10,5))
    QHP_history_no_design_days = QHP_history.iloc[192:,:]
    step1 = [s[:6] + "/" + str(place_holder_year) + " " + s[7:] for s in QHP_history_no_design_days["Date/Time"].values]
    step2 = []
    days_in_month = []
    # TODO - Dealing with non-year TMY3 dates is a pain - this code may be a useful tool!
    for date_string in step1:
        ds1 = date_string.strip().split(' ')
        ds2 = ds1[0].split("/")
        ds3 = ds1[1].split(":")
        if "24" in ds3[0]:
            ds3[0] = "00"
            try:
                if int(ds2[1]) == days_per_month[int(ds2[0])-1]:
                    if ds2[0] == '12': # increment the year reset the month and day
                        ds2[2] = str(int(ds2[2]) + 1)
                        ds2[0] = "01"
                        ds2[0] = "01"
                    else: # increment the month reset the day
                        ds2[0] = str(int(ds2[0]) + 1)
                        ds2[1] = "01"
                else: # increment the day
                    ds2[1] = str(int(ds2[1]) + 1)
            except:
                pass
        step2.append("/".join(ds2)+" "+":".join(ds3))
        
    
    QHP_history_no_design_days.index = pd.to_datetime(step2)
    ax2.plot(QHP_history_no_design_days.index[:-1],QHP_history_no_design_days["Heat Taken from Triplex Units (W)"].iloc[:-1])
    ax2.grid("on")
    start_of_month_indices = np.cumsum(np.array(days_per_month)*24*4)-np.array(days_per_month)*24*4
    ax2.set_xticks(QHP_history_no_design_days.index[start_of_month_indices])
    date_form = DateFormatter("%b")
    ax2.xaxis.set_major_formatter(date_form)
    ax2.set_xlim(ax2.get_xlim()[0]+10,ax2.get_xlim()[1]-15)
    ax2.set_ylabel(r"$Q_{hpe}$ Heat to TBL per triplex unit (W)")
    plt.savefig(os.path.join(figure_path,"energy_plus_heat_input.png"),dpi=300)

# CONTROL SCRIPT
def master_script():
    # GLOBAL PLOT CONTROLS

    stime = time.time()
    run_parallel = True
    frac_cpu_to_use = 0.95
    font = {'family':'normal',
            'weight':'normal',
            'size':14}
    rc('font', **font)
    create_plots = True
    
    if create_plots:
        plt.close("all")
    
    # PATHS and files input
    # Create an input structure that flows into the analysis and also writes to the 
    # documentation so that manual maintenance of parameter changes is not necessary.
    input_file_path = os.path.join(os.path.dirname(__file__),"..","SANDReport","from_python","input_table.tex")
    figure_path = os.path.join(os.path.dirname(__file__),"..","SANDReport","figures")
    base_path = os.path.dirname(__file__)
    data_path = os.path.join(base_path,"..","Data")
    weather_file = "Albuquerque_Weather_Data.csv" # this is hourly and needs to
                                                  # be resampled to 15 minutes.
    # triplex unit loads
    qhp_loads_file = r"MF+CZ4B+USA_NM_Albuquerque.Intl.AP.723650+hp+slab+IECC_2018_TriplexUnit_LOADS.csv"
    
    # TMY3 weather date axis handling to provide days_per_month
    place_holder_year = 2021
    days_per_month = [calendar.monthrange(place_holder_year,month+1)[1] for month in range(12)]

    # INPUT - this forms a latex table for the publication that also provides input
    #         values.
    #   GOTO input_table to change values of any input to the analysis!
    inputs = input_table(input_file_path)
    # build up input objects needed in igshpa.py
    (inp_gshp, inp_water, inp_tank, inp_loop, inp_pumping_gshp,
     inp_pumping_loop, inp_triplex, inp_hp, inpd) = create_input_objects(inputs)    
    

    # BEGIN PARAMETER STUDY STUFF                
    # parameter study controls:
    
    # Time step - do not change - this has not been tested with different
    #  time step sizes and much of the analysis assumes 15 minutes
    dt = 15*60.0 # 15 minute time steps     
    
    # troubleshooting/convenience controls
    resolve = False                 # True = run the entire analysis, 
                                    # False = refresh plots from the analysis
    percent_to_complete = 101         # 1-101 = percent of analysis (from skip_to_step)
                                    #         to complete 
    pause_simulation_time = None    # None = Never pause simulation
                                    # Value = seconds to simulation pause
                                    # use this to stop at a specific time #12*3600 = 12 hrs into the simulation
    break_on_troubleshoot = False   # Complete stop at pause_simulation_time
    skip_to_step = 0                # 15 minute steps to skip (allows starting at any time
                                    # in the year but starts with the same initial conditions regardless)
    
    dat1 = load(open("pump_polyfit_results_Loop2022-02-15.pickle",'rb'))
    dat2 = load(open("pump_polyfit_results_GSHP2022-02-15.pickle",'rb'))
    pump_poly_dict = {"Loop":dat1,
                     "GSHP":dat2}  #Loop,GSHP
    
    # parameter variations
    compressor_capacities = [10000]#,10000,15000]
    # finer resolution
    num_bore_holes = np.arange(45,125)   #[15,25,30,35,40,45,50]#,10,15,21]
    tank_height = [2,2.5,3,4,5,6] #np.arange(2.5,5.75,0.25)#,2,3,4]
    
    name_tags = []
    cps = []
    nbs = []
    ths = []
    for comp in compressor_capacities:
        for nb in num_bore_holes:
            for th in tank_height:
                name_tags.append("comp_{0:d}_nb_{1:d}_tankh_{2:0.1f}".format(comp,nb,th).replace(".","_"))
                cps.append(comp)
                nbs.append(nb)
                ths.append(th)
                
    # THIS IS THE FINAL CALIBRATED VALUE SEE ghx_calibration_to_xiaobing_data for more details
    # DO NOT CHANGE THIS. IT IS AN EMPIRICAL FACTOR. - THIS IS NOT THE frac 
    # that has to do with water consumption.
    
    # this is optimal for 0.3 cv MJ/m3-K
    #frac = [[1.2434658252815482e-05,2.924542480469096e-05]]#[4.19313e-9]   

    # this is optimal for 1.53 cv MJ/m3-K
    frac = [[0.00015791084966032944,1.1127693387715544e-07]]#[4.19313e-9]              
    
    # DYNAMIC TANK TARGET go to the function to change the time history.
    T_tank_target = form_tank_target_history(inpd["grou_temperature"],inp_triplex)

    
    # RUN THE PARAMETER STUDY    
    (master_sol,cop_air_avg,cop_wts_avg,compressor_capacities,
     num_bore_holes,tank_height,T_tank_target,
     cop_air,cop_wts,name_tags) = run_parameter_study(inpd,
                        inp_gshp,
                        inp_tank,
                        inp_water,
                        inp_loop,
                        inp_pumping_gshp,
                        inp_pumping_loop,
                        inp_triplex,
                        inp_hp,
                        qhp_loads_file,
                        weather_file,
                        data_path,
                        compressor_capacities,
                        num_bore_holes,
                        tank_height,
                        frac,
                        resolve,
                        percent_to_complete,
                        pause_simulation_time,
                        break_on_troubleshoot,
                        skip_to_step,
                        pump_poly_dict,
                        T_tank_target,name_tags,
                        run_parallel,
                        frac_cpu_to_use)
    
    # POST PROCESS/PLOT

        # create plots - all plots need to go straight to the publication
    # figures folder so that the publication updates if you are making
    # changes!
    if create_plots:
        hot_water_use_fractions_plot(inpd,figure_path)
        QHP_history_plot(data_path,qhp_loads_file,figure_path,place_holder_year,days_per_month)
        post_process_plot_1(master_sol,dt,cop_air,cop_wts,name_tags,cps,nbs,ths,figure_path,T_tank_target,skip_to_step)
    
    etime = time.time()
    import pdb;pdb.set_trace()
    print("The study took {0:0.1f} seconds to complete.".format(etime-stime))
    


if __name__ == "__main__":
    master_script()
    

    


