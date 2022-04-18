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

Created on Thu Mar 11 12:57:04 2021

@author: dlvilla
"""

import wntr

vol_demand = 0.001
L_total = 1000
pipe_rough = 120
pipe_diam = 0.05
del_elev = 10.0
minor_loss = 30.0
dt = 15*60
mu = 0.0010005 # Pa *s



wn = wntr.network.WaterNetworkModel()

# notice pump_parameter is listed as "Float value of power in KW."
help(wn.add_pump)

wn.add_reservoir("res1",base_head=0.0,coordinates=(0,0))
wn.add_junction("pump_outlet",base_demand=0,elevation=0.0,coordinates=(0,0))
wn.add_junction("farthest_point",base_demand=0,elevation=del_elev,coordinates=(L_total,0))
wn.add_pipe("pipe_from_res","farthest_point","pump_outlet",length=L_total,diameter=pipe_diam,roughness=pipe_rough,
            minor_loss=minor_loss,status="Open")
#
# 1.0 should be kW based on the documentation!!!
#
wn.add_pump("pump","res1","pump_outlet","Power",pump_parameter=1.0,speed=1.0)

wn.options.hydraulic.viscosity = mu
wn.options.time.duration=dt

nn = wn.get_node("farthest_point")
nn.demand_timeseries_list[0].base_value = vol_demand

        
sim = wntr.sim.EpanetSimulator(wn)

res = sim.run_sim()


# pressure is in meters head with rho_water =1000kg/m3
del_P_pump = res.node['pressure']['pump_outlet'].iloc[0] - res.node['pressure']['res1'].iloc[0]
vol_flow = res.link['flowrate']['pipe_from_res'].iloc[0]

# THIS WORKS and produces 1000 W

# units:                        m        kg/m3   m/s2    m3/s -> Watts
pump_power_by_del_P = np.abs(del_P_pump * 1000 * 9.81 * vol_flow)

if pump_power_by_del_P < 1000 + 1 and pump_power_by_del_P > 1000 - 1:
    print("WNTR power works correctly")
else:
    print("WNTR input is W even though it says it is in kW!")


