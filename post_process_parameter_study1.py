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


Created on Sat Mar 13 14:00:29 2021

@author: dlvilla
"""

from pickle import load

out = load(open("finished_parameter_study1.pickle","rb"))

cop_air_avg = out[0]
cop_wts_avg = out[-4]
compressor_capacities = out[-3]
num_bore_holes = out[-2]
tank_height = out[-1]

dict1 = {}
idx = 0
for cap in compressor_capacities:
    dict1[cap] = {}
    for num_bores in num_bore_holes:
        dict1[cap][num_bores] = {}
        for height in tank_height:
            dict1[cap][num_bores][height] = cop_wts_avg[idx]
            idx += 1
            
