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

Created on Sat Mar  6 11:28:28 2021

@author: dlvilla
"""

zone_names = ["Breezeway","living_unit1_FrontRow_BottomFloor",
                          "living_unit2_FrontRow_BottomFloor",
                          "living_unit3_FrontRow_BottomFloor",
                          "living_unit1_BackRow_BottomFloor",
                          "living_unit2_BackRow_BottomFloor",
                          "living_unit3_BackRow_BottomFloor",
                          "living_unit1_FrontRow_MiddleFloor",
                          "living_unit2_FrontRow_MiddleFloor",
                          "living_unit3_FrontRow_MiddleFloor",
                          "living_unit1_BackRow_MiddleFloor",
                          "living_unit2_BackRow_MiddleFloor",
                          "living_unit3_BackRow_MiddleFloor",
                          "living_unit1_FrontRow_TopFloor",
                          "living_unit2_FrontRow_TopFloor",
                          "living_unit3_FrontRow_TopFloor",
                          "living_unit1_BackRow_TopFloor",
                          "living_unit2_BackRow_TopFloor",
                          "living_unit3_BackRow_TopFloor",
                          "attic"]

meter_names = ["Fans:Electricity:Zone:","EnergyTransfer:Zone:",
               "Cooling:EnergyTransfer:Zone:","Heating:EnergyTransfer:Zone:":]

for meter in meter_names:
    for zone in zone_names:
        print("")
        print("Output:Meter,")
        print("    {0}{1} ,!- Name".format(meter,zone))
        print("    Detailed;")
