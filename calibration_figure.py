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

Created on Wed Jul 21 18:30:46 2021

@author: dlvilla
"""

# this is the calibration data 

import pandas as pd
import openpyxl as xl
from matplotlib import rc
from matplotlib import pyplot as plt
import numpy as np
import os


plt.close("all")

font = {'family':'normal',
        'weight':'normal',
        'size':16}
rc('font', **font)

linestyle_tuple = [
 ('dotted',                (0, (1, 1))),
 ('dashed',                (0, (5, 5))),
 ('dashdotted',            (0, (3, 5, 1, 5))),
 ('densely dashdotted',    (0, (3, 1, 1, 1))),
 ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10)))]

b_color = ['k','r','b']

wb = xl.load_workbook('WTS_GSHP_ModelCalibration.xlsx',data_only=True)
sheet = wb.get_sheet_by_name("Sheet1")

data_rows = []
for row in sheet['A3':'E21']:
    data_col = []
    for cell in row:
        data_col.append(cell.value)
    data_rows.append(data_col)
    
columns = data_rows[0]

df = pd.DataFrame(data_rows[1:],columns=columns)

err = np.sqrt(df["Error2"].min())

log_f = df[df["Error2"]==df["Error2"].min()]["Log10"].values[0]


fig,ax = plt.subplots(1,1,figsize=(10,8))



ax.plot(df["Log10"],df["Max GSHP temperature"]-273.15,label="Max Temp",color='k')
ax.plot(df["Log10"],df["Min GSHP temperature"]-273.15,label="Min Temp",color='k',linestyle=linestyle_tuple[2][1])
ax.plot([-10,0],[35,35],label="Target Max",color='b',linestyle=linestyle_tuple[1][1])
ax.plot([-10,0],[3,3],label="Target Min",color='b',linestyle=linestyle_tuple[0][1])

ax.plot([log_f],[err],marker="*",markersize=10)

ax.set_xlim([-10,0])


secax = ax.secondary_yaxis('right')

secax.set_ylabel('Root square error (K)')
ax.plot(df["Log10"],np.sqrt(df["Error2"]),color='r',linestyle=linestyle_tuple[3][1],label="Error")

ax.set_xlabel(r"$log_{10}(f_{bl})$")
ax.set_ylabel("Output temperature of GHX $T_{out}$ (K)")

ax.legend()
ax.grid("on")
plt.tight_layout()

plt.savefig(os.path.join("..","SANDReport","figures","model_calibration_results.png"),dpi=300)



