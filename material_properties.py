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

Created on Thu Mar 11 09:53:01 2021

@author: dlvilla
"""
import numpy as np


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