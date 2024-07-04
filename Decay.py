#########
# Decay #
#########

### Explanations ###
#This program computes the derivative contributions from decay.

#Comments
# - None

#Import standard
import math
import numpy as np



#Decay contribution to time derivative
def decayderiv(p1ComovingGrid, m1, f1, Gamma1, aScale, an, T):

  #Initialize derivatives
  df1dt = np.full((len(p1ComovingGrid)), 0.0)

  #Loop over all possible momenta
  for i in range(0, len(p1ComovingGrid) - 1):

    #Find the momenta and energies
    p1     = p1ComovingGrid[i]/aScale**an
    E1     = math.sqrt(p1**2 + m1**2)
    gamma1 = E1/m1

    #Equilibrium distribution
    f1eq = math.exp(-E1/T)

    #Compute derivatives
    df1dt[i] = -1/gamma1*Gamma1*(f1[i] - f1eq)

  #Return derivatives
  return df1dt




