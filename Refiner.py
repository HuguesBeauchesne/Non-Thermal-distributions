##########
# Refiner #
##########

### Explanations ###
#This program refines a grid and its corresponding density.

#Comments
# - None

#Import standard
import numpy as np

#Import custom
import Generic



#Write items
def GridRefiner(pGridIn, fGridIn, fmin, pMaxMin):

  #Find maximum acceptable momentum
  pMax = pGridIn[-1]

  for i in range(0, len(pGridIn) - 1):
    if fGridIn[i] < fmin and pGridIn[i] > pMaxMin:
      pMax = pGridIn[i]
      break

  #Refine the momentum grid
  pGridOut = np.linspace(0, pMax, len(pGridIn))

  #Refine the phase-space distribution grid
  fGridOut = []

  for ptemp in pGridOut:
    fGridOut.append(Generic.listInterpolation(ptemp, pGridIn, fGridIn))

  fGridOut = np.array(fGridOut)

  #Return the new grids
  return pGridOut, fGridOut




