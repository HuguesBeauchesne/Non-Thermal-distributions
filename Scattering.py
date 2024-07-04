##############
# Scattering #
##############

### Explanations ###
#This program computes the derivative contributions from scattering.

#Comments
# - None

#Import standard
import math
import numpy as np
from multiprocessing import Pool

#Import custom
from Generic import listInterpolation
from Generic import listInterpolation3D



#Scattering matrix
def scatteringderiv(return_list, nList, listi, p1ComovingGrid, p2ComovingGrid, p3ComovingGrid, p4ComovingGrid, p1Grid, p2Grid, p3Grid, p4Grid, an1, an2, an3, an4, m1, m2, m3, m4, f1, f2, f3, f4, W, aScale, nPointsum, T, scenario):

  #Rescaled vectors
  p1vector = p1ComovingGrid/aScale**an1
  p2vector = p2ComovingGrid/aScale**an2
  p4vector = p4ComovingGrid/aScale**an4

  E1vector = np.sqrt(p1vector**2 + m1**2)
  E2vector = np.sqrt(p2vector**2 + m2**2)
  E4vector = np.sqrt(p4vector**2 + m4**2)

  #Initialize variables
  df4dt = []

  #Early stop
  def earlystop(i, j, k):

    #Energy third particle
    E3 = E1vector[j] + E2vector[k] - E4vector[i]

    if E3 < m3:
      return True

    #Check integration range of t
    p3 = (E3**2 - m3**2)**0.5

    t1max = m1**2 + m4**2 - 2*E1vector[j]*E4vector[i] + 2*p1vector[j]*p4vector[i]
    t2min = m2**2 + m3**2 - 2*E2vector[k]*E3          - 2*p2vector[k]*p3

    if t1max < t2min:
      return True

    t2max = m2**2 + m3**2 - 2*E2vector[k]*E3          + 2*p2vector[k]*p3
    t1min = m1**2 + m4**2 - 2*E1vector[j]*E4vector[i] - 2*p1vector[j]*p4vector[i]

    if t2max < t1min:
      return True

    #Else, the event is fine
    return False

  #Find lower limits of j and k for a given i using a bisection method
  def minfinderj(i0, k0):

    #Initialize variables
    jminguess = 0
    jmaxguess = len(p1vector)

    #Do bisection
    while True:

      jmiddleguess = int(np.floor((jmaxguess - jminguess)/2)) + jminguess

      if earlystop(i0, jmiddleguess, k0):
        jminguess = jmiddleguess
      else:
        jmaxguess = jmiddleguess

      if jmaxguess - jminguess <= 1:
        break

    #Return lower boundary
    return jminguess

  def minfinderk(i0, j0):

    #Initialize variables
    kminguess = 0
    kmaxguess = len(p2vector)

    #Do bisection
    while True:

      kmiddleguess = int(np.floor((kmaxguess - kminguess)/2)) + kminguess

      if earlystop(i0, j0, kmiddleguess):
        kminguess = kmiddleguess
      else:
        kmaxguess = kmiddleguess

      if kmaxguess - kminguess <= 1:
        break

    #Return lower boundary
    return kminguess

  #Compute the integral for the different values of the momentum of p4
  for i in listi:

    #Initialize variables
    df4dti = 0.0

    #Find boundaries p1 and p2
    jmax = len(p1vector) - 1
    kmax = len(p2vector) - 1
    jmin = minfinderj(i, kmax)
    kmin = minfinderk(i, jmax)

    #Set up grids
    jstep = max(int(np.floor((jmax - jmin)/nPointsum)), 1)
    kstep = max(int(np.floor((kmax - kmin)/nPointsum)), 1)

    listj = np.flip(np.array([jmax - n*jstep for n in range(1, int(np.floor((jmax - jmin)/jstep)) + 1)]))
    listk = np.flip(np.array([kmax - n*kstep for n in range(1, int(np.floor((kmax - kmin)/kstep)) + 1)]))

    dp1array = [(p1vector[listj[n + 1]] - p1vector[listj[n]] if n < len(listj) - 1 else p1vector[listj[n]] - p1vector[listj[n - 1]]) for n in range(0, len(listj))]
    dp2array = [(p2vector[listk[n + 1]] - p2vector[listk[n]] if n < len(listk) - 1 else p2vector[listk[n]] - p2vector[listk[n - 1]]) for n in range(0, len(listk))]

    #Loop over other particles
    for n1 in reversed(range(0, len(listj))):
      for n2 in reversed(range(0, len(listk))):

        #Initialization
        j = listj[n1]
        k = listk[n2]

        #Potential early stopping
        if earlystop(i, j, k):
          break

        #Information particle 3
        E3 = E1vector[j] + E2vector[k] - E4vector[i]

        #W calculation
        Winterpolate = listInterpolation3D(p4vector[i], p1vector[j], p2vector[k], p4Grid, p1Grid, p2Grid, W)

        if Winterpolate == 0:
          continue

        #f3m information
        if scenario == 2:
          f3m = np.exp(-E3/T)
        else:
          f3m = listInterpolation(np.sqrt(E3**2 - m3**2)*aScale**an3, p3ComovingGrid, f3)

        #Gain/loss
        if scenario == 0:
          df4dti += Winterpolate*(f1[j]*f2[k] - f3m*f4[i])*dp1array[n1]*dp2array[n2]
        elif scenario == 1:
          df4dti += Winterpolate*(np.exp(-(E1vector[j] + E2vector[k])/T) - f3m*f4[i])*dp1array[n1]*dp2array[n2]
        elif scenario == 2:
          df4dti += Winterpolate*(np.exp(-E1vector[j]/T)*f2[k] - f3m*f4[i])*dp1array[n1]*dp2array[n2]

    #Return derivative at point i
    df4dt.append(df4dti)

  #Return derivatives
  return_list[nList] = df4dt




