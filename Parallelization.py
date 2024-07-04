###################
# Parallelization #
###################

### Explanations ###
#This program contains a function to parallelize computations of 2 -> 2 scatterings.

#Comments
# - None

#Import standard
import multiprocessing
import numpy as np

#Import custom
import Scattering



#Function to parallelize computation of 2 -> 2 scatterings
def parC(p1ComovingGrid, p2ComovingGrid, p3ComovingGrid, p4ComovingGrid, p1Grid, p2Grid, p3Grid, p4Grid, an1, an2, an3, an4, m1, m2, m3, m4, f1, f2, f3, f4, W, aScale, merging, T, scenario, maxProc):

  #Parallelization
  listlisti = [[] for _ in range(0, maxProc)]

  for i in range(0, len(p4ComovingGrid)):
    listlisti[i%maxProc].append(i)

  listProcesses = []
  return_list   = multiprocessing.Manager().list([[] for _ in range(0, maxProc)])

  for nList in range(0, maxProc):
    listProcesses.append(multiprocessing.Process(target=Scattering.scatteringderiv, args=(return_list, nList, listlisti[nList], p1ComovingGrid, p2ComovingGrid, p3ComovingGrid, p4ComovingGrid, p1Grid, p2Grid, p3Grid, p4Grid, an1, an2, an3, an4, m1, m2, m3, m4, f1, f2, f3, f4, W, aScale, merging, T, scenario)))
    listProcesses[nList].start()

  for nList in range(0, maxProc):
    listProcesses[nList].join()

  #Merge results
  dfdt = []

  for i in range(0, len(p4ComovingGrid)):
    dfdt.append(return_list[i%maxProc][int(np.floor(i/maxProc))])

  #Total
  return np.array(dfdt)




