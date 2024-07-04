################
# W calculator #
################

### Explanations ###
#This program computes the energy transfer coefficients.

#Comments
# - None

#Import standard
import math, multiprocessing, scipy
import numpy as np




#u definite integral for the case of a constant amplitude
def F1(aprime, bprime, cprime, umin, umax, M2):

  D = bprime**2/(4*aprime**2) - cprime/aprime

  uprimemax = umax + bprime/(2*aprime)
  uprimemin = umin + bprime/(2*aprime)

  if D > uprimemax**2:
    tempmax = np.arctan(uprimemax/(D - uprimemax**2)**0.5)
  else:
    tempmax = math.pi/2*np.sign(uprimemax)

  if D > uprimemin**2:
    tempmin = np.arctan(uprimemin/(D - uprimemin**2)**0.5)
  else:
    tempmin = math.pi/2*np.sign(uprimemin)
 
  return M2/(-aprime)**0.5*(tempmax - tempmin)



#Tensor computer for general case
def Wcalculator(return_list, nList, p1grid, p2grid, p4grid, m1, m2, m3, m4, M2, inIdentical, numberDiagrams, case, iGrid):

  #Initialize variables
  Wtensor = np.full((len(p4grid), len(p1grid), len(p2grid)), 0.0)

  #Symmetry factor
  Fint = 1/2 if inIdentical else 1

  #Loop over energies
  for i in iGrid:
    for j in range(0, len(p1grid)):
      for k in range(0, len(p2grid)):

        #Energies
        p1 = p1grid[j]
        p2 = p2grid[k]
        p4 = p4grid[i]

        E1 = (p1**2 + m1**2)**0.5
        E2 = (p2**2 + m2**2)**0.5
        E4 = (p4**2 + m4**2)**0.5
        E3 = E1 + E2 - E4

        if E3 < m3:
          continue 

        #Hat variables
        ahat = 1
        bhat = 4*E2*E3 - 2*(m2**2 + m3**2)
        chat = 4*(E2 - E3)*(E2*m3**2 - E3*m2**2) + (m2**2 - m3**2)**2

        #Check t variable has an acceptable range
        Deltahat = bhat**2 - 4*ahat*chat

        if Deltahat < 0:
          continue

        #Boundaries t
        tmax1 = m1**2 + m4**2 - 2*E1*E4 + 2*p1*p4
        tmin1 = m1**2 + m4**2 - 2*E1*E4 - 2*p1*p4

        tmax2 = (-bhat + Deltahat**0.5)/(2*ahat)
        tmin2 = (-bhat - Deltahat**0.5)/(2*ahat)

        tmax = min(tmax1, tmax2)
        tmin = max(tmin1, tmin2)

        if tmax < tmin:
          continue

        #Function to integrate over t
        def toIntegrate(t):

          #Generic
          costheta1 = (t - m1**2 - m4**2 + 2*E1*E4)/(2*p1*p4)
          sintheta1 = (1 - costheta1**2)**0.5
          tbar      = t + m2**2 - m3**2

          #Normal variables
          a = -4*p2**2*((E1 - E4)**2 - t)
          b = -8*p2*(tbar/2 + E1*E2 - E2*E4)*(p4 - p1*costheta1)
          c =  4*p1**2*p2**2*sintheta1**2 - 4*(tbar/2 + E1*E2 - E2*E4)**2

          Delta = b**2 - 4*a*c

          if Delta < 0:
            return 0

          #Prime variables (see notes)
          aprime = a                              /(4*p2**2*p4**2)
          bprime = a*(-m2**2 - m4**2 + 2*E2*E4)   /(2*p2**2*p4**2) + b                           /(2*p2*p4)
          cprime = a*(-m2**2 - m4**2 + 2*E2*E4)**2/(4*p2**2*p4**2) + b*(-m2**2 - m4**2 + 2*E2*E4)/(2*p2*p4) + c

          #Boundaries u
          umax1 = m2**2 + m4**2 - 2*E2*E4 + 2*p2*p4
          umin1 = m2**2 + m4**2 - 2*E2*E4 - 2*p2*p4

          xmax2 = (-b - Delta**0.5)/(2*a)
          xmin2 = (-b + Delta**0.5)/(2*a)

          umax2 = m2**2 + m4**2 - 2*E2*E4 + 2*p2*p4*xmax2
          umin2 = m2**2 + m4**2 - 2*E2*E4 + 2*p2*p4*xmin2

          umax = min(umax1, umax2)
          umin = max(umin1, umin2)

          if umax < umin:
            return 0

          #Function to return
          if case == 1:
            uIntegral = F1(aprime, bprime, cprime, umin, umax, M2)

          if not math.isnan(uIntegral):
            return 1/(256*math.pi**4*E4*p4**2)*uIntegral
          else:
            return 0

        #Results
        Wtensor[i, j, k] = numberDiagrams*Fint*(scipy.integrate.quad(toIntegrate, tmin, tmax)[0])*p1*p2/(E1*E2)

  #Return the energy exchange tensor
  print("Process done!")
  return_list[nList] = Wtensor



def Wparallelization(p1grid, p2grid, p4grid, m1, m2, m3, m4, M2, inIdentical, numberDiagrams, case, maxProc):

  #Separate points in grid
  listi         = np.array([i for i in range(0, len(p4grid))])
  listlisti     = np.array_split(listi, maxProc)
  listProcesses = []
  return_list   = multiprocessing.Manager().list([[]]*maxProc)

  #Start processes
  for nList in range(0, maxProc):
    listProcesses.append(multiprocessing.Process(target=Wcalculator, args=(return_list, nList, p1grid, p2grid, p4grid, m1, m2, m3, m4, M2, inIdentical, numberDiagrams, case, listlisti[nList])))
    listProcesses[nList].start()

  for nList in range(0, maxProc):
    listProcesses[nList].join()

  #Merge results
  Wtensor = return_list[0]

  for nList in range(1, maxProc):
    Wtensor += return_list[nList]

  #Total
  return np.array(Wtensor)




