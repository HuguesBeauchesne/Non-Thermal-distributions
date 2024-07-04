####################
# Thermal averages #
####################

### Explanations ###
#This program contains the functions to compute thermal averages and make the corresponding plots.

#Comments
# - The thermal averages of different quantities can be returned.

#Import standard
import math, os, scipy
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

#Import custom
import Generic

#Integration parameter
pInt = 20



#Thermal average for particles with identical temperatures
def thermal_average(CS, CSt, min1, min2, mout1, mout2, T, mode):

  #Function to integrate
  def CSint(s):

    #Common term
    common = (s - (min1 + min2)**2)*(s - (min1 - min2)**2)*CS(s)/(8*min1**2*min2**2*T)

    #Function to integrate for cross section
    if mode == "CS":
      return                1/s**0.5*Generic.K1on22(T, s, min1, min2)*common

    #Function to integrate for Eplus and Eplus'
    if mode == "Eplus" or mode == "EplusP":
      return                         Generic.K2on22(T, s, min1, min2)*common

    #Function to integrate for Eminus
    if mode == "Eminus":
      return (min1**2  -  min2**2)/s*Generic.K2on22(T, s, min1, min2)*common

    #Function to integrate for Eminus'
    if mode == "EminusP":
      return (mout1**2 - mout2**2)/s*Generic.K2on22(T, s, min1, min2)*common

  #Return integral
  temp = scipy.integrate.quad(CSint, max(min1 + min2, mout1 + mout2)**2, (max(min1 + min2, mout1 + mout2) + pInt*T)**2)[0]

  if math.isnan(temp):
    temp = 0

  return temp



#Thermal average for particles with different temperatures
def thermal_average2D(CS, CSt, min1, min2, mout1, mout2, T1, T2, mode):

  #Identical temperatures
  if T1 == T2:
    return thermal_average(CS, CSt, min1, min2, mout1, mout2, T1, mode)

  #Function to integrate
  def CSint(Ep, s):

    #Useful kinematic quantities
    TA     = 2*T1*T2/(T2 - T1)
    TS     = 2*T1*T2/(T2 + T1)
    pij    = (s - (min1  + min2 )**2)**0.5*(s - (min1  - min2 )**2)**0.5/(2*s**0.5)
    pmn    = (s - (mout1 + mout2)**2)**0.5*(s - (mout1 - mout2)**2)**0.5/(2*s**0.5)
    Emmax  = Ep*(min1**2 - min2**2)/s + 2*pij*(Ep**2/s - 1)**0.5
    Emmin  = Ep*(min1**2 - min2**2)/s - 2*pij*(Ep**2/s - 1)**0.5 
    Aplus  = Ep/TS + Emmin/TA
    Aminus = Ep/TS + Emmax/TA
    common = TA*s**0.5/(8*T1*T2*min1**2*min2**2*scipy.special.kn(2, min1/T1)*scipy.special.kn(2, min2/T2))

    #Numerator for cross section
    if mode == "CS":
      return (math.exp(-Aplus) - math.exp(-Aminus))*CS(s)*pij*common

    #Numerator for Eplus and Eplus'
    if mode == "Eplus" or mode == "EplusP":
      return Ep*(math.exp(-Aplus) - math.exp(-Aminus))*CS(s)*pij*common

    #Numerator for Eminus
    if mode == "Eminus":
      Bplus  = Emmin + TA
      Bminus = Emmax + TA
      return (Bplus*math.exp(-Aplus) - Bminus*math.exp(-Aminus))*CS(s)*pij*common

    #Numerator for Eminus'
    if mode == "EminusP":
      term1  = (mout1**2 - mout2**2)/s*Ep*(math.exp(-Aplus) - math.exp(-Aminus))*CS(s)*pij*common
      Cplus  = TA - 2*pij*(Ep**2/s - 1)**0.5
      Cminus = TA + 2*pij*(Ep**2/s - 1)**0.5
      term2   = (Cplus*math.exp(-Aplus) - Cminus*math.exp(-Aminus))*CSt(s)*pmn*common
      return term1 + term2

  #Return double integral
  temp = scipy.integrate.dblquad(CSint, max(min1 + min2, mout1 + mout2)**2, (max(min1 + min2, mout1 + mout2) + pInt*max(T1, T2))**2, lambda s: s**0.5, lambda s: s**0.5 + pInt*max(T1, T2))[0]

  if math.isnan(temp):
    temp = 0

  return temp



#Plot maker for the cross section for a single temperature
def plotMaker(Process, TlistCS, listCS, mode):

  originalDirectory = os.getcwd()
  os.chdir(originalDirectory + "/Plots")

  plt.plot(TlistCS, listCS)
  axes = plt.gca()
  axes.set_xlabel("T [GeV]")

  if mode == "CS":
    axes.set_ylabel("CS [$GeV^{-2}$]")

  if mode == "Eplus":
    axes.set_ylabel("CS $E_+$ [$GeV^{-1}$]")

  if mode == "Eminus":
    axes.set_ylabel("CS $E_-$ [$GeV^{-1}$]")

  if mode == "EplusP":
    axes.set_ylabel("CS $E_+'$ [$GeV^{-1}$]")

  if mode == "EminusP":
    axes.set_ylabel("CS $E_-' [$GeV^{-1}$]")

  plt.savefig(Process + "_" + mode + ".pdf")
  plt.clf()

  os.chdir(originalDirectory)



#Plot maker for the cross section at different temperatures
def plotMaker2D(Process, TlistCS1, TlistCS2, listCS, mode):

  originalDirectory = os.getcwd()
  os.chdir(originalDirectory + "/Plots")

  listCSnp = np.array(listCS)
  listCSff = np.flipud(listCSnp)
  im = plt.imshow(listCSff, extent=[TlistCS1[0], TlistCS1[-1], TlistCS2[0], TlistCS2[-1]], cmap='bwr', aspect='auto', vmax=np.max(listCSnp), vmin=np.min(listCSnp), interpolation="gaussian")
  plt.xlabel('$T_{P_{in_2}}$ [GeV]')
  plt.ylabel('$T_{P_{in_1}}$ [GeV]')
  axes = plt.gca()

  divider = make_axes_locatable(axes)
  cax = divider.append_axes("right", size="5%", pad=0.1)

  cbar = plt.colorbar(im, cax=cax)

  plt.savefig(Process + "_" + mode + ".pdf")
  plt.clf()

  os.chdir(originalDirectory)



#List makers
def listMaker(Process, CS, CSt, TlistCS, paramsInt, min1, min2, mout1, mout2, mode):

  #Create list of cross sections
  listCS = []

  #Compute cross sections on 1D grid
  for T in TlistCS:
    listCS.append(thermal_average(lambda s: CS(paramsInt, min1, min2, mout1, mout2, s), lambda s: CSt(paramsInt, min1, min2, mout1, mout2, s), min1, min2, mout1, mout2, T, mode))

  #Make plot of expectation value as a function of the temperature
  plotMaker(Process, TlistCS, listCS, mode)

  #Return list of cross sections on 1D grid
  return np.array(listCS)

def listMaker2D(Process, CS, CSt, TlistCS1, TlistCS2, paramsInt, min1, min2, mout1, mout2, mode):

  #Create list of cross sections
  listCS = []

  #Compute cross sections on 2D grid
  for T1 in TlistCS1:

    listtemp = []

    for T2 in TlistCS2:
      listtemp.append(thermal_average2D(lambda s: CS(paramsInt, min1, min2, mout1, mout2, s), lambda s: CSt(paramsInt, min1, min2, mout1, mout2, s), min1, min2, mout1, mout2, T1, T2, mode))

    listCS.append(listtemp)

  #Make plot of expectation value as a function of the temperature
  plotMaker2D(Process, TlistCS1, TlistCS2, listCS, mode)

  #Return list of cross sections on 2D grid
  return np.array(listCS)

intersect = lambda a, b: any(i in b for i in a)

def listMakerProcess(Process, CS, CSt, TlistCSlists, paramsInt, min1, min2, mout1, mout2, listElements):

  #Initialize the list of results
  listResults = []

  #Internal function to compute the expectation value lists
  if len(TlistCSlists) == 1:
    def listMakerInt(mode):
      return listMaker(Process, CS, CSt, TlistCSlists[0], paramsInt, min1, min2, mout1, mout2, mode)
  else:
    def listMakerInt(mode):
      return listMaker2D(Process, CS, CSt, TlistCSlists[0], TlistCSlists[1], paramsInt, min1, min2, mout1, mout2, mode)

  #Compute the different lists
  if intersect(listElements, ["CS"]):
    listCS      = listMakerInt("CS")
    listResults.append(["CS", listCS])

  if intersect(listElements, ["Eplus", "Ein1", "Ein2", "Eout1mEin1", "Eout2mEin2"]):
    listEplus   = listMakerInt("Eplus")
    if "Eplus" in listElements:
      listResults.append(["Eplus", listEplus])

  if intersect(listElements, ["Eminus", "Ein1", "Ein2", "Eout1mEin1", "Eout2mEin2"]):
    listEminus  = listMakerInt("Eminus")
    if "Eminus" in listElements:
      listResults.append(["Eminus", listEminus])

  if intersect(listElements, ["EplusP", "Eout1", "Eout2", "Eout1mEin1", "Eout2mEin2"]):
    listEplusP  = listMakerInt("EplusP")
    if "EplusP" in listElements:
      listResults.append(["EplusP", listEplusP])

  if intersect(listElements, ["EminusM", "Eout1", "Eout2", "Eout1mEin1", "Eout2mEin2"]):
    listEminusP  = listMakerInt("EminusP")
    if "EminusP" in listElements:
      listResults.append(["EminusP", listEminusP])

  if intersect(listElements, ["Ein1", "Eout1mEin1"]):
    listEin1    = np.add(listEplus,   listEminus )/2
    if "Ein1" in listElements:
      listResults.append(["Ein1", listEin1])

  if intersect(listElements, ["Ein2", "Eout2mEin2"]):
    listEin2    = np.add(listEplus,  -listEminus )/2
    if "Ein2" in listElements:
      listResults.append(["Ein2", listEin2])

  if intersect(listElements, ["Eout1", "Eout1mEin1"]):
    listEout1   = np.add(listEplusP,  listEminusP)/2
    if "Eout1" in listElements:
      listResults.append(["Eout1", listEout1])

  if intersect(listElements, ["Eout2", "Eout2mEin2"]):
    listEout2   = np.add(listEplusP, -listEminusP)/2
    if "Eout2" in listElements:
      listResults.append(["Eout2", listEout2])

  if intersect(listElements, ["Eout1mEin1"]):
    listEout1mEin1 = np.add(listEout1, -listEin1)
    listResults.append(["Eout1mEin1", listEout1mEin1])

  if intersect(listElements, ["Eout2mEin2"]):
    listEout2mEin2 = np.add(listEout2, -listEin2)
    listResults.append(["Eout2mEin2", listEout2mEin2])

  #Return a list of the requested quantities
  return listResults



#Class to compute the cross section of relevant scatterings
class CSInfo_Class_General():

  #Method to define methods that compute the expectation value of different thermal quantities at different temperatures
  def functions_generator_int(self, TlistCSlists, ExpectationValues, Name, rescale, nrescale):
    if len(TlistCSlists) == 1:
      setattr(self, Name, lambda T:      rescale**nrescale*Generic.listInterpolation(T, TlistCSlists[0], ExpectationValues))
    else:
      setattr(self, Name, lambda T1, T2: rescale**nrescale*Generic.listInterpolation2D(T1, T2, TlistCSlists[0], TlistCSlists[1], ExpectationValues))

  def functions_generator(self, Process, CS, CSt, TlistCSlists, paramsInt, min1, min2, mout1, mout2, listElements, rescale, nrescale):

    #If not previously defined, compute the cross sections and rates
    if not hasattr(self, Process + "_" + listElements[0]):
      setattr(self, Process + "List", listMakerProcess(Process, CS, CSt, TlistCSlists, paramsInt, min1, min2, mout1, mout2, listElements))
      print(Process + " process complete.")

    #Generate functions for each of them
    listExpectationValues = getattr(self, Process + "List")

    for i in range(0, len(listExpectationValues)):
      self.functions_generator_int(TlistCSlists, listExpectationValues[i][1], Process + "_" + listExpectationValues[i][0], rescale, nrescale)




