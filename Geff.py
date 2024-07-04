########
# geff #
########

### Explanations ###
#This program contains the computation of geff and related functions.

#Comments
# - None

#Import generic
import math, os, scipy
import numpy as np
import scipy.integrate
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

#Import custom
import Constants, Generic



#Make plots of geff?
makePlot = False



#Slope control for QCD phase transition
Asmoothing = 75



#Corrections for when a particle is non-relativisitc (BF=1: Boson, BF=-1: Fermion)
def NRCorrection(MonT, BF):

  #Very relativistic
  if MonT < 0.1:
    return 1

  #Very unrelativistic
  if MonT > 10:
    return 0

  #Not so relativistic 
  def Integrand(u):
    return 15/(math.pi**4)*(u**2 - MonT**2)**0.5*u**2/(math.exp(u) - BF)

  integral = scipy.integrate.quad(Integrand, MonT, 100)[0]

  #Apply proper statistic
  if BF == 1:
    return integral
  else:
    return integral*(8/7)



#geff computation using quarks
def geff_Calc_aboveQCD(T):

  #Initialize variables
  geff = 0

  #Fermions
  mU = [Constants.mu, Constants.mc,  Constants.mt  ]
  mD = [Constants.md, Constants.ms,  Constants.mb  ]
  mE = [Constants.me, Constants.mmu, Constants.mtau]

  for i in range(0, 3):
    geff += (7/8)*3*4*NRCorrection(mU[i]/T, -1)
    geff += (7/8)*3*4*NRCorrection(mD[i]/T, -1)
    geff += (7/8)*1*4*NRCorrection(mE[i]/T, -1)
    geff += (7/8)*1*2

  #Vector bosons
  geff += 1*2
  geff += 8*2
  geff += 2*3*NRCorrection(Constants.mW/T, +1)
  geff += 1*3*NRCorrection(Constants.mZ/T, +1)

  #Higgs
  geff += NRCorrection(Constants.mh/T, +1)

  #Return final result
  return geff



#geff computation using hadrons
def geff_Calc_belowQCD(T):

  #Initialize variables
  geff = 10.75

  #Pseudoscalar mesons
  geff += 2*NRCorrection(Constants.mpionC/T, +1)
  geff +=   NRCorrection(Constants.mpion0/T, +1)
  geff += 2*NRCorrection(Constants.mKC/T,    +1)
  geff += 2*NRCorrection(Constants.mK0/T,    +1)

  #Muon
  geff += (7/8)*1*4*NRCorrection(Constants.mmu/T, -1)

  #Return final result
  return geff



#Smoothing function for the QCD phase transition
def fsmooth(x, x0):
  if Asmoothing*(x - x0) >  200:
    return 1
  if Asmoothing*(x - x0) < -200:
    return 0
  return math.exp(Asmoothing*(x - x0))/(math.exp(Asmoothing*(x - x0)) + 1)



#Compute geff of the M sector at a given temperature
def geff_Calc_M(T):
  return geff_Calc_aboveQCD(T)*fsmooth(T, Constants.QCDscale) + (1 - fsmooth(T, Constants.QCDscale))*geff_Calc_belowQCD(T)



#Compute geff and heff
Tgefflist = 10**(np.linspace(-2, 3, 10000))
gefflist  = []

for T in Tgefflist:
  gefflist.append(geff_Calc_M(T))

hefflist = [gefflist[0]]

for i in range(0, len(Tgefflist) - 1):
  heffNext = hefflist[i] + (3/Tgefflist[i]*(gefflist[i] - hefflist[i]) + 3/4*(gefflist[i + 1] - gefflist[i])/(Tgefflist[i + 1] - Tgefflist[i]))*(Tgefflist[i + 1] - Tgefflist[i])
  hefflist.append(heffNext)

if makePlot:
  plt.plot(Tgefflist, gefflist)
  plt.plot(Tgefflist, hefflist)

  plt.xlabel(r"$T$",      fontsize=16)
  plt.ylabel(r"$g_\ast$", fontsize=16)

  plt.xscale('log')
  plt.yscale('log')

  plt.tick_params(axis='both', which='major', labelsize=16)

  plt.savefig("geff.pdf", bbox_inches='tight')
  plt.clf()



#Interpolation
def geff(T):
  return Generic.listInterpolation(T, Tgefflist, gefflist)

def heff(T):
  return Generic.listInterpolation(T, Tgefflist, hefflist)

def gstar(T):
  deltaT = 0.0001*T
  dhdT   = (heff(T + deltaT) - heff(T - deltaT))/(2*deltaT)
  return heff(T)**2/geff(T)*(1 + T/(3*heff(T))*dhdT)**2



#dt/dx
def dtdx(x, mRef):
  T = mRef/x
  return (45/(4*math.pi**3))**0.5*gstar(T)**0.5/heff(T)*Constants.mPl/(mRef*T)




