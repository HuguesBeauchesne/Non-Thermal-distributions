##############
# Plot maker #
##############

### Explanations ###
#This program creates plots of the time evolution of various quantities.

#Comments
# - None

#Import standard
import math, os, scipy, sys
import numpy as np
import scipy.integrate
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable



#Plot Omega
def PlotOmega(xList, OmegaList, PlotName):

  plt.plot(xList, OmegaList)

  plt.xlabel(r"$x$", fontsize=16)
  plt.ylabel(r"$\Omega$", fontsize=16)

  plt.yscale('log')

  plt.tick_params(axis='both', which='major', labelsize=16)

  plt.savefig(PlotName, bbox_inches='tight')
  plt.clf()



#Plot Y's
def PlotY(xList, YList, YEqList, PlotName):

  plt.plot(xList, YList, color='blue')
  plt.plot(xList, YEqList, color='green')

  plt.xlabel(r"$x$", fontsize=16)
  plt.ylabel(r"$Y$", fontsize=16)

  plt.yscale('log')

  plt.ylim([0.9*min(YList), 1.1*max(YList)])

  plt.tick_params(axis='both', which='major', labelsize=16)

  plt.savefig(PlotName, bbox_inches='tight')
  plt.clf()



#Plot all Y's
def PlotYMultiple(xListList, yListList, PlotName):

  for i in range(0, len(xListList)):
    plt.plot(xListList[i], yListList[i])

  plt.xlabel(r"$x$", fontsize=16)
  plt.ylabel(r"$Y$", fontsize=16)

  plt.yscale('log')

  plt.ylim([0.9*min(yListList[0]), 1.1*max(yListList[0])])

  plt.tick_params(axis='both', which='major', labelsize=16)

  plt.savefig(PlotName, bbox_inches='tight')
  plt.clf()



#Plot densities at different times
def plotf(pComovGridList, fListList, xList, PlotName):

  for i in range(0, len(fListList)):
    plt.plot(pComovGridList[i][:-1], fListList[i][:-1])

  plt.xlabel(r"$p_c$ [GeV]", fontsize=16)
  plt.ylabel(r"$f$", fontsize=16)

  plt.yscale('log')

  plt.tick_params(axis='both', which='major', labelsize=16)

  plt.savefig(PlotName, bbox_inches='tight')
  plt.clf()




