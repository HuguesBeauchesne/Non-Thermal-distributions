########
# main #
########

### Explanations ###
#This program evolves the phase-space distributions of the particles A and B.

#Comments
# - This code is written in Python3 format and should be executed as such.

#Import standard
import math, multiprocessing, os, scipy, shutil, sys
import numpy as np
from scipy  import special
from bisect import bisect_left
np.set_printoptions(threshold=sys.maxsize)

#Import custom
import Constants, Decay, Generic, Geff, IntegratedDensities_TnotTp, Parallelization, PlotMaker, Refiner, Scattering, Wcalc, Writer



#Physical parameters
mA              = 200.0              #Mass of phi_A
mB              = 198.0              #Mass of phi_B
mSM             = Constants.mh       #Mass of the Higgs boson
mRef            = 200.0              #Reference mass m_0
GammaB          = float(sys.argv[3]) #Width of phi_B
M2AAAA          = 0.0000             #Amplitude square for phi_A phi_A -> phi_A phi_A
M2AABB          = 0.0100             #Amplitude square for phi_A phi_A -> phi_B phi_B
M2BBBB          = 0.0000             #Amplitude square for phi_B phi_B -> phi_B phi_B
M2SMSMBB        = 0.0001             #Amplitude square for SM    SM    -> phi_B phi_B
AdjustDM        = True

#Comoving/normal momenta and refiner
anA             = 1                  #1: Comoving momenta for phi_A. 0: Regular momenta for phi_A.
anB             = 0                  #1: Comoving momenta for phi_B. 0: Regular momenta for phi_B.
anSM            = 0                  #1: Comoving momenta for SM.    0: Regular momenta for SM.
RefineA         = False              #Refinement of the phi_A grid
RefineB         = True               #Refinement of the phi_B grid
RefineFreq      = 1                  #Frequencey of the grid refinement

#Simulation parameters (recommended settings in ())
nameFolder      = sys.argv[2]        #Name of output folder (R1)
maxProc         = int(sys.argv[1])   #Number of threads to use (7)
nbins           = 400                #Number of bins for the W matrices (400)
nbinsCovariant  = 2001               #Number of bins for the momenta (2001)
nPointScan      = 50                 #Period of W sampling (50)
epsilon         = 0.001              #Precision tolerance for evolution by one step (0.001)
x0              = 5.0                #Starting x for the MB computation (5.0)
OmegaStart      = 100.               #Starting Omega for the no MB computation (100.)
alpha           = 1e-25              #Determination of the maximum initial momenta (1e-25)
Agrid           = 5.0                #Parameter to control the density of points on the W grid (5.0)
eFoldmin1       = 7.0                #MB: When to assume thermal equilibrium is broken (7.0)
eFoldmin2       = 7.0                #MB: When to assume chemical equilibrium is broken (7.0)
maxdx           = 0.1                #Max dx size (0.1)
epstermination  = 1e-3               #Tolerance for late time flatness (1e-3)
nNoChangeMax    = 30                 #Number of Delta x = 10 below tolerance to stop computation (30)
xTerminationMax = 5000.0             #Max x (5000.0)
maxProcW        = 128                #Extra condition on the maximum number of processes (128)
mReport         = False              #Report results (False)
ratioTrackB     = 2                  #When to stop using an adaptive grid for phi_B (2)

#Information storage
os.chdir('Results')
if os.path.isdir(nameFolder):
  shutil.rmtree(nameFolder)
os.mkdir(nameFolder)
os.chdir(nameFolder)
os.mkdir('Plots')



#Maxwell-Boltzmann computation with different temperatures
tempInfo = IntegratedDensities_TnotTp.DMabundanceMB(x0, mA, mB, mSM, mRef, M2AABB, M2SMSMBB, GammaB, eFoldmin1, eFoldmin2, AdjustDM)
xListMBDT, YAListMBDT, YBListMBDT, YAEqListMBDT, YBEqListMBDT, TAListMBDT, TBListMBDT, x0New, rescale, aScaleList, converged = tempInfo
PlotMaker.PlotY(xListMBDT, YAListMBDT, YAEqListMBDT, "YA_MBDT.pdf")
PlotMaker.PlotY(xListMBDT, YBListMBDT, YBEqListMBDT, "YB_MBDT.pdf")
Writer.WriteItem(xListMBDT,  "xListMBDT.txt" )
Writer.WriteItem(aScaleList, "aScaleList.txt")
Writer.WriteItem(YAListMBDT, "YAListMBDT.txt")
Writer.WriteItem(YBListMBDT, "YBListMBDT.txt")
Writer.WriteItem(TAListMBDT, "TAListMBDT.txt")
Writer.WriteItem(TBListMBDT, "TBListMBDT.txt")

M2AAAA *= rescale**2
M2AABB *= rescale**2
M2BBBB *= rescale**2

if not converged:
  print("Not possible to obtain a sufficiently low dark matter abundance")
  exit()



#Scatterings information
OmegaMBDT = np.array(YAListMBDT)*mA/Constants.fact
x0        = Generic.listInterpolation(OmegaStart, np.flip(OmegaMBDT), np.flip(np.array(xListMBDT)))

aScaleList = np.array(aScaleList)/Generic.listInterpolation(x0, xListMBDT, aScaleList)

TAtemp     = Generic.listInterpolation(x0, xListMBDT, TAListMBDT)
TBtemp     = Generic.listInterpolation(x0, xListMBDT, TBListMBDT)

pAcmax = (-2*mA*TAtemp*np.log(alpha) + (TAtemp*np.log(alpha))**2)**0.5
pBcmax = pAcmax

tempX        = np.linspace(0.001, 1, nbins)
pAGrid       = pAcmax*np.sinh(Agrid*tempX)/np.sinh(Agrid)
pBGrid       = pBcmax*np.sinh(Agrid*tempX)/np.sinh(Agrid)
pSMGrid      = pBcmax*np.sinh(Agrid*tempX)/np.sinh(Agrid)
pAComovGrid  = np.linspace(0.0, pAcmax, nbinsCovariant)
pBComovGrid  = np.linspace(0.0, pBcmax, nbinsCovariant)
pSMComovGrid = np.linspace(0.0, pBcmax, nbinsCovariant)

print(pAcmax, pBcmax)
print(pAComovGrid)

WAAtoAA   = Wcalc.Wparallelization(pAGrid,  pAGrid,  pAGrid, mA,  mA,  mA,  mA, M2AAAA,   False, 2, 1, min(maxProc, maxProcW))
WAAtoBB   = Wcalc.Wparallelization(pAGrid,  pAGrid,  pBGrid, mA,  mA,  mB,  mB, M2AABB,   False, 1, 1, min(maxProc, maxProcW))
WBBtoAA   = Wcalc.Wparallelization(pBGrid,  pBGrid,  pAGrid, mB,  mB,  mA,  mA, M2AABB,   True , 1, 1, min(maxProc, maxProcW))
WABtoAB   = Wcalc.Wparallelization(pAGrid,  pBGrid,  pBGrid, mA,  mB,  mA,  mB, M2AABB,   False, 2, 1, min(maxProc, maxProcW))
WBAtoBA   = Wcalc.Wparallelization(pBGrid,  pAGrid,  pAGrid, mB,  mA,  mB,  mA, M2AABB,   False, 1, 1, min(maxProc, maxProcW))
WBBtoBB   = Wcalc.Wparallelization(pBGrid,  pBGrid,  pBGrid, mB,  mB,  mB,  mB, M2BBBB,   True , 1, 1, min(maxProc, maxProcW))
WSMSMtoBB = Wcalc.Wparallelization(pSMGrid, pSMGrid, pBGrid, mSM, mSM, mB,  mB, M2SMSMBB, True,  1, 1, min(maxProc, maxProcW))
WSMBtoSMB = Wcalc.Wparallelization(pSMGrid, pBGrid,  pBGrid, mSM, mB,  mSM, mB, M2SMSMBB, False, 1, 1, min(maxProc, maxProcW))

#Initial conditions
aScale      = 1.0
x           = x0
T           = mRef/x
xLastCheck  = x0
nStep       = 0
dx          = 0.001
dxnext      = 0.001
nNoChange   = 0
rhoPrevious = 10000.
rhoNow      = 0.
deltarho    = 10000.

YA = Generic.listInterpolation(x, xListMBDT, YAListMBDT)
YB = Generic.listInterpolation(x, xListMBDT, YBListMBDT)

EATempGrid = (pAComovGrid**2 + mA**2)**0.5
EBTempGrid = (pBComovGrid**2 + mB**2)**0.5

fA = np.exp((mA - EATempGrid)/TAtemp)
fB = np.exp((mB - EBTempGrid)/TBtemp)

fA *= Generic.Entropy(T)*YA/Generic.ComovingDensityCalc(fA, pAComovGrid, 2)
fB *= Generic.Entropy(T)*YB/Generic.ComovingDensityCalc(fB, pBComovGrid, 1)

fAmin = fA[-1]
fBmin = fB[-1]



#Compute time derivatives of densities
def dfAdxCalc(fAint, fBint, aScaleInt, T):

  #Derivatives
  dfAdt = np.full([len(fA)], 0.0)

  if __name__ == "__main__":
    dfAdt += Parallelization.parC(pAComovGrid, pAComovGrid, pAComovGrid, pAComovGrid, pAGrid, pAGrid, pAGrid, pAGrid, anA, anA, anA, anA, mA, mA, mA, mA, fAint, fAint, fAint, fAint, WAAtoAA, aScaleInt, nPointScan, T, 0, maxProc)
    dfAdt += Parallelization.parC(pBComovGrid, pBComovGrid, pAComovGrid, pAComovGrid, pBGrid, pBGrid, pAGrid, pAGrid, anB, anB, anA, anA, mB, mB, mA, mA, fBint, fBint, fAint, fAint, WBBtoAA, aScaleInt, nPointScan, T, 0, maxProc)
    dfAdt += Parallelization.parC(pBComovGrid, pAComovGrid, pBComovGrid, pAComovGrid, pBGrid, pAGrid, pBGrid, pAGrid, anB, anA, anB, anA, mB, mA, mB, mA, fBint, fAint, fBint, fAint, WBAtoBA, aScaleInt, nPointScan, T, 0, maxProc)

  #Momentum shift
  if anA == 0:
    pshift = Generic.HubbleConstant(T)*pAComovGrid*Generic.derivDistribution(pAComovGrid, fAint)
  else:
    pshift = 0

  #Total
  return (dfAdt + pshift)*Geff.dtdx(x, mRef)

def dfBdxCalc(fAint, fBint, aScaleInt, T):

  #Derivatives
  dfBdt = np.full([len(fB)], 0.0)

  if __name__ == "__main__":
    dfBdt += Parallelization.parC(pBComovGrid,  pBComovGrid,  pBComovGrid,  pBComovGrid, pBGrid,  pBGrid,  pBGrid,  pBGrid, anB,  anB,  anB,  anB, mB,  mB,  mB,  mB, fBint, fBint, fBint, fBint, WBBtoBB,   aScaleInt, nPointScan, T, 0, maxProc)
    dfBdt += Parallelization.parC(pAComovGrid,  pAComovGrid,  pBComovGrid,  pBComovGrid, pAGrid,  pAGrid,  pBGrid,  pBGrid, anA,  anA,  anB,  anB, mA,  mA,  mB,  mB, fAint, fAint, fBint, fBint, WAAtoBB,   aScaleInt, nPointScan, T, 0, maxProc)
    dfBdt += Parallelization.parC(pAComovGrid,  pBComovGrid,  pAComovGrid,  pBComovGrid, pAGrid,  pBGrid,  pAGrid,  pBGrid, anA,  anB,  anA,  anB, mA,  mB,  mA,  mB, fAint, fBint, fAint, fBint, WABtoAB,   aScaleInt, nPointScan, T, 0, maxProc)
    dfBdt += Parallelization.parC(pSMComovGrid, pSMComovGrid, pBComovGrid,  pBComovGrid, pSMGrid, pSMGrid, pBGrid,  pBGrid, anSM, anSM, anB,  anB, mSM, mSM, mB,  mB, [],    [],    fBint, fBint, WSMSMtoBB, aScaleInt, nPointScan, T, 1, maxProc)
    dfBdt += Parallelization.parC(pSMComovGrid, pBComovGrid,  pSMComovGrid, pBComovGrid, pSMGrid, pBGrid,  pSMGrid, pBGrid, anSM, anB,  anSM, anB, mSM, mB,  mSM, mB, [],    fBint, [],    fBint, WSMBtoSMB, aScaleInt, nPointScan, T, 2, maxProc)

  #Bdecay
  dfBdtDecay = Decay.decayderiv(pBComovGrid, mB, fBint, GammaB, aScaleInt, anB, T)

  #Momentum shift
  if anB == 0:
    pshift = Generic.HubbleConstant(T)*pBComovGrid*Generic.derivDistribution(pBComovGrid, fBint)
  else:
    pshift = 0

  #Total
  return (dfBdt + dfBdtDecay + pshift)*Geff.dtdx(x, mRef)



#Evolution
while True:

  #Evolution
  kA1 = dfAdxCalc(fA, fB, aScale, mRef/x)
  kB1 = dfBdxCalc(fA, fB, aScale, mRef/x)

  while True:

    #One step of dx
    dx  = dxnext
    fA1 = fA + kA1*dx
    fB1 = fB + kB1*dx

    #Second order Runge-Kutta
    fA2 = fA + kA1*dx/2
    fB2 = fB + kB1*dx/2

    aScale2 = Generic.listInterpolation(x + dx/2, xListMBDT, aScaleList)

    fA3 = fA + dfAdxCalc(fA2, fB2, aScale2, mRef/(x + dx/2))*dx
    fB3 = fB + dfBdxCalc(fA2, fB2, aScale2, mRef/(x + dx/2))*dx

    #Error
    errA = np.max(np.array([abs(fA1[i] - fA3[i])/max(abs(fA1[i] + fA3[i]), 1e-15*fAmin) for i in range(0, len(fA1))]))
    errB = np.max(np.array([abs(fB1[i] - fB3[i])/max(abs(fB1[i] + fB3[i]), 1e-15*fBmin) for i in range(0, len(fB1))]))

    TE     = max(errA, errB) if YA*mA/Constants.fact > ratioTrackB*Constants.OmegaDM else errA
    dxnext = min(0.9*dx*(epsilon/TE)**0.33, maxdx, 1.1*dx) if TE > 0.0 else min(maxdx, 1.1*dx)

    if mReport:
      print("Old step size", TE, dx)
      print("New step size", TE, dxnext)

    if TE < epsilon:
      fA = np.copy(fA3)
      fB = np.copy(fB3)
      break

  #Safety in case of negative values
  fA = np.abs(fA)
  fB = np.abs(fB)

  #Update information
  aScale  = Generic.listInterpolation(x + dx, xListMBDT, aScaleList)
  x      += dx
  T       = mRef/x

  #Comoving density
  ComovDensityA = Generic.ComovingDensityCalc(fA, pAComovGrid, 2)
  ComovDensityB = Generic.ComovingDensityCalc(fB, pBComovGrid, 1)

  #Density
  densityA = ComovDensityA/aScale**(3*anA)
  densityB = ComovDensityB/aScale**(3*anB)

  sEntropy = Generic.Entropy(T)

  YA   = densityA/sEntropy
  YB   = densityB/sEntropy
  YAEq = Generic.NeqT(T, mA, 2)/sEntropy
  YBEq = Generic.NeqT(T, mB, 1)/sEntropy

  #Density of dark matter today if processes froze-out at that time
  OmegaA = YA*mA/Constants.fact
  OmegaB = YB*mB/Constants.fact

  #Ratio to standard computation
  ratioA = YA/Generic.listInterpolation(x, xListMBDT, YAListMBDT)
  ratioB = YB/Generic.listInterpolation(x, xListMBDT, YBListMBDT)

  #Report
  if mReport:
    print()
    print("Report:")
    print(x, deltarho, OmegaA, OmegaB, YA, YB, ratioA, ratioB, pAComovGrid[-1], pBComovGrid[-1])
    print(fA)
    print(fB)
    print()

  Writer.WriteProgress(x, deltarho, OmegaA, OmegaB, pAComovGrid[-1], pBComovGrid[-1], ratioA, ratioB)

  #Write information
  Writer.WriteItem(x,      "xList.txt"     )
  Writer.WriteItem(OmegaA, "OmegaAList.txt")
  Writer.WriteItem(OmegaB, "OmegaBList.txt")
  Writer.WriteItem(YA,     "YAList.txt"    )
  Writer.WriteItem(YB,     "YBList.txt"    )
  Writer.WriteItem(YAEq,   "YAEqList.txt"  )
  Writer.WriteItem(YBEq,   "YBEqList.txt"  )

  if nStep%10 == 0:
    Writer.WriteList(fA,          "fA.txt"         )
    Writer.WriteList(fB,          "fB.txt"         )
    Writer.WriteList(pAComovGrid, "pAComovGrid.txt")
    Writer.WriteList(pBComovGrid, "pBComovGrid.txt")

  #Termination conditions
  if x >= xLastCheck + 10:
    xLastCheck  = x
    rhoNow      = OmegaA
    deltarho    = (rhoPrevious - rhoNow)/(rhoPrevious + rhoNow)
    if deltarho < epstermination:
      nNoChange += 1
    rhoPrevious = rhoNow

  if nNoChange > nNoChangeMax or x > xTerminationMax or dxnext < 1e-6:
    break

  #Refinement
  if nStep%RefineFreq == 0:
    if RefineA:
      pAComovGrid, fA = Refiner.GridRefiner(pAComovGrid, fA, fAmin, 0.0)
    if RefineB:
      pBMaxMin = 1.2*(2*mA*abs(mA - mB))**0.5
      pBComovGrid, fB = Refiner.GridRefiner(pBComovGrid, fB, fBmin, pBMaxMin)

  #Increment step number
  nStep += 1




