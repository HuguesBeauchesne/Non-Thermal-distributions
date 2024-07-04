########################
# Integrated densities #
########################

### Explanations ###
#Evolve the integrated densities without assuming all temperatures are degenerate

#Comments
# - rescaleMax is the maximum value that we allow to rescale the couplings. It is set much larger than the unitarity limits to insure no useful information is lost.

#Import generic
import math, scipy
import numpy as np

#Import custom
import Constants, CrossSections, Generic, Geff, ThermalAv



#Class to compute the cross section of relevant scatterings
class CSInfo_Class(ThermalAv.CSInfo_Class_General):

  #Inheritance
  def __init__(self):
    super().__init__()

  #Function to generate the functions to compute the cross sections
  def update_CS(self, mA, mB, mSM, mRef, M2AABB, M2SMSMBB, rescaleInt):

    #List of temperatures
    self.TlistCS1A  = np.linspace(0.004, 40, 5000)
    self.TlistCS1B  = np.linspace(0.004, 40, 5000)
    self.TlistCS2A  = np.linspace(0.8,   40,   50)
    self.TlistCS2B  = np.linspace(0.8,   40,   50)
    self.TlistCS2SM = np.linspace(0.8,   40,   50)

    #Common processes
    self.functions_generator("AA_BB",   CrossSections.AA_BB_CS,   CrossSections.AA_BB_CS_t,   [self.TlistCS1A],                  [M2AABB],   mA,  mA, mB,  mB,  ["CS", "Eplus"],              rescaleInt, 2)
    self.functions_generator("BB_AA",   CrossSections.BB_AA_CS,   CrossSections.BB_AA_CS_t,   [self.TlistCS1B],                  [M2AABB],   mB,  mB, mA,  mA,  ["CS", "Eplus"],              rescaleInt, 2)
    self.functions_generator("AB_AB",   CrossSections.AB_AB_CS,   CrossSections.AB_AB_CS_t,   [self.TlistCS2A, self.TlistCS2B],  [M2AABB],   mA,  mB, mA,  mB,  ["Eout1mEin1", "Eout2mEin2"], rescaleInt, 2)
    self.functions_generator("BB_SMSM", CrossSections.BB_SMSM_CS, CrossSections.BB_SMSM_CS_t, [self.TlistCS1B],                  [M2SMSMBB], mB,  mB, mSM, mSM, ["CS", "Eplus"],              1,          2)
    self.functions_generator("SMB_SMB", CrossSections.SMB_SMB_CS, CrossSections.SMB_SMB_CS_t, [self.TlistCS2SM, self.TlistCS2B], [M2SMSMBB], mSM, mB, mSM, mB,  ["Eout2mEin2"],               1,          2)



#Computation of the dark matter abundance with Maxwell-Boltzmann distributions
def DMabundanceMB(x0, mA, mB, mSM, mRef, M2AABB, M2SMSMBB, GammaB, eFoldmin1, eFoldmin2, AdjustDM):

  #Compute T from rhoY
  listTrhoY    = []
  listrhoYA_nA = []
  listrhoYB_nB = []

  for i in range(-800, 600):
    Ttemp = 10**(i/100)
    listTrhoY.append(Ttemp)
    listrhoYA_nA.append(mA*Generic.K1on2(mA/Ttemp) + 3*Ttemp)
    listrhoYB_nB.append(mB*Generic.K1on2(mB/Ttemp) + 3*Ttemp)

  def TfromrhoYA(nA, rhoYA):
    if rhoYA > mA*nA and nA > 0:
      return max(Generic.listInterpolation(rhoYA/nA, listrhoYA_nA, listTrhoY), 1e-5)
    else:
      return 1e-5

  def TfromrhoYB(nB, rhoYB):
    if rhoYB > mB*nB and nB > 0:
      return max(Generic.listInterpolation(rhoYB/nB, listrhoYB_nB, listTrhoY), 1e-5)
    else:
      return 1e-5

  #Compute the thermally averaged cross sections
  print("Beginning cross section computations:")
  CSInfo = CSInfo_Class()
  CSInfo.update_CS(mA, mB, mSM, mRef, M2AABB, M2SMSMBB, 1)
  print("Cross section computations done!")
  print("")

  #Derivatives
  def derivInfo(YAint, YBint, rhoYAint, rhoYBint, x, chemicalEq=False):

    #Initialize variables
    dYAdt    = 0.0
    dYBdt    = 0.0
    drhoYAdt = 0.0
    drhoYBdt = 0.0
    Tint     = mRef/x
    TAint    = TfromrhoYA(YAint, rhoYAint)
    TBint    = TfromrhoYB(YBint, rhoYBint)
    dtdx     = Geff.dtdx(x, mRef)

    if chemicalEq:
      factchemicalEq = 0
    else:
      factchemicalEq = 1

    #Expansion of the Universe
    H         = Generic.HubbleConstant(Tint)
    drhoYAdt -= 3*H*TAint*YAint
    drhoYBdt -= 3*H*TBint*YBint

    #A A -> B B
    dYAdt    -= 2*Generic.Entropy(Tint)*CSInfo.AA_BB_CS(TAint)   *YAint**2*factchemicalEq
    dYBdt    += 2*Generic.Entropy(Tint)*CSInfo.AA_BB_CS(TAint)   *YAint**2*factchemicalEq
    drhoYAdt -=   Generic.Entropy(Tint)*CSInfo.AA_BB_Eplus(TAint)*YAint**2*factchemicalEq
    drhoYBdt +=   Generic.Entropy(Tint)*CSInfo.AA_BB_Eplus(TAint)*YAint**2*factchemicalEq

    dYAdt    += 2*Generic.Entropy(Tint)*CSInfo.BB_AA_CS(TBint)   *YBint**2*factchemicalEq
    dYBdt    -= 2*Generic.Entropy(Tint)*CSInfo.BB_AA_CS(TBint)   *YBint**2*factchemicalEq
    drhoYAdt +=   Generic.Entropy(Tint)*CSInfo.BB_AA_Eplus(TBint)*YBint**2*factchemicalEq
    drhoYBdt -=   Generic.Entropy(Tint)*CSInfo.BB_AA_Eplus(TBint)*YBint**2*factchemicalEq

    #A B -> A B
    drhoYAdt += Generic.Entropy(Tint)*CSInfo.AB_AB_Eout1mEin1(TAint, TBint)*YAint*YBint*factchemicalEq
    drhoYBdt += Generic.Entropy(Tint)*CSInfo.AB_AB_Eout2mEin2(TAint, TBint)*YAint*YBint*factchemicalEq

    #B B -> SM SM
    dYBdt    -= 2*Generic.Entropy(Tint)*CSInfo.BB_SMSM_CS(TBint)   *YBint**2
    drhoYBdt -=   Generic.Entropy(Tint)*CSInfo.BB_SMSM_Eplus(TBint)*YBint**2

    dYBdt    += 2*Generic.Entropy(Tint)*CSInfo.BB_SMSM_CS(Tint)    *Generic.YeqT(Tint, mB, 1)**2
    drhoYBdt +=   Generic.Entropy(Tint)*CSInfo.BB_SMSM_Eplus(Tint) *Generic.YeqT(Tint, mB, 1)**2

    #SM B -> SM B
    drhoYBdt  += Generic.Entropy(Tint)*CSInfo.SMB_SMB_Eout2mEin2(Tint, TBint)*YBint*Generic.YeqT(Tint, mSM, 1)

    #B -> SM SM
    dYBdt    -= GammaB*Generic.K1on2(mB/TBint)*YBint
    drhoYBdt -= GammaB*mB*YBint

    dYBdt    += GammaB*Generic.K1on2(mB/Tint)*Generic.YeqT(Tint, mB, 1)
    drhoYBdt += GammaB*mB*Generic.YeqT(Tint, mB, 1)

    #Return derivative information
    return dYAdt*dtdx, drhoYAdt*dtdx, dYBdt*dtdx, drhoYBdt*dtdx

  #Compute DM abundance
  def DMAbundanceComputer(rescaleInt, printInfo):

    #Update cross section information
    CSInfo.update_CS(mA, mB, mSM, mRef, M2AABB, M2SMSMBB, rescaleInt)

    #Initial conditions
    dx0     = 0.00001
    epsilon = 0.001

    T      = mRef/x0
    TA     = T
    TB     = T
    x      = x0
    YA     = Generic.YeqT(T, mA, 2)
    YB     = Generic.YeqT(T, mB, 1)
    rhoYA  = Generic.rhoYeqT(T, mA, 2)
    rhoYB  = Generic.rhoYeqT(T, mB, 1)
    x0New  = x0
    dx     = dx0
    dxnext = dx
    aScale = 1

    ThermaleqNeverBroken  = True
    ChemicaleqNeverBroken = True

    YAList     = []
    YBList     = []
    YAEqList   = []
    YBEqList   = []
    TAList     = []
    TBList     = []
    xList      = []
    aScaleList = []

    #Evolution
    while True:

      #Thermal equilibrium case
      if ThermaleqNeverBroken:

        x += 0.01
        T  = mRef/x
        TA = T
        TB = T

        YA    = Generic.YeqT(   T, mA, 2)
        YB    = Generic.YeqT(   T, mB, 1)
        rhoYA = Generic.rhoYeqT(T, mA, 2)
        rhoYB = Generic.rhoYeqT(T, mB, 1)

        if printInfo:
          print(x, YA, YB, T, TA, TB, "Thermal equilibrium!")

        if np.log(abs(Generic.Entropy(T)*CSInfo.BB_SMSM_CS(T)*YB)) < np.log(Generic.HubbleConstant(T)) + eFoldmin1:
          ThermaleqNeverBroken = False

      #Chemical equilibrium case
      elif ChemicaleqNeverBroken:

        kA1_1, kA2_1, kB1_1, kB2_1 = derivInfo(YA, YB, rhoYA, rhoYB, x, chemicalEq=True)

        denom1   = 2*Generic.YonY(mA, mRef, TA) + Generic.YonY(mB, mRef, TB)
        factA1_1 = 2*Generic.YonY(mA, mRef, TA)/denom1
        factB1_1 =   Generic.YonY(mB, mRef, TB)/denom1

        denom2   = 2*Generic.rho1onrho2(mA, mRef, TA) + Generic.rho1onrho2(mB, mRef, TB)
        factA2_1 = 2*Generic.rho1onrho2(mA, mRef, TA)/denom2
        factB2_1 =   Generic.rho1onrho2(mB, mRef, TB)/denom2

        while True:

          #First order Runge-Kutta
          dx        = dxnext
          Ytotal1   = YA    + YB    + (kA1_1 + kB1_1)*dx
          rhototal1 = rhoYA + rhoYB + (kA2_1 + kB2_1)*dx

          YA1    = Ytotal1  *factA1_1
          YB1    = Ytotal1  *factB1_1
          rhoYA1 = rhototal1*factA2_1
          rhoYB1 = rhototal1*factB2_1

          #Second order Runge-Kutta
          Ytotal2   = YA    + YB    + (kA1_1 + kB1_1)*dx/2
          rhototal2 = rhoYA + rhoYB + (kA2_1 + kB2_1)*dx/2

          YA2    = Ytotal2  *factA1_1
          YB2    = Ytotal2  *factB1_1
          rhoYA2 = rhototal2*factA2_1
          rhoYB2 = rhototal2*factB2_1

          TA2 = max(TfromrhoYA(YA2, rhoYA2), 0.000000001)
          TB2 = TA2

          kA1_2, kA2_2, kB1_2, kB2_2 = derivInfo(YA2, YB2, rhoYA2, rhoYB2, x + dx/2, chemicalEq=True)

          Ytotal3   = YA    + YB    + (kA1_2 + kB1_2)*dx
          rhototal3 = rhoYA + rhoYB + (kA2_2 + kB2_2)*dx

          denom3   = 2*Generic.YonY(mA, mRef, TA2) + Generic.YonY(mB, mRef, TB2)
          factA1_2 = 2*Generic.YonY(mA, mRef, TA2)/denom3
          factB1_2 =   Generic.YonY(mB, mRef, TB2)/denom3

          denom4   = 2*Generic.rho1onrho2(mA, mRef, TA2) + Generic.rho1onrho2(mB, mRef, TB2)
          factA2_2 = 2*Generic.rho1onrho2(mA, mRef, TA2)/denom4
          factB2_2 =   Generic.rho1onrho2(mB, mRef, TB2)/denom4

          YA3    = Ytotal3  *factA1_2
          YB3    = Ytotal3  *factB1_2
          rhoYA3 = rhototal3*factA2_2
          rhoYB3 = rhototal3*factB2_2

          #Error
          TE     = max(np.max(abs((YA1 - YA3)/YA)), np.max(abs((rhoYA1 - rhoYA3)/rhoYA)))
          dxnext = min(0.9*dx*(epsilon/TE)**0.33, 1.1*dx, 0.0001*x)

          if TE < epsilon:

            YA    = YA3
            YB    = YB3
            rhoYA = rhoYA3
            rhoYB = rhoYB3

            TA = max(TfromrhoYA(YA, rhoYA), 0.000000001)
            TB = TA

            x += dx
            T  = mRef/x

            aScale += Generic.HubbleConstant(T)*aScale*Geff.dtdx(x, mRef)*dx

            if printInfo:
              print(x, YA, YB, T, TA, TB, "Chemical equilibrium!")

            YAList.append(YA)
            YBList.append(YB)
            YAEqList.append(Generic.YeqT(T, mA, 2))
            YBEqList.append(Generic.YeqT(T, mB, 1))
            TAList.append(TA)
            TBList.append(TB)
            xList.append(x)
            aScaleList.append(aScale)

            break

        if np.log(np.abs(Generic.Entropy(T)*CSInfo.AA_BB_CS(TA)*YA)) < np.log(Generic.HubbleConstant(T)) + eFoldmin2:
          ChemicaleqNeverBroken = False
          x0New = x
          dx    = dx0 

      #No equilibrium case
      else:

        kA1_1, kA2_1, kB1_1, kB2_1 = derivInfo(YA, YB, rhoYA, rhoYB, x)

        while True:

          #First order Runge-Kutta
          dx     = dxnext
          YA1    = YA    + kA1_1*dx
          YB1    = YB    + kB1_1*dx
          rhoYA1 = rhoYA + kA2_1*dx
          rhoYB1 = rhoYB + kB2_1*dx

          #Second order Runge-Kutta
          YA2    = YA    + kA1_1*dx/2
          YB2    = YB    + kB1_1*dx/2
          rhoYA2 = rhoYA + kA2_1*dx/2
          rhoYB2 = rhoYB + kB2_1*dx/2

          kA1_2, kA2_2, kB1_2, kB2_2 = derivInfo(YA2, YB2, rhoYA2, rhoYB2, x + dx/2)

          YA3    = YA    + kA1_2*dx
          YB3    = YB    + kB1_2*dx
          rhoYA3 = rhoYA + kA2_2*dx
          rhoYB3 = rhoYB + kB2_2*dx

          #Error
          errA1 = np.max(abs((YA1 - YA3)/YA))
          errA2 = np.max(abs((rhoYA1 - rhoYA3)/rhoYA))
          errB1 = np.max(abs((YB1 - YB3)/YB))
          errB2 = np.max(abs((rhoYB1 - rhoYB3)/rhoYB))

          TE = max(errA1, errA2, errB1, errB2)

          dxnext = min(0.9*dx*(epsilon/TE)**0.33, 1.1*dx, 0.0001*x)

          if TE < epsilon:

            YA     = YA3
            YB     = YB3
            rhoYA  = rhoYA3
            rhoYB  = rhoYB3
            x     += dx

            T  = mRef/x
            TA = TfromrhoYA(YA, rhoYA)
            TB = TfromrhoYB(YB, rhoYB)

            aScale += Generic.HubbleConstant(T)*aScale*Geff.dtdx(x, mRef)*dx

            YAList.append(YA)
            YBList.append(YB)
            YAEqList.append(Generic.YeqT(T, mA, 2))
            YBEqList.append(Generic.YeqT(T, mB, 1))
            TAList.append(TA)
            TBList.append(TB)
            xList.append(x)
            aScaleList.append(aScale)

            if printInfo:
              print(x, YA, YB, T, TA, TB)

            break

        #Break condition
        if x > x0New + 0.5 and abs(kA1_1/YA) < 1E-6:
          break

    #Return results
    return xList, YAList, YBList, YAEqList, YBEqList, TAList, TBList, x0New, rescaleInt, aScaleList

  #Adjust DM abundance
  rescale     = 1.0
  stepRescale = 0.01
  YMGoal      = Constants.fact*Constants.OmegaDM
  info3       = DMAbundanceComputer(rescale, False)
  rescaleMax  = 4*(8*2**0.5*math.pi)/M2AABB**0.5
  nTry1       = 0

  while AdjustDM:

    #Check precision
    if abs(mA*info3[1][-1] - YMGoal)/YMGoal < 1.0e-4:
      converged = True
      break

    #Check if possible considering rescale limits
    if mA*info3[1][-1] > YMGoal and abs(rescale) > rescaleMax:
      converged = False
      break

    #Check for convergence
    if nTry1 > 1000:
      converged = False
      break

    #Newton's method
    info1 = info3
    info2 = DMAbundanceComputer(rescale + stepRescale, False)

    YMint  =  mA*info1[1][-1]
    dYMint = (mA*info2[1][-1] - mA*info1[1][-1])/stepRescale

    #Damping
    damping = 1.0

    while True:

      nTry1 += 1

      rescaleTemp = min(abs(rescale - damping*(YMint - YMGoal)/dYMint), 1.1*rescaleMax)
      info3       = DMAbundanceComputer(rescaleTemp, False)

      print("Info 1", abs(mA*info1[1][-1] - YMGoal)/YMGoal, abs(mA*info3[1][-1] - YMGoal)/YMGoal, damping, rescaleTemp)

      if abs(mA*info3[1][-1] - YMGoal)/YMGoal < abs(mA*info1[1][-1] - YMGoal)/YMGoal:
        rescale = rescaleTemp
        break
      else:
        damping *= 0.8

      if mA*info3[1][-1] > YMGoal and abs(rescaleTemp) > rescaleMax:
        rescale = rescaleTemp
        break

      if nTry1 > 1000:
        break

  #Return information on density evolution
  if AdjustDM:
    return DMAbundanceComputer(rescale, True) + (converged,)
  else:
    return DMAbundanceComputer(rescale, True) + (True,)




