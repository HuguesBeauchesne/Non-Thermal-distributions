###########
# Factors #
###########

### Explanations ###
#Miscalleneous functions

#Comments
# - None

#Import standard
import math, scipy
import numpy as np
from bisect import bisect_left

#Import custom
import Constants, Geff



#Ratio of temperature on mass below which to use asymptotic expansion
TonmMax = 0.05



#Entropy
def Entropy(T):
  return 2*math.pi**2/45*Geff.heff(T)*T**3



#Equilibrium density
def NeqT(T, m, g):
  return g*m**2*T/(2*math.pi**2)*scipy.special.kn(2, m/T)



#Equilibrium number density by entropy density
def YeqT(T, m, g):
  return NeqT(T, m, g)/Entropy(T)



#Equilibrium energy density by entropy density
def rhoYeqT(T, m, g):
  return g*m**2*T/(2*math.pi**2)*(m*scipy.special.kn(1, m/T) + 3*T*scipy.special.kn(2, m/T))/Entropy(T)



#Rho SM
def rhoSM(T):
  return math.pi**2/30*Geff.geff(T)*T**4



#Hubble constant
def HubbleConstant(T):
  return (8*math.pi*Constants.G/3*rhoSM(T))**0.5



#Y(m1, T)/Y(m2, T) = n2(m1, T)/n1(m2, T)
def YonY(m1, m2, T):
  if T/m1 > TonmMax or T/m2 > TonmMax:
     return (m1/m2)**2*scipy.special.kn(2, m1/T)/scipy.special.kn(2, m2/T)
  else:
    c1 = 1 + 15/8*(T/m1) + 105/128*(T/m1)**2 - 315/1024*(T/m1)**3 + (10395/32768)*(T/m1)**4
    c2 = 1 + 15/8*(T/m2) + 105/128*(T/m2)**2 - 315/1024*(T/m2)**3 + (10395/32768)*(T/m2)**4
    return (m1/m2)**1.5*math.exp(-(m1 -m2)/T)*c1/c2



#rho(m1, T)/rho(m2, T)
def rho1onrho2(m1, m2, T):
  if T/m1 > TonmMax or T/m2 > TonmMax:
     return (m1/m2)**2*(m1*scipy.special.kn(1, m1/T) + 3*T*scipy.special.kn(2, m1/T))/(m2*scipy.special.kn(1, m2/T) + 3*T*scipy.special.kn(2, m2/T))
  else:
    c1 = 1 + (27/8)*(T/m1) + (705/128)*(T/m1)**2 + (2625/1024)*(T/m1)**3 - (34965/32768)*(T/m1)**4
    c2 = 1 + (27/8)*(T/m2) + (705/128)*(T/m2)**2 + (2625/1024)*(T/m2)**3 - (34965/32768)*(T/m2)**4
    return (m1/m2)**2.5*math.exp(-(m1 -m2)/T)*c1/c2



#K1(x) on K2(x)
def K1on2(x):
  if 1/x < TonmMax:
    return 1 - 3/(2*x) + 15/(8*x**2) - 15/(8*x**3) + 135/(128*x**4)
  else: 
    return scipy.special.kn(1, x)/scipy.special.kn(2, x)



#K1 on K2 times K2 for different masses
def K1on22(T, s, m1, m2):

  ss = s**0.5

  if min(T/m1, T/m2) < TonmMax:
    term0  = 1
    term1  = -15/(8*m1) - 15/(8*m2) + 3/(8*ss)
    term2  = 345/(128*m1**2) + 345/(128*m2**2) + 225/(64*m1*m2) - 15/(128*ss**2) - 45/(64*m1*ss) - 45/(64*m2*ss)
    term3  = -3285/(1024*m1**3) - 3285/(1024*m2**3) - 5175/(1024*m1*m2**2) - 5175/(1024*m1**2*m2) + 105/(1024*ss**3) + 225/(1024*m1*ss**2) + 225/(1024*m2*ss**2) + 1035/(1024*m1**2*ss) + 1035/(1024*m2**2*ss) + 675/(512*m1*m2*ss)
    return ((2*m1*m2)/(math.pi*T*ss))**0.5*math.exp(-(ss - m1 - m2)/T)*(term0 + term1*T + term2*T**2 + term3*T**3)
  else: 
    return scipy.special.kn(1, ss/T)/(scipy.special.kn(2, m1/T)*scipy.special.kn(2, m2/T))



#K2 on K2 times K2 for different masses
def K2on22(T, s, m1, m2):

  ss = s**0.5

  if min(T/m1, T/m2) < TonmMax:
    term0  = 1
    term1  = -15/(8*m1) - 15/(8*m2) + 15/(8*ss)
    term2  = 345/(128*m1**2) + 345/(128*m2**2) + 225/(64*m1*m2) + 105/(128*ss**2) - 225/(64*m1*ss) - 225/(64*m2*ss)
    term3  = -3285/(1024*m1**3) - 3285/(1024*m2**3) - 5175/(1024*m1*m2**2) - 5175/(1024*m1**2*m2) - 315/(1024*ss**3) - 1575/(1024*m1*ss**2) - 1575/(1024*m2*ss**2) + 5175/(1024*m1**2*ss) + 5175/(1024*m2**2*ss) + 3375/(512*m1*m2*ss)
    result = ((2*m1*m2)/(math.pi*T*ss))**0.5*math.exp(-(ss - m1 - m2)/T)*(term0 + term1*T + term2*T**2 + term3*T**3)
  else: 
    result = scipy.special.kn(2, ss/T)/(scipy.special.kn(2, m1/T)*scipy.special.kn(2, m2/T))

  return result



#Comoving density
def ComovingDensityCalc(f, pGrid, g):
  ComovingDensity = 0.0
  for i in range(0, len(pGrid) - 1):
    ComovingDensity += g/(2*math.pi**2)*(pGrid[i + 1] - pGrid[i])*pGrid[i]**2*f[i]
  return ComovingDensity



#Interpolation 1D
def listInterpolation(x, listx, listy):

  #Find value of index to the left of the value
  idx = bisect_left(listx, x) - 1

  #Apply safety for too large values of x
  if x >= listx[-1]:
    return listy[-1]

  #Return value interpolated at linear order
  deltax1 = x - listx[idx]
  deltax2 = listx[idx + 1] - listx[idx]
  return listy[idx]*(1 - deltax1/deltax2) + listy[idx + 1]*deltax1/deltax2



#Function to retrieve the interpolated value from list in 2D
def listInterpolation2D(x1, x2, listx1, listx2, listy):

  #Find value of index to the left of the value
  idx1 = bisect_left(listx1, x1) - 1
  idx2 = bisect_left(listx2, x2) - 1

  #Apply safety for too large values of x1 or x2
  if x1 >= listx1[-1] and x2 >= listx2[-1]:
    return listy[-1, -1]

  if x1 >= listx1[-1]:
    return listy[-1, idx2]

  if x2 >= listx2[-1]:
    return listy[idx1, -1]

  #Return value interpolated at linear order
  delta1 = (x1 - listx1[idx1])/(listx1[idx1 + 1] - listx1[idx1])
  delta2 = (x2 - listx2[idx2])/(listx2[idx2 + 1] - listx2[idx2])

  t1 = listy[idx1,     idx2    ]*(1 - delta1)*(1 - delta2)
  t2 = listy[idx1 + 1, idx2    ]*(    delta1)*(1 - delta2)
  t3 = listy[idx1,     idx2 + 1]*(1 - delta1)*(    delta2)
  t4 = listy[idx1 + 1, idx2 + 1]*(    delta1)*(    delta2)

  return t1 + t2 + t3 + t4



#Function to retrieve the interpolated value from list in 3D
def listInterpolation3D(x1, x2, x3, listx1, listx2, listx3, listy):

  #Find value of index to the left of the value
  idx1 = bisect_left(listx1, x1) - 1
  idx2 = bisect_left(listx2, x2) - 1
  idx3 = bisect_left(listx3, x3) - 1

  #Apply safety for too large values of x1, x2 or x3
  if x1 >= listx1[-1] or x2 >= listx2[-1] or x3 >= listx3[-1]:
    return 0

  #Return value interpolated at linear order
  delta1 = (x1 - listx1[idx1])/(listx1[idx1 + 1] - listx1[idx1])
  delta2 = (x2 - listx2[idx2])/(listx2[idx2 + 1] - listx2[idx2])
  delta3 = (x3 - listx3[idx3])/(listx3[idx3 + 1] - listx3[idx3])

  t1 = listy[idx1    , idx2    , idx3    ]*(1 - delta1)*(1 - delta2)*(1 - delta3)
  t2 = listy[idx1 + 1, idx2    , idx3    ]*(    delta1)*(1 - delta2)*(1 - delta3)
  t3 = listy[idx1    , idx2 + 1, idx3    ]*(1 - delta1)*(    delta2)*(1 - delta3)
  t4 = listy[idx1    , idx2    , idx3 + 1]*(1 - delta1)*(1 - delta2)*(    delta3)
  t5 = listy[idx1 + 1, idx2 + 1, idx3    ]*(    delta1)*(    delta2)*(1 - delta3)
  t6 = listy[idx1 + 1, idx2    , idx3 + 1]*(    delta1)*(1 - delta2)*(    delta3)
  t7 = listy[idx1    , idx2 + 1, idx3 + 1]*(1 - delta1)*(    delta2)*(    delta3)
  t8 = listy[idx1 + 1, idx2 + 1, idx3 + 1]*(    delta1)*(    delta2)*(    delta3)

  return t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8



#Derivative of distribution
def derivDistribution(x, y):

  #Initialize variables
  deriv = np.full([len(y)], 0.0)

  #Compute derivatives at different points
  for i in range(0, len(y) - 1):
    if i == 0:
      deriv[i] = 0
    else:
      dx1 = x[i    ] - x[i - 1]
      dx2 = x[i + 1] - x[i    ]
      a =  dx1/(dx2*(dx1 + dx2))
      b = (-dx1 + dx2)/(dx1*dx2)
      c = -dx2/(dx1*(dx1 + dx2))
      deriv[i] = a*y[i + 1] + b*y[i] + c*y[i - 1]

  #Return derivative
  return deriv




