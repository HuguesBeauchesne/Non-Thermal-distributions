##################
# Cross sections #
##################

### Explanations ###
#This program computes cross sections for the computations with Maxwell-Boltzmann distributions.

#Comments
# - A is assumed complex and B real.
# - The cross-sections is averaged over incoming degrees of freedom and summed over outgoing ones.

#Import generic
import math



#Cross section p1 p1 > p2 p2
def GenericCS1(paramsInt, m1, m2, m3, m4, s):
  if s > 4*m1**2 and s > 4*m3**2:
    return paramsInt[0]/(16*math.pi*s)*(s - 4*m3**2)**0.5/(s - 4*m1**2)**0.5
  else:
    return 0

#Cross section p1 p2 > p1 p2
def GenericCS2(paramsInt, m1, m2, m3, m4, s):
  if s > (m1 + m2)**2:
    return paramsInt[0]/(16*math.pi*s)
  else:
    return 0



#Cross sections (including factors for identical particles)
def AA_BB_CS(paramsInt, m1, m2, m3, m4, s):
  return (1/8)*GenericCS1(paramsInt, m1, m2, m3, m4, s)

def BB_AA_CS(paramsInt, m1, m2, m3, m4, s):
  return (1/2)*GenericCS1(paramsInt, m1, m2, m3, m4, s)

def AB_AB_CS(paramsInt, m1, m2, m3, m4, s):
  return       GenericCS2(paramsInt, m1, m2, m3, m4, s)

def BB_SMSM_CS(paramsInt, m1, m2, m3, m4, s):
  return (1/4)*GenericCS1(paramsInt, m1, m2, m3, m4, s)

def SMB_SMB_CS(paramsInt, m1, m2, m3, m4, s):
  return       GenericCS2(paramsInt, m1, m2, m3, m4, s)



#Partial wave cross sections
def AA_BB_CS_t(paramsInt, m1, m2, m3, m4, s):
  return 0

def BB_AA_CS_t(paramsInt, m1, m2, m3, m4, s):
  return 0

def AB_AB_CS_t(paramsInt, m1, m2, m3, m4, s):
  return 0

def BB_SMSM_CS_t(paramsInt, m1, m2, m3, m4, s):
  return 0

def SMB_SMB_CS_t(paramsInt, m1, m2, m3, m4, s):
  return 0




