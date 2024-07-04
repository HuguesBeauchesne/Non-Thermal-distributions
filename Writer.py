##########
# Writer #
##########

### Explanations ###
#This program writes relevant information to files.

#Comments
# - None

#Import standard
import os



#Write items
def WriteItem(item, fileName):
  fstream = open(fileName, "a")
  fstream.write(str(item) + '\n')
  fstream.close()



#Write list
def WriteList(itemList, fileName):
  fstream = open(fileName, "a")
  for item in itemList:
    fstream.write(str(item) + " ")
  fstream.write('\n')
  fstream.close()



#Write progress updates
def WriteProgress(x, rateT1, OmegaA, OmegaB, pAMax, pBMax, ratioA, ratioB):
  fstream = open("Progress.txt", "a")
  fstream.write(str(x) + ' ' + str(rateT1) + ' ' + str(OmegaA) + ' ' + str(OmegaB) + ' ' + str(pAMax) + ' ' + str(pBMax) + ' ' + str(ratioA) + ' ' + str(ratioB))
  fstream.write('\n')
  fstream.close()




