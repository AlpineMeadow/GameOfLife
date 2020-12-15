#! /usr/bin/env python3

#A program to simulate The Game of Life by John Conway.


def getNeighborsState(i, j, gOL) :
        
  #Get the inital conditions.
  iup = i - 1
  idown = i + 1
  jleft = j - 1
  jright = j + 1

  currentCellState = gOL[i, j]  #Either live or dead.      
  uleft = gOL[iup, jleft]
  ucenter = gOL[iup, j]
  uright = gOL[iup, jright]
  cleft = gOL[i, jleft]
  cright = gOL[i, jright]
  lleft = gOL[idown, jleft]
  lcenter = gOL[idown, j]
  lright = gOL[idown, jright]

  #Sum up the surrounding cells states.
  liveOrDie = uleft + ucenter + uright + cleft + cright + lleft + lcenter + lright

  return  currentCellState, liveOrDie
#End of the function getNeighborsState.py
######################################################################################

######################################################################################

def getArgs(parser) :
  import numpy as np
    
  #Get the parameters
  parser.add_argument('-nI', '--numIterations', default = 50,
                      help = 'Choose how many iterations are made before stopping.', type = int)

  numGridPointsStr1 = ('Choose the number of grid points. ')
  numGridPointsStr2 = ('This value will give numGridPoints Squared points.')
  numGridPointsStr = numGridPointsStr1 + numGridPointsStr2
  
  parser.add_argument('-nG', '--numGridPoints', default = 10, help = numGridPointsStr,
                      type = int)

  args = parser.parse_args()

  #Generate variables from the inputs.
  numIterations = args.numIterations
  numGridPoints = args.numGridPoints

  #Generate the initial set of cells.
  initialSet = np.rint(np.random.rand(numGridPoints, numGridPoints))

  return initialSet, numIterations
#End of the function getArgs(parser).py

#################################################################################

#################################################################################

def getGameOfLife(gOL) :
  import numpy as np

  #Get the shape of the game of life array.
  m, n = gOL.shape

  #Allocate a new Game Of Life array.
  newGOL = np.ndarray((m, n))
  
  #Loop through the cells.
  for i in range(1, m - 1) :  #Do not count the outer boundary.
    for j in range(1, n - 1) :  #Do not count the outer boundary.

      #Determine the live or die parameter for each cell.
      currentCellState, liveOrDie = getNeighborsState(i, j, gOL)

      #Apply the rules of the game.

      #Rule #1.
      if(liveOrDie < 2) :
        if(currentCellState == 1) :
          newGOL[i, j] = 0  #Cell dies.
        else :
          newGOL[i, j] = 0  #Cell dies.
        #End of if(currentState == 1) : statement.
      #End of if(liveOrDie < 2) : statement.

      #Rule #2.
      if((liveOrDie == 2) or (liveOrDie == 3)) :
        if(currentCellState == 1) :
          newGOL[i, j] = 1  #Cell lives.
        else :
          newGOL[i, j] = 0  #Cell dies
        #End of if(currentCellState == 1) : statement.
      #End of if((liveOrDie == 2) or (liveOrDie == 3)) : statement.

      #Rule #3.
      if(liveOrDie > 3) :
        if(currentCellState == 1) :
          newGOL[i, j] = 0 #Cell dies.
        else :
          newGOL[i, j] = 0 #Cell dies.
        #End of if(currentCellState == 1) : statement.
      #End of if(liveOrDie > 3) : statement.

      #Rule #4.
      if((currentCellState == 0) and (liveOrDie == 3)) :
        newGOL[i, j] = 1 #Cell is born.
      #End of if((currentCellState == 0) and (liveOrDie == 3)) : statement.
        
    #End of for loop - for j in range(m) :
  #End of for loop - for i in range(n) :
  
  return newGOL
#End of the function getGameOfLife.py

#####################################################################################

#####################################################################################

#Gather our code in a main() function.
def main() :
  import argparse
  import matplotlib.pyplot as plt 
  import numpy as np
  import cv2
#  from moviepy.editor import VideoClip 
#  from moviepy.video.io.bindings import mplfig_to_npimage
  
  #Set up the argument parser.
  parser = argparse.ArgumentParser()

  #Set up the location of domain to be investigated.
  #Set up the number of iterations to be done on the point.
  #Create a number of points.  This will give numGridPoints^2 of values to be plotted.  
  gOL, numIterations = getArgs(parser)

  #Get the shape of the domain.
  m, n = gOL.shape

  #Create a output file name to where the plot will be saved.
  outfilepath = '/home/jdw/Computer/GameOfLife/Movies/'
  filename = ('GameOfLife.mp4')
  outfile = outfilepath + filename     

  #initialize video writer
#  fourCC possibilities are DIVX, XVID, MJPG, X264, WMV2, mp4v
  fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
  framesPerSecond = 1
  out = cv2.VideoWriter(outfile, fourcc, framesPerSecond, (m, n), True)
  
  #Loop through the iterations.
  for i in range(numIterations) :

    #Get the Game Of Life set.
    gOL = 255*getGameOfLife(gOL)
    
    gray = cv2.normalize(gOL, None, 255, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    gray_3c = cv2.merge([gray, gray, gray])

    out.write(gray_3c) # Write out frame to video
  #End of for loop - for i in range(numIterations):

  # Release everything if job is finished
#  out.release()
#  cv2.destroyAllWindows()

# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
  main()
