import sys
import time
import numpy as np

## ###################################################
## ###################################################
## ################################################################
## ################################################################
def polynomial_function(m, n, ny=10, norder=2, nrank=10, ibias=1):
  """
  Task: to generate a vector valued random polynomial function on an input matrix 
  Input:  m         the number of examples, the rows in the input matrix
          n         the number of variables, the columns in the input matrix
          ny        the number of output variables
          norder    the maximum degree of the polynomial
          nrank     the rank of the tensor generating the polynomial
          ibias     =1 add a column to X with 1s, homogeneous coordinates to enumerate 
                       the lower degree terms
                    =0 no
  Output: X         2d array of input examples
          Y         2d array of output examples
  """
  rng_seed = 12345
  rng = np.random.default_rng(rng_seed)
  
  X = rng.standard_normal(size=(m,n))
  X /= np.max(np.abs(X))
  if ibias==1:
    X = np.hstack((X, np.ones((m,1))))
    nx = n+1
  else:
    nx = n

    

  ## random polynomial parameters
  xP = rng.standard_normal(size=(norder,nrank,nx))
  xQ = rng.standard_normal(size=(nrank,ny))

  ## row wise normalization of the parameters for ranks
  for d in range(norder):
    xnorm = np.sqrt(np.sum(xP[d]**2,1))
    xnorm = xnorm + 1*(xnorm == 0)
    xP[d] /= np.outer(xnorm,np.ones(nx))

  xnorm = np.sqrt(np.sum(xQ**2,1))
  xnorm = xnorm + 1*(xnorm == 0)
  xQ /= np.outer(xnorm, np.ones(ny))
  
  ## output
  Y=np.zeros((m,ny))
  ## rank-wise scale 
  xlambda = rng.standard_normal(size=(nrank,))
  xlambda = xlambda[np.argsort(-xlambda)]  ## sort in a decreasing order
    
  ## generate the polynomial function
  Y0 = np.ones((m,nrank)) 
  for d in range(norder):
    PX = np.dot(X, xP[d].T)
    Y0 *= PX
  Y = np.dot(Y0, (xQ.T*xlambda).T)

  ## scale the output
  Y /= np.outer(np.max(np.abs(Y),1),np.ones(ny))
  
  return(X,Y)

## ################################################################
def add_noise(y, noise_type=0, scale=1):
  """
  Task: to add noise to a matrix(vector)
  Input:   y           matrix(vector) input data matrix
           noise_type  =-1 no noise =0 gaussian =1 unifoem
           scale       noise std = scale * input data std     
  Output:  ynoisy      matrix of noisy data, if y is vector it has shape (m,1)
  """
  rng_seed = 12345
  rng = np.random.default_rng(rng_seed)
  
  m = len(y)
  if len(y.shape) < 2:
    ynoisy = y.reshape((m,1))
  else:
    ynoisy = np.copy(y)

  m,n = ynoisy.shape
  if noise_type == 0:   ## Gaussian noise
    xstd = np.std(y)
    ynoisy += scale*xstd*rng.standard_normal(size = (m,n))
  elif noise_type == 1:  ## Uniform noise
    xstd = np.std(y)
    ynoisy += scale*xstd*rng.uniform(size = (m,n))

  return(ynoisy)
  
## ###################################################
## ################################################################
## if __name__ == "__main__":
##   if len(sys.argv)==1:
##     iworkmode=0
##   elif len(sys.argv)>=2:
##     iworkmode=eval(sys.argv[1])
##   main(iworkmode)
