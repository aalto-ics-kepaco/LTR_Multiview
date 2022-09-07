## LTR example main file 
## data generation by random polynomial function
import sys
import time

import numpy as np

## ###################################################
## for demonstration
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cmplot
import sklearn.model_selection 
from sklearn.metrics import mean_squared_error
## ###################################################

import data_generator 
import ltr_multiview as ltr

## ################################################################
## ################################################################
## ###################################################
def acc_eval(yobserved, ypredicted):
  """
  Task: to report some statistics, 
        f1, precision and recall is meaningfull if the data is binary coded (1,0,-1). 

  Input: yobserved     2d array of observed data
         ypredicted    2d array of predicted data
         ibinary       binary(=0,1) =1 binary coded data (=1,0,-1), =0 rela values 
  Output: pcorr        Pearson correlation
          rmse         Root mean square error
          prec         precision    
          recall       recall        
          f1           f1 measure
  """

  ## nobject,nclass=yobserved.shape

  tp=np.sum((yobserved>0)*(ypredicted>0))     ## true positive
  fp=np.sum((yobserved<=0)*(ypredicted>0))    ## false positive
  fn=np.sum((yobserved>0)*(ypredicted<=0))    ## false negative
  tn=np.sum((yobserved<=0)*(ypredicted<=0))   ## true negative

  print('Tp,Fn,Fp,Tn:',tp,fn,fp,tn)

  if tp+fp>0:
    prec=tp/(tp+fp)
  else:
    prec=0
    
  if tp+fn>0:
    recall=tp/(tp+fn)
  else:
    recall=0
  
  if prec+recall>0:
    f1=2*prec*recall/(prec+recall)
  else:
    f1=0
  
  ## Pearson correlation
  pcorr= np.corrcoef(yobserved.ravel(), ypredicted.ravel())[0, 1]
  ## Root mean sqaure error
  rmse = np.sqrt(mean_squared_error(yobserved.ravel(), ypredicted.ravel()))
  
  return(pcorr,rmse,prec,recall,f1)

## ################################################################
def print_result(xstat):
  """
  xstat = (pcorr,scorr,rmse,precision,recall,f1)
  """
  
  nitem = len(xstat)
  stext = ['P-corr','RMSE','Precision','Recall','F1']
  if len(stext) == nitem:
    for i in range(nitem):
      print(stext[i]+':','%7.4f'%(xstat[i]))

  return

## ################################################################
def main(iworkmode=None):
  """
  Task: to run the LTR solver on randomly generated polynomial functions
  """

  igraph=1   ## =0 no graph =1 graph is shown

  nfold=5    ## number of folds in the cross validation
  
  ## -------------------------------------
  ## Parameters to learn
  ## the most important parameter
  norder=2      ## maximum power
  rank=50      ## number of rows
  rankuv=20   ## internal rank for bottlenesck if rankuv<rank
  sigma=0.01  ## learning step size
  nsigma=10      ## step size correction interval
  gammanag=0.9     ## discount for the ADAM method
  gammanag2=0.9    ## discount for the ADAM method norm

  # mini-bacht size,
  mblock=500

  ## number of epochs
  nrepeat=10

  ## regularizatin constant for xlambda optimization parameter
  cregular=0.000001  

  ## activation function
  iactfunc = 0  ## =0 identity, =1 arcsinh, =2 2*sigmoid-1, =3 tanh, =4 relu

  ## cmodel.lossdegree = 0  ## =0 L_2^2, =1 L^2, =0.5 L_2^{0.5}, ...L_2^{z}
  lossdegree=0  ## default L_2^2 =0
  regdegree=1   ## regularization degree, Lasso

  norm_type  = 0 ## parameter normalization =0 L2 =1 L_{infty} =2 arcsinh + L2 
                 ## =3 RELU, =4 tanh + L_2 

  perturb = 0 ## gradient perturbation

  report_freq = 100 ## frequency of the training reports

  ## --------------------------------------------
  cmodel=ltr.ltr_solver_cls(norder=norder,rank=rank,rankuv=rankuv)

  ## set optimization parameters
  cmodel.update_parameters(nsigma=nsigma, \
                           mblock=mblock, \
                           sigma0=sigma, \
                           gammanag=gammanag, \
                           gammanag2=gammanag2, \
                           cregular=cregular, \
                           iactfunc=iactfunc, \
                           lossdegree=lossdegree, \
                           regdegree=regdegree, \
                           norm_type =norm_type, \
                           perturb = perturb, \
                           report_freq = report_freq)
                           
  print('Order:',cmodel.norder)
  print('Rank:',cmodel.nrank)
  print('Rankuv:',cmodel.nrankuv)
  print('Step size:',cmodel.sigma0)
  print('Step freq:',cmodel.nsigma)
  print('Step scale:',cmodel.dscale)
  print('Epoch:',nrepeat)
  print('Mini-batch size:',mblock)
  print('Discount:',cmodel.gamma)
  print('Discount for NAG:',cmodel.gammanag)
  print('Discount for NAG norm:',cmodel.gammanag2)
  print('Bag size:',cmodel.mblock)
  print('Regularization:',cmodel.cregular)
  print('Gradient max ratio:',cmodel.sigmamax)
  print('Type of activation:',cmodel.iactfunc)
  print('Degree of loss:',cmodel.lossdegree)
  print('Degree of regularization:',cmodel.regdegree)
  print('Normalization type:',cmodel.norm_type)
  print('Gradient perturbation:', cmodel.perturb)
  print('Activation:', cmodel.iactfunc)
  print('Input centralization:', cmodel.ixmean)
  print('Input L_infty scaling:', cmodel.ixscale)
  print('Quantile regression:',cmodel.iquantile)
  print('Quantile alpha:',cmodel.quantile_alpha)    ## 0.5 for L_1 norm loss
  print('Quantile smoothing:',cmodel.quantile_smooth)

  #####################################
  ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  ## In an application this data loading needs to be changed 

  ## load randomly generated polynomial data
  mdatasize = 50000         ## number of examples, input rows 
  ninputvariable = 100     ## number of input variables, columns in X
  noutputvariable = 10     ## number of output variables columns in Y
  dataorder = 2            ## order of the data generating polynomial
  datarank = 10            ## rank of the data generating polynomial function
  ibias = 1                ## for homogeneous coordinate, a column of 1s appended to X
  X,Y = data_generator.polynomial_function(mdatasize, ninputvariable, \
                                             ny = noutputvariable, \
                                             norder = dataorder, \
                                             nrank = datarank, \
                                             ibias = ibias)
                                             
  ## for structured output, e.g.  multiclass
  ibinary_output = 0        ## =1 binary output, =0 no
  if ibinary_output == 1:
    Y = np.sign(Y)

  ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

  if len(Y.shape) == 1:
    ndimy = 1
    mfull = Y.shape[0]
  else:
    mfull,ndimy = Y.shape
  ndimx = X.shape[1]
  print('Full data shape:',mfull,ndimy,ndimx)

  ## --------------------------------------------
  ## collect the prediction for all folds
  yprediction=np.zeros((mdatasize,ndimy))
  ## collect the statistics for all folds
  xstats = np.zeros((nfold,5))   ## P-corr, rmse, precision, recall, f1

  ifold=0

  time00=time.time()

  ## training and test splits 
  cselection=sklearn.model_selection.KFold(n_splits=nfold,shuffle = True, \
                                             random_state = None) 

  for itrain,itest in cselection.split(X):

    xtrain = X[itrain]
    ytrain = Y[itrain]
    xtest = X[itest]
    ytest = Y[itest]
    mtrain = len(itrain)
    mtest = len(itest)
    print('Training:', mtrain)
    print('Test:', mtest)
    
    time0 = time.time()
    
    ## training
    ## some design configuration
    idesign  = 0   ## some designs are demonstrated here
    
    if idesign == 0: ## simplest default
      ## the order of the model is equal to norder
      cmodel.fit(xtrain, ytrain, llinks = None, xindex = None, \
                           nepoch=nrepeat)
      time1 = time.time()

      ## prediction
      ypred_test = cmodel.predict(xtest, llinks = None, \
                                        xindex = None)
    elif idesign == 1:
      ## the order of the model is the number of views
      lxtrain = [xtrain]
      llinks = [ 0 for _ in range(norder)]
      cmodel.fit(lxtrain, ytrain, llinks = llinks, xindex = None, \
                           nepoch=nrepeat)
      time1 = time.time()

      ## prediction
      lxtest = [xtest]
      ypred_test = cmodel.predict(lxtest, llinks = llinks, \
                                        xindex = None)
    elif idesign == 2:
      ## the order of the model is the number of views
      lxtrain = [xtrain, xtrain]
      llinks = [ 0, 1]
      cmodel.fit(lxtrain, ytrain, llinks = llinks, xindex = None, \
                           nepoch=nrepeat)
      time1 = time.time()

      ## prediction
      lxtest = [xtest, xtest]
      ypred_test = cmodel.predict(lxtest, llinks = llinks, \
                                        xindex = None)

    elif idesign == 3:
      ## the order of the model is the number of views
      nhalf = int(ninputvariable/2)
      lxtrain = [xtrain[:,:nhalf], xtrain[:,nhalf:]]
      ## combine the variables of the first half to the second one, 
      ## where each variable can have power up to 2
      ## X = [X0,X1] = > 1, X0, X1, X0*X1, X1*X1
      llinks = [ [0,1], [1]]
      cmodel.fit(lxtrain, ytrain, llinks = llinks, xindex = None, \
                           nepoch=nrepeat)
      time1 = time.time()

      ## prediction
      lxtest = [xtest[:,:nhalf], xtest[:,nhalf:]]
      ypred_test = cmodel.predict(lxtest, llinks = llinks, \
                                        xindex = None)
  
    print('Training time:', time1 - time0)

    ## save the prediction computed in the current fold
    yprediction[itest] = ypred_test

    ## save the accuracy results
    xstats[ifold] = acc_eval(ytest.ravel(),ypred_test.ravel())

    print(">>>> TEST result on fold:", ifold)
    print_result(xstats[ifold])

    ifold += 1

    sys.stdout.flush()

  time2 = time.time()
  print('Total training time(s): ', time2 - time00)

  print(10*'=')
  print('Average on all folds')
  xstatmean = np.mean(xstats,0)
  print_result(xstatmean)

  sys.stdout.flush()

  if igraph == 1:
    
    fig=plt.figure(figsize=(6,6))

    fig.suptitle('Observed values versus prediction')

    ax=plt.subplot2grid((1,1),(0,0),colspan=1,rowspan=1)
    ax.scatter(Y.ravel(),yprediction.ravel(),s=2)
    ax.set_xlabel('Observed output components')
    ax.set_ylabel('Predicted output components')
    ax.set_title('Order:'+str('%3d'%norder)+', '+'Rank:'+str('%4d'%rank))
    ax.grid('on')
    ## ax.set_aspect(aspect='equal')

    plt.tight_layout(pad=1)
    plt.show()

  print('Bye')

  return(0)

## ###################################################
## ################################################################
if __name__ == "__main__":
  if len(sys.argv)==1:
    iworkmode=0
  elif len(sys.argv)>=2:
    iworkmode=eval(sys.argv[1])
  main(iworkmode)
