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
import ltr_solver_multiview as ltr

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
  iactfunc = 0  ## =0 identity, =1 arcsinh =2 2*sigmoid-1 =3 tanh

  ## cmodel.lossdegree = 0  ## =0 L_2^2, =1 L^2, =0.5 L_2^{0.5}, ...L_2^{z}
  lossdegree=0  ## default L_2^2 =0
  regdegree=1   ## regularization degree, Lasso

  norm_type  = 0 ## parameter normalization =0 L2 =1 L_{infty} =2 arcsinh + L2 
                 ## =3 RELU, =4 tanh + L_2 

  perturb = 0 ## gradient perturbation

  report_freq = 50 ## frequency of the training reports

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

  ## MULTIVIEW setting, direct sum of X
  ## 2 Views, 
  ## Input: all rows of X are joint, concatenated, to all rows 
  ## Output: all rows Y are summed to all rows  
  
  ## load randomly generated polynomial data for X and Y
  ## first data table
  mdata1 = 120         ## number of examples, input rows 
  ninputvariable1 = 30     ## number of input variables, columns in X
  noutputvariable1 = 10     ## number of output variables columns in Y
  dataorder = 2            ## order of the data generating polynomial
  datarank = 10            ## rank of the data generating polynomial function
  ibias = 1                ## for homogeneous coordinate, a column of 1s appended to X
  X1,Y1 = data_generator.polynomial_function(mdata1, ninputvariable1, \
                                             ny = noutputvariable1, \
                                             norder = dataorder, \
                                             nrank = datarank, \
                                             ibias = ibias)

  ## second data table
  mdata2 = 80         ## number of examples, input rows 
  ninputvariable2 = 20     ## number of input variables, columns in X
  X2,Y2 = data_generator.polynomial_function(mdata2, ninputvariable2, \
                                             ny = noutputvariable1, \
                                             norder = dataorder, \
                                             nrank = datarank, \
                                             ibias = ibias)

                                             
  ##  join, direct sum, operation might be given this way
  ## X = np.hstack((np.kron(X1, np.ones(mdata2)), np.kron(np.ones(mdata1), X2)))
  ## Since the join is carried out within the LTR we can use the original tables X1,X2
  
  ## The output is given by
  Y = np.kron(Y1, np.ones((mdata2,1))) + np.kron(np.ones((mdata1,1)), Y2)
  
  ## xindex will be constructed when the designs are given
  ## @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

  ## --------------------------------------------
  ## collect the prediction for all folds
  yprediction=np.zeros(Y.shape)
  ## collect the statistics for all folds
  xstats = np.zeros((nfold,5))   ## P-corr, rmse, precision, recall, f1

  ifold=0

  time00=time.time()

  ## training and test splits 
  cselection=sklearn.model_selection.KFold(n_splits=nfold,shuffle = True, \
                                             random_state = None) 

  xselection = np.ones(Y.shape[0])

  for itrain,itest in cselection.split(xselection):

    ## we need to split the xindex and the output only
    ## the selection of input is based on the xindex array
    ytrain = Y[itrain]
    ytest = Y[itest]
    mtrain = len(itrain)
    mtest = len(itest)
    print('Training:', mtrain)
    print('Test:', mtest)
    
    time0 = time.time()
    
    ## training
    ## some design configuration
    idesign  = 1   ## some designs are demonstrated here
    
    if idesign == 0: ## simplest case
      ## variables of X1 combined with variables of X2
      ## the order of the model is the length of llinks, thus 2
      llinks = [0, 1]
      lxtrain = [X1, X2]

      ## We need to construct the join index array: xindex
      ## The rows of xindex contain the index pairs, we have 2 views, 
      ## pointing to X1 and X2 
      ## the indexes have to be integers
      xindex = np.hstack((np.kron(np.arange(mdata1,dtype=int).reshape((mdata1,1)), \
                                np.ones((mdata2,1),dtype=int)), \
                      np.kron(np.ones((mdata1,1),dtype=int), \
                                np.arange(mdata2,dtype=int).reshape((mdata2,1)))))

      xindex_train = xindex[itrain]
      xindex_test = xindex[itest]

      cmodel.fit(lxtrain, ytrain, llinks = llinks, xindex = xindex_train, \
                           nepoch=nrepeat)
      time1 = time.time()

      ## prediction
      lxtest = [X1, X2]
      ypred_test = cmodel.predict(lxtest, llinks = llinks, \
                                        xindex = xindex_test)
    elif idesign == 1:  
      ## variables up degree 2 of X1 combined with variables of X2, 
      ## and also all variables X2 with all variables of X2
      ## the order of the model is the length of llinks, thus 3
      llinks = [[0, 2], [1,2], [1,2]]
      lxtrain = [X1, X2]

      ## The rows of xindex contain the index triplets, we have 3 views, 
      ## pointing to X1 and twice X2 to join X2 with itself 
      ## the indexes have to be integers
      xindex = np.hstack((np.kron(np.arange(mdata1,dtype=int).reshape((mdata1,1)), \
                                np.ones((mdata2,1),dtype=int)), \
                      np.kron(np.ones((mdata1,1),dtype=int), \
                                np.arange(mdata2,dtype=int).reshape((mdata2,1))), \
                      np.kron(np.ones((mdata1,1),dtype=int), \
                                np.arange(mdata2,dtype=int).reshape((mdata2,1)))))

      xindex_train = xindex[itrain]
      xindex_test = xindex[itest]

      cmodel.fit(lxtrain, ytrain, llinks = llinks, xindex = xindex_train, \
                           nepoch=nrepeat)
      time1 = time.time()

      ## prediction
      lxtest = [X1, X2]
      ypred_test = cmodel.predict(lxtest, llinks = llinks, \
                                        xindex = xindex_test)

  
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
