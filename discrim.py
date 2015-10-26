#!/usr/bin/env python
# https://github.com/dsmbgu8/discrim

import sys, os, warnings

from util.aliases import *

from progressbar import ProgressBar, ETA, Bar, Percentage

from sklearn.base import clone
from sklearn.externals.joblib import load as jlload, dump as jldump
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.utils import check_X_y
from sklearn.cross_validation import LeaveOneOut, StratifiedKFold, ShuffleSplit
from sklearn.grid_search import GridSearchCV

from models import *

##### Feature scaling method ###################################################
scaling_method = 'MinMax' # 'Standard' # None # 'Normalize' #

##### Serialization params #####################################################
output_dir = './discrim_output'
jlcompress = 0 # warning: nonzero values disable memory mapping
jlmmap     = None # 'r' # memmap mode for loading (None=disabled)
jlcache    = 500 # cache size in mb for joblib IO

def tuned_fit(clf,tuned_params):
    '''
    collect parameters in tuned_params that were tuned to the given clf
    '''
    clf_fit = {}
    clf_params = clf.get_params()
    for param_list in tuned_params:
        for param in param_list:
            clf_fit[param] = clf_params[param]
            
    return clf_fit

def model_train(X_train,y_train,model_clf,model_tuned):
    '''
    train a model via gridsearchcv with tuning parameters model_tuned
    '''

    # make a copy to ensure we don't invalidate parameters
    clf = clone(model_clf)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # disable sklearn convergence warnings
        # tune params where desired
        if model_tuned is not None and len(model_tuned) != 0: 
            cv = GridSearchCV(clf,model_tuned,cv=gridcv_folds,scoring=gridcv_score,
                              n_jobs=gridcv_jobs,verbose=gridcv_verbose,refit=True)
            cv.fit(X_train, y_train)
            clf = cv.best_estimator_
        else: # otherwise train with the default parameters (no tuning)
            clf.fit(X_train,y_train)

    return clf
    
def discrim_cv(X,y,cv,output={}):
    '''    
    cross validate all models in model_eval list on samples X labels y
    '''    
    N,n = X.shape
    uy  = []
    if pred_mode == 'clf':
        uy = unique(y)
        K,ymin = len(uy),uy[0]
        if K<2 or K>2:
            print 'Error: need 2 classes for classification!'
            return {}        
        pstr = ', '.join(['%d: %d (%4.2f%%)'%(yi, sum(y==yi), sum(y==yi)/float(N)) for yi in uy])    
    elif pred_mode == 'reg':
        ymin,ymax  = extrema(y)
        ymed,ymean = median(y), mean(y)
        pstr = 'ymin=%g, ymean=%g, ymedian=%g, ymax=%g'%(ymin,ymean,ymed,ymax)
    
    for model_id in model_eval:
        print 'Evaluating %s model "%s" on %d %d-dimensional samples (%s)'%(pred_mode,
                                                                              model_id,
                                                                              N,n,pstr)
        model_clf,model_tuned,model_coef = model_params[model_id]

        # check to make sure new model is different from old model
        if model_id in output:
            output_cv    = output['cv']
            output_tuned = output[model_id].get('model_tuned',None)
            if output_tuned == model_tuned and \
               ([trte for trte in cv] == [trte for trte in output_cv]):
                print 'Model %s trained with the same data and tuned parameters already exists, skip?'%model_id
                yn = raw_input()
                if yn.strip() in ('y','Y'):
                    continue
                
        if cv_id == 'loo':
            y_pred = zeros(N)

        n_folds = len(cv)

        # bookkeeping variables
        models,preds,errors = [],[],[]
        scores = dict([(key,[]) for key in scorefn.keys()])
        errors = dict([(key,[]) for key in errorfn.keys()])
        
        widgets = ['%s cv: '%cv_id, Percentage(), ' ', Bar('='), ' ', ETA()]
        pbar = ProgressBar(widgets=widgets, maxval=n_folds+(cv_id=='loo')).start()
        for i,(train_index,test_index) in enumerate(cv):
            pbar.update(i)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # xgb assumes labels \in {0,1}
            if model_id == 'xgb' and ymin == -1:                
                y_train[y_train==-1] = 0                

            # train/predict as usual
            clf = model_train(X_train,y_train,model_clf,model_tuned)
            clf_pred = clf.predict(X_test)

            if model_id == 'xgb' and ymin == -1:
                clf_pred[clf_pred==0] = -1

            # loo predicts one label per 'fold'
            if cv_id == 'loo':
                y_pred[test_index] = clf_pred
            else:
                # collect output for all test samples in this fold
                for score,score_fn in scorefn.iteritems():
                    scorei = score_fn(y_test,clf_pred,uy)
                    scores[score].append(scorei)                
                preds.append(clf_pred)                
                models.append(clf)
                for error,error_fn in errorfn.iteritems():
                    errors[error].append(error_fn(y_test,clf_pred))
                
        if cv_id == 'loo':
            for score,score_fn in scorefn.iteritems():                
                scorei = score_fn(y,y_pred,uy)
                scores[score].append(scorei)
            preds  = [y_pred]
            for error,error_fn in errorfn.iteritems():
                errors[error].append(error_fn(y,y_pred))
            
            models = [model_train(X,y,model_clf,model_tuned)]
            pbar.update(i+1)
        pbar.finish()

        for score,vals in scores.iteritems():
            print 'mean %s: %7.4f'%(score, mean(vals))
            
        output[model_id] = {'preds':preds,'scores':scores,'errors':errors,
                            'models':models,'model_tuned':model_tuned}
    return output

def discrim_state(input_statefile,output_statefile,update_output=False):
    '''
    run discrim_cv for an experiment (input_statefile) and serialize the output 
    (output_statefile), scales features using scaling_method, and excludes
    unlabeled (y=0) samples
    '''

    input_state  = jlload(input_statefile,mmap_mode=jlmmap)
    output_state = {}
    if update_output and pathexists(output_statefile):
        print 'Loading existing state from', output_statefile
        output_state = jlload(output_statefile,mmap_mode=None)
    
    X = input_state['X_exp'].copy()
    y = input_state['y_exp'].copy()

    multi_output = len(y.shape) > 1 and min(y.shape) > 1

    # remove incompatible models
    if multi_output and ('linsvm' in model_eval or 'rbfsvm' in model_eval):
        print 'Error: SVR (currently) incompatible with multi-output labels'
        return input_state,{}
    
    if scaling_method=='Normalize':
        scale_fn = Normalizer(norm='l2').fit_transform
    elif scaling_method=='MinMax':
        scale_fn = MinMaxScaler().fit_transform
    elif scaling_method=='Standard':
        scale_fn = StandardScaler().fit_transform
    elif scaling_method==None:
        scale_fn = lambda X: X
    else:
        print 'Error: unknown scaling method "%s"'%scaling_method
        return input_state,{}

    print 'Scaling features using method "%s"'%scaling_method
    X = scale_fn(X)

    # remove unlabeled samples
    if multi_output:
        labmask = (y!=0).any(axis=1)
        y = y[labmask,:]
    else:
        labmask = y!=0
        y = y[labmask].ravel()
    X = X[labmask,:]

    # make sure X,y are valid after scaling/masking operations
    check_X_y(X,y,multi_output=multi_output)

    # get number of *labeled* samples
    N = len(y)

    if cv_id == 'loo':
        cv = LeaveOneOut(N)
    elif pred_mode == 'clf':
        cv = StratifiedKFold(y,n_folds=cv_folds,random_state=train_state)
    elif pred_mode == 'reg':
        cv = ShuffleSplit(n=N,n_iter=cv_folds,test_size=int(N/cv_folds))
        
    output_state = discrim_cv(X,y,cv,output_state)
    output_state.update({'cv':cv,'cv_id':cv_id,'model_eval':model_eval,
                         'labmask':labmask,'scaling_method':scaling_method})

    jldump(output_state,output_statefile,compress=jlcompress,cache_size=jlcache)

    return input_state, output_state

def discrim_exp(X_exp,y_exp,exp_name,exp_dir=output_dir):
    '''
    given input data X_exp labels y_exp \in {-1,0,1} (0=unlabeled),
    define input/output state files and serialize input data
    '''
    if not pathexists(exp_dir):
        os.makedirs(exp_dir)

    input_statefile  = pathjoin(exp_dir,exp_name+'_input.pkl')
    output_statefile = pathjoin(exp_dir,exp_name+'_output.pkl')
    jldump({'X_exp':X_exp,'y_exp':y_exp,'exp_name':exp_name},
           input_statefile,compress=jlcompress,cache_size=jlcache)
    input_state, output_state = discrim_state(input_statefile,output_statefile)
    return input_state, output_state            

def discrim_coef(exp_name,exp_dir=output_dir):    
    input_statefile  = pathjoin(exp_dir,'_'.join([exp_name,'input.pkl']))
    output_statefile = pathjoin(exp_dir,'_'.join([exp_name,scaling_method,pred_mode,'output.pkl']))
    try:
        input_state  = jlload(input_statefile,mmap_mode=None)
        output_state = jlload(output_statefile,mmap_mode=None)
    except:
        print 'Error: unable to load input/output state files in exp_dir=%s'%exp_dir
        return {}

    coef = {}
    for model_id in output_state['model_eval']:
        models = output_state[model_id]['models']
        coef_fn = model_params[model_id][-1]
        model_coef = []
        for i,model in enumerate(models):
            if model_id == 'xgb': # fix stupid xgb dimensionality issue
                w = coef_fn(model,input_state['X_exp'].shape[1])
            else:
                w = coef_fn(model)
            model_coef.append(w)

        model_coef = asarray(model_coef)
        model_mean = mean(model_coef,axis=0)
        model_std  = std(model_coef,axis=0)
        
        coef[model_id] = {'coef':model_coef,
                          'mean':model_mean,
                          'std':model_std}
        
    return coef

if __name__ == '__main__':
    from sklearn.datasets import load_digits

    data  = load_digits()
    X_exp = data.data
    y_exp = data.target
    mask5  = (y_exp==5)
    mask8  = (y_exp==8)
    
    y_exp[:]     =  0
    y_exp[mask5] = -1
    y_exp[mask8] =  1
    if pred_mode == 'reg':
        y_exp = c_[y_exp==1,y_exp==-1].astype(float)
    discrim_exp(X_exp,y_exp,'digits5v8',exp_dir='/tmp/discrim_test')
