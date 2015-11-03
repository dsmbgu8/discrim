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

from defaults import *


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

def model_train(X_train,y_train,model_clf,model_tuned,gridcv_folds,gridcv_score):
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

class DISCRIM_EXP:
    def __init__(self,**kwargs):
        ##### Prediction mode (clf=classification, reg=regression) ##############
        self.pred_mode = kwargs.pop('pred_mode','clf')
        self.model_eval = kwargs.pop('model_eval',default_models[self.pred_mode])
        self.cv_folds = kwargs.pop('cv_folds',cv_folds)
        self.cv_id = 'loo' if self.cv_folds == inf else '%d-fold'%self.cv_folds

        ##### Feature scaling method ############################################
        self.scaling_method = kwargs.pop('scaling_method','MinMax') # 'Standard' # None # 'Normalize' #
        self.gridcv_folds = kwargs.pop('gridcv_folds',gridcv_folds)
        self.gridcv_score = kwargs.pop('gridcv_score',default_gridcv[self.pred_mode])
        
        ##### Models to evaluate/cv objective function ##########################
        ##### pred_mode specific parameters #####################################
        self.model_params = {}

        for model in self.model_eval:
            if self.pred_mode=='clf':
                model_id = model.replace('svm','svc').replace('rf','rfc')
            else:
                model_id = model.replace('svm','svr').replace('rf','rfr')
            self.model_params[model] = model_params[model_id]

        self.scorefn = {}
        for score in default_scores[self.pred_mode]:
            self.scorefn[score] = scorefn[score]

        self.errorfn = {}            
        for error in default_errors[self.pred_mode]:
            self.errorfn[error] = errorfn[error]
            
    def _run_cv(self,X,y,cv,output={}):
        '''    
        cross validate all models in model_eval list on samples X labels y
        '''    
        N,n = X.shape
        uy  = []
        if self.pred_mode == 'clf':
            uy = unique(y)
            K,ymin = len(uy),uy[0]
            if K<2 or K>2:
                print 'Error: need 2 classes for classification!'
                return {}        
            pstr = ', '.join(['%d: %d (%4.2f%%)'%(yi, sum(y==yi), sum(y==yi)/float(N)) for yi in uy])    
        elif self.pred_mode == 'reg':
            ymin,ymax  = extrema(y)
            ymed,ymean = median(y), mean(y)
            pstr = 'ymin=%g, ymean=%g, ymedian=%g, ymax=%g'%(ymin,ymean,ymed,ymax)

        for model_id in self.model_eval:
            print 'Evaluating %s model "%s" on %d %d-dimensional samples (%s)'%(self.pred_mode,
                                                                                  model_id,
                                                                                  N,n,pstr)
            model_clf,model_tuned,model_coef = self.model_params[model_id]

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

            if self.cv_id == 'loo':
                y_pred = zeros(N)

            n_folds = len(cv)

            # bookkeeping variables
            models,preds,errors = [],[],[]
            scores = dict([(key,[]) for key in self.scorefn.keys()])
            errors = dict([(key,[]) for key in self.errorfn.keys()])

            widgets = ['%s cv: '%self.cv_id, Percentage(), ' ', Bar('='), ' ', ETA()]
            pbar = ProgressBar(widgets=widgets, maxval=n_folds+(self.cv_id=='loo')).start()
            for i,(train_index,test_index) in enumerate(cv):
                pbar.update(i)
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # xgb assumes labels \in {0,1}
                if model_id == 'xgb' and ymin == -1:                
                    y_train[y_train==-1] = 0                

                # train/predict as usual
                clf = model_train(X_train,y_train,model_clf,model_tuned,
                                  self.gridcv_folds,self.gridcv_score)
                clf_pred = clf.predict(X_test)

                if model_id == 'xgb' and ymin == -1:
                    clf_pred[clf_pred==0] = -1

                # loo predicts one label per 'fold'
                if self.cv_id == 'loo':
                    y_pred[test_index] = clf_pred
                else:
                    # collect output for all test samples in this fold
                    for score,score_fn in self.scorefn.iteritems():
                        scorei = score_fn(y_test,clf_pred,uy)
                        scores[score].append(scorei)                
                    preds.append(clf_pred)                
                    models.append(clf)
                    for error,error_fn in self.errorfn.iteritems():
                        errors[error].append(error_fn(y_test,clf_pred))

            if self.cv_id == 'loo':
                for score,score_fn in self.scorefn.iteritems():                
                    scorei = score_fn(y,y_pred,uy)
                    scores[score].append(scorei)
                preds  = [y_pred]
                for error,error_fn in self.errorfn.iteritems():
                    errors[error].append(error_fn(y,y_pred))

                models = [model_train(X,y,model_clf,model_tuned,
                                      self.gridcv_folds,self.gridcv_score)]
                pbar.update(i+1)
            pbar.finish()

            for score,vals in scores.iteritems():
                print 'mean %s: %7.4f'%(score, mean(vals))

            output[model_id] = {'preds':preds,'scores':scores,'errors':errors,
                                'models':models,'model_tuned':model_tuned}
        return output

    def _collect_state(self,input_statefile,output_statefile,update_output=False):
        '''
        run _cv for an experiment (input_statefile) and serialize the output 
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
        if multi_output and ('linsvm' in self.model_eval or 'rbfsvm' in self.model_eval):
            print 'Error: SVR (currently) incompatible with multi-output labels'
            return input_state,{}

        if self.scaling_method=='Normalize':
            scale_fn = Normalizer(norm='l2').fit_transform
        elif self.scaling_method=='MinMax':
            scale_fn = MinMaxScaler().fit_transform
        elif self.scaling_method=='Standard':
            scale_fn = StandardScaler().fit_transform
        elif self.scaling_method==None:
            scale_fn = lambda X: X
        else:
            print 'Error: unknown scaling method "%s"'%self.scaling_method
            return input_state,{}

        print 'Scaling features using method "%s"'%self.scaling_method
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

        if self.cv_id == 'loo':
            cv = LeaveOneOut(N)
        elif self.pred_mode == 'clf':
            cv = StratifiedKFold(y,n_folds=cv_folds,random_state=train_state)
        elif self.pred_mode == 'reg':
            cv = ShuffleSplit(n=N,n_iter=cv_folds,test_size=int(N/cv_folds))

        output_state = self._run_cv(X,y,cv,output_state)
        output_state.update({'cv':cv,'cv_id':self.cv_id,'model_eval':self.model_eval,
                             'labmask':labmask,'scaling_method':self.scaling_method})

        jldump(output_state,output_statefile,compress=jlcompress,
               cache_size=jlcache)

        return input_state, output_state

    def run(self,X_exp,y_exp,exp_name,exp_dir=output_dir):
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
        input_state, output_state = self._collect_state(input_statefile,
                                                        output_statefile)
        return input_state, output_state            

    def model_coef(self,exp_name,exp_dir=output_dir):    
        input_statefile  = pathjoin(exp_dir,'_'.join([exp_name,'input.pkl']))
        output_statefile = pathjoin(exp_dir,'_'.join([exp_name,self.scaling_method,self.pred_mode,'output.pkl']))
        try:
            input_state  = jlload(input_statefile,mmap_mode=None)
            output_state = jlload(output_statefile,mmap_mode=None)
        except:
            print 'Error: unable to load input/output state files in exp_dir=%s'%exp_dir
            return {}

        coef = {}
        for model_id in output_state['model_eval']:
            models = output_state[model_id]['models']
            coef_fn = self.model_params[model_id][-1]
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
    cv_folds = 10
    
    data  = load_digits()
    X_exp = data.data.copy()
    y_exp = data.target.copy()
    mask5  = (y_exp==5)
    mask8  = (y_exp==8)

    y_clf        =  y_exp
    y_clf[:]     =  0
    y_clf[mask5] = -1
    y_clf[mask8] =  1
    y_reg = c_[y_clf==1,y_clf==-1].astype(float)

    clf_models = ['linsvc']
    clf_exp = DISCRIM_EXP(pred_mode='clf',model_eval=clf_models,cv_folds=cv_folds)
    clf_exp.run(X_exp,y_clf,'digits5v8_clf',exp_dir='/tmp/discrim_test')
    
    reg_models = ['linreg']
    reg_exp = DISCRIM_EXP(pred_mode='reg',model_eval=reg_models,cv_folds=cv_folds)
    reg_exp.run(X_exp,y_reg,'digits5v8_reg',exp_dir='/tmp/discrim_test')
