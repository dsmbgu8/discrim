import sys, os, warnings

from util.aliases import *

from progressbar import ProgressBar, ETA, Bar, Percentage

from sklearn.base import clone
from sklearn.externals.joblib import load as jlload, dump as jldump
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.utils import check_X_y
from sklearn.cross_validation import  LeaveOneOut, StratifiedKFold, ShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import precision_score, recall_score, mean_squared_error

from sklearn.linear_model import LinearRegression, OrthogonalMatchingPursuit
from sklearn.svm import LinearSVC, SVC, LinearSVR, SVR
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from xgboost import XGBClassifier


##### Prediction mode (clf=classification, reg=regression) #####################
pred_mode = 'clf' #  'reg' # 

##### Feature scaling method ###################################################
scaling_method = 'MinMax' # 'Standard' # None # 'Normalize' #

##### Models to evaluate #######################################################
if pred_mode=='clf':
    model_eval = ['linsvm','rbfsvm','rf','xgb']
else:
    model_eval  = ['linreg','omp','rf']

##### Serialization params #####################################################
output_dir = './discrim_output'
jlcompress = 0 # warning: nonzero values disable memory mapping
jlmmap     = None # 'r' # memmap mode for loading (None=disabled)
jlcache    = 500 # cache size in mb for joblib IO

##### Training / cross-validation params #######################################
gridcv_score   = 'roc_auc' if pred_mode=='clf' else 'mean_squared_error'
cv_folds       =  inf # (integer->KFold, inf->LeaveOneOut)
cv_id          = 'loo' if cv_folds == inf else '%d-fold'%cv_folds
train_verbose  = 0 # verbosity level of training algorithm output
train_jobs     = 1 # -1 = all cores
train_state    = 42 # random state

##### Grid search params for parameter tuning ##################################
gridcv_folds   =  2 # number of cross-validation folds per gridcv parameter 
gridcv_jobs    = -1 if train_jobs == 1 else 1 # -1 = use all cores
gridcv_verbose =  0 # verbosity level of model-tuning cross-validation output

##### Default model parameters #################################################

# Linear SVM    
svmC = logspace(-3,8,12) 
linsvm_tuned = {'C':svmC}
linsvm_default = {
    'C':1.0,'verbose':train_verbose,'random_state':train_state
}
linsvm_coef = lambda model: model.coef_.squeeze()

# RBF SVM
rbfsvm_tuned = {'C':svmC,'gamma':logspace(-7,4,12)}
rbfsvm_default = {
    'C':1.0,'kernel':'rbf','gamma':1,'verbose':train_verbose,
    'max_iter':10e5,'random_state':train_state
}
# rbf coef in dual space, inaccessible in primal, and thus == nan
rbfsvm_coef = lambda model: ones(model.support_vectors_.shape[1])*nan 

# Random Forest
rf_features = linspace(0.1,1,10)
rf_depth    = [3,5,10,25,250,5000,25000]
rf_tuned = {'max_features':rf_features,'max_depth':rf_depth}
rf_default = {
    'n_estimators': 400,'max_features':'sqrt',
    'verbose':train_verbose,'random_state':train_state
}
rf_coef = lambda model: model.feature_importances_


model_params  = {} # map from model_id -> tuple(default, tuning_params, coef)
scorefn = {} # map from name -> function 
##### pred_mode specific parameters ############################################
if pred_mode == 'clf':
    LinSVM = LinearSVC
    linsvm_default.update({'penalty':'l2', 'loss':'squared_hinge'})

    RBFSVM = SVC
    rbfsvm_default.update({'probability':False}) 

    RF = ExtraTreesClassifier
    rf_default.update({'criterion':'gini','class_weight':'subsample'})    
    
    # XGBoost
    xgb_default = {
        'n_estimators':400,'max_delta_step':1,'learning_rate':0.1,
        'objective':'binary:logistic','max_depth':3,'subsample':0.5,
        'colsample_bytree':1,'subsample':1,'silent':(not train_verbose),
        'seed':train_state,'nthread':train_jobs
    }
    xgb_tuned = {'learning_rate':[0.001,0.01,0.05,0.1,0.25,0.4],
                 'max_depth':rf_depth,'subsample':[0.25,0.5,0.75]}

    def xgb_coef(model,dim):
        booster = model.booster()
        fscores = booster.get_fscore()
        findex  = map(lambda f: int(f[1:]), fscores.keys())
        fscore  = zeros(dim)
        fscore[findex] = fscores.values()
        return fscore / double(fscore.sum())

    model_params['xgb']  = (XGBClassifier(**xgb_default),[xgb_tuned],xgb_coef)

    ul = [-1,1]
    scorefn['precision'] = lambda te,pr,ul=ul: precision_score(te,pr,labels=ul)
    scorefn['recall']    = lambda te,pr,ul=ul: recall_score(te,pr,labels=ul)
    errorfn              = lambda y_true,y_pred: y_true==y_pred
    
elif pred_mode == 'reg':
    linreg_default = {'fit_intercept':True,'normalize':False,'copy_X':True,
                      'n_jobs':train_jobs}
    linreg_tuned   = {}
    linreg_coef = lambda model: model.coef_
    model_params['linreg']  = (LinearRegression(**linreg_default),
                               [linreg_tuned],linreg_coef)

    omp_default = {'fit_intercept':True,'normalize':False,
                   'n_nonzero_coefs':None, 'tol':None}
    omp_tuned = {}
    omp_coef = lambda model: model.coef_
    model_params['omp']  = (OrthogonalMatchingPursuit(**omp_default),
                               [omp_tuned],omp_coef)
    
    svmEps = logspace(-4,4+1)
    LinSVM = LinearSVR
    linsvm_default.update({'loss':'epsilon_insensitive','epsilon':0})
    linsvm_tuned.update({'epsilon':svmEps})

    RBFSVM = SVR
    rbfsvm_default.update({'epsilon':0})
    rbfsvm_tuned.update({'epsilon':svmEps})

    RF = ExtraTreesRegressor
    rf_default.update({'criterion':'mse'})

    scorefn['mse'] = lambda te,pr,_=None: mean_squared_error(te,pr)
    errorfn        = lambda y_true,y_pred: y_true-y_pred

model_params['linsvm'] = (LinSVM(**linsvm_default),[linsvm_tuned],linsvm_coef)
model_params['rbfsvm'] = (RBFSVM(**rbfsvm_default),[rbfsvm_tuned],rbfsvm_coef)
model_params['rf']     = (RF(**rf_default),[rf_tuned],rf_coef)

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
                errors.append(errorfn(y_test,clf_pred))
                
        if cv_id == 'loo':
            for score,score_fn in scorefn.iteritems():                
                scorei = score_fn(y,y_pred,uy)
                scores[score].append(scorei)
            preds  = [y_pred]
            errors = [errorfn(y,y_pred)]
            models = [model_train(X,y,model_clf,model_tuned)]
            pbar.update(i+1)
        pbar.finish()

        for score,vals in scores.iteritems():
            print 'mean %s: %7.4f'%(score, mean(vals))
            
        output[model_id] = {'preds':preds,'errors':errors,'scores':scores,
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
        cv = ShuffleSplit(cv_folds)
        
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