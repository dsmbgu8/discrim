from util.aliases import *

from sklearn.linear_model import LinearRegression, OrthogonalMatchingPursuit, LassoLarsCV
from sklearn.svm import LinearSVC, SVC, LinearSVR, SVR
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier

from sklearn.metrics import precision_score, recall_score, mean_squared_error, \
    explained_variance_score, coverage_error

##### Prediction mode (clf=classification, reg=regression) #####################
pred_mode = 'reg' # 'clf' #  

##### Models to evaluate/cv objective function #################################
if pred_mode=='clf':
    model_eval = ('linsvm','rbfsvm','rf','xgb')
    gridcv_score = 'roc_auc'
else:
    model_eval = ('linreg','omp','rf')
    gridcv_score = 'mean_squared_error'
    
##### Training / cross-validation params #######################################
cv_folds       = 10 # (integer->KFold, inf->LeaveOneOut)
cv_id          = 'loo' if cv_folds == inf else '%d-fold'%cv_folds
train_verbose  = 0 # verbosity level of training algorithm output
train_jobs     = 1 # -1 = all cores
train_state    = 42 # random state

##### Grid search params for parameter tuning ##################################

gridcv_folds   =  2 # number of cross-validation folds per gridcv parameter 
gridcv_jobs    =  -1  # -1 = use all cores
gridcv_verbose =  1 # verbosity level of model-tuning cross-validation output


##### Default model parameters #################################################

# generic model template
model_default = {'verbose':train_verbose,'random_state':train_state}
model_tuned = {}
model_coef = lambda model: model.coef_.squeeze()


# Linear SVM    
svmC = logspace(-3,8,12) 
linsvm_tuned = {'C':svmC}
linsvm_default = {
    'C':1.0,'verbose':train_verbose,'random_state':train_state
}
linsvm_coef = model_coef

# RBF SVM
rbfsvm_tuned = {'C':svmC,'gamma':logspace(-7,4,12)}
rbfsvm_default = {
    'C':1.0,'kernel':'rbf','gamma':1,'verbose':train_verbose,'max_iter':10e5
}
# rbf coef in dual space, inaccessible in primal, and thus == nan
rbfsvm_coef = lambda model: ones(model.support_vectors_.shape[1])*nan 

# Random Forest
rf_trees = 400
rf_feats = linspace(0.1,1,5)
rf_depth = [2,4,10,25,100]

rf_tuned = {'max_features':rf_feats,'max_depth':rf_depth}
rf_default = {
    'n_estimators': rf_trees,'max_features':'sqrt','n_jobs':train_jobs,
    'verbose':train_verbose,'random_state':train_state
}
rf_coef = lambda model: model.feature_importances_

# XGBoost
xgb_default = {
    'n_estimators':rf_trees,'max_delta_step':1,'learning_rate':0.1,
    'objective':'binary:logistic','max_depth':3,'subsample':0.5,
    'colsample_bytree':1,'subsample':1,'silent':(not train_verbose),
    'seed':train_state,'nthread':train_jobs
}
xgb_subsample = linspace(0.1,1,5)
xgb_tuned = {'learning_rate':[0.001,0.01,0.05,0.1,0.25,0.33],
             'max_depth':rf_depth,'subsample':xgb_subsample}

def xgb_coef(model,dim):
    fscores = model.booster().get_fscore()
    findex  = map(lambda f: int(f[1:]), fscores.keys())
    fscore  = zeros(dim)
    fscore[findex] = fscores.values()
    return fscore / double(fscore[findex].sum())

# linear regression 
linreg_default = {'fit_intercept':True,'normalize':False,'copy_X':true,
                  'n_jobs':train_jobs}
linreg_tuned   = {}
linreg_coef = model_coef

# Orthogonal Matching Pursuit
omp_default = {'fit_intercept':True,'normalize':False,
               'n_nonzero_coefs':None, 'tol':None}
omp_tuned = {}
omp_coef = model_coef

# LassoLars
# use the lassolarscv object to xvalidate rather than gridsearch
lassolars_default = {'fit_intercept':True,'normalize':False,'copy_X':True,
                     'cv':cv_folds,'n_jobs':gridcv_jobs,'max_n_alphas':1000}
lassolars_tuned   = {}
lassolars_coef    = model_coef    

model_params  = {} # map from model_id -> tuple(default, tuning_params, coef_fn)
scorefn = {} # map from name (e.g., mse) -> f(y_true,y_pred)
errorfn = {} # map from name (e.g., diff) -> f(y_true,y_pred)
##### pred_mode specific parameters ############################################
if pred_mode == 'clf':
    ulab = [-1,1]
    scorefn['precision'] = lambda te,pr,ul=ulab: precision_score(te,pr,labels=ul)
    scorefn['recall']    = lambda te,pr,ul=ulab: recall_score(te,pr,labels=ul)
    errorfn['equal']     = lambda y_true,y_pred: y_true==y_pred
    
    LinSVM = LinearSVC
    linsvm_default.update({'penalty':'l2','loss':'squared_hinge'})

    RBFSVM = SVC
    rbfsvm_default.update({'random_state':train_state,'probability':False}) 

    RF = RandomForestClassifier # ExtraTreesClassifier #
    rf_default.update({'criterion':'gini','class_weight':'subsample'})    

    model_params['xgb']  = (XGBClassifier(**xgb_default),[xgb_tuned],xgb_coef)

elif pred_mode == 'reg':
    scorefn['mse']  = lambda te,pr,_=None: mean_squared_error(te,pr)
    #scorefn['exp']  = lambda te,pr,_=None: explained_variance_score(te,pr)
    errorfn['diff'] = lambda y_true,y_pred: y_true-y_pred
    
    model_params['linreg']     = (LinearRegression(**linreg_default),
                                  [linreg_tuned],linreg_coef)
    model_params['omp']        = (OrthogonalMatchingPursuit(**omp_default),
                                  [omp_tuned],omp_coef)
    model_params['lassolars']  = (LassoLarsCV(**lassolars_default),
                                  [lassolars_tuned],lassolars_coef)
    
    svmEps = logspace(-4,4+1,8)
    LinSVM = LinearSVR
    linsvm_default.update({'loss':'epsilon_insensitive','epsilon':0.0})
    linsvm_tuned.update({'epsilon':svmEps})

    RBFSVM = SVR
    rbfsvm_default.update({'epsilon':0.0})
    rbfsvm_tuned.update({'epsilon':svmEps})

    RF = RandomForestRegressor # ExtraTreesRegressor # 
    rf_default.update({'criterion':'mse'})


model_params['linsvm'] = (LinSVM(**linsvm_default),[linsvm_tuned],linsvm_coef)
model_params['rbfsvm'] = (RBFSVM(**rbfsvm_default),[rbfsvm_tuned],rbfsvm_coef)
model_params['rf']     = (RF(**rf_default),[rf_tuned],rf_coef)
    
