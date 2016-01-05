from util.aliases import *

from sklearn.linear_model import LinearRegression, OrthogonalMatchingPursuit, LassoLarsCV
from sklearn.svm import LinearSVC, SVC, LinearSVR, SVR
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier

from sklearn.metrics import precision_score, recall_score, mean_squared_error, \
    explained_variance_score, coverage_error

# 1. SERIALIZATION PARAMS ######################################################
output_dir = './discrim_output'
jlcompress = 9 # int \in [0,9] 
jlcache    = 500 # cache size in mb for joblib IO


# 2. TRAINING / CROSS-VALIDATION PARAMS ########################################
train_folds    = inf # (integer->KFold, inf->LeaveOneOut)
train_verbose  = 0 # verbosity level of training algorithm output
train_jobs     = -1 # -1 = all cores
train_state    = 42 # random state


# 3. GRID SEARCH PARAMS FOR PARAMETER TUNING ###################################
gridcv_folds   =  2 # number of cross-validation folds per gridcv parameter 
gridcv_jobs    =  -1  # -1 = use all cores
gridcv_verbose =  1 # verbosity level of model-tuning cross-validation output
default_gridcv = {'clf':'roc_auc','reg':'mean_squared_error'}


# 4. SCORING AND ERROR FUNCTIONS ###############################################
default_scores = {'clf':['precision','recall'],'reg':['mse']}
default_errors = {'clf':['match'],'reg':['diff']}

clf_ulab             = [-1,1] # assume class labels \in {-1, 1}

# score functions
scorefn = {} # map from name (e.g., mse) -> f(y_true,y_pred)
scorefn['precision'] = lambda te,pr,ul=clf_ulab: precision_score(te,pr,labels=ul)
scorefn['recall']    = lambda te,pr,ul=clf_ulab: recall_score(te,pr,labels=ul)
scorefn['mse']       = lambda te,pr,_=None: mean_squared_error(te,pr)
scorefn['exp']       = lambda te,pr,_=None: explained_variance_score(te,pr)

# error/loss functions
errorfn = {} # map from name (e.g., diff) -> f(y_true,y_pred)
errorfn['match'] = lambda y_true,y_pred: y_true==y_pred
errorfn['diff']  = lambda y_true,y_pred: y_true-y_pred


# 5. DEFAULT MODEL PARAMETERS ##################################################
default_models = {'clf':('linsvm','rbfsvm','rf','xgb'),
                  'reg':('linreg','omp','rf')}

# flag which models cannot handle n-dimensional vectors as labels
model_nomulti = ('linsvm','rbfsvm','xgb','lassolars')

# generic model template: default+tuned parameters, function to retreive coefs
model_default = {'verbose':train_verbose,'random_state':train_state}
model_coef = lambda model: model.coef_.squeeze()

# 6. MODEL SPECIFICATIONS ######################################################
### Linear SVM #################################################################
svmC = logspace(-3,8,12) 
linsvm_tuned = {'C':svmC}
linsvm_default = {
    'C':1.0,'verbose':train_verbose,'random_state':train_state
}
linsvm_coef = model_coef
svmEps = logspace(-4,4+1,8)
linsvr_default = linsvm_default.copy()

linsvc_default = linsvm_default.copy()
linsvc_default.update({'penalty':'l2','loss':'squared_hinge'})
linsvc_tuned = linsvm_tuned.copy()

linsvr_default = linsvm_default.copy()
linsvr_default.update({'loss':'epsilon_insensitive','epsilon':0.0})
linsvr_tuned = linsvm_tuned.copy()
linsvr_tuned.update({'epsilon':svmEps})

### RBF SVM ####################################################################
rbfsvm_tuned = {'C':svmC,'gamma':logspace(-7,4,12)}
rbfsvm_default = {
    'C':1.0,'kernel':'rbf','gamma':1,'verbose':train_verbose,'max_iter':10e5
}
# rbf coef in dual space, inaccessible in primal, and thus == nan
rbfsvm_coef = lambda model: ones(model.support_vectors_.shape[1])*nan 

rbfsvc_default = rbfsvm_default.copy()
rbfsvc_default.update({'random_state':train_state,'probability':False})
rbfsvc_tuned = rbfsvm_tuned.copy()

rbfsvr_default = rbfsvm_default.copy()
rbfsvr_default.update({'epsilon':0.0})

rbfsvr_tuned = rbfsvm_tuned.copy()
rbfsvr_tuned.update({'epsilon':svmEps})

### Random Forest ##############################################################
rf_trees = 400
rf_feats = linspace(0.1,1,5)
rf_depth = [2,4,10,25,100]
rf_jobs  = 1 # multiprocessing + randomforest can cause race conditions

rf_tuned = {'max_features':rf_feats,'max_depth':rf_depth}
rf_default = {
    'n_estimators': rf_trees,'max_features':'sqrt','n_jobs':rf_jobs,
    'verbose':train_verbose,'random_state':train_state
}
rf_coef = lambda model: model.feature_importances_

rfc_default = rf_default.copy()
rfc_default.update({'criterion':'gini','class_weight':'subsample'})

rfr_default = rf_default.copy()
rfr_default.update({'criterion':'mse'})

### XGBoost ####################################################################
xgb_depth = [3,4,5,10,25]
xgb_subsample = linspace(0.1,1,5)
xgb_default = {
    'n_estimators':rf_trees,'max_delta_step':1,'learning_rate':0.1,
    'objective':'binary:logistic','max_depth':3,'subsample':0.5,
    'colsample_bytree':1,'subsample':1,'silent':(not train_verbose),
    'seed':train_state,'nthread':train_jobs
}
xgb_tuned = {'learning_rate':[0.001,0.01,0.05,0.1,0.25,0.33],
             'max_depth':xgb_depth,'subsample':xgb_subsample}

def xgb_coef(model):
    fscores = model.booster().get_fscore()
    return dict([(int(key[1:]),val) for key,val in fscores.iteritems()])


### Linear regression ##########################################################
linreg_default = {'fit_intercept':True,'normalize':False,'copy_X':true,
                  'n_jobs':train_jobs}
linreg_tuned   = {}
linreg_coef = model_coef

### Orthogonal Matching Pursuit ################################################
omp_default = {'fit_intercept':True,'normalize':False,
               'n_nonzero_coefs':None, 'tol':None}
omp_tuned = {}
omp_coef = model_coef

### LassoLars ##################################################################
# use the lassolarscv object to xvalidate rather than gridsearch
lassolars_default = {'fit_intercept':True,'normalize':False,'copy_X':True,
                     'cv':train_folds,'n_jobs':gridcv_jobs,'max_n_alphas':1000}
lassolars_tuned   = {}
lassolars_coef    = model_coef    

# 7. ADD MODELS TO MODEL_PARAMS ################################################

# model_params: map from model_id -> tuple(default, tuning_params, coef_fn)
model_params  = {}

model_params['linsvc'] = (LinearSVC(**linsvc_default),[linsvc_tuned],linsvm_coef)
model_params['linsvr'] = (LinearSVR(**linsvr_default),[linsvr_tuned],linsvm_coef)
model_params['rbfsvc'] = (SVC(**rbfsvc_default),[rbfsvc_tuned],rbfsvm_coef)
model_params['rbfsvr'] = (SVR(**rbfsvr_default),[rbfsvr_tuned],rbfsvm_coef)

model_params['rfc'] = (RandomForestClassifier(**rfc_default),[rf_tuned],rf_coef) 
model_params['rfr'] = (RandomForestRegressor(**rfr_default),[rf_tuned],rf_coef) 
model_params['xgb'] = (XGBClassifier(**xgb_default),[xgb_tuned],xgb_coef)

model_params['linreg'] = (LinearRegression(**linreg_default),
                          [linreg_tuned],linreg_coef)
model_params['omp'] = (OrthogonalMatchingPursuit(**omp_default),
                       [omp_tuned],omp_coef)
model_params['lassolars']  = (LassoLarsCV(**lassolars_default),
                              [lassolars_tuned],lassolars_coef)

