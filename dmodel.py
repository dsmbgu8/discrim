## TODO: replace model_params tuple with DMODEL class instances
class DMODEL:
    def __init__(self,model,**kwargs):
        self.model          = model
        self.model_coef     = kwargs.pop('model_coef',model_coef)
        self.default_params = kwargs.pop('default_params',model_default)
        self.tuning_params  = kwargs.pop('tuning_params',{})
        self.multi_output   = False

    #@abstractmethod
    def coef(self):
        """Return model coefs."""
        return self.model_coef(self.model)

    def fit(self,*args,**kwargs):
        return self.model.fit(*args,**kwargs)

    def predict(self,*args,**kwargs):
        return self.model.predict(*args,**kwargs)
