
import numpy as np 

class Tonic_DA_model:
    def __init__(self, EC501=1000, EC502=10, DA_range = [1.1,3],DA_step = 0.05,delta_DA = 3,beta=0.001):
        self.ec50_1 = EC501
        self.ec50_2 = EC502
        self.DA_range = DA_range
        self.DA_step = DA_step
        self.delta_DA = delta_DA
        self.beta = beta
        self.DA = 10**(np.arange(self.DA_range[0],self.DA_range[1],step =self.DA_step))
    
    def compute_dose_occupancy(self, DA_range=None):
        if DA_range is None:
            DA = self.DA
        else:
            DA = 10**(np.arange(DA_range[0],DA_range[1],step =self.DA_step))
        ec1 = self.ec50_1
        ec2 = self.ec50_2

        occ_D1 = DA/(DA+ec1)
        occ_D2 = DA/(DA+ec2)
        self.occ_D1 = occ_D1
        self.occ_D2 = occ_D2

        return occ_D1, occ_D2, DA
        
    def compute_sensitivities(self, type_scale = 'log'):
        dDA = self.delta_DA
        DA = self.DA
        ec1 = self.ec50_1
        ec2 = self.ec50_2
        slope_D1 = np.zeros(len(DA))
        slope_D2 = np.zeros(len(DA))
        for (ic,c) in enumerate(DA):
            if type_scale=='log':
                x1,x2 = c,c/dDA
                y1,y2 = (x1/(ec2+x1)),(x2/(ec2+x2))
                slope_D2[ic] = -(y2-y1)/(x1/x2)

                x1,x2 = c,c*dDA
                y1,y2 = (x1/(ec1+x1)),(x2/(ec1+x2))
                slope_D1[ic] = (y2-y1)/(x2/x1)

            elif type_scale =='lin':
                x1,x2 = c,c-dDA
                y1,y2 = (x1/(ec2+x1)),(x2/(ec2+x2))
                slope_D2[ic] = -(y2-y1)/(x1-x2)

                x1,x2 = c,c+dDA
                y1,y2 = (x1/(ec1+x1)),(x2/(ec1+x2))
                slope_D1[ic] = (y2-y1)/(x2-x1)

            else:
                raise(NotImplementedError)
        
        if type_scale=='log':
            self.slope_D1_log = slope_D1
            self.slope_D2_log = slope_D2
        elif type_scale =='lin':
            self.slope_D1_lin = slope_D1
            self.slope_D2_lin = slope_D2
        else:
            raise(NotImplementedError)

        return slope_D1, slope_D2
    
    def compute_taus(self,type_scale='log'):
        if type_scale=='log':
            taus = self.slope_D1_log/( self.slope_D1_log+ self.slope_D2_log)
            self.taus_log = taus
        elif type_scale =='lin':
            taus = self.slope_D1_lin/( self.slope_D1_lin+ self.slope_D2_lin)
            self.taus_lin = taus
        else:
            raise(NotImplementedError)

        return taus
    
    def compute_closed_form_bernoulli(self, p = 0.1,r=1):
        taus = self.compute_taus(type_scale='log')
        beta = self.beta
        dec_term = beta/((1-taus)*(1-p))
        num_ = (taus/(1-taus))*(p/(1-p))*r
        den_ = (taus/(1-taus))*(p/(1-p))+1
        v_prediction = num_/(den_+dec_term)

        return v_prediction
    


