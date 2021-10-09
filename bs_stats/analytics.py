#%%
#%%
import sys
sys.path.append('D:\\BaseballProject\\python')
import numpy as np
import scipy.stats as stats
from bs_stats import sample as sp
#%%
s = sp.Sample()
#%%
class Analytics():
    
    def __init__(self):
        self.total_len = 19
        
        self.run_idx = 10
        self.run_xr_idx = 11
        self.x_idx = tuple([i for i in range(12,20)])
        
        return
    
    
        
    def get_score(self,y,x,params,link,alpha = 3):
    
        y = y.reshape(-1,1)
        y_hat = np.dot(x,params).reshape(-1,1)
        if link == 'inverse':
            y_hat = 1/y_hat
            
        elif link =='log':
            y_hat = np.exp(y_hat)
            
            
        params = list(params)
        
        beta = y_hat / alpha
        
        
        rmse = np.round(np.sqrt(np.mean(((y - y_hat)**2))),4)
        cp = np.mean(s.gamma_pdf(y,[alpha,beta]))
        
        #msle = np.round(np.mean((np.log(y+1) - np.log(y_hat+1))**2),4)
        #mae = np.round(np.mean(abs(y-y_hat)),4)
        
        
        
        return [rmse, cp]#msle, mae, mge]
    
    def get_mean(self,x,params,link):
        y_hat = np.dot(x,params).reshape(-1,1)
        if link == 'inverse':
            y_hat = 1/y_hat
            
        elif link =='log':
            y_hat = np.exp(y_hat)
        return y_hat
    
    def get_input(self, total_array, min_sp_len = 0, max_sp_len = 100):
        
        sp_len_idx = 16
        len_array = total_array[:,sp_len_idx].astype(np.float)
        len_mask = (len_array >= min_sp_len) & (len_array <= max_sp_len)
        total_array = total_array[len_mask,:]
        
        
        run = total_array[:,self.run_idx].reshape(-1,1).astype(np.float)
        run_xr = total_array[:,self.run_xr_idx].reshape(-1,1).astype(np.float)
        X = total_array[:,12:].astype(np.float)
        
        xr = X[:,0].reshape(-1,1)
        
        inn = X[:,1].reshape(-1,1)
        #inn = np.where(inn<=4,4,inn)
        
        sp_era = X[:,2].reshape(-1,1)
        sp_fip = X[:,3].reshape(-1,1)
        sp_len = X[:,4].reshape(-1,1)
        rp_era = X[:,5].reshape(-1,1)
        rp_fip = X[:,6].reshape(-1,1)
        
        
        
        sp_era = np.where(sp_era>=8,8,sp_era)
        sp_fip = np.where(sp_fip>=8,8,sp_fip)
        #sp_fip = (sp_era*5 + sp_fip*5) / 10
        
        
        rp_era = np.where(rp_era>=8,8,rp_era)
        rp_fip = np.where(rp_fip>=8,8,rp_fip)

        #rp_fip = (rp_era*5 + rp_fip*5) / 10
        
        fip_ratio = sp_fip/4.5
        fip_ratio = np.where(fip_ratio <=0.7,0.7,fip_ratio)
        fip_ratio = np.where(fip_ratio >=1.3,1.3,fip_ratio)
        
        
        
        sp_era = (inn*sp_era) / 9 
        sp_fip = (inn*sp_fip) / 9 
        rp_fip = ((9-inn)*rp_fip)/9 
        rp_era = ((9-inn)*rp_era)/9
        
        X_home = np.where(total_array[:,7] == 'home',1,0).reshape(-1,1)
        X = np.hstack([xr, sp_fip, rp_fip, X_home,sp_len])
        
        #X = np.hstack([xr, inn, sp_fip, rp_fip, X_home,sp_len])
        
        X_data = np.hstack([np.ones(len(X)).reshape(-1,1),X])
        
        result_array = np.hstack([run,run_xr,X_data])  # result_array(n x 3): xr , fip, sp_len
        return result_array
    
    