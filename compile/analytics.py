#%%
import sys
sys.path.append('C:\\Users\\Chan\\Desktop\\BaseballProject\\python')

#%%
import numpy as np
import pandas as pd
import math
import scipy
import scipy.stats as stats

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
#%%
from baseball import base as bs
from baseball import preprocess as md
from baseball import sample as sp

#%%


s = sp.Sample()
d = bs.Database()
d.load_data_all()
#%%
# data dictionary 기본 세팅
b = md.Modification() 

b.game_info_array = d.game_info_array
b.batter_array = d.batter_array
b.pitcher_array = d.pitcher_array
b.score_array = d.score_array

b.set_dic_all()

#%%
br_range = [20,30]
sp_range = [7,10]
rp_range = [30,50]


#%%
import time
start_time = time.time()
b.is_park_factor = True
for br in br_range:
    b.br_range = br
    b.set_xr_dic(1)
for sr in sp_range:
    b.sp_range = sr
    b.set_xr_dic(2)
for rr in rp_range:
    b.rp_range = rr
    b.set_xr_dic(3)
    


range_data_dic = dict()    
for br in br_range:
    for sr in sp_range:
        for rr in rp_range:
            
            b.set_record_total_dic(br,sr,rr)
            range_data_dic[(br,sr,rr)] = b.record_total_dic
            print(br,sr,rr)
            '''
            except:
                print('error!!!!!!!!!: ',br,sr,rr)
            '''
print(time.time() - start_time)

#%%
def get_params(y,x,params):
    y_hat = np.dot(x,params).reshape(-1,1)
    params = list(np.round(params,3))
    mse = np.round(np.mean((y - y_hat)**2),4)
    msle = np.round(np.mean((np.log(y+1) - np.log(y_hat+1))**2),4)
    return params + [mse] + [msle]


    
#%%


total_params_dic = dict()
for year in range(2017,2022):
    total_params_dic[year] = list()

x_idx = (14,19,20,21)
t2_dic = dict()
t_dic = dict()
for br in br_range:
    for sr in sp_range:
        
        for rr in rp_range:
            
            total_data = np.zeros((1,22))
            for year in range(2017,2021):
                range_data = np.zeros((1,22))
                for team_num in range(1,11):
                    team_data = range_data_dic[(br,sr,rr)][year][team_num]
                    range_data = np.vstack([range_data,team_data])
                range_data = range_data[1:]
                total_data = np.vstack([total_data,range_data])
                
                
                
                Y = range_data[:,10].reshape(-1,1).astype(np.float)
                X = range_data[:,x_idx].astype(np.float) #공격
                
                inn = X[:,1].reshape(-1,1)
                sp_fip = X[:,2].reshape(-1,1)
                rp_fip = X[:,3].reshape(-1,1)
                
                sp = (inn*sp_fip)/9
                rp = ((9-inn)*rp_fip)/9
                
                new_X = np.sum([sp,rp],axis = 0)
                X = np.hstack([X,new_X])
                X_home = np.where(range_data[:,7] == 'home',1,0).reshape(-1,1)
                X = np.hstack([X,X_home])
                X_data = sm.add_constant(X[:,(0,-2,-1)])    
                model = sm.GLM(Y,X_data,family = sm.families.Gamma(link = sm.genmod.families.links.identity)).fit()
                params = model.params
                
                
                year_params = get_params(Y, X_data, params)
                year_params = [[br,sr,rr],year_params]
                total_params_dic[year].append(year_params)
                
            total_data = total_data[1:,:]
            Y = total_data[:,10].reshape(-1,1).astype(np.float)
            X = total_data[:,x_idx].astype(np.float) #공격
            inn = total_data[:,18].reshape(-1,1).astype(np.float)
            
            sp_fip = X[:,2].reshape(-1,1)
            rp_fip = X[:,3].reshape(-1,1)
            '''
            inn = np.where(inn>7,7,inn)
            inn = np.where(inn<2,2,inn)
            
            sp_fip = np.where(sp_fip>6,6,sp_fip)
            sp_fip = np.where(sp_fip<1,1,sp_fip)
                            
            rp_fip = np.where(rp_fip>6,6,rp_fip)
            rp_fip = np.where(rp_fip<1,1,rp_fip)
            '''
            sp = (inn*sp_fip)/9
            rp = ((9-inn)*rp_fip)/9
            
            new_X = np.sum([sp,rp],axis = 0)
            X = np.hstack([X,new_X])
            X_home = np.where(total_data[:,7] == 'home',1,0).reshape(-1,1)
            X = np.hstack([X,X_home])
            XX = X[:,(0,-2,-1)]
            X_data = sm.add_constant(XX)
            
            model = sm.GLM(Y,X_data,family = sm.families.Gamma(link = sm.genmod.families.links.identity)).fit()
            params = model.params
            total_params = get_params(Y,X_data,params)
            total_params = [[br,sr,rr],total_params]
            t_dic[(br,sr,rr)] = total_params
            
            model = sm.GLM(Y,XX,family = sm.families.Gamma(link = sm.genmod.families.links.identity)).fit()
            params = model.params
            total_params = get_params(Y,XX,params)
            total_params = [[br,sr,rr],total_params]
            t2_dic[(br,sr,rr)] = total_params
#%%
            
t_dic
#%%
import time
year = 2021

start_time = time.time()

result_total_list = [0] + [np.arange(1,b.max_game_dic[year][i]+1).reshape(-1,1) for i in range(1,11)]


mse_dic = dict()
mean_dic = dict()

range_list = list()
s.size = 10000
params_list = list()
Y_idx = 10
foe_X_idx = (19,20,21)
park_idx = 8
mean_dic = [ dict() for i in range(0,11)]
for br in br_range:
    for sr in sp_range:
        for rr in rp_range:
            range_list.append([br,sr,rr])
            data_list = range_data_dic[(br,sr,rr)][year]
            
            
            sample_total_list = [0]
            record_list = [0]
            for team_num in range(1,11):
                new_data_array = data_list[team_num]
                
                sample_list= list()
                mean_list = list()
                max_game_num = b.max_game_dic[year][team_num]
                
                
                
                foe_X =  new_data_array[:,foe_X_idx].astype(np.float)
                inn = foe_X[:,0].reshape(-1,1)
                sp_fip = foe_X[:,1].reshape(-1,1)
                rp_fip = foe_X[:,2].reshape(-1,1)
                
                
                #new_Y = np.where(new_Y>12,12,new_Y)
                '''
                inn = np.where(inn>7,7,inn)
                inn = np.where(inn<2,2,inn)
                
                sp_fip = np.where(sp_fip>6,6,sp_fip)
                sp_fip = np.where(sp_fip<1,1,sp_fip)
                                
                rp_fip = np.where(rp_fip>6,6,rp_fip)
                rp_fip = np.where(rp_fip<1,1,rp_fip)
                '''
                new_X2 = (inn*sp_fip)/9
                new_X3 = ((9-inn)*rp_fip)/9
                
                new_X4 = np.sum([new_X2,new_X3],axis = 0 )
                
                
                new_data_array = np.hstack([new_data_array,new_X4])
                X_home = np.where(new_data_array[:,7] == 'home',1,0).reshape(-1,1)
                new_data_array = np.hstack([new_data_array,X_home])
                for i in range(max_game_num):
                    
                    
                    range_data = new_data_array[:i+1,:]
                    
         
                    
                    Y = range_data[10:-1,10].reshape(-1,1).astype(np.float)
                    
                    X = range_data[10:-1,(14,-2,-1)].astype(np.float)
                    
                    
                    
                    
                   
                    x = range_data[-1,(14,-2,-1)].reshape(1,-1).astype(np.float)
                    c = np.ones((1,1))
                    x_data = np.hstack([c,x])
                    
                    
                    m_list = list()
                    
                    
                    if i <=10:
                        m_list.append(np.dot(x_data,np.array(t_dic[(br,sr,rr)][1][:4])))
                        var = 12.5
                    else:
                        
                        C = np.ones((i-10,1))
                        X_data = np.hstack([C,X])
                        
                        nY = Y - X[:,1].reshape(-1,1)
                        nX = X[:,0].reshape(-1,1) - X[:,1].reshape(-1,1)
                        
                        try:
                            
                            if i <50:
                            #m_list.append(np.dot(x,np.array(t2_dic[(br,sr,rr)][1][:3])))
                                m_list.append(np.dot(x_data,np.array(t_dic[(br,sr,rr)][1][:4])))
                                
                                '''
                                model1 = sm.GLM(Y,X,family = sm.families.Gamma(link = sm.genmod.families.links.identity)).fit()
                                params = model1.params
                                mean = np.dot(x,params)
                                m_list.append(mean)
                                
                                model = sm.GLM(Y,X_data,family = sm.families.Gamma(link = sm.genmod.families.links.identity)).fit()
                                params = model.params
                                mean = np.dot(x_data,params)
                                m_list.append(mean)
                                
                                
                                model = sm.OLS(Y,X).fit()
                                params = model.params
                                mean = np.dot(x,params)
                                m_list.append(mean)
                                
                                '''
                            else:
                                m_list.append(np.dot(x_data,np.array(t_dic[(br,sr,rr)][1][:4])))
                                '''
                                model = sm.OLS(Y,X_data).fit()
                                params = model.params
                                mean= np.dot(x_data,params)
                                
                                
                                m_list.append(mean)
                                
                                var = model.mse_resid
                            
                            model = sm.OLS(nY,sm.add_constant(nX)).fit()
                            params = model.params
                            mean = params[0] + params[1]*x[0,0] + (1-params[1])*x[0,1]
                            m_list.append(mean)
                            '''
                            
                        except:
                            
                            '''
                            model1 = sm.OLS(Y,X_data).fit()
                            params = model1.params
                            mean1 = np.dot(x_data,params)
                            var = model1.mse_resid
                            m_list = [mean1]
                            '''
                            m_list.append(np.dot(x_data,np.array(t_dic[(br,sr,rr)][1][:4])))
                            print('error!!')
                            
                    '''
                    params = total_params_dic[(br,sr,rr)]
                    
                    mean2 = params[0] + params[1]*x1 + params[2]*x2
                    '''
                    mean = np.mean(m_list)
                    
                    if mean_dic[team_num].get(i)==None:
                        mean_dic[team_num][i] = [mean]
                    else:
                        mean_dic[team_num][i].append(mean)
                    mean = np.mean(mean_dic[team_num][i])
                    
                    '''
                    if mean<=1: mean=1
                    
                    if mean>=6: mean=6
                    '''
                    #beta = var/mean
                    beta = 2.5
                    alpha = mean / beta
                    if alpha < 0.5:
                        alpha = 0.5
                    
                    
                    sample = s.gamma_sample(theta = [alpha,beta])
                    #sample = s.norm_sample(theta = [mean,np.sqrt(var)])
                    
                    sample_list.append(sample)
                    mean_list.append(mean)
                    
                
                sample_total_list.append(sample_list)
                mean_array = np.array(mean_list).reshape(-1,1)
                
                
                new_data_array = np.hstack([new_data_array,mean_array])
                record_list.append(new_data_array)
                    
                    
            for team_num  in range(1,11):
                sample_list = sample_total_list[team_num]
                record_array = record_list[team_num]
                result_list = list()
                foe_mean_list = list()
                max_game_num = b.max_game_dic[year][team_num]
                for i in range(max_game_num):
                    date = record_array[i,0]
                    team_sample = sample_list[i]
                    foe_num = record_array[i,5]
                    foe_data = record_list[foe_num]
                    foe_game_num = foe_data[foe_data[:,0]==date,6][0]
                    foe_sample = sample_total_list[foe_num][foe_game_num-1]
                    result_gamma = round(np.sum(team_sample>foe_sample)/s.size,3)
                    foe_mean = foe_data[foe_game_num-1,-1]
                    result_list.append(result_gamma)
                    foe_mean_list.append(foe_mean)
                    
                    
                
                
                result_array = np.array(result_list).reshape(-1,1)
                foe_mean_array = np.array(foe_mean_list).reshape(-1,1)
                record_array = np.hstack([record_array,result_array,foe_mean_array])

                exp_rate = record_array[:,-2].reshape(-1,1)
                
                
                result_total_list[team_num] = np.hstack([result_total_list[team_num],exp_rate])
                        
                        
            print([br,sr,rr])



#%%
            
            
basic_list = b.game_info_dic[year]
b.set_toto_dic()
toto_list = b.toto_dic[2020]


for team_num in range(1,11):
    #team_win = np.where(toto_list[team_num][:,6]>=toto_list[team_num][:,7],1,0)
    basic_array = basic_list[team_num][:,:10]
    toto_rate_array = toto_list[team_num][:b.max_game_dic[year][team_num],-4].reshape(-1,1)
    #basic_list[team_num][:,-1] = team_win
    
    basic_list[team_num] = np.hstack([basic_array,toto_rate_array,result_total_list[team_num][:,1:]])
    
    #basic_list[team_num] = np.hstack([basic_array,np.mean(basic_list[team_num][:,11:],axis = 1).reshape(-1,1)])

new_range_list = ["toto"] + range_list + ["mean"]





len_record = len(basic_list[1][0]) 
result_idx = 9


max_range_list = [0,0,0,0,0]
max_rate_list = [0,0,0,0,0]
max_count_list = [0,0,0,0,0]
max_list = [0,0,0,0,0]
min_list = [1000]*5
min_range_list = [0,0,0,0,0]

for i,num in enumerate(range(result_idx+1, len_record)):
    
    count = 0
    toto_count = 0
    toto_count_win = [0 for i in range(20)]
    toto_count_total = [0 for i in range(20)]
    toto_mse = 0
    toto_count_sum = 0
    total_array = np.zeros((1,2))
    for team_num in range(1,11):
        record_team = basic_list[team_num][10:]
        Y = record_team[:,result_idx].reshape(-1,1)
        
        X = record_team[:,num].reshape(-1,1).astype(np.float)
        
        
        new_array = np.hstack([Y,X])
        total_array = np.vstack([total_array,new_array])
        '''
        if i !=0:
            for i in range(len(record_team)):
                if record_team[i,7] == "home":
                    record_team[i,num] +=0.03
                else:
                    record_team[i,num] -=0.03
        '''
        record_team[:,num] = np.where(record_team[:,num]<0.25,0.25,record_team[:,num])
        record_team[:,num] = np.where(record_team[:,num]>0.75,0.75,record_team[:,num])
        


        win_count_mask = record_team[:,num] > 0.5
        wc_array = np.where(win_count_mask,1,0)
        toto_count += sum([1 for wc,y in zip(wc_array,Y) if wc==y])
        
        toto_mse += np.sum(record_team[win_count_mask,num] - record_team[win_count_mask,result_idx])
        count+=len(record_team[:,0])
        
        for i in range(20): # 구간 나누기
            toto_count_win[i]+=np.sum(record_team[(record_team[:,num]>i*0.05) & (record_team[:,num]<=(i+1)*0.05),result_idx])
            toto_count_total[i]+= len(record_team[(record_team[:,num]>i*0.05) & (record_team[:,num]<=(i+1)*0.05),num])

    
    pred_range = new_range_list[num-10]
    pred_rate_list = np.round(np.divide(toto_count_win,toto_count_total),3)[10:]
    
    pred_count_list = toto_count_total[10:]
    pred_rate = np.round(toto_count/count,3)
    
    toto_mse = np.round(toto_mse,5)
    if min(max_list) < pred_rate:
        new_idx = max_list.index(min(max_list))
        max_range_list[new_idx] = pred_range
        max_list[new_idx] = pred_rate
        max_rate_list[new_idx] = pred_rate_list
        max_count_list[new_idx] = pred_count_list
    
    if max(min_list) > toto_mse:
        new_idx = min_list.index(max(min_list))
        min_range_list[new_idx] = pred_range
        min_list[new_idx] = toto_mse
        
    result_Y = total_array[:,0].astype(np.int)
    result_X = total_array[:,1].astype(np.float)
    
    auc = np.round(metrics.roc_auc_score(result_Y,result_X),3)
    
    
    
    print(pred_range)
    print(['50','55','60','65','70','75'])
    print(pred_rate_list)
    print(pred_count_list)
    print(pred_rate)
    print(auc)
    print(np.round(toto_mse/count,3))
    #print(' ')



