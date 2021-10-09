#%%
import sys
sys.path.append('D:\\BaseballProject\\python')

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
from sklearn.model_selection import KFold
from sklearn import metrics
#%%
from bs_database import base as bs
from bs_stats import preprocess as pr
from bs_stats import sample as sp
from bs_stats import analytics as an
from bs_personal import personal_code as cd
#%%
a = an.Analytics()
b = pr.Preprocess() 
d = bs.Database()
s = sp.Sample()


#%%
# data dictionary 기본 세팅

d.load_data_all(db_address = cd.db_address ,code = cd.local_code , file_address = cd.file_address)

b.game_info_array = d.game_info_array
b.batter_array = d.batter_array
b.pitcher_array = d.pitcher_array
b.score_array = d.score_array

b.set_dic_all()


#("mysql+pymysql://root:" + "dudrn1" + "@127.0.0.1/baseball",encoding = 'utf-8')
#%%

br_range = [45]
sp_range = [i for i in range(3,21)]
rp_range = [50]


#%%
import time
start_time = time.time()
b.is_iv = False
b.is_new_game = False
b.is_park_factor = True
b.is_pa = False
b.is_epa_xr = True
b.is_epa_sp = True
b.is_epa_rp = False
for br in br_range:
    b.br_range = br
    b.set_range_dic(1)
for sr in sp_range:
    b.sp_range = sr
    b.set_range_dic(2)
for rr in rp_range:
    b.rp_range = rr
    b.set_range_dic(3)

range_data_dic = dict()
for br in br_range:
    for sr in sp_range:
        for rr in rp_range:
            
            b.set_record_total_dic(br,sr,rr)
            range_data_dic[(br,sr,rr)] = b.record_total_dic
            print(br,sr,rr)
            '''
                print('error!!!!!!!!!: ',br,sr,rr)
            except:
            '''
print(time.time() - start_time)


#%%
sm.families.Gamma.safe_links.append(sm.genmod.families.links.identity)
sm.families.Gamma.safe_links.append(sm.genmod.families.links.inverse_power)

    
year_list = [i for i in range(2017,2022)]

#record_array(n x 18): hName(0), aName(1), hRun(2), aRun(3), home_array(4:11), away_array(11:18)
#home_array(n x 7) = hXR, aInn, aSp_era, aSp_fip, aSp_len, aRp_era, aRp_fip


#%%
def get_total_params(range_data_dic,br_range,sp_range,rp_range,year_list, get_type = 'params',is_print = True):

    total_params_list = [dict() for i in range(4)]
    total_scale_list = [dict() for i in range(4)]
    total_score_dic = dict()
    for br in br_range:
        for sr in sp_range:
            for rr in rp_range:
                
                
                min_sp_len = sr
            
                max_sp_len = sr
                
                
                total_array = np.zeros((1,19))
                
                for year in year_list:
                    
                    for team_num in range(1,11):
                        
                        team_array = range_data_dic[(br,sr,rr)][year][team_num]
                        
                        total_array = np.vstack([total_array,team_array])
                
                
                total_array = total_array[1:,:]
                input_array = a.get_input(total_array,min_sp_len,max_sp_len)
                
                run = input_array[:,0]
                run_xr = input_array[:,1]
                X = input_array[:,2:-1]
                cv = KFold(5,shuffle =True)
                
                
                
                model_len = 3
                model_list = [0 for i in range(model_len)]
                model_type_list = ['inverse','log','','']
                score_list = list()
                score_len = 2
                score_array = np.zeros((1,score_len))
                
                params_list = list()
                params_len = len(X[0])
                params_array = np.zeros((1,params_len))

                scale_list = list()                
                
                scale_array = np.zeros((1))
                for i in range(model_len):
                    score_list.append(score_array)
                    params_list.append(params_array)
                    scale_list.append(scale_array)
                
                
                for (train_idx, test_idx) in cv.split(X):
                    X_train = X[train_idx]
                    X_test = X[test_idx]
                    Y_train = run[train_idx]
                    Y_test = run[test_idx]
    
                    model_list[0] = sm.GLM(Y_train,X_train,family = sm.families.Gamma(link = sm.genmod.families.links.inverse_power())).fit()
                    model_list[1] = sm.GLM(Y_train,X_train,family = sm.families.Gamma(link = sm.genmod.families.links.log())).fit()
                    model_list[2] = sm.GLM(Y_train,X_train,family = sm.families.Gamma(link = sm.genmod.families.links.identity())).fit()
                    #model_list[3] = sm.OLS(Y_train,X_train).fit()
                    
                    
                    for i, model in enumerate(model_list):
                        params_list[i] = np.vstack([params_list[i], model.params])
                        
                        new_scale = 1/model.scale
                        scale_list[i] = np.vstack([scale_list[i],new_scale])
                        
                        new_score = a.get_score(Y_test, X_test, model.params, model_type_list[i],alpha = new_scale)
                        score_list[i] = np.vstack([score_list[i],new_score])
                        
                
                i = 0
                for scale, params in zip(scale_list, params_list):
                    
                    params_mean = np.mean(params[1:],axis = 0)
                    
                    total_params_list[i][(br,sr,rr)] = params_mean
                    
                    
                    scale_mean = np.mean(scale[1:])
                    
                    total_scale_list[i][(br,sr,rr)] = scale_mean
                    i+=1
                    
                
                # 2017~2020 데이터 cv-fold 결과
                total_score_array = np.zeros((1,score_len))
                for i, score in enumerate(score_list):
                    total_score_array = np.vstack([total_score_array, np.mean(score,axis = 0)])
                    #print([br,sr,rr], i, '=', np.mean(score,axis = 0))
                
                total_score = np.round(np.mean(total_score_array[1:], axis = 0 ),4)
                total_score_dic[(br,sr,rr)] = total_score
                if is_print:
                    print(len(input_array))
                    print([br,sr,rr], total_score)
                    print(params_mean)
                    print(' ')
    
    if get_type == 'score':
        return [total_score_dic]
    
    else:
        return [total_params_list,total_scale_list]
#%%
total_dic = dict()
for i in range(10):
    score_dic = get_total_params(range_data_dic, br_range, sp_range, rp_range, year_list,get_type = 'score',is_print = False)[0]
    for br in br_range:
        for sr in sp_range:
            for rr in rp_range:
                total_dic[(br,sr,rr)] = total_dic.get((br,sr,rr),0)
                total_dic[(br,sr,rr)] += score_dic[(br,sr,rr)]

print(total_dic)

#%%
total_params_list,total_scale_list = get_total_params(range_data_dic, br_range, sp_range, rp_range, year_list,get_type = 'params',is_print = False)

#%%

range_data_dic[(br,sr,rr)][2021][1][:,18]
#%%
import time

br = 45
sr = 20
rr = 50
model_type_list = ['inverse','log','','']
start_time = time.time()
result_total_dic = dict()
year_list = [i for i in range(2021,2022)]
range_game_num = 170
for year in year_list:
    result_total_list = [0] + [np.arange(1,b.max_game_dic[year][i]+1).reshape(-1,1) for i in range(1,11)]
    team_params_list = [0]
    team_scale_list = [0]
    mse_dic = dict()
    mean_dic = dict()
    
    s.size = 100000
    
    #x_idx = tuple([i for i in range(14,21)])
    #y_idx = 12
    park_idx = 8
    mean_dic = [ dict() for i in range(0,11)]
    
    t_count = 0
    
    a_count = 0
    
    sample_total_list = [0]
    record_list = [0]
    is_range_list = [False for i in range(11)]
    sp_ratio_dic= dict()
    sp_ratio_dic[year] =  [0 for i in range(11)]
    
    for team_num in range(1,11):
        max_game_num = b.max_game_dic[year][team_num]
        sample_list= list()
        mean_list = [[] for i in range(max_game_num)]
        scale_list=  [[] for i in range(max_game_num)]
        params_list = [[] for i in range(max_game_num)]
        
        #br,sr,rr = team_best_num_list[team_num]
        team_array = range_data_dic[(br,sr,rr)][year][team_num]
        
                        
        stat_data = a.get_input(team_array,0)
       
        
        range_mean_list = list()
        
        for game_num in range(max_game_num):#( max_game_num-1,max_game_num):
            
            
            range_data = stat_data[:game_num,:]
            x = stat_data[game_num,2:-1].reshape(1,-1).astype(np.float)
            
            run = range_data[:,0].reshape(-1,1).astype(np.float)
            Y = range_data[:,1].reshape(-1,1).astype(np.float)
            X = range_data[:,2:-1].astype(np.float)
            
            sp_len = int(stat_data[game_num,-1])
            if sp_len <= 7:
                sp_len = 7
            if sp_len >=sr:
                sp_len = sr
            m_list = list()
            s_list = list()
            p_list = [[0] for p in range(3)]
            
            tp_list = list()
            
            for i in range(3):
                tp_list.append(np.array(total_params_list[i][(br,sp_len,rr)]))
            
            
                
            
            if game_num <=range_game_num:
                for i in range(3):
                    m_list.append(a.get_mean(x,tp_list[i],model_type_list[i]))
                pass
                
                
            else:
                
                    
                    
                    
                cv = KFold(5,shuffle = True)
                model_len = 2
                model_list = [0 for i in range(model_len)]
                model_type_list = ['inverse','log','','']
                
                params_list = list()
                params_len = len(X[0])
                params_array = np.zeros((1,params_len))
                
                for j in range(model_len):
                    params_list.append(params_array)
                
                range_score_list = [0,0,0]
                middle_score_list = [0,0,0]
                total_score_list = [0,0,0] 
                
                for (train_idx, test_idx) in cv.split(X):
                    X_train = X[train_idx]
                    X_test = X[test_idx]
                    Y_train = run[train_idx]
                    Y_test = run[test_idx]
                    
                    
                    model_list[0] = sm.GLM(Y_train,X_train,family = sm.families.Gamma(link = sm.genmod.families.links.inverse_power())).fit()
                    model_list[1] = sm.GLM(Y_train,X_train,family = sm.families.Gamma(link = sm.genmod.families.links.log())).fit()
                    #model_list[2] = sm.GLM(Y_train,X_train,family = sm.families.Gamma(link = sm.genmod.families.links.identity())).fit()
                    
                    
                    
                    for j, model in enumerate(model_list):
                        params = model.params
                        params_list[j] = np.vstack([params_list[j], params])
                        params_mean = (params + total_params_list[j][br,sr,rr])/2
                        range_score = a.get_score(Y_test, X_test, params , model_type_list[j])[0]
                        middle_score = a.get_score(Y_test, X_test, params_mean , model_type_list[j])[0]
                        total_score = a.get_score(Y_test, X_test, total_params_list[j][(br,sr,rr)], model_type_list[j])[0]
                        range_score_list[j] += range_score
                        middle_score_list[j] += middle_score
                        total_score_list[j] += total_score
                        
                
                
                        
                for j, params in enumerate(params_list):
                    
                    range_score = range_score_list[j]
                    middle_score = middle_score_list[j]
                    total_score = total_score_list[j]
                    
                    params_mean = np.mean(params[1:],axis = 0)
                    
                    score_sum = middle_score + total_score
                    
                    
                    range_mean = (1-(range_score/ score_sum)) * a.get_mean(x,params_mean,model_type_list[j])
                    total_mean = (1-(total_score / score_sum)) * a.get_mean(x,tp_list[j],model_type_list[j])
                    m_list.append((range_mean + total_mean)/2)
                    
                    #print(team_num, game_num, j,  range_score,middle_score,total_score)
                    #m_list.append(a.get_mean(x,params_mean,model_type_list[j]))
                    #m_list.append(a.get_mean(x,tp_list[j],model_type_list[j]))
                    
                    
                    
                    '''
                    min_score = min(range_score,middle_score,total_score)
                    min_idx = 0
                    if min_score == range_score:
                        m_list.append(a.get_mean(x,params_mean,model_type_list[j]))
                        min_idx = 0
                    elif min_score == middle_score:
                        m_list.append(a.get_mean(x,params_mean,model_type_list[j]))
                        m_list.append(a.get_mean(x,tp_list[j],model_type_list[j]))
                        min_idx = 1
                    elif min_score == total_score:
                        m_list.append(a.get_mean(x,tp_list[j],model_type_list[j]))
                        min_idx = 2
                    
                    print(team_num, game_num, min_idx)    
                    '''
                '''
                
                model_1 = sm.GLM(Y,X,family = sm.families.Gamma(link = sm.genmod.families.links.inverse_power())).fit()
                model_2 = sm.GLM(Y,X,family = sm.families.Gamma(link = sm.genmod.families.links.log())).fit()
                model_3 = sm.GLM(Y,X,family = sm.families.Gamma(link = sm.genmod.families.links.identity())).fit()
                
                r_mean1 = 1/np.dot(x,model_1.params)[0]
                r_mean2 = np.exp(np.dot(x,model_2.params)[0])
                r_mean3 = np.dot(x,model_3.params)[0]
                
                m_list.append(r_mean1)
                m_list.append(r_mean2)
                m_list.append(r_mean3)
                
                p_list.append(model_1.params)
                p_list.append(model_2.params)
                p_list.append(model_3.params)
                '''
                
                '''
                except:
                    for j in range(3):
                        m_list.append(a.get_mean(x,tp_list[j],model_type_list[j]))
                    print('--- error ---')    
                    '''
            range_mean_list.append(m_list[:3])
            mean = np.mean(m_list)
            
            #print(team_num, game_num, m_list,sp_len)
            #scale = np.mean(s_list)
            mean_list[game_num].append(mean)
            
            total_scale = total_scale_list[0][(br,sp_len,rr)]
            if game_num < range_game_num:
                scale_list[game_num].append(total_scale)
                
            else:
                
                scale_model = sm.GLM(run,X,family = sm.families.Gamma(link = sm.genmod.families.links.inverse_power())).fit()
                scale_list[game_num].append(1/scale_model.scale)    
                scale_list[game_num].append(total_scale)
                '''
                score1 = a.get_score(run,X,scale_model.params,link = 'inverse',alpha = 1/scale_model.scale)[1]
                score2 = a.get_score(run,X,scale_model.params,link = 'inverse',alpha = total_scale)[1]
                
                if score1>=score2:
                    scale_list[game_num].append(total_scale)
                else:
                    scale_list[game_num].append(1/scale_model.scale)    
                '''
                #scale_list[game_num].append(total_scale_list[2][(br,sp_len,rr)])
            #scale_list[game_num].append(np.var(run)/mean)
            
            
           
        for i,ml in enumerate(mean_list):
            mean = np.mean(ml)
            
            scale = np.mean(scale_list[i])
            
            alpha = scale
            beta = mean / alpha
            
            
            if beta <=0.5:
                beta = 0.5
            

            sample = s.gamma_sample(theta = [alpha,beta])
            
            
            sample_list.append(sample)
            
        
        team_params_list.append(p_list)
        team_scale_list.append(alpha)
        sample_total_list.append(sample_list)
        
        record_list.append(team_array)
        
    #난수뽑는함수

    
    for team_num  in range(1,11):
        sample_list = sample_total_list[team_num]
        record_array = record_list[team_num]
        result_list = list()
        foe_mean_list = list()
        max_game_num = b.max_game_dic[year][team_num]
        for i in range(max_game_num):
            game_idx = record_array[i,1]
            team_sample = sample_list[i]
            foe_num = record_array[i,5]
            foe_data = record_list[foe_num]
            foe_game_num = foe_data[foe_data[:,1]==game_idx,6][0]
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
                
    result_total_dic[year] = result_total_list
    print([year, time.time() - start_time])
    


#%%
    

for year in year_list:

    basic_list = b.game_info_dic[year]
    b.set_toto_dic()
    toto_list = b.toto_dic[year]
    
    
    '''
    if year ==2021:
        toto_list = b.toto_dic[2020]
        
        
        
    '''
    
    
    total_basic_array = np.zeros((1,3))
    for team_num in range(1,11):
        #team_win = np.where(toto_list[team_num][:,6]>=toto_list[team_num][:,7],1,0)
        basic_array = basic_list[team_num][:,:10]
        toto_rate_array = toto_list[team_num][:b.max_game_dic[year][team_num],-4].reshape(-1,1)
        #basic_list[team_num][:,-1] = team_win
        basic_list[team_num] = np.hstack([basic_array,toto_rate_array,result_total_dic[year][team_num][:,1:]])
        #basic_list[team_num] = np.hstack([basic_array,np.ean(basic_list[team_num][:,11:],axis = 1).reshape(-1,1)])
        total_basic_array = np.vstack([total_basic_array,basic_list[team_num][:,-3:]])
    new_range_list = ["toto","LYC"]
    
    
    len_record = len(basic_list[1][0]) 
    result_idx = 9
    
    
    
    
    
    
    for i, num in enumerate(range(result_idx+1, len_record)):
        
        total_count = 0
        total_correct_count = 0
        total_sum_error = 0
        
        year_count_result_list = [0 for i in range(20)]
        year_count_pred_list = [0 for i in range(20)]
        
        year_array = np.zeros((1,2))
        for team_num in range(1,11):
            
            count_result_list = [0 for i in range(20)]
            count_pred_list = [0 for i in range(20)]
            
            
            record_team = basic_list[team_num][70:]
            record_team = record_team[record_team[:,result_idx]!=0.5,:]
            Y = record_team[:,result_idx].reshape(-1,1).astype(np.int)
            
            X = record_team[:,num].reshape(-1,1).astype(np.float)
            
            new_array = np.hstack([Y,X])
            year_array = np.vstack([year_array,new_array])
            
            win_count_mask = X > 0.5
            wc_array = np.where(win_count_mask,1,0)
            
    
            count = len(Y)
            correct_count = sum(wc_array==Y)[0]
            sum_error = np.round(np.sum(X[win_count_mask] - Y[win_count_mask]),5)
            
    
            total_count += count
            total_correct_count += correct_count
            total_sum_error += sum_error        
            
            #최대 최솟값보정       
            X = np.where(X < 0.25,0.25,X)
            X = np.where(X > 0.75,0.75,X)
            
            #구간 나누기
            for i in range(20): 
                
                range_result = np.sum(Y[(X>i*0.05) & (X<(i+1)*0.05)])
                range_pred = len(X[(X>i*0.05) & (X<(i+1)*0.05)])
                count_result_list[i]= range_result
                count_pred_list[i]= range_pred
                
                year_count_result_list[i] += range_result
                year_count_pred_list[i] += range_pred
            
            pred_range = new_range_list[num-10]
            pred_rate_list = np.round(np.divide(count_result_list,count_pred_list),3)[5:15] 
            
            pred_count_list = count_pred_list[5:15]
            pred_rate = np.round(correct_count/count,3)
            
    
        
        
        pred_range = new_range_list[num-10]
        year_pred_rate_list = np.round(np.divide(year_count_result_list,year_count_pred_list),3)[5:15]
        year_pred_count_list = year_count_pred_list[5:15]
        year_result_count_list = year_count_result_list[5:15]
        year_pred_rate = np.round(total_correct_count/total_count,3)
        
        total_Y = year_array[1:,0]
        total_X = year_array[1:,1]
             
        auc = np.round(metrics.roc_auc_score(total_Y,total_X),3)
            
        
        print(year)
        print(['25','30','35','40','45','50','55','60','65','70','75'])
        
        print(year_pred_rate_list)
        print(year_result_count_list)
        
        print(year_pred_count_list)
        print(year_pred_rate)
        print(auc)
        print(np.round(total_sum_error/total_count,3))
        #print(' ')


    
    #%%

total_array = np.zeros((1,12))
total_basic_array = np.zeros((1,3))
for year in [2017,2018,2019,2020]:

    basic_list = b.game_info_dic[year]
    b.set_toto_dic()
    toto_list = b.toto_dic[year]
    
    
    '''
    if year ==2021:
        toto_list = b.toto_dic[2020]
    '''
    
    
    
    for team_num in range(1,11):
        #team_win = np.where(toto_list[team_num][:,6]>=toto_list[team_num][:,7],1,0)
        basic_array = basic_list[team_num][:,:10]
        toto_rate_array = toto_list[team_num][:b.max_game_dic[year][team_num],-4].reshape(-1,1)
        #basic_list[team_num][:,-1] = team_win
        basic_list[team_num] = np.hstack([basic_array,toto_rate_array,result_total_dic[year][team_num][:,1:]])
        #basic_list[team_num] = np.hstack([basic_array,np.ean(basic_list[team_num][:,11:],axis = 1).reshape(-1,1)])
        total_basic_array = np.vstack([total_basic_array,basic_list[team_num][30:,-3:]])
    new_range_list = ["toto","LYC"]
    
    
    len_record = len(basic_list[1][0]) 
    result_idx = 0
    
    
    
    
    
for num in range(1,3):
    year_count_result_list = [0 for i in range(20)]
    year_count_pred_list = [0 for i in range(20)]
    
    year_array = np.zeros((1,2))

    total_count = 0
    total_correct_count = 0
    total_sum_error = 0
    
    
     
    count_result_list = [0 for i in range(20)]
    count_pred_list = [0 for i in range(20)]
    
    
    record_team = total_basic_array
    record_team = record_team[record_team[:,result_idx]!=0.5,:]
    Y = record_team[:,result_idx].reshape(-1,1).astype(np.int)
    
    X = record_team[:,num].reshape(-1,1).astype(np.float)
    
    new_array = np.hstack([Y,X])
    year_array = np.vstack([year_array,new_array])
    
    win_count_mask = X > 0.5
    wc_array = np.where(win_count_mask,1,0)
    

    count = len(Y)
    correct_count = sum(wc_array==Y)[0]
    sum_error = np.round(np.sum(X[win_count_mask] - Y[win_count_mask]),5)
    

    total_count += count
    total_correct_count += correct_count
    total_sum_error += sum_error        
    
    #최대 최솟값보정       
    X = np.where(X < 0.25,0.25,X)
    X = np.where(X > 0.75,0.75,X)
    
    #구간 나누기
    for i in range(20): 
        
        range_result = np.sum(Y[(X>i*0.05) & (X<(i+1)*0.05)])
        range_pred = len(X[(X>i*0.05) & (X<(i+1)*0.05)])
        count_result_list[i]= range_result
        count_pred_list[i]= range_pred
        
        year_count_result_list[i] += range_result
        year_count_pred_list[i] += range_pred
    
    pred_range = new_range_list[num-1]
    pred_rate_list = np.round(np.divide(count_result_list,count_pred_list),3)[5:15] 
    
    pred_count_list = count_pred_list[5:15]
    pred_rate = np.round(correct_count/count,3)
    



    
    year_pred_rate_list = np.round(np.divide(year_count_result_list,year_count_pred_list),3)[5:15]
    year_pred_count_list = year_count_pred_list[5:15]
    year_result_count_list = year_count_result_list[5:15]
    year_pred_rate = np.round(total_correct_count/total_count,3)
    
    total_Y = year_array[1:,0]
    total_X = year_array[1:,1]
         
    auc = np.round(metrics.roc_auc_score(total_Y,total_X),3)
        
    
    print(pred_range)
    print(['25','30','35','40','45','50','55','60','65','70','75'])
    print(year_pred_rate_list)
    print(year_result_count_list)
    print(year_pred_count_list)
    print(year_pred_rate)
    print(auc)
    print(np.round(total_sum_error/total_count,3))
    #print(' ')














#%%

info_dic = b.game_info_dic # info_array(n x 10) : (date,game_num,total_game_num,year,team_num,foe_num,game_num,home&away,stadium,result)  



#%%
len_info = len(info_dic[2017][1][0]) # len_info = 10
len_info_record = 2 # 추가적인 레코드 개수(최근성적, 홈원정)
len_X = 4
foe_idx = 5
for year in range(2017,2021):
    team_list = [0]
    for team_num in range(1,11):
        info_array = info_dic[year][team_num][:,:10]
        record_dic = range_data_dic[(br,sr,rr)]
        record_array = record_dic[year][team_num]                    
        
        old_array = np.zeros((1,1 + len_X))
        recent_result_list = list() # 최근전적
        foe_result_list = list() # 상대전적
        
            
        for i,record in enumerate(record_array):
            
            
            #상대전적
            foe_num  = record_array[i,foe_idx]
            past_array = record_array[:i]
            try:
                foe_run_array = past_array[(past_array[:,foe_idx] == foe_num),11]
                run_array = past_array[:,11]
                len_foe_run = len(foe_run_array)
                if len_foe_run <= 3:
                    foe_run = np.mean(run_array)
                else:
                    foe_run = np.mean(foe_run_array)
            except:
                
                foe_run = 5
            foe_result_list.append(foe_run)
        
        home_array = np.where(info_array[:,7] == 'home',1,0)

        info_array = np.hstack([info_array,np.array(foe_result_list).reshape(-1,1)])
        info_array = np.hstack([info_array,home_array.reshape(-1,1)])
        info_dic[year][team_num] = info_array
        

        




#%%
        record_array[0]
        #%%

total_array = np.zeros((1,21))
for year in range(2017,2021):
    for team_num in range(1,11):
        record_array = range_data_dic[(br,sr,rr)][year][team_num]
        info_array = info_dic[year][team_num]
        new_array = np.hstack([info_array,record_array[:,10:]])
        total_array = np.vstack([total_array, new_array[10:]])
        
total_array = total_array[1:]

Y = total_array[:,12].astype(np.float)
X_1 = total_array[:,(10,11)].astype(np.float) #(recent, home/away, )
X_2 = total_array[:,14:].astype(np.float) #(recent, home/away, )
X = np.hstack([X_1,X_2])




X_data = sm.add_constant(X)
model = sm.GLM(Y,X_data,family = sm.families.Gamma(link = sm.genmod.families.links.identity())).fit()
model.summary()

            #%%
            

