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
from sklearn.linear_model import LogisticRegression, LinearRegression, HuberRegressor, Ridge, Lasso, BayesianRidge

import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn import ensemble
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

from xgboost import XGBRegressor, XGBClassifier
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

d.load_data_all(db_address = cd.db_address ,code = cd.aws_code , file_address = cd.file_aws_address)

b.game_info_array = d.game_info_array
b.batter_array = d.batter_array
b.pitcher_array = d.pitcher_array

b.score_array = d.score_array

b.set_dic_all()


#%%
# 기본값 설정

import time
start_time = time.time()
b.is_iv = False
b.is_new_game = False
b.is_park_factor = True
b.is_pa = False
b.is_epa_xr = False
b.is_epa_sp = False
b.is_epa_rp = False
# 타격, 투수, 계투 경기 수 범위 설정
br_range = [20]#[i for i in range(5,101,5)]#[50]#[30]#[50]
sp_range = [i for i in range(1,21)]#[10]
rp_range = [20]#[i for i in range(5,101,5)]#[50]#[50]#[30]"
year_list = [i for i in range(2017,2025)]


#%%
# 기본 데이터 셋 만들기(딕셔너리)
br_time = time.time()
for br in br_range:
    b.br_range = br
    b.set_range_dic(1)
print('br_time: ', time.time()-br_time)

sr_time = time.time()
for sr in sp_range:
    b.sp_range = sr
    b.set_range_dic(2)
print('sr_time: ', time.time()-sr_time)

rp_time = time.time()

b.set_rp_data_dic()

for rr in rp_range:
    b.rp_range = rr
    b.set_range_dic(3)
print('rp_time: ', time.time()-rp_time)
#%%
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



#record_array(n x 18): hName(0), aName(1), hRun(2), aRun(3), home_array(4:11), away_array(11:18)
#home_array(n x 7) = hXR, aInn, aSp_era, aSp_fip, aSp_len, aRp_era, aRp_fip

#%%

#%%
a.br_range_list = br_range
a.sp_range_list = sp_range
a.rp_range_list = rp_range
a.year_list = [i for i in range(2017,2025)]
           

#%%
a.sr_type = 1
a.xr_type = 0
a.sp_type = 0
s.random_state = 1
a.len_model = 3

#%%
new_score_dic = dict()

i = 0
for random_state in range(1,11):
    a.random_state = random_state
    a.set_total_params(range_data_dic,is_print = False)
    for br in br_range:
        for sr in sp_range:
            for rr in rp_range:
                try:
                    new_score_dic[(br,sr,rr)] += a.total_score_dic[(br,sr,rr)]
                except:
                    new_score_dic[(br,sr,rr)] = a.total_score_dic[(br,sr,rr)]
                print(br,sr,rr, random_state, a.total_score_dic[(br,sr,rr)])
  #%%
#a.year_list = [i for i in range(2017,2021)]
a.len_model = 3
s.random_state = 10

a.set_total_params(range_data_dic,is_print = False)


#%%
total_score_dic = a.total_score_dic
total_params_list = a.total_params_list
total_scale_list = a.total_scale_list
total_data_len_dic = a.total_data_len_dic

#%%
from pycaret.regression import *
from pycaret.datasets import get_data

a.sr_type = 0
a.xr_type = 0
a.sp_type = 0
s.random_state = 10
total_array = np.zeros((1,a.len_total))

br = 20
sr = 10
rr = 20

for year in b.year_list:
    
    for team_num in range(1,11):
        
        team_array = range_data_dic[(br,sr,rr)][year][team_num]
        
        total_array = np.vstack([total_array,team_array])



total_array = total_array[1:,:]



#%%





train_array = total_array[total_array[:,3]<2021]
test_array = total_array[total_array[:,3]==2021]

    
model_list = [list() for i in range(5)]
a.start_game_idx = 0
a.min_recent_sp = 0

train_array= a.get_input(train_array, sr = sr)
test_array = a.get_input(test_array, sr = sr)


Y = train_array[:,0].reshape(-1,1)

X = train_array[:,2:-1]
Y_test = test_array[:,0].reshape(-1,1)
X_test = test_array[:,2:-1]

score_list = [list() for i in range(5)]
cv = KFold(5,shuffle =True,random_state= 13)
gbr = GradientBoostingRegressor()
xgb = XGBRegressor(n_estimators = 300, learning_rate = 0.1, reg_lambda = 3, reg_alpha = 3)


for (train_idx, valid_idx) in cv.split(X):
     
     new_model_list = list()
     X_train = X[train_idx]
     X_valid = X[valid_idx]
     Y_train = Y[train_idx]
     Y_valid = Y[valid_idx]
     
     
     
     pf_list = list()
     '''
     pca = PCA(n_components=20)
     X_train = pca.fit_transform(X_train)
     X_valid = pca.transform(X_valid)
     '''
     new_model_list.append(sm.GLM(Y_train,X_train,family = sm.families.Gamma(link = sm.genmod.families.links.identity())).fit())
     new_model_list.append(sm.OLS(Y_train,X_train).fit())
     new_model_list.append(HuberRegressor().fit(X_train, Y_train))
     new_model_list.append(Ridge().fit(X_train,Y_train))
     new_model_list.append(gbr.fit(X_train,Y_train))
     for i, model in enumerate(new_model_list):
         
         model_list[i].append(model)
         
         y_hat = model.predict(X_valid).reshape(-1,1)
         
         
         
         score_list[i].append(a.get_score(Y_valid, y_hat)[0])

for score in score_list:
    print(np.round(np.mean(score),4))


#%%
for i in range(5):
    score_list = list()
    for j in range(5):
        y_hat = model_list[i][j].predict(X_test)
                
       
        score = a.get_score(Y_test,y_hat)[0]
        score_list.append(score)
    print(np.round(np.mean(score_list),4))
#%%
X_train = pd.DataFrame(X_train)
X_train.columns = X_columns
model = Ridge().fit(X_train,Y_train)

#model.summary()
model.coef_
#%%
batter_columns = ['h1','h2','h3','hr','bb','hbp','ibb','sac','sf','so','go','fo','gidp', 'etc', 'h', 'tbb', 'ab', 'pa', 'xr']
sp_columns = ['ip','tbf','np','ab','h','hr','tbb','so','r','er','fip','fip_var']
rp_columns = ['rp_tbf','rp_np','rp_ab','rp_h','rp_hr','rp_tbb','rp_so','rp_r','rp_er','rp_fip']  
etc_columns = ['Home','잠실','사직','광주','삼성','대전','문학','고척','창원','수원']
X_columns = ['1'] + batter_columns[:13] + sp_columns[:8] + rp_columns[:7] + etc_columns
#%%
Y_valid = Y_valid.reshape(-1)
y_hat = y_hat.reshape(-1)
np.mean(Y_valid - y_hat)
        
#%%
old_array = np.zeros((1,39))
for i in range(4):
    if i <=1:
        new_array = np.round(new_model_list[i].params,3)
    else:
        new_array = np.round(new_model_list[i].coef_,3)        
    old_array = np.vstack([old_array,new_array])
z = pd.DataFrame(old_array[1:])
z.columns = X_columns
    #%%
    b.sp_range = 30
    b.sp_by_game(2022,1,115)
    #%%
from pycaret.regression import *
from pycaret.datasets import get_data
#%%
input_array = a.get_input(total_array, sr = sr)

run = input_array[:,0]

run_xr = input_array[:,1]
X = input_array[:,2:-1]
train = np.hstack([run.reshape(-1,1),X])
train_df = pd.DataFrame(train)
train_df.columns = ['target'] + [str(i+1) for i in range(53)]
train_df.columns
#%%
setup_reg = setup(data = train_df,target = 'target',fold =5,session_id = 1)
#%%
best = compare_models(sort='mse')
#%%
evaluate_model(best)
#%%
result = pull()
print(result)
#%%

#%%
cv = KFold(5,shuffle =True,random_state = 1)
#%%

#%%
for i,j in cv.split(X):
    print(len(i))
    print(len(j))
#%%
from xgboost import plot_importance
plot_importance(model)
    
    

#%%
import time

br = 50
sr = 20
rr = 50
model_len = 3
a.min_recent_sp = 0

start_time = time.time()
result_total_dic = dict()
year_list = [i for i in range(2017,2023)]
range_game_num = 150
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
        run_list = [[] for i in range(max_game_num)]
        mean_list = [[] for i in range(max_game_num)]
        scale_list=  [[] for i in range(max_game_num)]
        params_list = [[] for i in range(max_game_num)]
        
        #br,sr,rr = team_best_num_list[team_num]
        team_array = range_data_dic[(br,sr,rr)][year][team_num]
        
        a.sr_type = 0
        
        stat_data = a.get_input(team_array,sr = sr)
        sp_len_data = stat_data[:,-1]
        
        range_mean_list = list()
        
        for game_num in range(max_game_num):#( max_game_num-1,max_game_num):
            
            
            range_data = stat_data[:game_num,:-1]
            x = stat_data[game_num,2:-1].reshape(1,-1).astype(np.float)
            
            run = range_data[:,0].reshape(-1,1).astype(np.float)
        
            Y = range_data[:,1].reshape(-1,1).astype(np.float)
            X = range_data[:,-4:].astype(np.float)
            X = np.hstack([np.ones(len(X)).reshape(-1,1),X])
            
            sp_len =sp_len_data[game_num]
            #if sp_len> sr:
            #    sp_len = sr
            
            if sp_len <= 1:
                sp_len = 1
            
            if sp_len >= 20:
                sp_len = 20
                
            #sp_len = 10
            
            
            
            
            m_list = list()
            s_list = list()
            p_list = [[0] for p in range(3)]
            
            
            
            
            ground = team_array[game_num,8]

            for i in range(model_len):#range(model_len):
                for j in range(5):
                    model = total_params_list[i][(br,sp_len,rr)][j]
                    mean  = model.predict(x)
                    '''
                    pf = b.park_factor_total.get(ground)
                    if pf != None:
                        mean= mean * pf
                    '''
                    mean= mean[0]
                    '''
                    if mean<=2:
                        mean = 2
                    elif mean >= 8:
                        mean = 8
                    '''
                    m_list.append(mean)
                    
            
            
            model_mean = np.mean(m_list)
            
            #print(team_num, game_num, m_list,sp_len)
            #scale = np.mean(s_list)
            mean_list[game_num].append(model_mean)
            run_list[game_num].append(run)
            total_scale = total_scale_list[0][(br,sp_len,rr)]
            scale_list[game_num].append(1/total_scale)
            

        for i,ml in enumerate(mean_list):
            
            mean = np.mean(ml)
            
        
            scale = np.mean(scale_list[i])
            
            
            beta = 2.5
            alpha = mean/beta
            if alpha <=0.5:
                alpha = 0.5
            
            alpha = 2
            beta = mean/alpha
            if beta <=0.5:
                beta = 0.5
            #print(year,team_num, i, [alpha,beta])
            '''
            beta = scale/mean
            alpha= mean/beta
            #print(year,team_num, i, [alpha,beta])
            
            
            alpha = scale
            beta = mean / alpha
            if beta <=0.5:
                beta = 0.5
            
            
            #print('origin: ', team_num, i, round(alpha,2), round(beta,2), round(alpha*beta,2))
            
            MAP사용한 난수생성
            if year == 2022:
                if i>=2:
                    
                    run = run_list[i][0].reshape(-1)
                    run = s.mod_array_zero_to_num(run,0.6)
                    s.set_data(run)
                    s.start_theta = s.mom_gamma_theta
                    s.alpha_theta = [alpha, 1]
                    s.beta_theta = [beta, 1]
    
                    alpha, beta = s.fit(dist='map')[-1,1:]
                    #print('new   : ', team_num, i, round(alpha,2), round(beta,2), round(alpha*beta,2))
            '''
            
             
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
        record_array = np.hstack([record_array, result_array, foe_mean_array])
    
        exp_rate = record_array[:,-2].reshape(-1,1)
        
        result_total_list[team_num] = np.hstack([result_total_list[team_num],exp_rate])
                
    result_total_dic[year] = result_total_list
    print([year, time.time() - start_time])
    


#%%
len_record = 3
total_rate_list = [list() for i in range(len_record)]
total_pred_count_list = [list() for i in range(len_record)]
total_result_count_list = [list() for i in range(len_record)]
for year in year_list:
    
    basic_list = b.game_info_dic[year]
    b.set_toto_dic()

    if year ==2022:
        toto_list = b.toto_dic[2022]
        
    else:
        toto_list = b.toto_dic[year]
    
    if year == 2022:
        continue
    
    total_basic_array = np.zeros((1,3))
    for team_num in range(1,11):
        
        #team_win = np.where(toto_list[team_num][:,6]>=toto_list[team_num][:,7],1,0)
        toto_rate_array = toto_list[team_num][:b.max_game_dic[year][team_num],-4].reshape(-1,1)
        #basic_list[team_num][:,-1] = team_win
        len_toto = len(toto_rate_array)
        basic_array = basic_list[team_num][:len_toto,:10]
        
        new_logis = logis_model[(year-2017)*1440:(year-2016)*1440][(team_num-1)*144:team_num*144]
        basic_list[team_num] = np.hstack([basic_array,toto_rate_array,result_total_dic[year][team_num][:len_toto,1:],new_logis])
        #basic_list[team_num] = np.hstack([basic_array,np.ean(basic_list[team_num][:,11:],axis = 1).reshape(-1,1)])
        total_basic_array = np.vstack([total_basic_array,basic_list[team_num][:,-3:]])
    
    new_range_list = ["toto","LYC","Logistic"]
    
    
    
    result_idx = 9
        
    for j, num in enumerate(range(result_idx+1, len_record + result_idx + 1)):
        
        year_count = 0
        year_correct_count = 0
        year_sum_error = 0
        
        year_count_result_list = [0 for i in range(20)]
        year_count_pred_list = [0 for i in range(20)]
        
        year_array = np.zeros((1,2))
        for team_num in range(1,11):
            
            count_result_list = [0 for i in range(20)]
            count_pred_list = [0 for i in range(20)]
            
            
            record_team = basic_list[team_num][10:]
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
            
    
            year_count += count
            year_correct_count += correct_count
            year_sum_error += sum_error        
            
            #최대 최솟값보정       
            #X = np.where(X < 0.25,0.25,X)
            #X = np.where(X > 0.75,0.75,X)
            
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
        year_pred_rate = np.round(year_correct_count/year_count,3)
        
        year_Y = year_array[1:,0]
        year_X = year_array[1:,1]
             
        year_auc = np.round(metrics.roc_auc_score(year_Y,year_X),3)
            
        
        print(year)
        print(['25','30','35','40','45','50','55','60','65','70','75'])
        
        print(year_pred_rate_list)
        print(year_result_count_list)
        
        print(year_pred_count_list)
        print(year_pred_rate)
        print(year_auc)
        print(np.round(year_sum_error/year_count,3))
        #print(' ')
        
        total_pred_count_list[j].append(year_pred_count_list)
        total_result_count_list[j].append(year_result_count_list)
        total_rate_list[j].append(year_array)
      
for j in range(len_record):
    new_pred = [i for i in range(10)]
    new_result = [i for i in range(10)]
    new_rate = np.zeros((1,2))
    new_pred_rate = [i for i in range(10)]
    for total_pred, total_result, total_rate in zip(total_pred_count_list[j], total_result_count_list[j], total_rate_list[j]):
        
        new_rate = np.vstack([new_rate, total_rate])
        
        for i in range(10):
            
            new_pred[i]+=total_pred[i]
            new_result[i]+=total_result[i]
            
    for i in range(10):
        
        new = np.round(new_result[i] / new_pred[i],3)
        new_pred_rate[i] = new

    Y = new_rate[:,0].astype(np.int)
    X = new_rate[:,1]
    win_count_mask = X > 0.5
    wc_array = np.where(win_count_mask,1,0)
    
    
    count = len(Y)
    correct_count = sum(wc_array==Y)
    
    correct_rate = np.round(correct_count / count,3)
    auc = np.round(metrics.roc_auc_score(Y,X),3)
    
    print('')
    if j == 0:
        print('Total - toto')
    else:
        print('Total - Lee')
        
    print(['25','30','35','40','45','50','55','60','65','70','75'])
    
    print(new_pred_rate)
    print(new_result)
    
    print(new_pred) 
    print(correct_rate)
    print(auc)
    
#%%

a.sr_type = 0
total_array = np.zeros((1,68))

for year in range(2017,2022):
    year_list = range_data_dic[(50,10,50)][year]
    for team_num in range(1,11):
        team_array = year_list[team_num]
        old_array = np.zeros((1,30))
        for data in team_array:
            
            game_idx_num = data[1]
            foe_num = data[5]
            record_by_foe = year_list[foe_num]
            record_by_foe = record_by_foe[record_by_foe[:,1] == game_idx_num]
            new_array = a.get_input(record_by_foe, sr= 20)[:,:30]
            old_array = np.vstack([old_array, new_array])
        foe_array = old_array[1:]
        year_array = team_array[:,3].reshape(-1,1)
        team_array = a.get_input(team_array,sr = 20)
        result_array = team_array[:,0]>=foe_array[:,0]
        result_array = np.where(result_array==True,1,0).reshape(-1,1)
        new_total_array = np.hstack([year_array, result_array, team_array[:,3:], foe_array[:,3:]])
        
        total_array = np.vstack([total_array, new_total_array])
    
total_array = total_array[1:]


#%%
Y = total_array[:,1].reshape(-1,1)
X = total_array[:,2:]
Year = total_array[:,0].reshape(-1,1)



old = np.zeros((1,1))
new_model_list = list()
for year in range(2017,2022):
     
     train_idx = (Year!=year).reshape(-1)
     valid_idx = (Year==year).reshape(-1)
     X_train = X[train_idx]
     X_valid = X[valid_idx]
     Y_train = Y[train_idx].astype(np.int)
     Y_valid = Y[valid_idx]
     
     `  
     
     pf_list = list()
     '''
     pca = PCA(n_components=20)
     X_train = pca.fit_transform(X_train)
     X_valid = pca.transform(X_valid)
     '''
     
     model = LogisticRegression().fit(X_train,Y_train)
     new = model.predict_proba(X_valid)[:,1:]
     old = np.vstack([old,new])
logis_model = old[1:]
#%%
Y_train.shape
         
         

#%%



