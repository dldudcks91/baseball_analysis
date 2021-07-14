#%%
import sys
sys.path.append('C:\\Users\\Chan\\Desktop\\BaseballProject\\python')

#%%
import numpy as np
import pandas as pd
import math
import scipy
import scipy.stats as stats
import pymysql


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
from baseball_2021 import base as bs
from baseball_2021 import modification as md
from baseball_2021 import sample as sp

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
b.is_new_game = True
b.is_park_factor = True
b.is_pa = True
b.least_pa = 0
b.br_range = 20
b.sp_range = 10
b.rp_range = 50
#%%
b.is_new_game = True
b.load_today_array()
lineup_record = [0]
for i in range(1,11):
    game_num = b.today_array[b.today_array[:,3]==i,5][0]
    lineup_record.append([b.xr_by_game(2021,i,game_num),list(b.sp_by_game(2021,i,game_num)),b.rp_by_game(2021,i,game_num)[0]])
#%%
lineup_record
#%%
b.is_new_game = True
b.xr_by_game(2021,1,71)
#%%
for i,j in zip(b.run_by_team(2021,9), b.xr_by_team(2021,9)):
    print(i,j)
#%%
np.mean(b.run_by_team(2021,3)), np.mean(b.xr_by_team(2021,3))
#%%
for i in range(1,11):
    print(round(np.mean(b.xr_by_team(2021,i)),3), round(np.mean(b.run_by_team(2021,i)),3))

#%%
z = pd.DataFrame(b.batter_dic[2020][8])
np.sum(b.batter_dic[2020][8][:,-1])/144
b.xr_by_team(2020,8)
#%%

toto_array = np.zeros((1,10))
for game_info in b.today_game_info:
    date = game_info[0][:8]
    time = game_info[4]
    site_name = 'lyc'
    away_name = game_info[2]
    away_num = b.team_dic[away_name]
    home_name = game_info[1]
    home_num = b.team_dic[home_name]
    
    ground = game_info[3]
    def get_odds(lineup_record,team_num,foe_num,is_home):
        x1 = lineup_record[team_num][0]
        foe_list = lineup_record[foe_num]
        inn = float(foe_list[1][1])
        sp = float(foe_list[1][2])
        rp = foe_list[2]
        x2 = (inn * sp)/9 + (9-inn)*rp/9
        if is_home:
            x3 = 1
        else:
            x3 = 0
        param_list = [0.928, 0.416, 0.454, 0.479]
        result  = np.dot(param_list ,[1,x1,x2,x3])
        
        return result
    away_run = get_odds(lineup_record,away_num,home_num,False)
    if b.park_factor_total.get(ground)!=None:
        away_run = away_run * b.park_factor_total[ground]
    away_alpha = away_run / 2.5
    away_sample = s.gamma_sample([away_alpha,2.5])
    
    home_run = get_odds(lineup_record,home_num,away_num,True)
    if b.park_factor_total.get(ground)!=None:
        home_run = home_run * b.park_factor_total[ground]
    home_alpha = home_run / 2.5
    home_sample = s.gamma_sample([home_alpha,2.5])
    
    away_odds = round(sum(away_sample > home_sample)/s.size,3)
    home_odds = round(sum(away_sample < home_sample)/s.size,3)
    
    total_run = away_sample + home_sample
    total_run.sort()
    total_mean = total_run[int(s.size/2)]
    under = sum(total_run < total_mean) / s.size
    over = sum(total_run >= total_mean) / s.size
    
    for win_type in [1,3]:
        info_list = [date,time,site_name,win_type,away_name,home_name]
        if win_type ==1:
            record_list = [away_odds,home_odds,0,'00:00']
        else:
            record_list = [under,over,round(total_mean,1),'00:00']
        new_toto_list = info_list + record_list
        toto_array = np.vstack([toto_array,new_toto_list])
    print([away_run,home_run])
toto_array = toto_array[1:]

#%%
lineup_record
#%%
#conn  = pymysql.connect(host='localhost', user='root', password='dudrn1', db='baseball', charset='utf8')
conn = pymysql.connect(host='lyc-baseball.cgh18xdnf8rj.ap-northeast-2.rds.amazonaws.com', user='LYC', password='3whddpdltm!', db='baseball', charset='utf8')
#c.set_last_game_num_list(2021,conn)
cursor = conn.cursor()
#%%
for i,data in enumerate(toto_array):
    data_str = str(tuple(data))
    data_str = data_str.replace('None','Null')
    sql = 'insert into'+ ' today_toto' + ' values ' + data_str
    cursor.execute(sql)
conn.commit()
conn.close()

        
#%%
br_range = [18,19,20]#[7,10,15,20,30]
sp_range = [12,13,14]#[5,7,15]
rp_range = [30,50,70]

#%%
import time
start_time = time.time()
for br in br_range:
    b.br_range = br
    b.set_xr_dic(1)
for sr in sp_range:
    b.sp_range = sr
    b.set_xr_dic(2)
for rr in rp_range:
    b.rp_range = rr
    b.set_xr_dic(3)


#%%
record_dic = dict()
        
for year in range(2017,2021):
    team_list =[0]
    for team_num in range(1,11):
        max_game = b.max_game_dic[year][team_num]
        
        #분석에 사용할 변수 데이터 불러오기
        old_array = np.zeros((max_game,1))
        
        old_array = np.hstack([old_array,b.run_by_team(year,team_num)]) # run(n x 1) : run
        for br in br_range:
            old_array = np.hstack([old_array,b.xr_dic[br][year][team_num]]) # xr(n x 1) : xr
        for sr in sp_range:
            old_array = np.hstack([old_array,b.sp_dic[sr][year][team_num][:,1:]])# sp(n x 3) : name, inn , sp_fip
            
        for rr in rp_range:
            old_array = np.hstack([old_array,b.rp_dic[rr][year][team_num]])# rp_fip 
        
        old_array = old_array [:,(0,1,2,3,4,5,7,9,6,8,10,11,12,13)]
        total_record_array = old_array[:,1:] # total_record_array(n x 6): run, xr, inn, sp_fip, rp_fip       
        team_list.append(total_record_array)
        
    record_dic[year] = team_list


#%%
'''
우리팀 공격력 상대팀 수비력 지표 가져오기
'''
info_dic = b.game_info_dic # info_array(n x 10) : (date,game_num,total_game_num,year,team_num,foe_num,game_num,home&away,stadium,result)  
record_dic = record_dic 

len_info = 10

len_info_record = 1

len_record = 1 + 3*4

len_team_X = 3
len_foe_X = 9

date_idx = 0
foe_num_idx = 5

record_total_dic = dict()
for year in range(2017,2021):
    team_list = [0]
    for team_num in range(1,11):
        
        record_array = record_dic[year][team_num]
        info_array = info_dic[year][team_num]
        
        old_array = np.zeros((1,1 + len_team_X + len_foe_X))
        recent_result_list = list()
        foe_result_list = list()
        for i,info in enumerate(info_array):
            if i < 10:
                recent_result = (sum(info_array[:i,-1]) + 1) / (i + 2)
            else:
                recent_result = (sum(info_array[i-10:i,-1]) + 1) / 12
            
            
            
            foe_num  = info_array[i,5]
            try:
                foe_result = info_array[(info_array[:i,5] == foe_num),-1]
                foe_result = (sum(foe_result) + 1) / (len(foe_result) + 2)
            except:
                foe_result = 0.5
            recent_result_list.append(recent_result)
            foe_result_list.append(foe_result)
        info_array = np.hstack([info_array,np.array(recent_result_list).reshape(-1,1)])
        info_array = np.hstack([info_array,np.array(foe_result_list).reshape(-1,1)])
            
        for info,record in zip(info_array,record_array):
            
            date = info[date_idx]
            foe_num = info[foe_num_idx]
            
            info_by_foe = info_dic[year][foe_num]
            record_by_foe = record_dic[year][foe_num]
            record_by_foe = np.hstack([info_by_foe,record_by_foe])                    
            record_by_foe = record_by_foe[record_by_foe[:,date_idx] == date][0,len_info:]

            team_record_array = np.hstack([record[:len_team_X+1],record_by_foe[-len_foe_X:]]).reshape(1,-1)
            
            old_array = np.vstack([old_array,team_record_array])
        old_array = old_array[1:,:] # record_array(n x 13): run xr1 xr2 xr3 inn1 inn2 inn3 sp1 sp2 sp3 rp1 rp2 rp3
        
        total_array = np.hstack([info_array[10:],old_array[10:]])
        
        team_list.append(total_array)
    record_total_dic[year] = team_list

#%%
len_info = 10
len_info_record = 2
len_record = 1 + 3*4

len_total = len_info + len_info_record + len_record

old_array = np.zeros((1, len_total))
for year in range(2017,2021):
    for team_num in range(1,11):
        old_array = np.vstack([old_array,record_total_dic[year][team_num]])

total_array = old_array[1:]
#%%



#%%
x_idx = len_info + len_info_record + 1
y_idx = len_info + len_info_record 
Y = total_array[:,y_idx].astype(np.float).reshape(-1,1)
#new_Y = np.where(Y>12,12,Y)

X_att = total_array[:, x_idx:x_idx+3].astype(np.float).reshape(-1,3)
X_dep = total_array[:,-9:].astype(np.float)

inn = X_dep[:,:3].reshape(-1,3)
sp_fip = X_dep[:,3:6].reshape(-1,3)
rp_fip = X_dep[:,6:9].reshape(-1,3)
'''
inn = np.where(inn>7,7,inn)
inn = np.where(inn<2,2,inn)

sp_fip = np.where(sp_fip>5.5,5.5,sp_fip)
sp_fip = np.where(sp_fip<1,1,sp_fip)
                
rp_fip = np.where(rp_fip>5.5,5.5,rp_fip)
rp_fip = np.where(rp_fip<1,1,rp_fip)
'''
for i in range(3):
    for j in range(3):
        
        x1 = (inn[:,i]*sp_fip[:,i])/9
        x2 = (9-inn[:,i])*rp_fip[:,j]/9
        x3 = np.sum([x1,x2],axis = 0)
        X_dep = np.hstack([X_dep,x3.reshape(-1,1)])

X_home = np.where(total_array[:,7] == 'home',1,0).reshape(-1,1)
X_recent = total_array[:,10].astype(np.float).reshape(-1,1)
X_foe_result = total_array[:,11].astype(np.float).reshape(-1,1)


#X = np.hstack([X_att,x1[:,-1].reshape(-1,1),X_dep[:,-1].reshape(-1,1)])
for i in range(3):
    for j in range(9):
        X = np.hstack([X_att[:,i].reshape(-1,1),X_dep[:,(-9+j)].reshape(-1,1)]) #타자, 투수 기록
        X = np.hstack([X,X_home])  # 홈 & 어웨이 
        X = np.hstack([X,X_recent])
        X = np.hstack([X,X_foe_result])
        X_data = sm.add_constant(X)
        model1 = sm.GLM(Y,X_data,family = sm.families.Gamma(link = sm.genmod.families.links.identity)).fit()
        model2 = sm.OLS(Y,X_data).fit()

        def get_params(y,x,params):
            y_hat = np.dot(x,params).reshape(-1,1)
            
            mse = np.round(np.square(np.subtract(y,y_hat)).mean(),4)
            msle = np.round(np.mean((np.log(y+1) - np.log(y_hat+1))**2),4)
            return [mse] + [msle]
        print(get_params(Y, X_data, model1.params),i,j)
    

      
#%%
i = 0
j = 8
#Y = np.where(total_array[:,9] == 1,1,0)
X = np.hstack([X_att[:,i].reshape(-1,1),X_dep[:,(-9+j)].reshape(-1,1)])
X = np.hstack([X,X_home])  # 홈 & 어웨이 
X = np.hstack([X,X_recent])
X = np.hstack([X,X_foe_result])
X_data = sm.add_constant(X)

model1 = sm.GLM(Y,X_data,family = sm.families.Gamma(link = sm.genmod.families.links.identity)).fit()
model2 = sm.OLS(Y,X_data).fit()
model2.summary()
#model3 = LogisticRegression()
#model3.fit(X_data,Y)


