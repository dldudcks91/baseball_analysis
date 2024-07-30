#%%

# 불러오기 위치 설정
import sys
sys.path.append('D:\\BaseballProject\\python')



import numpy as np
import pandas as pd
import pymysql
from sqlalchemy import create_engine

from bs_database import base as bs

from bs_stats import preprocess as pr
from bs_stats import sample as sp
from bs_personal import personal_code as cd
from bs_stats import analytics as an

import datetime
#%%
import pickle

with open('C:/Users/82109/Desktop/LYC/git/baseball_analysis/bs_crontab/total_params_list.pkl','rb') as f:
    total_params_list = pickle.load(f)
#%%
a = an.Analytics()
s = sp.Sample()
d = bs.Database()
d.load_data_all(db_address = cd.db_address, code = cd.aws_code, file_address = cd.file_aws_address)

#%%
b = pr.Preprocess() 

b.game_info_array = d.game_info_array
b.batter_array = d.batter_array
b.pitcher_array = d.pitcher_array
b.score_array = d.score_array

b.set_dic_all()

#%%
b.is_new_game = True
b.is_iv = False

#%%

b.is_park_factor = True
b.is_pa = False
b.is_epa_xr = True
b.is_epa_sp = False
b.is_epa_rp = False
# 타격, 투수, 계투 경기 수 범위 설정
br_range = [20]#[1] + [i for i in range(5,101,5)]#[50]#[30]#[50]
sp_range = [i for i in range(1,21)]
rp_range = [20]#[50]#[30]
a.sr_type = 1
a.xr_type = 0
a.sp_type = 0
b.set_rp_data_dic()
#%%
br = 20
sr = 10
rr = 20
#%%

#%%
b.load_today_array(db_address = cd.db_address, code = cd.aws_code, file_address = cd.file_aws_address)
lineup_record = [0]
#%%
idx = pd.isnull(b.today_array[:,-1])
new_idx = idx == False
b.today_array = b.today_array[new_idx]
#%%
team_num = 1
z = b.today_array
zz = list(b.today_array[b.today_array[:,3] == team_num,-1])

#%%
for i in range(1,11):
    
    game_num = b.today_array[b.today_array[:,3]==i,5][0]
    lineup_record.append([b.xr_by_game(2024,i,game_num),b.sp_by_game(2024,i,game_num).reshape(-1),b.rp_by_game(2024,i,game_num)])
    


        
#%%
def get_new_input(lineup_record, ground_array, team_num, foe_num, is_home, xr_type = 0, sp_type = 0):
    
        batter_array = lineup_record[team_num][0].reshape(1,-1)
        
        
        foe_list = lineup_record[foe_num]
        sp_array = foe_list[1][1:].astype(np.float64).reshape(1,-1)
        rp_array = foe_list[2].astype(np.float64).reshape(1,-1)
        
        
        
        
        
        xr_array = batter_array[-1].reshape(-1,1)
        
        
        
        
        inn = sp_array[:,0].reshape(-1,1)
        inn = np.where(inn<= 4, 4, inn)
        
        # 선발, 계투 데이터 세부분리
        sp_era = sp_array[:,-4].reshape(-1,1) #선발투수 era
        sp_fip = sp_array[:,-3].reshape(-1,1) #선발투수 fip
        sp_var = sp_array[:,-2]
        sp_len = sp_array[:,-1].reshape(-1,1) #선발투수 등판 수
        
        sp_era = sp_era *sp_len
        rp_era = rp_array[:,-2].reshape(-1,1) #계투 era
        rp_fip = rp_array[:,-1].reshape(-1,1) #계투 fip
        
        
    
        
        '''
        #fip ratio 설정
        fip_ratio = sp_fip/4.5
        fip_ratio = np.where(fip_ratio <=0.7,0.7,fip_ratio)
        fip_ratio = np.where(fip_ratio >=1.3,1.3,fip_ratio)
        '''
        
        
        
        # 홈원정 변수 생성        
        if is_home:
            X_home = np.array([1]).reshape(-1,1)
        else:
            X_home = np.array([0]).reshape(-1,1)
        
        
       
        
        if xr_type == 0:
            x_batter = batter_array[:,:-6]
            
        elif xr_type ==1:
            x_batter = xr_array
        
        
        if sp_type == 0:
            x_sp = sp_array
            x_rp = rp_array
        elif sp_type ==1:
            x_sp = (sp_array *inn) / 9
            x_sp = x_sp[:,1:]
            x_rp = ((9-inn)*rp_array) / 9 
        elif sp_type == 2:
            x_sp = (inn*sp_fip) / 9 
            x_sp = x_sp[:,1:]
            x_rp = ((9-inn)*rp_array) / 9 
        elif sp_type == 3:
            x_sp = (inn*sp_array) / 9 
            x_sp = x_sp[:,1:]
            x_rp = ((9-inn)*rp_fip) / 9 
        
        x_sp = x_sp[:,:-5]
        x_rp = x_rp[:,:-3]
        
        X = np.hstack([x_batter, x_sp, x_rp, X_home, ground_array, sp_len])
        X = np.hstack([np.ones(len(X)).reshape(-1,1),X])
        
            
        
        
        
        
        
        return X

   

        
#%%
import datetime
today = datetime.datetime.today()

now = str(today.hour).zfill(2) + ":" +str(today.minute).zfill(2)

s.size = 100000
toto_array = np.zeros((1,10))

sr_type = a.sr_type

MODEL_CV = 1

for game_info in b.today_game_info:
    date = game_info[0][:8]
    time = game_info[4]
    if time == '경기취소':
        continue
    site_name = 'lyc'
    away_name = game_info[2]
    away_num = b.team_dic[away_name]
    home_name = game_info[1]
    home_num = b.team_dic[home_name]
    
    ground = game_info[3]
    park_factor = {'잠실': 0, '사직': 1,'광주': 2, '대구': 3, '대전': 4,'문학': 5,'고척': 6,'마산': 7,'수원': 8}
    
    pf = park_factor.get(ground)
    ground_array = np.zeros((1,9))
    if pf==None:
        pass
    else:
        ground_array[0,pf] = 1
    
    
    away_input = get_new_input(lineup_record, ground_array, away_num, home_num, is_home = False, xr_type = a.xr_type, sp_type = a.sp_type)
    
    sp_len = away_input[0,-1]
    if sp_len <=1: sp_len = 1
    away_run = 0
    for i, total_params in enumerate(total_params_list):
        for j in range(MODEL_CV):
            
            
            if sr_type == 0:
                model = total_params[br,sr,rr][j]
                new_run = np.dot(away_input[:,:-1],model)
                
                
    
            else:
                model = total_params[br,sp_len,rr][j]
                new_run = np.dot(away_input[:,:-1],model)
            if new_run<=2:
                new_run = 2    
            if new_run>= 8:
                new_run = 8
            
            away_run+= new_run
            
    away_run = np.round(away_run/((i+1)*MODEL_CV ),3)
    
    
    if b.park_factor_total.get(ground)!=None:
        away_run = away_run * b.park_factor_total[ground]
        
    
    '''
    if sr_type == 0:
        away_alpha = 1/total_scale_list[0][(br,sr,rr)][0]
    else:
        away_alpha = 1/total_scale_list[0][(br,sp_len,rr)][0]
    '''
    away_alpha = 2
    away_beta = away_run / away_alpha
    away_sample = s.gamma_sample([away_alpha,away_beta])
    
    
    
    home_input = get_new_input(lineup_record, ground_array, home_num, away_num, is_home = True, xr_type = a.xr_type, sp_type = a.sp_type)
    sp_len = home_input[0,-1]
    if sp_len <=1: sp_len = 1
    home_run = 0
    for i, total_params in enumerate(total_params_list):
        for j in range(MODEL_CV ):
            if sr_type == 0:
                model = total_params[br,sr,rr][j]
                new_run = np.dot(home_input[:,:-1],model)
                
                
    
            else:
                model = total_params[br,sp_len,rr][j]
                new_run = np.dot(home_input[:,:-1],model)
            if new_run<=2:
                new_run = 2   
            if new_run >=8:
                new_run = 8
            home_run+= new_run
                
    home_run = np.round(home_run/((i+1)*MODEL_CV ),3)
    
    
    
    if b.park_factor_total.get(ground)!=None:
        home_run = home_run * b.park_factor_total[ground]
    
    #home_alpha =2
    '''
    if sr_type == 0:
        home_alpha = 1/total_scale_list[0][(br,sr,rr)][0]
    else:
        home_alpha = 1/total_scale_list[0][(br,sp_len,rr)][0]
    '''
    home_alpha =2
    home_beta = home_run / home_alpha
    
    home_sample = s.gamma_sample([home_alpha,home_beta])
    
    
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
            record_list = [away_odds,home_odds,0,now]
        else:
            record_list = [under,over,round(total_mean,1),now]
        new_toto_list = info_list + record_list
        toto_array = np.vstack([toto_array,new_toto_list])
    print(away_name,": ", np.round(away_run,4)," ", home_name,": ", np.round(home_run,4))
toto_array = toto_array[1:]
#%%

print(np.array(lineup_record))
#%%
print(toto_array)
#%%
#onn_local = pymysql.connect(host= cd.local_host, user=cd.local_user, password= cd.local_code, db= cd.db, charset='utf8')
conn_aws = pymysql.connect(host = cd.aws_host, user = cd.aws_user, password = cd.aws_code , db = cd.db, charset = cd.charset)

#c.set_last_game_num_list(2021,conn)


#%%
d.insert_table(conn_local, 'today_toto',toto_array)
conn_local.commit()
conn_local.close()
#%%  M
d.insert_table(conn_aws, 'today_toto',toto_array)
conn_aws.commit()
conn_aws.close()