#%%

# 불러오기 위치 설정
import sys
sys.path.append('D:\\BaseballProject\\python')


#%%
import numpy as np
import pandas as pd
import pymysql
from sqlalchemy import create_engine

from bs_database import base as bs
from bs_stats import preprocess as pr
from bs_stats import sample as sp
from bs_personal import personal_code as cd
import datetime
#%%
s = sp.Sample()
d = bs.Database()
d.load_data_all(db_address = cd.db_address, code = cd.local_code, file_address = cd.file_address)

#%%
b = pr.Preprocess() 

b.game_info_array = d.game_info_array
b.batter_array = d.batter_array
b.pitcher_array = d.pitcher_array
b.score_array = d.score_array

b.set_dic_all()

#%%
b.is_new_game = True
b.is_park_factor = True
b.is_iv = False
b.is_pa = False
b.is_epa_xr = True
b.is_epa_sp = True
b.is_epa_rp = False

b.br_range = 45
b.sp_range = 20
b.rp_range = 50

#%%

#%%
b.load_today_array(db_address = cd.db_address, code = cd.local_code, file_address = cd.file_address)
lineup_record = [0]
for i in range(1,11):
    try:
        game_num = b.today_array[b.today_array[:,3]==i,5][0]
        lineup_record.append([b.xr_by_game(2021,i,game_num),list(b.sp_by_game(2021,i,game_num)),b.rp_by_game(2021,i,game_num)])
        
    except:
        lineup_record.append(0)
#%%
import datetime
today = datetime.datetime.today()

now = str(today.hour).zfill(2) + ":" +str(today.minute).zfill(2)

s.size = 100000
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
    
    def get_odds(lineup_record,team_num,foe_num,is_home,total_params_list,team_params_list):
        x1 = lineup_record[team_num][0]
        foe_list = lineup_record[foe_num]
        inn = float(foe_list[1][0][1])
        sp_era = float(foe_list[1][0][2])
        sp_fip = float(foe_list[1][0][3])
        
        sp_len = int(foe_list[1][0][4])
        sp_era = np.where(sp_era>=8,8,sp_era)
        sp_fip = np.where(sp_fip>=8,8,sp_fip)
        
        sp = sp_fip
        
        rp_era = foe_list[2][0]
        rp_fip = foe_list[2][1]
        
        
        rp_era = np.where(rp_era>=8,8,rp_era)
        rp_fip = np.where(rp_fip>=8,8,rp_fip)
        
        rp = rp_fip
        #fip_mean = sp_ratio_dic[2021][team_num]
        if inn <=5:
            inn = 5
        
        x2 = (inn * sp)/9
        x3 = (9-inn)*rp/9
        if is_home:
            x4 = 1
        else:
            x4 = 0
        x5 = sp_len
        
        
        if sp_len <= 3:
            sp_len = 3
        
            
        print(x1,sp,rp,x4,sp_len)
        
        team_params = team_params_list[team_num]
        result = 0 
        count = 0
        for i in range(3):
            
            params = total_params_list[i][(45,sp_len,50)]
            pred = np.dot(params, [1,x1,x2,x3,x4])
            if i == 0:
              result+=  1/pred
            elif i == 1:
                result+= np.exp(pred)
            else:
                result+= pred
            count+=1
        
        for i, params in enumerate(team_params):
            
            if params[0] == 0:
                pass
            
            else:
                count+=1
                
                pred = np.dot(params,[1,x1,x2,x3,x4])
                if i == 0:
                    result+=  1/pred
                elif i == 1:
                    result+= np.exp(pred)
                else:
                    result+= pred
                    
        result = result / count
        
        
        #result = pred_total[0]
        
        return result 
    away_alpha = total_scale_list[0][(45,sp_len,50)] + 0.5
    away_run = get_odds(lineup_record,away_num,home_num,False,total_params_list,team_params_list)
    if b.park_factor_total.get(ground)!=None:
        away_run = away_run * b.park_factor_total[ground]
    away_beta = away_run / away_alpha
    away_sample = s.gamma_sample([away_alpha,away_beta])
    
    home_run = get_odds(lineup_record,home_num,away_num,True,total_params_list,team_params_list)
    if b.park_factor_total.get(ground)!=None:
        home_run = home_run * b.park_factor_total[ground]
    
    home_alpha= total_scale_list[0][(45,sp_len,50)] + 0.5
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
#%%
print(np.array(lineup_record))
#%%
print(toto_array)
#%%
conn_local = pymysql.connect(host= cd.local_host, user=cd.local_user, password= cd.local_code, db= cd.db, charset='utf8')
conn_aws = pymysql.connect(host = cd.aws_host, user = cd.aws_user, password = cd.aws_code , db = cd.db, charset = cd.charset)

#c.set_last_game_num_list(2021,conn)

#%%

d.insert_table(conn_local, 'today_toto',toto_array)
conn_local.commit()
conn_local.close()
#%%
d.insert_table(conn_aws, 'today_toto',toto_array)
conn_aws.commit()
conn_aws.close()