#%%

# 불러오기 위치 설정
import sys
sys.path.append('D:\\BaseballProject\\python')

from bs_stats import base as bs


import numpy as np
import pandas as pd
import pymysql
from sqlalchemy import create_engine
#%%
d = bs.Database()
d.load_data_all(db_address = "mysql+pymysql://root:",code = "dudrn1", file_address = "@127.0.0.1/baseball")

#%%
conn  = pymysql.connect(host='localhost', user='root', password='dudrn1', db='baseball', charset='utf8')
game_info_array = np.array(d.fetch_sql('select * from game_info',conn))
team_game_info_array = np.array(d.fetch_sql('select * from team_game_info',conn))
score_array = np.array(d.fetch_sql('select * from score_record',conn))
batter_array = np.array(d.fetch_sql('select * from batter_record',conn))
pitcher_array = np.array(d.fetch_sql('select * from pitcher_record',conn))
conn.close()
#%%

range_game_idx = game_info_array[game_info_array[:,0]>='20210610000000',0]

tgi_list = list()
for game_idx in range_game_idx:
    team_game_idx = team_game_info_array[team_game_info_array[:,0]==game_idx,1]
    game_info_array = game_info_array[game_info_array[:,0]!=game_idx]
    for t in team_game_idx:
        tgi_list.append(str(t))

#%%
for tgi in tgi_list:
    
    def ss(array,i,tgi):
        array = array[array[:,i,]!=tgi]
        return array
    team_game_info_array = ss(team_game_info_array,1,tgi)
    score_array = ss(score_array,0,tgi)
    batter_array = ss(batter_array,0,tgi)
    pitcher_array = ss(pitcher_array,0,tgi)
    
#%%
conn  = pymysql.connect(host='localhost', user='root', password='dudrn1', db='baseball', charset='utf8')

cursor = conn.cursor()
#%%
for data in pitcher_array:
    data_str = str(tuple(data))
    data_str = data_str.replace('None','Null')
    sql = 'insert into'+ ' pitcher_record' + ' values ' + data_str
    cursor.execute(sql)
#%%
conn.commit()
conn.close()

#%%

#%%
new_list = list()
for i,idx in enumerate([54,53,52,54,52,53,52,55,53,52]):
    new_list.append([str(idx) ,str(2021),str(i+1)])
#%%
sql = ''' UPDATE team_info SET total_game_num= %s WHERE year= %s AND team_num= %s; ''' 
cursor.executemany(sql,new_list)
#%%
conn.commit()
conn.close()

