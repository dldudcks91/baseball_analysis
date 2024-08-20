#%%
import sys
sys.path.extend(['D:\BaseballProject\\python', 'C:\\Users\\82109\\Desktop\\LYC\\git\\baseball_analysis'])
import requests
from bs4 import BeautifulSoup 
import json
import time
import pandas as pd
import numpy as np
from bs_crawling import base as cb
from datetime import datetime, timedelta

import pymysql
from bs_personal import personal_code as cd
from bs_crawling import kbo_request as kr


#%%
conn_aws = pymysql.connect(host = cd.aws_host, user=cd.aws_user, password= cd.aws_code, db= cd.db, charset='utf8')
#%%
conn = conn_aws
#%%
ck = kr.Crawling_kbo_request()
YEAR = 2024
ck.year = YEAR

#%%

sql = 'select max(game_idx) from game_info '
last_game_idx_tuple = ck.fetch_sql(sql,conn)
last_game_idx = last_game_idx_tuple[0][0]
last_game_date_str = last_game_idx[:8]
#%%

#%%
ck.set_last_game_num_list(ck.year,conn) #ck.last_num_list = [0 for i in range(10)]#
date_time = datetime.strptime(last_game_date_str, "%Y%m%d") + timedelta(days = 1)
end_date = 20240814
end_date_time = datetime.strptime(str(end_date),"%Y%m%d")
while date_time < end_date_time:
    date_str = date_time.strftime("%Y%m%d")
    date = int(date_str)
    
    ck.year = int(date_str[:4])
    ck.craw_game_info(date)
    ck.craw_box_score(date)
    ck.craw_score_board(date)  
    if ck.game_dic[date]:
        ck.set_date_total(date)    
    
    date_time = datetime.strptime(date_str, "%Y%m%d")
    date_time+= timedelta(days = 1)
    
    print(date, ck.game_info_array.shape)
    
#%%
print(ck.game_info_array.shape)
print(ck.team_game_info_array.shape)
print(ck.batter_array.shape)
print(ck.pitcher_array.shape)
print(ck.score_array.shape)



#%%
try:
    ck.array_to_db(conn, ck.game_info_array, 'game_info')  
    ck.array_to_db(conn, ck.team_game_info_array, 'team_game_info')
    ck.array_to_db(conn, ck.batter_array, 'batter_record')
    ck.array_to_db(conn, ck.pitcher_array, 'pitcher_record')
    ck.array_to_db(conn, ck.score_array, 'score_record')
    ck.update_team_info(conn, YEAR, ck.last_game_num_list, update_type = 'game_num')
except:
    conn.rollback()
    print('error')



conn.commit()
conn.close()

#%%

date = 20240726
date_str = str(date)

ck.year = int(date_str[:4])
ck.set_last_game_num_list(ck.year,conn)
ck.craw_game_info(date)
ck.craw_lineup(date)
ck.set_date_start(date)
#%%
print(ck.game_info_array.shape)
print(ck.team_game_info_array.shape)
print(ck.lineup_array.shape)
#%%

ck.delete_table_data(conn, 'today_lineup')
ck.delete_table_data(conn, 'today_team_game_info')
ck.delete_table_data(conn, 'today_game_info')




#%%
try:
    ck.array_to_db(conn, ck.game_info_array, 'today_game_info')  
    ck.array_to_db(conn, ck.team_game_info_array, 'today_team_game_info')
    ck.array_to_db(conn, ck.lineup_array, 'today_lineup')
    
except pymysql.InterfaceError as e:
    conn.rollback()
    print(f"InterfaceError 발생: {e}")



conn.commit()

conn.close()
#%%
z1 = ck.game_dic[20240724]

z = ck.lineup_dic[20240724]


#%%
ck.team_game_info_array


















#%%
#%%

from bs_database import update as db_update

YEAR = 2024
#conn_local = pymysql.connect(host= cd.local_host, user=cd.local_user, password= cd.local_code, db= cd.db, charset='utf8')
#conn = conn_local

conn_aws = pymysql.connect(host = cd.aws_host, user=cd.aws_user, password= cd.aws_code, db= cd.db, charset='utf8')
conn = conn_aws
ck.load_data_all(db_address = cd.db_address ,code = cd.aws_code , file_address = cd.file_aws_address)
ck.set_last_game_num_list(YEAR,conn)

#%%


#%%
d = db_update.Update()
win_rate_list = d.get_new_win_rate(ck.game_info_array, ck.score_array)
#%%
ck.update_team_info(conn, YEAR, ck.last_game_num_list, update_type = 'game_num')
ck.update_team_info(conn, YEAR, win_rate_list, update_type = 'record')
conn.commit()
conn.close()
#%%