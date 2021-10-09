#%%

# 불러오기 위치 설정
import sys
sys.path.append('D:\\BaseballProject\\python')

import numpy as np
import pandas as pd
import pymysql
from sqlalchemy import create_engine
#%%
from bs_database import base as db_base
from bs_database import update as db_update
from bs_personal import personal_code as cd
#%%

d = db_update.Update()

#%%
'''
    크롤링한 데이터를 통해 최근 동향(run_graph) 데이터 갱신 및 team_info 테이블 update
'''
# run-graph-data
conn_local  = pymysql.connect(host = cd.local_host, user = cd.local_user, password = cd.local_code , db = cd.db, charset = cd.charset)
conn_aws = pymysql.connect(host = cd.aws_host, user = cd.aws_user, password = cd.aws_code , db = cd.db, charset = cd.charset)

d.load_data_all(db_address = cd.db_address ,code = cd.local_code , file_address = cd.file_address)
d.set_last_game_num_list(2021,conn_local)

sql = 'select * from run_graph_data'
old_graph_array = np.array(d.fetch_sql(sql,conn_local))


new_graph_array = d.get_recent_data(2021, old_graph_array)

d.array_to_db(conn_local,new_graph_array,'run_graph_data')
d.array_to_db(conn_aws,new_graph_array,'run_graph_data')
conn_aws.commit()
conn_aws.close()

conn_local.commit()
conn_local.close()

#%%
#update_team_info
conn_local  = pymysql.connect(host = cd.local_host, user = cd.local_user, password = cd.local_code , db = cd.db, charset = cd.charset)
win_rate_list = d.get_new_win_rate(d.game_info_array, d.score_array)

d.update_team_info(conn_local, 2021, win_rate_list, update_type = 'record')
conn_local.commit()
conn_local.close()

#%%
'''
local_db to aws_db

local db에서 aws db로 데이터 보내기
'''


#%%

insert_table_list = ['game_info','team_game_info','batter_record','pitcher_record','score_record']#'run_graph_data']
update_table_list = ['team_info']



#%%
try:
    conn_local = pymysql.connect(host= cd.local_host, user=cd.local_user, password= cd.local_code, db= cd.db, charset='utf8')
    conn_aws = pymysql.connect(host= cd.aws_host, user=cd.aws_user, password= cd.aws_code, db= cd.db, charset='utf8')
    for table in insert_table_list:
        
        local_array = d.load_table(conn_local ,table=table)
        aws_array = d.load_table(conn_aws, table=table)
        
        if table == "game_info":
            last_idx = aws_array[-1,0]
            insert_array = local_array[local_array[:,0]>last_idx,:]
        elif table == "team_game_info":
            
            insert_array = local_array[local_array[:,0]>last_idx,:]
            team_game_idx_list = insert_array[:,1] 
        else:
                
            for i, team_game_idx in enumerate(team_game_idx_list):
                
                if i == 0:
                    old_array = local_array[local_array[:,0]==team_game_idx]
                else:
                    new_array = local_array[local_array[:,0]==team_game_idx]
                    old_array = np.vstack([old_array,new_array])
            
            insert_array = old_array
        d.insert_table(conn_aws, table=table, data_array = insert_array)
        print(table)
    conn_local.close()
    conn_aws.commit()
    conn_aws.close()
    print('-- the end commit --')
        
except:
    conn_local.close()
    conn_aws.rollback()
    conn_aws.close()
    print('-- error! --')

#%%
try:
    conn_local = pymysql.connect(host= cd.local_host, user=cd.local_user, password= cd.local_code, db= cd.db, charset='utf8')
    conn_aws = pymysql.connect(host= cd.aws_host, user=cd.aws_user, password= cd.aws_code, db= cd.db, charset='utf8')
    for table in update_table_list:
    
        local_array = d.load_table(conn_local ,table=table)[:,(4,5,6,7,8,0,1)]
        local_list = list(local_array)
        for i, local in enumerate(local_list):
            local_list[i] = list(local)
        d.update_team_info(conn_aws, 2021, local_list, 'local_to_aws')
    conn_local.close()
    conn_aws.commit()
    conn_aws.close()
    print('-- the end commit --')
        
except:
    conn_local.close()
    conn_aws.rollback()
    conn_aws.close()
    print('-- error! --')
#%%

'''
today-table  데이터 옮기
'''
conn_local = pymysql.connect(host= cd.local_host, user=cd.local_user, password= cd.local_code, db= cd.db, charset='utf8')
conn_aws = pymysql.connect(host = cd.aws_host, user = cd.aws_user, password = cd.aws_code , db = cd.db, charset = cd.charset)

conn_aws.begin()

d.delete_table_data(conn_aws,'today_lineup')
d.delete_table_data(conn_aws,'today_team_game_info')
d.delete_table_data(conn_aws,'today_game_info')




for table in ['today_game_info','today_team_game_info','today_lineup']:
    insert_array = d.load_table(conn_local ,table=table)
    d.insert_table(conn_aws, table=table, data_array = insert_array)

conn_local.close()    
conn_aws.commit()
conn_aws.close()
#%%

#%%



