#%%

# 불러오기 위치 설정
import sys
sys.path.append('C:\\Users\\Chan\\Desktop\\BaseballProject\\python')

from baseball_2021 import base as bs


import numpy as np
import pandas as pd
import pymysql
from sqlalchemy import create_engine

#%%
d = bs.Database()
d.load_data_all()

#%%
import time
start= time.time()
old_array = np.zeros((1,9))
game_info_array = d.game_info_array[(d.game_info_array[:,2] == 2021),:]


#%%

for i,team_game_info in enumerate(game_info_array):
    team_game_idx = team_game_info[1]
    year = team_game_idx[:4]
    team_num = team_game_idx[4:6]
    game_num = team_game_idx[6:]
    
    stadium = team_game_info[-1]
    park_factor = d.park_factor_total.get(stadium)
    if park_factor == None: park_factor = 1
    
    # run-graph 데이터 생성
    score_array = d.score_array[d.score_array[:,1]==team_game_idx,:][0]
    game_inn = 12 - list(score_array[-16:-4]).count('-')
    run = int(score_array[-4]) * 9 / game_inn
    run = run / park_factor
    
    pitcher_array = d.pitcher_array[d.pitcher_array[:,1]==team_game_idx,:]
    rp_array = pitcher_array[1:]
    rp_fip = np.sum(rp_array[:,-1])
    rp_fip = rp_fip / park_factor
    rp_inn = np.sum(rp_array[:,-11])
    
    if int(game_num) == 1:
        
        count = 0
        run_array = np.zeros((1,3))
        
    new_run_list = [run,rp_inn,rp_fip]
    run_array = np.vstack([run_array,new_run_list])
    
    if count < 20:
        count+=1
   
    count_5 = min(5,count)
    count_20 = min(20,count)
    run_5 = np.round(sum(run_array[-count_5:,0]) / count_5,3)
    run_20 = np.round(sum(run_array[-count_20:,0]) / count_20,3)
    
    rp_inn_5 = sum(run_array[-count_5:,1])
    if rp_inn_5 == 0: rp_inn_5 = 1
    rp_inn_20 = sum(run_array[-count_20:,1])
    if rp_inn_20 == 0: rp_inn_20 = 1
    
    rp_fip_5 = np.round((sum(run_array[-count_5:,2]) / rp_inn_5 ) + 3.2,3)
    rp_fip_20 = np.round((sum(run_array[-count_20:,2]) / rp_inn_20 ) + 3.2,3)
    
    
    
    new_array = np.array([team_game_idx,year,team_num,game_num,run,run_5,run_20,rp_fip_5,rp_fip_20]).reshape(1,-1)
    old_array = np.vstack([old_array,new_array])
    
print(time.time()-start)
#%%
old_array[old_array[:,2]=='09',:]
    #%%
    conn  = pymysql.connect(host='localhost', user='root', password='dudrn1', db='baseball', charset='utf8')
    d.set_last_game_num_list(2021,conn)
    conn.close()
    #%%
    engine = create_engine("mysql+pymysql://root:" + "dudrn1" + "@127.0.0.1/baseball",encoding = 'utf-8')
    conn = engine.connect()
    old_graph_array = np.array(pd.read_sql_table('run_graph_data',conn))

    #%%
    graph_array =  np.zeros((1,9))
    
    for team_num in range(1,11):
        last_game_num = d.last_game_num_list[team_num]
        graph_last_game_num = int(old_graph_array[(old_graph_array[:,1]==2021)&(old_graph_array[:,2]==team_num),:][-1,3])

        for game_num in range(graph_last_game_num+1,last_game_num+1):
            new_array = old_array[(old_array[:,2] == str(team_num).zfill(2))&(old_array[:,1]=='2021')&(old_array[:,3]==str(game_num).zfill(3)),:]
            graph_array = np.vstack([graph_array,new_array])
    
    graph_array = graph_array[1:]
    conn.close()

    #%%
    conn = pymysql.connect(host='lyc-baseball.cgh18xdnf8rj.ap-northeast-2.rds.amazonaws.com', user='LYC', password='3whddpdltm!', db='baseball', charset='utf8')
    #conn  = pymysql.connect(host='localhost', user='root', password='dudrn1', db='baseball', charset='utf8')
    #c.set_last_game_num_list(2021,conn)
    cursor = conn.cursor()
#%%
for i,data in enumerate(graph_array):
    data_str = str(tuple(data))
    data_str = data_str.replace('None','Null')
    sql = 'insert into'+ ' run_graph_data' + ' values ' + data_str
    cursor.execute(sql)
conn.commit()
conn.close()
        


#%%
gia = d.game_info_array
sa = d.score_array
old_array = np.zeros((1,6))
for year in range(2021,2022):
    for team_num in range(1,11):
        
        
        team_array = sa[(sa[:,2]== year) & (sa[:,3] == team_num),:]
        team_score_list = list(team_array[:,-4])

        game_idx = team_array[:,0]
        win_list = list(team_array[:,7])
        win_count = win_list.count('win')
        foe_score_list = list()
        for gi in game_idx:
            
            foe_array = sa[(sa[:,0] == gi) & (sa[:,2]== year) & (sa[:,3] != team_num),:]
            foe_score = foe_array[0,-4]
            foe_score_list.append(foe_score)
        
        
        tsl = team_score_list[::-1]
        fsl = foe_score_list[::-1]
        win = 0
        draw = 0 
        lose = 0
        length = len(tsl)
        for i in range(length):
            ts = tsl.pop()
            fs = fsl.pop()
            if ts>fs:
                win+=1
            elif ts==fs:
                draw+=1
            else:
                lose+=1
        win_rate = round(win / (win+lose+draw),3)
        new_list = [win,lose,draw,win_rate, year,team_num]
        old_array = np.vstack([old_array,new_list])
        

#%%
record_array = old_array[1:].astype(str)
record_list = list()
for record in record_array:
    record_list.append(list(record))


#%%
#conn = pymysql.connect(host='lyc-baseball.cgh18xdnf8rj.ap-northeast-2.rds.amazonaws.com', user='LYC', password='3whddpdltm!', db='baseball', charset='utf8')
conn = pymysql.connect(host='localhost', user='root', password='dudrn1', db='baseball', charset='utf8')
cursor = conn.cursor()
#%%

sql = ''' UPDATE team_info SET win= %s, lose= %s, draw= %s, win_rate=%s WHERE year= %s AND team_num= %s; ''' 
cursor.executemany(sql,record_list)
#%%
conn.commit()
conn.close()
#%%
conn = pymysql.connect(host='lyc-baseball.cgh18xdnf8rj.ap-northeast-2.rds.amazonaws.com', user='LYC', password='3whddpdltm!', db='baseball', charset='utf8')
 
conn = pymysql.connect(host='localhost', user='root', password='dudrn1', db='baseball', charset='utf8')
cursor = conn.cursor()

sql = 'select * from today_game_info'
cursor.execute(sql)
today_game_info_array = cursor.fetchall()
conn.close()
self.today_game_info_array = today_game_info_array

#%%

insert_table_list = ['game_info','team_game_info','batter_record','pitcher_record','score_record','run_graph_data']
update_table_list = ['team_info']

def load_table(host, user, password, db, table):
    conn = pymysql.connect(host=host, user= user, password= password, db= db, charset='utf8')
    cursor = conn.cursor()
    sql = 'select * from ' + table
    cursor.execute(sql)
    result_array = np.array(cursor.fetchall())
    conn.close()
    return result_array
def insert_table(host, user, password, db, table, data_array):
    conn = pymysql.connect(host=host, user= user, password= password, db= db, charset='utf8')
    cursor = conn.cursor()
    
    for i,data in enumerate(data_array):
        data_str = str(tuple(data))
        data_str = data_str.replace('None','Null')
        sql = 'insert into '+ table + ' values ' + data_str
        cursor.execute(sql)
    conn.commit()
    conn.close()
    return
    
for table in insert_table_list:
    local_array = load_table(host='localhost', user='root', password='dudrn1', db='baseball',table=table)
    ser_array = load_table(host='lyc-baseball.cgh18xdnf8rj.ap-northeast-2.rds.amazonaws.com', user='LYC', password='3whddpdltm!', db='baseball', table=table)
    
    if table == "game_info":
        last_idx = ser_array[-1,0]
        insert_array = local_array[local_array[:,0]>last_idx,:]
    elif table == "team_game_info":
        
        insert_array = local_array[local_array[:,0]>last_idx,:]
        team_game_idx_list = insert_array[:,1] 
    else:
        old_array = np.zeros((1,1))
        for team_game_idx in team_game_idx_list:
            
            new_array = local_array[local_array[:,0]==team_game_idx]
            if old_array[0,0] == 0:
                old_array = new_array
            else:
                old_array = np.vstack([old_array,new_array])
        insert_array = old_array
    insert_table(host='lyc-baseball.cgh18xdnf8rj.ap-northeast-2.rds.amazonaws.com', user='LYC', password='3whddpdltm!', db='baseball', table=table,data_array = insert_array)
    print(table)
    #%%
    team_game_idx_list = ['202107077','202108079','202104076','202110073']
#%%

conn = pymysql.connect(host='lyc-baseball.cgh18xdnf8rj.ap-northeast-2.rds.amazonaws.com', user='LYC', password='3whddpdltm!', db='baseball', charset='utf8')    

conn.begin()

d.delete_table_data(conn,'today_lineup')
d.delete_table_data(conn,'today_team_game_info')
d.delete_table_data(conn,'today_game_info')


conn.commit()
conn.close()

for table in ['today_game_info','today_team_game_info','today_lineup']:
    insert_array = load_table(host='localhost', user='root', password='dudrn1', db='baseball',table=table)
    insert_table(host='lyc-baseball.cgh18xdnf8rj.ap-northeast-2.rds.amazonaws.com', user='LYC', password='3whddpdltm!', db='baseball', table=table,data_array = insert_array)
#%%

today_toto = np.loadtxt('today_toto.csv',dtype = str,delimiter = ',')
#%%
conn.thread_id()
#%%
conn = pymysql.connect(host='lyc-baseball.cgh18xdnf8rj.ap-northeast-2.rds.amazonaws.com', user='LYC', password='3whddpdltm!', db='baseball', charset='utf8')
d.delete_table_data(conn,'today_toto')

conn.commit()
conn.close()
#%%
insert_table(host='lyc-baseball.cgh18xdnf8rj.ap-northeast-2.rds.amazonaws.com', user='LYC', password='3whddpdltm!', db='baseball', table="today_toto",data_array = today_toto)

