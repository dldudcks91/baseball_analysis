#%%
import sys
sys.path.append('D:\\BaseballProject\\python')



#%%

import pymysql
import numpy as np
from bs_crawling import base as cb
from bs_crawling import kbo
from bs_crawling import wisetoto_all
from bs_crawling import today_toto

from bs_personal import personal_code as cd


#%%
'''
Crawling_KBO

경기결과 가져오기 : c.end_game_crawling
경기 시작 전 라인업 가져오기: c.start_game_crawling

'''
#%%
def craw_kbo(conn_type, date, url, game_type, start_idx = 0):
    
    ck.driver.get(url + str(date))
    
    if conn_type == 'local':
        conn  = pymysql.connect(host = cd.local_host, user = cd.local_user, password = cd.local_code , db = cd.db, charset = cd.charset)        
    elif conn_type == 'aws':
        conn = pymysql.connect(host = cd.aws_host, user = cd.aws_user, password = cd.aws_code , db = cd.db, charset = cd.charset)
    else:
        print('plz right conn_type')
        
        
    if game_type == 'end':    
        
        ck.end_game_crawling(conn)
        
    elif game_type == 'start':

        ck.start_game_crawling(conn, start_idx = start_idx)
    else:
        print('plz right game_type')
                
def click_next():
    next_button = ck.driver.find_element_by_xpath('//*[@id="contents"]/div[2]/ul/li[3]')
    next_button.click()   
    
ck = kbo.Crawling_kbo()


ck.driver_start()


ck.date = 20220913

url = 'https://www.koreabaseball.com/Schedule/GameCenter/Main.aspx?gameDate='

import datetime
import time
start_date = 20220913
datetime.datetime.strptime(start_date,"%Y%m%d")


craw_kbo('local',start_date, url,'end', start_idx = 0)  
ck.driver.close()

#%%

#%%

#%%
'''
Crawling_today_toto

오늘 나온 도박사들의 배당을 크롤링
'''

#%%
c = today_toto.Crawling_today_toto()
#%%

c.craw_toto_all(url_first = cd.ls_url, url_second = cd.f1_url, login_id = cd.f1_id, login_code = cd.f1_code)
#%%
conn_local  = pymysql.connect(host = cd.local_host, user = cd.local_user, password = cd.local_code , db = cd.db, charset = cd.charset)
conn_aws = pymysql.connect(host = cd.aws_host, user = cd.aws_user, password = cd.aws_code , db = cd.db, charset = cd.charset)
 for conn in [conn_local, conn_aws]:
    cursor = conn.cursor()
    record_array = c.toto_array
    for i,data in enumerate(record_array):
        data_str = str(tuple(data))
        sql = 'insert into'+ ' today_toto' + ' values ' + data_str
        cursor.execute(sql)

    conn.commit()
    conn.close()
    
    
#%%

    #%%

'''
Scheduling of today toto


'''
import time
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.base import JobLookupError

sched = BackgroundScheduler()

count = 0
def job_craw_odds():
    
    c.craw_toto_all(url_first = cd.ls_url, url_second = cd.f1_url, login_id = cd.f1_id, login_code = cd.f1_code)
    #c.update_craw_time()
    print(c.toto_array)
    conn_local  = pymysql.connect(host = cd.local_host, user = cd.local_user, password = cd.local_code , db = cd.db, charset = cd.charset)
    conn_aws = pymysql.connect(host = cd.aws_host, user = cd.aws_user, password = cd.aws_code , db = cd.db, charset = cd.charset)

    for conn in [conn_local, conn_aws]:
        cursor = conn.cursor()
        record_array = c.toto_array
        for i,data in enumerate(record_array):
            data_str = str(tuple(data))
            sql = 'insert into'+ ' today_toto' + ' values ' + data_str
            cursor.execute(sql)
        
        conn.commit()
        conn.close()
    
    
   
  
def job_craw_endgame():

    ck.driver_start(is_headless = True)
    
    for conn_type in ['local','aws']:
        ck.date = int(time.strftime('%Y%m%d',time.localtime()))
        url = 'https://www.koreabaseball.com/Schedule/GameCenter/Main.aspx?gameDate='
        craw_kbo(conn_type,ck.date, url,'end')  
        ck.driver.close()
        
def job_craw_startgame():
    
    ck.driver_start(is_headless = True)
    
    for conn_type in ['local','aws']:
        ck.date = int(time.strftime('%Y%m%d',time.localtime()))+1
        url = 'https://www.koreabaseball.com/Schedule/GameCenter/Main.aspx?gameDate='
        craw_kbo(conn_type,ck.date, url,'start')  
        ck.driver.close()

sched.add_job(job_craw_odds,'cron', minute = "10",id = 'odds')
#sched.add_job(job_craw_endgame,'cron', hour = '23', minute = '00', id='endgame')
#sched.add_job(job_craw_startgame,'cron', hour = '23', minute = '00', id='startgame')
sched.start()
#%%
sched.shutdown()

#%%
'''
Crawling Wise toto

wise-toto에 있는 과거 도박사 배당 크롤


'''
c = wisetoto_all.Crawling_wisetoto()

c.driver_start()

c.driver.get(cd.wise_url)
c.set_main()

        
#%%
year = 2022

error_list = list()
while year<=2022:
    c.set_year(year)    
    game_round = 41
    while True:
        
        try:
            while True:
                c.set_round(game_round)
                if len(c.round_data)!=17:
                    break
            game_round +=1
            
        except:
            
            break
        
        data_len = len(c.round_data)
        data_num = 0
        error_list.append([game_round,data_len])
        while True:
            if data_num >= data_len:
                break
            data = c.round_data[data_num]
            c.find_data_by_game(data)
            data_num+=1
            
        
    year+=1
#%%

new_toto_array = c.toto_array
#%%
len(old_toto_array)

#%%
toto_array = np.vstack([old_toto_array[1:],new_toto_array[1:]])
#%%
toto_array = c.toto_array[1:,:]
#%%
#toto_array = toto_array[toto_array[:,7] == 1,:]
toto_array = np.hstack([toto_array[:,0].reshape(-1,1),toto_array[:,3:5],toto_array[:,7:]])
#%%
team_dic = {'LG':1,'롯데':2,'KIA':3,'삼성':4,'두산':5,'한화':6,'SK':7,'SSG':7,'키움':8,'넥센':8,'NC':9,'KT':10,'kt':10}
home = toto_array[:,5]
away = toto_array[:,6]
old_array = np.zeros((1,2))
for h, a in zip(home, away):
    h_num = team_dic[h]
    a_num = team_dic[a]
    new_array = np.array([h_num,a_num]).reshape(-1,2)
    old_array = np.vstack([old_array,new_array]) 

team_num_array = old_array[1:]

toto_array = np.hstack([toto_array,team_num_array])

#%%
import pandas as pd
toto_df = pd.DataFrame(toto_array)

#%%
toto_df.to_csv('C:\\Users\\Chan\\Desktop\\crawling_toto_baseball20221.csv',encoding = 'cp949')

#%%
new_toto_array[:,1]