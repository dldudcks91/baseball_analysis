#%%
import sys
sys.path.append('D:\\BaseballProject\\python')

#%%


from crawling import crawling_base as cb
from crawling import crawling_kbo as ck
from crawling import crawling_wisetoto_all as cw
#from baseball_2021.crawling import crawling_today_toto
#%%

'''
Crawling_KBO

'''


c = crawling_kbo.Crawling_baseball()
#%%
c.driver_start()
c.date = 20210712
c.driver.get('https://www.koreabaseball.com/Schedule/GameCenter/Main.aspx?gameDate=' + str(c.date))
#c.end_game_crawling()
c.start_game_crawling()
c.driver.close()

#%%
'''
Crawling_toto
'''
c = crawling_today_toto.Crawling_toto()
#%%
c.craw_toto_all()
#c.update_craw_time()
print(c.toto_array)

#%%
c = Crawling_today_toto()
#%%
c.craw_toto_all()
#c.update_craw_time()
print(c.toto_array)

#%%
conn = pymysql.connect(host='localhost', user='root', password='dudrn1', db='baseball', charset='utf8')
cursor = conn.cursor()

#%%
conn = pymysql.connect(host='lyc-baseball.cgh18xdnf8rj.ap-northeast-2.rds.amazonaws.com', user='LYC', password='3whddpdltm!', db='baseball', charset='utf8')
cursor = conn.cursor()
#%%
record_array = c.toto_array
for i,data in enumerate(record_array):
    data_str = str(tuple(data))
    sql = 'insert into'+ ' today_toto' + ' values ' + data_str
    cursor.execute(sql)

conn.commit()
conn.close()

#%%
sql = 'select * from today_toto'
cursor.execute(sql)
local_today_game_info_array = cursor.fetchall()
conn.close()
#%%
last_idx = np.array(local_today_game_info_array)[-1,0]
t = np.array(today_game_info_array)
new_array = t[t[:,0]>last_idx,:]
#%%
record_array = new_array
for i,data in enumerate(record_array):
    data_str = str(tuple(data))
    sql = 'insert into'+ ' today_toto' + ' values ' + data_str
    cursor.execute(sql)

conn.commit()
conn.close()

#%%
import time
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.base import JobLookupError

sched = BackgroundScheduler()

count = 0
def job_craw_toto():
    t = time.localtime()
    c.craw_toto_all()
    #c.update_craw_time()
    print(c.toto_array)
    conn = pymysql.connect(host='lyc-baseball.cgh18xdnf8rj.ap-northeast-2.rds.amazonaws.com', user='LYC', password='3whddpdltm!', db='baseball', charset='utf8')
    cursor = conn.cursor()

    record_array = c.toto_array
    for i,data in enumerate(record_array):
        data_str = str(tuple(data))
        sql = 'insert into'+ ' today_toto' + ' values ' + data_str
        cursor.execute(sql)
    
    conn.commit()
    conn.close()
    
    
def job():
    t = time.localtime()
    
    global count, sched
    count+=1
    print(str(t.tm_min) + ":" + str(t.tm_sec))
    if t.tm_min >50:
        sched.remove_job('test')
        sched.shutdown()
    
#sched.add_job(job_craw_toto, 'cron', minute ="40", id="craw")
sched.add_job(job_craw_toto,'cron', minute = "25",id = 'test')
sched.start()
#%%
sched.shutdown()
#%%
time.localtime()



#%%
c = Crawling_wisetoto()

c.driver_start()
#%%
year = 2021

error_list = list()
while year<=2021:
    c.set_year(year)    
    game_round = 1
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
toto_array = c.toto_array[1:,:]
rate_mask = (toto_array[:,7]==1)
year_mask = (toto_array[:,0]==2020)
lg_home_mask = (toto_array[:,9]=="LG")
z1 = pd.DataFrame(toto_array[rate_mask&year_mask])
#%%
toto_array = c.toto_array[1:,:]

toto_df = pd.DataFrame(toto_array)

#%%
toto_df.to_csv('C:\\Users\\Chan\\Desktop\\crawling_toto_baseball14.csv',encoding = 'cp949')

#%%
z = pd.DataFrame(c.toto_array)
#%%
'https://livescore.co.kr/sports/score_board/baseball_score.php'

'http://fo-ac.com/'