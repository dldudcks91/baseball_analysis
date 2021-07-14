#%%
import sys
sys.path.append('C:\\Users\\Chan\\Desktop\\BaseballProject\\python')


#%%
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import requests
import chromedriver_autoinstaller

import datetime
import pandas as pd
import numpy as np 

import pymysql

from baseball_2021.crawling import crawling

#%%

class Crawling_toto(crawling.Crawling):
    def __init__(self):
        
        self.toto_array = np.empty((1,21))
        self.driver = None
        self.year = None
        self.game_round = None
        
        
        self.today_game_info_array = None
        self.livescore_array = None
        self.fieldone_array = None
        self.toto_array = None
        
        
        
        
        
    def set_today_game_info(self):
        conn = pymysql.connect(host='localhost', user='root', password='dudrn1', db='baseball', charset='utf8')
        cursor = conn.cursor()

        sql = 'select * from today_game_info'
        cursor.execute(sql)
        today_game_info_array = cursor.fetchall()
        conn.close()
        self.today_game_info_array = today_game_info_array

    def craw_livescore(self):
        self.driver.get('https://livescore.co.kr/sports/score_board/baseball_score.php')
        prev_button = self.driver.find_element_by_xpath('//*[@id="score_top"]/ul[2]/li[5]/a[1]')
        next_button = self.driver.find_element_by_xpath('//*[@id="score_top"]/ul[2]/li[5]/a[3]')
        
        
        page_source = self.driver.page_source
        soup = BeautifulSoup(page_source,'html.parser')
        elements = soup.find_all("b",text="KBO")
        
        old_array = np.zeros((1,10))
        for i, element in enumerate(elements):
            record_table = element.parent.parent.parent.parent
            game_time = record_table.find('strong').text[3:]
            records = record_table.find_all(attrs={'class':'teaminfo'})
            
            new_list = [self.today_date,game_time,'livescore',1]
            name_list = list()
            odds_list = list()
            for record in records:
                
                name = record.find('strong').text
                try:
                    odds = record.find('b').text
                except:
                    odds = '-'
                
                    
                
                name_list.append(name)
                odds_list.append(odds)
                
            new_list.extend(name_list)
            try:
                odds1 = float(odds_list[0])
                odds2 = float(odds_list[1])
                
                away_rate = round(1 - (odds1 / (odds1+odds2)),3)
                home_rate = round(1 - (odds2 / (odds1+odds2)),3)
            except:
                away_rate = 0
                home_rate = 0
            
            new_list.append(away_rate)
            new_list.append(home_rate)
            new_list.append(0)
            new_list.append(self.craw_time)
            old_array = np.vstack([old_array,new_list])
        result_array = old_array[1:]
        
        
        self.livescore_array = result_array
            
    def craw_fieldone(self):
        self.driver.get('http://fo-ac.com/')

        self.driver.find_element_by_xpath('//*[@id="memid"]').send_keys('dldudcks91')
        self.driver.find_element_by_xpath('//*[@id="mempwd"]').send_keys('dudrn1')
        self.driver.find_element_by_xpath('//*[@id="btnLogin"]').click()
        WebDriverWait(self.driver,5).until(expected_conditions.presence_of_element_located((By.XPATH,'//*[@id="menuJoinBet"]')))
        self.driver.find_element_by_xpath('//*[@id="menuJoinBet"]').click()
        
        page_source = self.driver.page_source
        soup = BeautifulSoup(page_source,'html.parser')

        elements = soup.find_all('tr',{'class':'game_set'})
        count = 0
        old_array = np.zeros((1,10))
        for i,element in enumerate(elements):   
            
            location = element.find("div",{'class':'flag_locatoin'})
            if location:
                if count == 0:
                    is_KBO = location.find("span",text='KBO')
                    if is_KBO:
                        count = 1
                        continue
                else:
                    count = 0
            
            
            if count == 1:
                
                game_time = element.find("td",{"class":'date'}).text[-5:]
                new_list = [self.today_date, game_time, 'field-one']
                
                away_name = element['awayname']
                away_name = away_name[:away_name.index('[')]
                away_odds = element['awayrate']
                
                home_name = element['homename']
                home_name = home_name[:home_name.index('[')]
                home_odds = element['homerate']
                
                win_type = element['wintype']
                
                new_list.append(win_type)
                new_list.append(away_name.strip())
                new_list.append(home_name.strip())
                
                
                odds1 = float(away_odds)
                odds2 = float(home_odds)
                
                away_rate = round(1 - (odds1 / (odds1+odds2)),3)
                home_rate = round(1 - (odds2 / (odds1+odds2)),3)
                
                handicap = element['drawrate']
                
                new_list.append(away_rate)
                new_list.append(home_rate)
                new_list.append(handicap)
                new_list.append(self.craw_time)
                old_array = np.vstack([old_array,new_list])
        result_array = old_array[1:]
        self.fieldone_array = result_array
        
    def craw_toto_all(self):
        self.set_craw_time(2)
        self.driver_start()
        self.craw_livescore()
        self.craw_fieldone()
        toto_array = np.zeros((1,10))
        toto_array = np.vstack([toto_array,self.livescore_array])
        toto_array = np.vstack([toto_array,self.fieldone_array])
        self.toto_array = toto_array[1:]
        self.driver.close()
#%%

c = Crawling_toto()
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
