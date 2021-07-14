#%%
import sys
sys.path.append('C:\\Users\\Chan\\Desktop\\BaseballProject\\python')

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

from baseball_2021 import base as bs
#%%

class Crawling(bs.Database):
    
    def __init__(self):
        self.today_date = None
        self.craw_time = None
        self.update_time_array = None
    
    def driver_start(self):
        '''
        Start driver
        
            구글 드라이버 실행
        
        Parameter
        -----------------------------
        
            start_date: 지정한 날짜부터 크롤링 시작
        
        '''
        
        try:
            driver = webdriver.Chrome('C:\\Users\\Chan\\.conda\\envs\\baseball\\lib\\site-packages\\chromedriver_autoinstaller\\90\\chromedriver')
            
            self.driver = driver
        except:
            chromedriver_autoinstaller.install()
        
    def set_craw_time(self, craw_type):
        today = datetime.datetime.today()
        self.today_date = str(today.year) + str(today.month).zfill(2) + str(today.day).zfill(2)
        self.craw_time = str(today.hour).zfill(2) + ":" +str(today.minute).zfill(2)
        self.update_time_array = np.array([self.today_date, self.craw_time, craw_type]).reshape(1,-1)
        
    def update_craw_time(self):
        conn = pymysql.connect(host='localhost', user='root', password='dudrn1', db='baseball', charset='utf8')
        cursor = conn.cursor()
        
        for data in self.update_time_array:
        
            data_str = str(tuple(data))
            sql = 'insert into'+ ' update_time' + ' values ' + data_str
            cursor.execute(sql)
    
        conn.commit()
        conn.close()