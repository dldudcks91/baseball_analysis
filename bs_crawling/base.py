#%%
import sys
sys.path.append('D:\\BaseballProject\\python')
#%%

# 크롤링 관련 library
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import requests
import chromedriver_autoinstaller

# 계산 및 기타 library
import datetime
import pandas as pd
import numpy as np 
import pymysql

from bs_database import base as bs
#%%

class Crawling(bs.Database):
    '''
    크롤링 기본 base
    
    '''
    def __init__(self):
        self.today_date = None
        self.craw_time = None
        self.update_time_array = None
        self.driver = None
        
        
    def driver_start(self,is_headless = False):
        '''
        Start driver
        
            구글 드라이버 실행
        
        Parameter
        -----------------------------
        
            start_date: 지정한 날짜부터 크롤링 시작
        
        '''
        
        version = chromedriver_autoinstaller.get_chrome_version()[:3]
        if is_headless:
            webdriver_options = webdriver.ChromeOptions()
            webdriver_options.add_argument('headless')
            
            try:
                self.driver = webdriver.Chrome(options = webdriver_options)
                
            except:
                chromedriver_autoinstaller.install()
                
        else:
            try:
                self.driver = webdriver.Chrome()
                
            except:
                chromedriver_autoinstaller.install()
                
                
        
    def set_craw_time(self, craw_type):
        today = datetime.datetime.today()
        self.today_date = str(today.year) + str(today.month).zfill(2) + str(today.day).zfill(2)
        self.craw_time = str(today.hour).zfill(2) + ":" +str(today.minute).zfill(2)
        self.update_time_array = np.array([self.today_date, self.craw_time, craw_type]).reshape(1,-1)
        
    def update_craw_time(self,conn):
        cursor = conn.cursor()
        
        for data in self.update_time_array:
        
            data_str = str(tuple(data))
            sql = 'insert into'+ ' update_time' + ' values ' + data_str
            cursor.execute(sql)
    
        conn.commit()
        conn.close()