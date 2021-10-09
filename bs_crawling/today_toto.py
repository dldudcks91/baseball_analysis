#%%
import sys
sys.path.append('D:\\BaseballProject\\python')


# 크롤링 관련 library
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

# 계산 및 기타 library
import pandas as pd
import numpy as np 

from bs_crawling import base as cb
#%%

class Crawling_today_toto(cb.Crawling):
    def __init__(self):
        
        
        self.driver = None
        self.year = None
        self.game_round = None
        
        
        self.today_game_info_array = None
        self.toto_first_array = None
        self.toto_second_array = None
        self.toto_array = None
        

        
    def set_today_game_info(self, conn):
        conn = conn
        cursor = conn.cursor()

        sql = 'select * from today_game_info'
        cursor.execute(sql)
        today_game_info_array = cursor.fetchall()
        conn.close()
        self.today_game_info_array = today_game_info_array

    def craw_odds_first(self, url):
        self.driver.get(url)
        
        self.driver.find_element_by_xpath('//*[@id="score_menu"]/ul/li[2]').click()

        #prev_button = self.driver.find_element_by_xpath('//*[@id="score_top"]/ul[2]/li[5]/a[1]')
        #next_button = self.driver.find_element_by_xpath('//*[@id="score_top"]/ul[2]/li[5]/a[3]')
        
        #self.driver.switch_to_frame(self.driver.find_elements_by_css_selector('#frame > iframe'))
        self.driver.switch_to_frame(self.driver.find_elements_by_tag_name('iframe',)[5])
        page_source = self.driver.page_source
        soup = BeautifulSoup(page_source,'html.parser')
        #WebDriverWait(self.driver,5).until(expected_conditions.presence_of_element_located((By.XPATH,'//*[@id="score_top"]/ul[2]/li[2]/a')))
        
        #kbo_button = self.driver.find_element_by_xpath('//*[@id="score_top"]/ul[2]/li[2]/a')
        
        #kbo_button.click()
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
                continue
            
            new_list.append(away_rate)
            new_list.append(home_rate)
            new_list.append(0)
            new_list.append(self.craw_time)
            old_array = np.vstack([old_array,new_list])
        result_array = old_array[1:]
        
        
        self.toto_first_array = result_array
            
    def craw_odds_second(self, url, login_id, login_code):
        self.driver.get(url)
        WebDriverWait(self.driver,5).until(expected_conditions.presence_of_element_located((By.XPATH,'//*[@id="memid"]')))
        self.driver.find_element_by_xpath('//*[@id="memid"]').send_keys(login_id)
        self.driver.find_element_by_xpath('//*[@id="mempwd"]').send_keys(login_code)
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
        self.toto_second_array = result_array
        
    def craw_toto_all(self, url_first, url_second, login_id, login_code):
        self.set_craw_time(2)
        self.driver_start(is_headless = True)
        self.craw_odds_first(url_first)
        self.craw_odds_second(url_second, login_id, login_code)
        toto_array = np.zeros((1,10))
        toto_array = np.vstack([toto_array,self.toto_first_array])
        toto_array = np.vstack([toto_array,self.toto_second_array])
        self.toto_array = toto_array[1:]
        print(self.toto_array)
        print('--- end craw_odds ---')
        self.driver.close()


