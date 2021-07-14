#%%
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np 


#%%

class Crawling_wisetoto:
    def __init__(self):
        
        self.toto_array = np.empty((1,21))
        self.driver = None
        self.year = None
        self.game_round = None
        self.round_data = None
        
        
        
    def driver_start(self):
        
        driver = webdriver.Chrome('C:\\Users\\Chan\\.conda\\envs\\baseball\\lib\\site-packages\\chromedriver_autoinstaller\\90\\chromedriver')
        driver.get('http://www.wisetoto.com/index.htm')
        
        WebDriverWait(driver,10).until(expected_conditions.presence_of_element_located((By.XPATH,'//*[@id="tab02_01"]')))
        self.driver = driver
    
        
        # Select year
    def set_year(self,year):
        select_year = Select(self.driver.find_element_by_xpath('//*[@id="tab02_01"]/div[1]/div[1]/select[1]'))
        select_year.select_by_value(str(year))
        self.year = year
    
    def set_round(self, game_round):
        
        select_round = Select(self.driver.find_element_by_xpath('//*[@id="tab02_01"]/div[1]/div[1]/select[2]'))
        select_round.select_by_value(str(game_round))
        WebDriverWait(self.driver,10).until(expected_conditions.presence_of_element_located((By.XPATH,'//*[@id="tab02_01"]')))
        source = self.driver.page_source
        bs = BeautifulSoup(source,'html.parser')#BeautifulSoup(source,'lxml')
        round_data = bs.find_all('ul')
        self.game_round = game_round
        self.round_data = round_data
        
        
        
    def find_data_by_game(self,data):
        new_list = [self.year,self.game_round]
        self.new_list = new_list
        li_list = data.find_all('li')
        
        try:
            if li_list[11].string =='취소':
                return
            if li_list[3].string !='KBO':
                return
        except:
            return
        
        for i,li in enumerate(li_list):
            
        
            # 게임번호 by round
            if i == 0: 
                self.new_list.append(int(li.string))

            
            # 날짜 & 요일
            elif i == 1:
                
                string = li.string
                month = string[:2]
                day = string[3:5]
                date = int(str(self.year) + month + day)
                self.new_list.append(date)
                self.new_list.append(string[6])
            
            # 종목
            elif i == 2:
                self.new_list.append(li['class'][1])
            
            # 리그
            elif i == 3: 
                
                self.new_list.append(str(li.string))
                
            # 핸디 ('' = 노핸디 / H = 점수 핸디 / U = 언오버)
            elif i == 4: 
                
                
                
                if li['class'][0] =='hm':
                    if li.string:
                        handi_num = 2
                        self.new_list.append(handi_num)
                        self.new_list.append(li.string[2:])
                    else:
                        handi_num = 1
                        self.new_list.append(handi_num)
                        self.new_list.append(0)
                elif li['class'][0] =='hp':
                    handi_num = 2
                    self.new_list.append(handi_num)
                    self.new_list.append(li.string[2:])
                elif li['class'][0] == 'un':
                    handi_num = 3
                    self.new_list.append(handi_num)
                    self.new_list.append(li.string[2:])
                    
                
            # 홈팀 이름 & 원정팀 이름
            elif i == 5:
                
                # 홈팀 이름
                self.new_list.append(str(li.contents[0].string))
                
                # 원정팀 이름
                if handi_num == 3:
                    
                    self.new_list.append(str(li_list[7].contents[0].string))
                else:
                    self.new_list.append(str(li_list[7].contents[2].string))
                
                
            # 홈팀 점수 & 원정팀 점수
            elif i == 6:
                if handi_num == 3:
                    self.new_list.append(float(li.contents[0].string))
                    self.new_list.append(None)
                else:
                    self.new_list.append(float(li_list[5].contents[2].string))
                    self.new_list.append(float(li_list[7].contents[0].string))
                    
            # 승배당
            elif i == 8:
                try:
                    self.new_list.append(float(li.find(class_ = 'pt').string))
                except:
                    self.new_list.append('1.74')
                
                
            # 무배당
            elif i == 9:
                
                find_pt = li.find(class_ = 'pt')
                
                if find_pt :
                    self.new_list.append(float(find_pt.string))
                    
                else:
                    self.new_list.append(None)
                
            # 패배당
            elif i == 10:
                try:
                    self.new_list.append(float(li.find(class_ = 'pt').string))
                except:
                    self.new_list.append('1.74')
            
            # 결과
            elif i == 11:
                self.new_list.append(str(li.string))
                
            # 변경 전 배당
            elif i == 12:
                
                bc_string = li.find(class_ = 'bc')
                if bc_string:
                    bc_string = bc_string['onmouseover']
                    
                    where_win = bc_string.find('승')
                    where_draw = bc_string.find('무')
                    where_lose = bc_string.find('패')
                    where_handi = bc_string.find('핸디')
                    where_unover = bc_string.find('U/O')
                    where_over = bc_string.find('O')
                    where_under = bc_string.find('U')
                    
                    if where_win != -1:
                        self.new_list.append(bc_string[where_win+7:where_win+11])
                    elif where_over !=-1 and where_unover == -1:
                        self.new_list.append(bc_string[where_over+7:where_over+11])
                    else:
                        self.new_list.append(None)
                    
                    if where_draw != -1:
                        self.new_list.append(bc_string[where_draw+7:where_draw+11])
                    else:
                        self.new_list.append(None)
                    
                    if where_lose != -1:
                        self.new_list.append(bc_string[where_lose+7:where_lose+11])
                        
                    elif where_under !=-1 and where_unover == -1:
                        self.new_list.append(bc_string[where_under+7:where_under+11])
                    else:
                        self.new_list.append(None)
                    
                    if where_handi != -1:
                        new_string = bc_string[where_handi+10:]
                        where_handi_end = new_string.find(' ')
                        self.new_list.append(new_string[:where_handi_end])
                        
                    elif where_unover !=-1:
                        new_string = bc_string[where_unover+11:]
                        where_unover_end = new_string.find(' ')
                        self.new_list.append(new_string[:where_unover_end])
                    else:
                        self.new_list.append(None)
                        
                    
                    
                    
                    
                else:
                    self.new_list.extend([None]*4)
                            
                            
                            
                        
                
        self.toto_array = np.vstack([self.toto_array,new_list])     

                    
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