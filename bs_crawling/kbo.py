#%%
import sys
sys.path.append('D:\BaseballProject\\python')


# crawling관련 library 불러오기

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup


# 데이터 전처리 및 저장에 필요한 library 불러오기
import numpy as np 
import pymysql
import datetime

from bs_crawling import base as cb
#%%

class Crawling_kbo(cb.Crawling):
    '''
    Class Crawling
    
        KBO에서 제공하는 경기 기록 크롤링
        
    '''
    
    
    def __init__(self):
        
        
        # DB에 넣을 table array생성
        self.game_info_array = np.zeros((1,6))
        self.team_game_info_array = np.zeros((1,7))
        self.score_array = np.zeros((1,18))
        self.batter_raw_array = np.zeros((1,14))
        self.pitcher_raw_array = np.zeros((1,19))
        
        self.lineup_array = np.zeros((1,4))
        # 기타 쓰이는 변수
        self.date = None
        self.is_start = True # 현재 페이지 시작여부
        self.end = None # 경기종료결과(경기종료, 우천취소 등)
        
        self.team_game_idx_dic = dict() # team_game_idx 생성 후 Home, Away구분해 저장하는 dictionary(데이터 분석용)


    def ready_by_round(self):
        '''
        Set basic setting by round
        
            
            해당 라운드(날짜)에 시작하는 경기들 기본정보 가져오기
            
        '''
        
        '''
        # 현재 페이지에서 crawling 할건지 다음페이지로 넘길건지 # 구간 길게해서 한번에 크롤링 사용할때 쓰임 
        
        if self.is_start: 
            self.is_start = False
        
        else: 
            next_button = self.driver.find_element_by_xpath('//*[@id="contents"]/div[2]/ul/li[3]')
            next_button.click()
        '''
        
        
        # 페이지소스 가져오기
        page_source = self.driver.page_source
        soup = BeautifulSoup(page_source,'html.parser')
        
        # 경기들에 대한 정보를 가지고있는 elements 가져오기 - 나중에 element로 가져와 매치 별로 사용
        self.elements = self.driver.find_elements_by_class_name('game-cont')
        
        # 날짜 생성        
        date = soup.find('span',{'id':'lblGameDate'})
        date = date.string
        year = date[:4]
        month = date[5:7]
        day = date[8:10]
        self.date_str = year + month + day
        self.year_str = year
        
        # 게임정보 list 생성
        view_port = soup.find('div', {'class' : 'bx-viewport'})
        view_list = view_port.find_all('li')
        self.view_list = view_list
        
        
    def ready_by_match(self,match_num,element,is_end):
        '''
        Set pagesource by match
        
            매치별 페이지 소스 가져오기
        
        Parameter
        --------------------------------
        
            match_num: 매치번호
            
            element: 매치번호에 맞는 element
            
        '''
        
        # 6번쨰 경기 이상일경우 찾아서 클릭
        if match_num > 4:
            
            self.driver.execute_script("arguments[0].click();", element)
        else:
            element.click()
            
        if is_end:
            WebDriverWait(self.driver,5).until(expected_conditions.presence_of_element_located((By.XPATH,'//*[@id="tabDepth2"]/li[2]')))
            review = self.driver.find_element_by_xpath('//*[@id="tabDepth2"]/li[2]')
        
        
        
            review.click()
            try:
                WebDriverWait(self.driver,5).until(expected_conditions.presence_of_element_located((By.CLASS_NAME,'box-score-area')))
            except:
                review.click()
        else:
            WebDriverWait(self.driver,5).until(expected_conditions.presence_of_element_located((By.ID,'tabPreview')))
            try:
                lineup = self.driver.find_element_by_xpath('//*[@id="tabPreview"]/li[3]/a')
            except:
                lineup = self.driver.find_element_by_xpath('//*[@id="tabPreview"]/li[2]/a')
        
            
            lineup.click()
            
            try:
                WebDriverWait(self.driver,5).until(expected_conditions.presence_of_element_located((By.CLASS_NAME,'mt50')))
            except:
                lineup.click()
            
        page_source = self.driver.page_source
        self.soup = BeautifulSoup(page_source,'html.parser')
        
    
    def craw_game_info(self,view):
        '''
        Set game_info array
        
            game_info table 데이터 전처리
            
            columns = ['game_idx', 'home_name', 'away_name', 'stadium', 'end', 'etc']
        
        
        Parameter
        ---------------------
        
            view: 매치별 기본정보 담은 페이지소스
        
        '''
        
        view_str = str(view)
        date_str = self.date_str
        try:
            home_name = view_str[view_str.index('home_nm'):view_str.index('home_p_id')].strip()[-4:-1].replace('"','')
            away_name = view_str[view_str.index('away_nm'):view_str.index('away_p_id')].strip()[-4:-1].replace('"','')
        
            
        except:
            home_name = view_str[view_str.index('home_nm'):view_str.index('kbot_se')].strip()[-4:-1].replace('"','')
            away_name = view_str[view_str.index('away_nm'):view_str.index('entry_ck')].strip()[-4:-1].replace('"','')
        
        self.home_name = home_name
        self.away_name = away_name
        
        home_num_str = '%02d' % self.team_dic[home_name]
        away_num_str = '%02d' % self.team_dic[away_name]
        
        today_game_num = '0' + view_str[view_str.index('game_sc')-3]
        game_idx = date_str + home_num_str + away_num_str + today_game_num
        self.game_idx = game_idx
        
        
        stadium = view.find('span',{'class' : 'place'}).string
        end = view.find('span',{'class':'time'}).string
        
        self.end = end
        
        etc = None
        

            
            
        
        game_info_array = np.array([game_idx,home_name,away_name,stadium,end,etc]).reshape(1,-1)
        self.game_info_array = game_info_array
        
    def craw_team_game_info(self):
        '''
        Set team_game_info_array
        
            team_game_info table 데이터 전처리
            
            columns = ['game_idx', 'team_game_idx', 'year', 'team_num', 'foe_num', 'game_num', 'home_away']
            
        '''
        year_str = self.year_str
        game_idx = self.game_idx
        
        home_name = self.home_name
        away_name = self.away_name
        
        
        home_num = self.team_dic[home_name]
        away_num = self.team_dic[away_name]
        
        self.home_away_num_dic = dict()
        self.home_away_num_dic['Home'] = self.team_dic[home_name] 
        self.home_away_num_dic['Away'] = self.team_dic[away_name]
        
        
        # home_away_game_idx 생성 및 dictionary 저장
        last_home_game_num = self.last_game_num_list[home_num]
        last_away_game_num = self.last_game_num_list[away_num]
        
        home_game_num = last_home_game_num + 1
        away_game_num = last_away_game_num + 1
        
        home_game_idx = year_str + '%02d' % home_num + '%03d' % (home_game_num)
        away_game_idx = year_str + '%02d' % away_num + '%03d' % (away_game_num)
        
        
        self.team_game_idx_dic['Home'] = home_game_idx
        self.team_game_idx_dic['Away'] = away_game_idx
        
        
        home_game_info_array = np.array([game_idx, home_game_idx, int(year_str), home_num, away_num, home_game_num,'home'])
        away_game_info_array = np.array([game_idx, away_game_idx, int(year_str), away_num, home_num, away_game_num,'away'])
        team_game_info_array = np.vstack([home_game_info_array, away_game_info_array])
        
        self.team_game_info_array = team_game_info_array
        
    def craw_score_array(self):
        '''
        Set score_array
        
            score_record table 데이터 가져오기
            
            columns = ['team_game_idx', 'result', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6',
                         'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'r', 'h', 'e', 'b']
            
        '''
        # 1~12회 이닝별 점수 리스트 생성
        boxscore_find= self.soup.find('div',{'class' : 'tbl-box-score data2'}).find_all('td')
        boxscore_away_list = list()
        boxscore_home_list = list()
        for i,boxscore in enumerate(boxscore_find):
            boxscore = boxscore.string
            
            if i < 12:
                boxscore_away_list.append(boxscore)
            else:
                boxscore_home_list.append(boxscore)
                
        # 경기 득점 / 안타 / 에러 / 볼넷 리스트 생성
        boxscore_run_find= self.soup.find('div',{'class' : 'tbl-box-score data3'}).find_all('td')
        
        boxscore_run_away_list = list()
        boxscore_run_home_list = list()
        for i,boxscore_run in enumerate(boxscore_run_find):
            boxscore_run = boxscore_run.string
            
            if i < 4:
                boxscore_run_away_list.append(boxscore_run)
            else:
                boxscore_run_home_list.append(boxscore_run)
                
        
        # 박스스코어 리스트로 결합
        boxscore_home_list= boxscore_home_list + boxscore_run_home_list
        boxscore_away_list= boxscore_away_list + boxscore_run_away_list
        
        
        
        # 경기 결과 
        self.result_dic = dict()
        hr = int(boxscore_home_list[12])
        ar = int(boxscore_away_list[12])
        if hr > ar:
            home_result = 'win'
            away_result = 'lose'
        elif hr < ar:
            home_result = 'lose'
            away_result = 'win'
        else:
            home_result = 'draw'
            away_result = 'draw'
            
        
        self.result_dic['Home'] = home_result
        self.result_dic['Away'] = away_result
        
        # 홈팀 / 어웨이팀 정보 및 결과 리스트 생성
        
        
        score_away_list = list()
        score_away_list.append(self.team_game_idx_dic['Away'])
        score_away_list.append(away_result)
        score_away_list.extend(boxscore_away_list)
        
        score_home_list = list()
        score_home_list.append(self.team_game_idx_dic['Home'])
        score_home_list.append(home_result)
        score_home_list.extend(boxscore_home_list)
        
        # score_array에 경기결과list 부착 
        score_array = np.vstack([score_away_list,score_home_list])
        self.score_array = score_array
        
        
        
    def craw_batter_array(self,home_away):
        '''
        Set batter_array
        
            batter_record table 데이터 가져오기
            
            columns = ['team_game_idx','bo','po','name','b1','b2','b3','hr','bb',
                       'hbp','ibb','sac','sf','so','go','fo','gidp','etc','h','tbb','ab','pa','xr']
            
        
        Parameter
        --------------------------------
        
            home_away: 홈, 원정 - 데이터가 페이지소스에 Home, Away로 구분됨
            
        '''
        
        # 기록 위치 추적
        batter_basic_find = self.soup.find('table',{'id' : 'tbl' + home_away + 'Hitter1'})
        batter_record_find = self.soup.find('div',{'id':'tbl' + home_away + 'Hitter2'})
        
        batter_num_find  = batter_basic_find.find_all('th')
        batter_name_find = batter_basic_find.find_all('td')
        batter_record_find = batter_record_find.find_all('tr')
        
                
        # 타순 리스트 생성
        num_list = list()
        count = 0
        
        for i,num in enumerate(batter_num_find):
            
            if i < 3:
                continue
            
            num = num.string
            count+=1
            if count % 2 != 0:
                num_list.append(num)
        
        len_batter = len(num_list)

        # 타자 이름 리스트 생성
        name_list = list()
        for i,name in enumerate(batter_name_find):
            if i >= len_batter:
                continue
            
            name = name.string
            name_list.append(name)
            
        # 타자 기록 리스트 생성    
        record_list = list()
        last_round = int(batter_record_find[0].find_all('th')[-1].string)
        
        
        for i,batter in enumerate(batter_record_find):
            if i == 0 or i > len_batter:
                continue
            record_by_batter=  batter.find_all('td')
            record_by_batter_list = list()
            for record in record_by_batter:
                record = str(record)
                where_td = record.find('</td')
                record_by_batter_list.append(record[4:where_td])
                
            record_by_batter_list = record_by_batter_list + ([' '] * (12-last_round))    
            record_list.append(record_by_batter_list)
            
        # 기본정보와 타격기록 취합
        
        name_array = np.array([name_list]).reshape(-1,1)
        num_array = np.array([num_list]).reshape(-1,1)
        record_array = np.array([record_list]).reshape(-1,12)
        batter_array = np.hstack([num_array,name_array,record_array])        
        
        self.batter_raw_array = batter_array
        
    def set_batter_array(self,team_game_idx):
        '''
        Set batter_array(n x 18)
        
            타자 raw-data 전처리
            
            이닝별로 기록된 타자데이터 -> 기록별 데이터 
            
        Parameter
        ---------
        
            team_game_idx: 팀 게임 index - ex) 2020시즌 LG(01) 15경기(015) = 202001015
        
        Values
        ---------
            
            batter_raw_array(n x 12): [bo, po, name, x1, x2, x3, x4, x5, x6, x7, x8, x9]
        
            batter_array(n x 23): [team_game_idx, team, bo, name, h1, h2, h3, hr, bb, hbp, ibb, sac, sf, so, go, fo, gidp, etc, h, tbb, ab, pa, xr]
            
        '''
        
        batter_array = self.batter_raw_array
        info_array = batter_array[:,:2]
        batter_record_array = batter_array[:,2:]
        
        # 기록을 저장하는 record_dic생성 및 데이터 할당(ex: {우중안:0, 우중2:1})
        # batter_records: 경기별 타자의 1~12회까지 기록 (1 x 12)
        record_dic = dict()
        old_record_array = np.zeros((1,14))
        for batter_records in batter_record_array:
            new_record_array = np.zeros(14)
            for records in batter_records:
                records = str(records)
                records = records.split('<br/>/ ')
                for record in records:
                    
                    record_num = record_dic.get(record)
                    
                    
                    if record_num != None:
                        new_record_array[record_num]+=1
                        
                    # record_dic에 값이 존재하지 않으면 add to key
                    else:
                        
                        if record[-1] == '안': record_dic[record] = 0 # 1루타: 0
                        elif record[-1] == '2': record_dic[record] = 1 # 2루타: 1
                        elif record[-1] == '3': record_dic[record] = 2 # 3루타: 2
                        elif record[-1] == '홈': record_dic[record] = 3 # 홈런: 3    
                        elif record == '4구': record_dic[record] = 4 # 볼넷: 4
                        elif record == '사구': record_dic[record] = 5 # 사구: 5
                        elif record == '고4': record_dic[record] = 6 # 고의사구: 6
                        elif record[-2:] in ['희번','희실','희선']: record_dic[record] = 7 # 희생번트: 7
                        elif record[-2:] == '희비' : record_dic[record] = 8 # 희생타: 8
                        elif record == '삼진' or record == '스낫': record_dic[record] = 9 # 삼진: 9
                        elif record[-1] in ['땅','번'] or record in ['포실','투실','1실','2실','3실','유실','야선']: record_dic[record] = 10 # 그라운드아웃: 10
                        elif record[-1] in ['파','비','직','실']: record_dic[record] = 11 # 플라이아웃: 11
                        elif record[-1] == '병' or record[-2:] == '삼중': record_dic[record] = 12 # 병살타: 12
                        elif record not in ['nan', '', ' ']: record_dic[record] = 13 # ETC: 13   
                        
                        if record not in ['nan','',' ']:
                            new_record_array[record_dic[record]] +=1
            
            old_record_array = np.vstack([old_record_array,new_record_array])
            
        
        record_array = old_record_array[1:,:] # record_array = (n x 14)
        
        # h,tbb,ab,pa 계산을 위한 new_matrix생성(14x4) 및 new_record_array생성 (n x 14) x (14 x 4) = (n x 4)
        
        # h = h1 + h2 + h3 + h4
        h_array = np.hstack([np.ones(4),np.zeros(10)])
        # tbb = bb + hbp + ibb
        tbb_array = np.hstack([np.zeros(4),np.ones(3),np.zeros(7)])
        # ab = h + so + go + fo + gidp
        ab_array = np.hstack([np.ones(4),np.zeros(5),np.ones(4),np.zeros(1)])
        # pa = ab + tbb + sf
        pa_array = ab_array + tbb_array
        pa_array[8] = 1
        
        new_matrix_array = np.transpose(np.vstack([h_array,tbb_array,ab_array,pa_array]))
        
        new_record_array = record_array.dot(new_matrix_array) # new_record_array (n x 4)
        
        total_record_array = np.hstack([record_array,new_record_array]) # total_record_array (n x 18)
        
        # GO값 보정: go+= sac + gidp
        total_record_array[:,10] += total_record_array[:,7] + total_record_array[:,12]
        
        # FO값 보정: fo+= sf
        total_record_array[:,11] += total_record_array[:,8]
        
        # XR생성
        
        xr_factor_array = np.array([0.5, 0.72, 1.04, 1.44, 0.34, 0.34, -0.09, 0.04, 0.37, -0.008, 0, 0, -0.37, 0, 0.09, 0, -0.09, 0])
        xr_array = np.round(np.dot(total_record_array,xr_factor_array).reshape(-1,1),3)
        
        po_list = list()
        last_bo = 0
        po = 1
        for batter in batter_array:
            bo = int(batter[0])
            
            if last_bo == bo:
                po +=1
                
            else:
                po = 1
            
            last_bo = bo
            po_list.append(po)
        
        po_array = np.array(po_list).reshape(-1,1)
        info_array = np.hstack([po_array,info_array])
        info_array[:,(0,1)] = info_array[:,(1,0)] # info_array(n x 3)
        
        
        len_batter_array = len(po_list)
        team_game_idx_array = np.full((len_batter_array,1),team_game_idx)
        
        batter_array = np.hstack([team_game_idx_array, info_array,total_record_array,xr_array])            
        self.batter_array = batter_array       
        
        
    def craw_pitcher_array(self, home_away):
        '''
        Set pitcher_array
        
            pitcher_record table 데이터 가져오기
            
            columns = ['team_game_idx','name', 'po', 'inn', 'tbf', 'np', 'ab', 'h', 'hr', 'tbb', 'so', 'r','er', 'fip']
            
        Parameter
        --------------------------------
        
            home_away: 홈, 원정 - 데이터가 페이지소스에 Home, Away로 구분됨
            
        '''
        
        pitcher_find = self.soup.find('table',{'id' : 'tbl' + home_away + 'Pitcher'}).find_all('tr')

        record_list = list()
        len_pitcher = len(pitcher_find)
        count = 0
        record_by_pitcher_list = list()
        for i,pitcher in enumerate(pitcher_find):
            if i == 0 or i == (len_pitcher-1):
                continue
            pitcher_record_find = pitcher.find_all('td')
            for record in pitcher_record_find:
                record = record.string
                record_by_pitcher_list.append(record)
                count +=1
                if count == 17: # 16가지 투수 기록 다 가져오면 record_list에 투수 기록 append
                    record_list.append(record_by_pitcher_list)
                    record_by_pitcher_list = list()
                    count = 0
        
        len_record = len_pitcher-2
        

        pitcher_array = np.array([record_list]).reshape(len_record,17)
        
        self.pitcher_raw_array = pitcher_array
    
    def set_pitcher_array(self,team_game_idx):
        '''
        Set pitcher_array
        
            투수 raw-data 전처리    
            
            이닝별로 기록된 타자데이터 -> 기록별 데이터 
            
        Parameter
        ----------
        
            team_game_idx: 팀 게임 index - ex) 2020시즌 LG(01) 15경기(015) = 202001015
            


        Values
        -------
        
            pitcher_raw_array(n x 19) : ['name','position','result','win','lose','save',ip','tbf','np','ab','h','hr','tbb','so','r','er','era']
            
            pitcher_array(n x 14) : ['team_game_idx','team','name','position','ip','tbf','np','ab','h','hr','tbb','so','r','er']
            
            
        '''
        
        pitcher_array = self.pitcher_raw_array
        info_array = pitcher_array[:,:2]
        inn_array = pitcher_array[:,6]
        record_array = pitcher_array[:,7:-1].astype(np.float)
        
        
        # inn_array 소수점으로 변경
        new_list = list()
        for inn in inn_array:
            if len(inn)==1:
                inn = int(inn)
            elif len(inn)==3:
                inn = int(inn[0]) * 0.333
            elif len(inn):
                inn = int(inn[0]) + int(inn[2])*0.333
            new_list.append(inn)
        inn_array = np.array(new_list).reshape(-1,1)
        
        # SP, RP 구분
        position_array = info_array[:,1]
        po_list = list()
        for po,position in enumerate(position_array):
            po_list.append(po+1)
            
        
        info_array[:,1] = np.array(po_list)
        
        '''
        # csv파일의 IP column의 1/3, 2/3을 날짜로 인식해 숫자로 변경
        # DB에 저장하기전 csv에 저장할 때 사용
        
        inn_array = pitcher_record_array[:,0]
        inn_array = np.where(inn_array == '43833',str(1/3),inn_array)
        inn_array = np.where(inn_array == '43864',str(2/3),inn_array)
        pitcher_record_array[:,0] = inn_array
        '''
        
        
        # fip = (((13 * HR) + (3 * TBB) - (2 * SO)) / IP) + 3.2
        # 나중에 IP로 나누고 3.2더하는 작업필요
        fip = (record_array[:,4]*13 + record_array[:,5]*3 - record_array[:,6]*2)
        
        fip_array = fip.reshape(-1,1)
        
        len_pitcher_array = len(inn_array)
        team_game_idx_array = np.full((len_pitcher_array,1),team_game_idx)
        
        pitcher_array = np.hstack([team_game_idx_array, info_array, inn_array, record_array,fip_array])
        
        self.pitcher_array = pitcher_array
    
    def craw_lineup(self,view, home_away):
    
        lineup_array = np.zeros((1,4))
        team_game_idx = self.team_game_idx_dic[home_away]
        
        home_away_lower = home_away.lower()
        pitcher_find = view.find('div', {'class':'team ' + home_away_lower}).get_text().strip()
        
        batter_find = self.soup.find('table',{'id':'tbl' + home_away + 'LineUp'}).find_all('td')[::-1]
        
        
        for i in range(10):
            lineup_list = list()
            lineup_list.append(team_game_idx)
            
            if i == 0:
                lineup_list.extend(['0','선발투수',pitcher_find])
            else:
                for j in range(4):
                    batter = batter_find.pop()
                    if j !=3:
                        lineup_list.extend(batter)
                        
                              

            lineup_array =np.vstack([lineup_array,lineup_list])
        
        
        self.lineup_array = lineup_array[1:]
    
    
    def end_game_crawling(self, conn):
        '''
        끝난 경기 데이터 크롤링
        '''
        self.set_craw_time(1)
        year = self.date//10000
        
        conn = conn
        conn.begin()
    
    
        self.ready_by_round()
        self.set_last_game_num_list(year,conn)
    
        #self.delete_table_data(conn,'today_team_game_info')
        #self.delete_table_data(conn,'today_game_info')
        
        
        try:
            
            for j, element in enumerate(self.elements):
                
                # last_game_num list 생성
                
                view = self.view_list[j]
                
                # game_info 크롤링 및 저장
                self.craw_game_info(view)
                self.craw_team_game_info()
                
                if self.end == '경기종료':
                    self.array_to_db(conn, self.game_info_array, 'game_info')   
                    self.array_to_db(conn, self.team_game_info_array,'team_game_info')
                    
                    self.ready_by_match(j,element,True) # 매치별 기본정보 구하기
                    
                    # score_record 크롤링 및 저장
                    
                    self.craw_score_array()
                    self.array_to_db(conn, self.score_array,'score_record')
                    for home_away in ['Home', 'Away']:
                        
                        team_game_idx = self.team_game_idx_dic[home_away]
                        
                        # batter_record 크롤링 및 저장
                        self.craw_batter_array(home_away)
                        self.set_batter_array(team_game_idx)
                        self.array_to_db(conn,self.batter_array,'batter_record')
                        
                        # pitcher_record 크롤링 및 저장
                        self.craw_pitcher_array(home_away)
                        self.set_pitcher_array(team_game_idx) 
                        self.array_to_db(conn, self.pitcher_array,'pitcher_record')
                        
                        # 게임번호 +1
                        team_num = self.home_away_num_dic[home_away]
                        self.last_game_num_list[team_num]+=1

                
                else: pass
                 
            self.update_total_game_num(conn, year, self.last_game_num_list)
            conn.commit()
            conn.close()
            
        except:
            conn.rollback()
            conn.close()
            print('error')


    def start_game_crawling(self, conn):
        
        self.set_craw_time(1)
        year = self.date//10000
        conn = conn 
        conn.begin()
        
        
        self.ready_by_round()
        self.set_last_game_num_list(year,conn)
        
        self.delete_table_data(conn,'today_lineup')
        self.delete_table_data(conn,'today_team_game_info')
        self.delete_table_data(conn,'today_game_info')
        '''
        try:
            
            for j, element in enumerate(self.elements):
                
                # last_game_num list 생성
                
                view = self.view_list[j]
                
                # game_info 크롤링 및 저장
                self.craw_game_info(view)
                self.craw_team_game_info()
                
                self.array_to_db(conn, self.game_info_array, 'today_game_info')
                self.array_to_db(conn, self.team_game_info_array,'today_team_game_info')
                
                self.ready_by_match(j,element,False)
                
                for home_away in ['Home', 'Away']:
                        
                    self.craw_lineup(view,home_away)
                    self.array_to_db(conn, self.lineup_array, 'lineup_info')
                
            conn.commit()
            conn.close()
        except:
            conn.rollback()
            conn.close()
            print('error')
        '''
        for j, element in enumerate(self.elements):
                
            # last_game_num list 생성
            
            view = self.view_list[j]
            
            # game_info 크롤링 및 저장
            self.craw_game_info(view)
            self.craw_team_game_info()
            
            
            
            if self.end == '경기종료': continue
                
            elif '취소' in self.end: 
                self.array_to_db(conn, self.game_info_array, 'today_game_info')
                self.array_to_db(conn, self.team_game_info_array,'today_team_game_info')
                for home_away in ['Home', 'Away']:
                    team_num = self.home_away_num_dic[home_away]
                    self.last_game_num_list[team_num]+=1
                
                
            else:
                self.array_to_db(conn, self.game_info_array, 'today_game_info')
                self.array_to_db(conn, self.team_game_info_array,'today_team_game_info')
                
                
                self.ready_by_match(j,element,False)    
                
                for home_away in ['Home', 'Away']:
                        
                    self.craw_lineup(view,home_away)
                    self.array_to_db(conn, self.lineup_array, 'today_lineup')

                    team_num = self.home_away_num_dic[home_away]
                    self.last_game_num_list[team_num]+=1
                
                    
        conn.commit()
        conn.close()
#%%

