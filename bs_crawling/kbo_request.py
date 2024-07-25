#%%
import sys
sys.path.extend(['D:\BaseballProject\\python', 'C:\\Users\\82109\\Desktop\\LYC\\git\\baseball_analysis'])
import requests
from bs4 import BeautifulSoup 
import json
import time
import pandas as pd
import numpy as np
from bs_crawling import base as cb
from datetime import datetime, timedelta

import pymysql
from bs_personal import personal_code as cd
#%%

class Crawling_kbo_request(cb.Crawling):
    '''
    Class Crawling
    
        KBO에서 제공하는 경기 기록 크롤링
        
        request 버전
        
    '''
    
    
    def __init__(self):
        
        
        
        self.year = 2024
        # DB에 넣을 table array생성
        
        
        self.game_info_array = np.zeros((0,6))
        self.team_game_info_array = np.zeros((0,7))
        self.score_array = np.zeros((0,18))
        self.batter_array = np.zeros((0,23))
        self.pitcher_array = np.zeros((0,14))
        
        
        self.last_game_num_dic = dict()
        self.last_game_num_list = [0 for i in range(11)]
        
        
        self.lineup_array = np.zeros((0,5))
        # 기타 쓰이는 변수
        self.date = None
        self.is_start = True # 현재 페이지 시작여부
        self.end = None # 경기종료결과(경기종료, 우천취소 등)
        
        self.team_game_idx_dic = dict() # team_game_idx 생성 후 Home, Away구분해 저장하는 dictionary(데이터 분석용)
        
        self.game_dic = dict()
        self.game_id_dic = dict()
        
        self.lineup_dic = dict()
        
        self.box_score_dic = dict()
        self.score_board_dic = dict()
        
        
        self.game_info_dic = dict()
        self.team_game_info_dic = dict()
        self.request_headers = {'referer': f'https://www.koreabaseball.com/Schedule/GameCenter/Main.aspx?gameDate={self.date}'
            
            }
    def craw_game_info(self, date: int) -> None:
        

        match_params = {'leId': 1, 'srId': [0,1,3,4,5,7,9], 'date': date}
        response = requests.post('https://www.koreabaseball.com/ws/Main.asmx/GetKboGameList', headers = self.request_headers, data = match_params)
        match_dic = response.json()
        game_list = match_dic['game']
        self.game_dic[date] = game_list
        self.game_id_dic[date] = [game["G_ID"] for game in game_list if (game['GAME_SC_NM']=='정규경기')&(game['CANCEL_SC_NM']=='정상경기')]
        
         
    def craw_box_score(self, date: int) -> None:
        
        
        
        game_id_list= self.game_id_dic[date]
        year = self.year
        
        box_score_list = list()
        for i, game_id in enumerate(game_id_list):
            
            review_params = {'leId': 1, 'srId': 0, 'seasonId': year, 'gameId': game_id}
            response = requests.post('https://www.koreabaseball.com/ws/Schedule.asmx/GetBoxScoreScroll', headers = self.request_headers, data = review_params)
            data_dic = response.json()
            # soup = BeautifulSoup(response.text,'html.parser')  
            # data_dic = json.loads(soup.text)
            box_score_list.append(data_dic)
        self.box_score_dic[date] = box_score_list
            
        
    
    def craw_score_board(self, date: int) -> None:
        game_id_list = self.game_id_dic[date]
        year = self.year
        
        score_board_list = list()
        for game_id in game_id_list:
            review_params = {'leId': 1, 'srId': 0, 'seasonId': year, 'gameId': game_id}
            response = requests.post('https://www.koreabaseball.com/ws/Schedule.asmx/GetScoreBoardScroll', headers = self.request_headers, data = review_params)
            score_dic = response.json()
            # soup = BeautifulSoup(response.text,'html.parser')
            # score_dic = json.loads(soup.text)
            score_board_list.append(score_dic)
        self.score_board_dic[date] = score_board_list
    
    
    def craw_lineup(self, date: int) -> None:
        
        game_id_list = self.game_id_dic[date]
        year = self.year
        
        lineup_list = list()
        for i, game_id in enumerate(game_id_list):
            
            review_params = {'leId': 1, 'srId': 0, 'seasonId': year, 'gameId': game_id}
            response = requests.post('https://www.koreabaseball.com/ws/Schedule.asmx/GetLineUpAnalysis', headers = self.request_headers, data = review_params)
            data_dic = response.json()
            # soup = BeautifulSoup(response.text,'html.parser')  
            # data_dic = json.loads(soup.text)
            lineup_list.append(data_dic)
        self.lineup_dic[date] = lineup_list
        
    
    
    def pre_game_info(self, game_dic: dict) -> list: 
        
        
        game_type = game_dic['GAME_SC_NM']
        game_result_type = game_dic['CANCEL_SC_NM']
        date_str = game_dic['G_DT']
        home_name = game_dic['HOME_NM']
        away_name = game_dic['AWAY_NM']
        
        home_num_str = '%02d' % self.team_dic[home_name]
        away_num_str = '%02d' % self.team_dic[away_name]
        
        today_game_num = '0' + str(game_dic['HEADER_NO'])
        game_idx = date_str + home_num_str + away_num_str + today_game_num
        
        stadium = game_dic['S_NM']
        
        game_info_list = [game_idx,home_name,away_name,stadium,game_result_type,game_type]
        return game_info_list

    def pre_team_game_info(self, game_info: list) -> np.array:
        '''
        Set team_game_info_array
        
            team_game_info table 데이터 전처리
            
            columns = ['game_idx', 'team_game_idx', 'year', 'team_num', 'foe_num', 'game_num', 'home_away']
            
        '''
        last_game_num_list = self.last_game_num_list 
        year_str = str(self.year)
        game_idx = game_info[0]
        
        home_name = game_info[1]
        away_name = game_info[2]
        
        
        
        home_num = self.team_dic[home_name]
        away_num = self.team_dic[away_name]
        
        
        # home_away_game_idx 생성 및 dictionary 저장
        last_home_game_num = last_game_num_list[home_num]
        last_away_game_num = last_game_num_list[away_num]
        
        home_game_num = last_home_game_num + 1
        away_game_num = last_away_game_num + 1
        
        home_game_idx = year_str + '%02d' % home_num + '%03d' % (home_game_num)
        away_game_idx = year_str + '%02d' % away_num + '%03d' % (away_game_num)
        
        
        home_game_info_array = np.array([game_idx, home_game_idx, int(year_str), home_num, away_num, home_game_num,'home'])
        away_game_info_array = np.array([game_idx, away_game_idx, int(year_str), away_num, home_num, away_game_num,'away'])
        team_game_info_array = np.vstack([home_game_info_array, away_game_info_array])
        
        self.last_game_num_list[home_num]+=1
        self.last_game_num_list[away_num]+=1
        return team_game_info_array
    
    def pre_batter_array(self, team_game_idx, batter_box_1, batter_box_2):
        old_record_array = np.zeros((0,14))
        old_info_list = list()
        for h1, h2 in zip(batter_box_1,batter_box_2):
            
            info_list = h1['row']
            new_info_list = list()
            for i,info in enumerate(info_list):
                if i != 1:
                    new_info_list.append(info['Text'])
            old_info_list.append(new_info_list)
            
            batter_record_list = h2['row']
            new_record_array = np.zeros(14)
            for records in batter_record_list:
                
                
                
                record = records['Text']
                
                
                    
                if record[-1] == '안': new_record_array[0]+=1 # 1루타: 0
                elif record[-1] == '2': new_record_array[1]+=1 # 2루타: 1
                elif record[-1] == '3': new_record_array[2]+=1 # 3루타: 2
                elif record[-1] == '홈': new_record_array[3]+=1 # 홈런: 3    
                elif record == '4구': new_record_array[4]+=1 # 볼넷: 4
                elif record == '사구': new_record_array[5]+=1 # 사구: 5
                elif record == '고4': new_record_array[6]+=1 # 고의사구: 6
                elif record[-2:] in ['희번','희실','희선']: new_record_array[7]+=1 # 희생번트: 7
                elif record[-2:] == '희비' : new_record_array[8]+=1 # 희생타: 8
                elif record == '삼진' or record == '스낫': new_record_array[9]+=1 # 삼진: 9
                elif record[-1] in ['땅','번'] or record in ['포실','투실','1실','2실','3실','유실','야선']: new_record_array[10]+=1 # 그라운드아웃: 10
                elif record[-1] in ['파','비','직','실']: new_record_array[11]+=1 # 플라이아웃: 11
                elif record[-1] == '병' or record[-2:] == '삼중': new_record_array[12]+=1 # 병살타: 12
                elif record not in ['nan', '', ' ','\xa0']: new_record_array[13]+=1 # ETC: 13   
                    
            old_record_array = np.vstack([old_record_array,new_record_array])
        
        info_array = np.array(old_info_list)
        record_array = old_record_array # record_array = (n x 14)
            
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
        for batter in info_array:
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
        return batter_array
        #self.team_game_info_array = team_game_info_array
        
    def pre_pitcher_array(self, team_game_idx, pitcher_box):
        old_pitcher_list = list()
        
        for p1 in pitcher_box:
            new_pitcher_list = list()
            for pitchers in p1['row']:
                
                
                new_pitcher_list.append(pitchers['Text'])
            old_pitcher_list.append(new_pitcher_list)            
        
        pitcher_array = np.array(old_pitcher_list)
        
        info_array = pitcher_array[:,:2]
        inn_array = pitcher_array[:,6]
        record_array = pitcher_array[:,7:-1].astype(np.float64)
        
        
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
        
        
        
        
        # fip = (((13 * HR) + (3 * TBB) - (2 * SO)) / IP) + 3.2
        # 나중에 IP로 나누고 3.2더하는 작업필요
        fip = (record_array[:,4]*13 + record_array[:,5]*3 - record_array[:,6]*2)
        
        fip_array = fip.reshape(-1,1)
        
        len_pitcher_array = len(inn_array)
        team_game_idx_array = np.full((len_pitcher_array,1),team_game_idx)
        
        pitcher_array = np.hstack([team_game_idx_array, info_array, inn_array, record_array,fip_array])

        return pitcher_array
    
    def pre_score_board(self, team_game_idx_list, score_box_1, score_box_2, score_box_3):
        
        old_score_total_list = list()
        for i, (s1, s2, s3) in enumerate(zip(score_box_1,score_box_2, score_box_3)):
            
            result = s1['row'][0]['Text']
            
            if result == '승': result = ['win']
            elif result == '패': result = ['lose']
            elif result == '무': result = ['draw']
            else: result = ["etc"]
                
            
            new_score_list = list()
            for score in s2['row']:
                new_score_list.append(score['Text'])
            
            new_score_result_list = list()
            for score_result in s3['row']:
                new_score_result_list.append(score_result['Text'])
            
            team_game_idx= team_game_idx_list[i]
            
            new_score_total_list = [team_game_idx]
            new_score_total_list.extend(result)
            new_score_total_list.extend(new_score_list)
            new_score_total_list.extend(new_score_result_list)
            
            old_score_total_list.append(new_score_total_list)
        
        score_array = np.array(old_score_total_list)
        return score_array 
    
    def pre_lineup(self, lineup_list, team_game_info_array, game_list) -> None:
        
        lineup_columns = ['team_game_idx','bo','po','name','xr']
        lineup_data = pd.DataFrame()
        for i, lineup in enumerate(lineup_list):
            game_info = game_list[i]
            
            for home_away_idx in range(2):
                team_lineup_list = json.loads(lineup[home_away_idx+3][0])['rows']
                team_game_idx = team_game_info_array[i*2 + home_away_idx,1]
                
                
                pitcher_name = game_info['B_PIT_P_NM'] if home_away_idx == 0 else game_info['T_PIT_P_NM']
                pitcher_name = pitcher_name.strip()
                old_lineup_data = pd.DataFrame([0,'선발투수', pitcher_name,0]).transpose()
                
                for team in team_lineup_list:
                    new_lineup_data = pd.DataFrame(pd.DataFrame(team['row'])['Text']).transpose()
                    old_lineup_data = pd.concat([old_lineup_data, new_lineup_data])
                
                
                old_lineup_data = old_lineup_data.reset_index(drop = True)
                
                old_lineup_data.columns = lineup_columns[1:]
                old_lineup_data['team_game_idx'] = team_game_idx
                
                
                lineup_data = pd.concat([lineup_data,old_lineup_data])
                
        lineup_data = lineup_data[lineup_columns]
        return lineup_data
    
    def get_game_info_array(self, date) -> np.array:
        
        game_list = self.game_dic[date]
        old_game_info_list = list()
        for game_dic in game_list:
            new_game_info = self.pre_game_info(game_dic)
            if (new_game_info[4] != '정상경기') | (new_game_info[5] !="정규경기"):
                continue
            old_game_info_list.append(new_game_info)
            
        return np.array(old_game_info_list)
        
        
        
    def get_team_game_info_array(self,date) -> np.array:
        
        
        game_info_array = self.game_info_dic[date]
        old_team_game_info_array = np.zeros((0,self.team_game_info_array.shape[1]))
        for game_info in game_info_array:
            
            new_team_game_info_array = self.pre_team_game_info(game_info)
            old_team_game_info_array = np.vstack([old_team_game_info_array, new_team_game_info_array])
        return old_team_game_info_array
        
    def get_batter_array(self, date) -> np.array:
        box_score_list = self.box_score_dic[date]
        game_info_array = self.game_info_dic[date]
        team_game_info_array = self.team_game_info_dic[date]
        
        old_batter_array = np.zeros((0,self.batter_array.shape[1]))
        for i, box_score_dic in enumerate(box_score_list):
            
            arrHitter_list = box_score_dic['arrHitter']
            for home_away_idx in range(2):
                batter_box_1 = json.loads(arrHitter_list[home_away_idx]['table1'])['rows']
                batter_box_2 = json.loads(arrHitter_list[home_away_idx]['table2'])['rows']
                
                team_game_idx = team_game_info_array[i*2 + 1-home_away_idx,1]
                new_batter_array = self.pre_batter_array(team_game_idx,batter_box_1, batter_box_2)
                old_batter_array = np.vstack([old_batter_array,new_batter_array])
                
        return old_batter_array
        
    def get_pitcher_array(self,date) -> np.array:
        
        box_score_list = self.box_score_dic[date]
        game_info_array = self.game_info_dic[date]
        team_game_info_array = self.team_game_info_dic[date]
        
        old_pitcher_array = np.zeros((0,self.pitcher_array.shape[1]))
        
        for i, box_score_dic in enumerate(box_score_list):
            
            arrPitcher_list = box_score_dic['arrPitcher']
            for home_away_idx in range(2):

                pitcher_box = json.loads(arrPitcher_list[home_away_idx]['table'])['rows']
                team_game_idx = team_game_info_array[i*2 + 1-home_away_idx,1]
                new_pitcher_array = self.pre_pitcher_array(team_game_idx, pitcher_box)
                old_pitcher_array = np.vstack([old_pitcher_array, new_pitcher_array])
                
        return old_pitcher_array
    def get_score_array(self, date) -> np.array:
        
        score_board_list = self.score_board_dic[date]
        game_info_array = self.game_info_dic[date]
        team_game_info_array = self.team_game_info_dic[date]
        
        old_score_array = np.zeros((0,self.score_array.shape[1]))
        for i, score_board_dic in enumerate(score_board_list):
            
            score_box_1 = json.loads(score_board_dic['table1'])['rows']
            score_box_2 = json.loads(score_board_dic['table2'])['rows']
            score_box_3 = json.loads(score_board_dic['table3'])['rows']
            
            team_game_idx_list = list(team_game_info_array[i*2 : (i+1)*2 ,1].reshape(-1))[::-1]
            new_score_array = self.pre_score_board(team_game_idx_list, score_box_1, score_box_2, score_box_3)
            old_score_array = np.vstack([old_score_array, new_score_array])
        return old_score_array
   
    def get_lineup_array(self, date):
        
        game_list = self.game_dic[date]
        team_game_info_array = self.team_game_info_array
        lineup_list = self.lineup_dic[date]
        lineup_data = self.pre_lineup(lineup_list,team_game_info_array, game_list)
        
        lineup_array = np.array(lineup_data)
        return lineup_array
    
    def set_date_total(self,date) -> None:
        
        try:
            game_info_array = self.get_game_info_array(date)
        except:
            return
        if game_info_array.size ==0:
            return
        
        self.game_info_dic[date] = game_info_array
        self.game_info_array = np.vstack([self.game_info_array, game_info_array])
         
        team_game_info_array = self.get_team_game_info_array(date)
        self.team_game_info_dic[date] = team_game_info_array
        self.team_game_info_array = np.vstack([self.team_game_info_array, team_game_info_array])
        
        batter_array = self.get_batter_array(date)
        self.batter_array = np.vstack([self.batter_array, batter_array])
        
        pitcher_array = self.get_pitcher_array(date)
        self.pitcher_array = np.vstack([self.pitcher_array, pitcher_array])
        
        score_array = self.get_score_array(date)
        self.score_array = np.vstack([self.score_array, score_array])

    def set_date_start(self, date) -> None:
        try:
            game_info_array = self.get_game_info_array(date)
        except:
            return
        if game_info_array.size ==0:
            return
        
        self.game_info_dic[date] = game_info_array
        self.game_info_array = np.vstack([self.game_info_array, game_info_array])
         
        team_game_info_array = self.get_team_game_info_array(date)
        self.team_game_info_dic[date] = team_game_info_array
        self.team_game_info_array = np.vstack([self.team_game_info_array, team_game_info_array])
        
        lineup_array = self.get_lineup_array(date)
        self.lineup_array = lineup_array
        


