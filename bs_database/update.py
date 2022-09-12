#%%

# 불러오기 위치 설정
import sys
sys.path.append('D:\\BaseballProject\\python')

import numpy as np
from bs_database import base as db

#%%

class Update(db.Database):
    '''
    
        크롤링 후 추가적인 계산과정이 필요한 DB에 업데이트 후 업로드
    '''
    
    def __init__(self):
        return
    
    def get_recent_data(self, year, old_graph_array):
        '''
        최근 5, 20 경기 득점, 불펜실점 구하기
        run_graph_data table에 업로드 
        
        '''
        game_info_array = self.game_info_array
        score_array = self.score_array
        pitcher_array = self.pitcher_array
        park_factor_total = self.park_factor_total
        
        last_game_num_list = self.last_game_num_list
        
        old_array = np.zeros((1,9))
        game_info_array = game_info_array[(game_info_array[:,2] == 2022),:]
        
        for i,team_game_info in enumerate(game_info_array):
            team_game_idx = team_game_info[1]
            year = team_game_idx[:4]
            team_num = team_game_idx[4:6]
            game_num = team_game_idx[6:]
            
            stadium = team_game_info[-1]
            park_factor = park_factor_total.get(stadium)
            if park_factor == None: park_factor = 1
            
            # run-graph 데이터 생성
            
            range_score_array = score_array[score_array[:,1]==team_game_idx,:][0]
            
            game_inn = 12 - list(range_score_array[-16:-4]).count('-')
            run = int(range_score_array[-4]) * 9 / game_inn
            run = run / park_factor
            
            range_pitcher_array = pitcher_array[pitcher_array[:,1]==team_game_idx,:]
            rp_array = range_pitcher_array[1:]
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
        
        graph_array =  np.zeros((1,9))
        
        for team_num in range(1,11):
            last_game_num = last_game_num_list[team_num]
            old_team_array = old_graph_array[(old_graph_array[:,1]=='2022')&(old_graph_array[:,2]==str(team_num)),:]
            if len(old_team_array) == 0:
                graph_last_game_num = 0 
            else:
                graph_last_game_num = int(old_team_array[-1,3])
    
            for game_num in range(graph_last_game_num+1,last_game_num+1):
                new_array = old_array[(old_array[:,2] == str(team_num).zfill(2))&(old_array[:,1]=='2022')&(old_array[:,3]==str(game_num).zfill(3)),:]
                graph_array = np.vstack([graph_array,new_array])
        
        graph_array = graph_array[1:]
        return graph_array

    def get_new_win_rate(self, game_info_array, score_array):
        '''
            팀 별 승률 계산
            team_info 테이블에 업로드
        
        '''
        
        
        old_array = np.zeros((1,6))
        for year in range(2022,2023):
            for team_num in range(1,11):
                
                
                team_array = score_array[(score_array[:,2]== year) & (score_array[:,3] == team_num),:]
                team_score_list = list(team_array[:,-4])
        
                game_idx = team_array[:,0]
                foe_score_list = list()
                for gi in game_idx:
                    
                    foe_array = score_array[(score_array[:,0] == gi) & (score_array[:,2]== year) & (score_array[:,3] != team_num),:]
                    foe_score = foe_array[0,-4]
                    foe_score_list.append(foe_score)
                
                
                tsl = team_score_list[::-1]
                fsl = foe_score_list[::-1]
                win = 0
                draw = 0 
                lose = 0
                length = len(tsl)
                print(length)
                for i in range(length):
                    ts = tsl.pop()
                    fs = fsl.pop()
                    if ts>fs:
                        win+=1
                    elif ts==fs:
                        draw+=1
                    else:
                        lose+=1
                win_rate = round(win / (win+lose),3)
                new_list = [win,lose,draw,win_rate, year,team_num]
                
                old_array = np.vstack([old_array,new_list])
        
        record_array = old_array[1:].astype(str)
        record_list = list()
        for record in record_array:
            record_list.append(list(record))
            
        return record_list

