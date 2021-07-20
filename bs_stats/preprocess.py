#%%

# 불러오기 위치 설정
import sys
sys.path.append('D:\\BaseballProject\\python')

import numpy as np
import pandas as pd
from bs_database import base as bs

#%%
class Preprocess(bs.Database):
    '''
    분석을 위한 데이터 전처리 class
    
    
    
    '''
    
    def __init__(self):
        
        #분석에 사용할 dictionary
        self.game_info_dic = dict()
        self.info_dic = dict()
        self.max_game_dic= dict()
        
        self.batter_dic = dict()
        self.pitcher_dic = dict()
        self.score_dic = dict()
        
        self.iv_batter_dic = dict()
        self.iv_pitcher_dic = dict()
        
        self.lineup_dic = dict()
        self.batter_game_num_dic = dict()
        self.pitcher_game_num_dic = dict()
        
        self.xr_dic = dict()
        self.sp_dic = dict()
        self.rp_dic = dict()
        
        
        #분석에 사용할 variable
        self.year = 2017 # year: 년도
        self.team_num = 1 # team_num: 팀 번호
        self.br_range = 18 # game_num_range_batter : 타자 경기 범위
        self.sp_range = 7 # game_num_range_pitcher : 선발투수 경기 범위
        self.rp_range = 30 # 계투 경기 범위
        
        self.start_fip = 5 # 투수 init fip
        self.start_inn = 4 # 투수 init inn
        self.start_pa = 4.43
        
        self.least_inn = 10
        self.least_pa = 10
        
        self.is_pa = True
        self.is_park_factor = True
        self.is_new_game = False
        
    def set_basic_params(self,**kwargs):
        
       
        if kwargs.get('br_range')!= None:
            self.br_range = kwargs['br_range']
            
        if kwargs.get('sp_range')!= None:
            self.sp_range = kwargs['sp_range']
            
        if kwargs.get('rp_range')!=None:
            self.rp_range = kwargs['rp_range']    
            
        if kwargs.get('start_fip')!= None:
            self.start_fip = kwargs['start_fip']
            
        if kwargs.get('start_pa')!= None:
            self.start_inn = kwargs['start_inn']
        
        if kwargs.get('start_inn')!= None:
            self.start_pa = kwargs['start_pa']
        
        if kwargs.get('least_pa')!= None:
            self.least_pa = kwargs['least_pa']
        
        if kwargs.get('least_inn')!= None:
            self.least_inn = kwargs['least_inn']
            
    def set_dic_all(self):
        '''
        Set game_info_dic / batter_dic / pitcher_dic / score_dic / max_game_dic
        
            년도와 팀에 따른 분석용 데이터를 빠르게 불러오기 위해 dictionary 생성 후 저
        

        game_info_dic[year][team] (n x 7) : 기본정보 
        ------------------------------------    
            date:0, game_idx:1, year:2, team_num:3, foe_num:4, game_num:5, home&away:6, stadium: 7

        
        batter_dic[year][team] (n x 30) : 타자정보
        ----------------------------------    
            date:0, game_idx:1, year:2, team_num:3, foe_num:4, game_num:5, home&away:6, stadium: 7 
            
            bo:8, po:9, name:10, h1:11, h2:12, h3:13, hr:14, bb:15, hbp:16, ibb:17, sac:18
            
            sf:19, so:20, go:21, fo:22, gidp:23, etc:24, h:25, tbb:26, ab:27, pa:28, xr:29
        
        
        pitcher_dic[year][team] (n x 21) : 투수정보
        ----------------------------------
            date:0, game_idx:1, year:2, team_num:3, foe_num:4, game_num:5, home&away:6, stadium: 7
            
            name:8, po:9, ip:10, tbf:11, np:12, ab:13, h:14, hr:15, tbb:16, so:17, r:18, er:19, fip:20
            
            
        score_dic[year][team] (n x 25) : 스코어보드
        --------------------------------
        
            date:0, game_idx:1, year:2, team_num:3, foe_num:4, game_num:5, home&away:6, stadium: 7
            
            result:8, x1:9, x2:10, x3:11, x4:12, x5:13, x6:14, x7:15, x8:16, x9:17
            
            x10:18, x11:19, x12:20, R:21, H:22, E:23, B:24
            
            
        max_num[year][team] (float) : 각 팀별 경기 수(껍데기 만들때 필요)
        -----------------------------
        '''        
        
        
        self.set_game_info_dic()
        self.batter_dic = self.div_by_year_team(self.batter_array)
        self.pitcher_dic = self.div_by_year_team(self.pitcher_array)
        self.score_dic = self.div_by_year_team(self.score_array)
        self.set_iv()
        self.set_game_num_dic()
        self.set_lineup_dic()
        
    def set_game_info_dic(self):
        '''
        게임기초정보 dictionary 만들기
        
        '''
        game_info_dic = dict()
        max_game_dic = dict()
        
        #game_info_array의 game_idx를 활용해 date생성 및 취합
        game_idx_array = self.game_info_array[:,0]
        date_df = pd.DataFrame(game_idx_array)
        date_df.columns = ['game_idx']
        date_array = np.array(date_df.game_idx.str[:8]).reshape(-1,1)
        self.game_info_array = np.hstack([date_array,self.game_info_array[:,:]])
        
        score_df =pd.DataFrame(self.score_array)
        score_df.columns = self.team_game_info_columns[1:] + self.score_columns
        
        for year in self.year_list:
            year_array = self.game_info_array[(self.game_info_array[:,3])==year,:]
            game_info_list = [0]
            max_game_list = [0]
            for team_num in range(1,11):
                team_array = year_array[(year_array[:,4]==team_num),:]
                team_df = pd.DataFrame(team_array)
                team_df.columns = self.team_game_info_columns
                inner_array = np.array(pd.merge(team_df , score_df, on = 'game_idx',how = 'left'))
                
                count=0
                win_list = list()
                for inner in inner_array:
                    if inner[11]==team_num:
                        tr = int(inner[-4])
                        
                    else:
                        fr = int(inner[-4])
                    count +=1
                    if count ==2:
                        if tr > fr:
                            win_list.append(1)
                        elif tr < fr:
                            win_list.append(0)
                        else:
                            win_list.append(0.5)
                        count = 0
                win_array = np.array(win_list).reshape(-1,1)
                #print(year,team_num,len(win_list),sum(win_list))
                new_team_array = np.hstack([team_array,win_array])
                game_info_list.append(new_team_array)
                max_game_list.append(len(new_team_array))
            game_info_dic[year] = game_info_list
            max_game_dic[year] = max_game_list
            
        self.game_info_dic = game_info_dic
        self.max_game_dic = max_game_dic
        
    def div_by_year_team(self, data_array):
        '''
        주어진 array를 년도와 팀별 dictionary로 바꾸는 함수
        '''
        data_dic = dict()
        
        
        for year in self.year_list:
            data_list = [0]
            
            
            for team_num in range(1,11):
                new_data_array = data_array[(data_array[:,2] == year) & (data_array[:,3]==team_num)]
                data_list.append(new_data_array)
                      
            data_dic[year] = data_list
        return data_dic
    
    
        
    def set_iv(self):
        '''
        Get Sum by name : XR
        
            전년 기록을 통해 다음년도의 XR,fip 초기값을 구하는 함수

        iv_batter_array (n x 20) : 년도별타자정보합산(index-4)
        ----------------------------------    
            name:4, h1:5, h2:6, h3:7, hr:8, bb:9, hbp:10, ibb:11, sac:12
            
            sf:13, so:14, go:15, fo:16, gidp:17, etc:18, h:19, tbb:20, ab:21, pa:22, xr:23
            
        iv_pitcher_array[year][team] (n x 13) : 전년도타자정보합산(index-2)
        ----------------------------------
            name:2, po:3, ip:4, tbf:5, np:6, ab:7, h:8, hr:9, tbb:10, so:11, r:12, er:13, fip:14
        '''
        year_list = self.year_list
        
        #batter initial value 구하기
        iv_batter_dic = dict()
        name_idx = 10
        for year in year_list:
            iv_array = np.zeros((1,20))
            for team_num in range(1,11):
                batter_array = self.batter_dic[year][team_num]
                names = np.unique(batter_array[:,name_idx])
                sum_by_name = np.array([np.sum(batter_array[batter_array[:,name_idx]==name],0) for name in names])
                new_iv_array = np.column_stack((names,sum_by_name[:,name_idx+1:]))
                iv_array = np.vstack([iv_array,new_iv_array])
            iv_batter_dic[year] = iv_array[1:,:]
        
        
        #pitcher initial value 구하기
        iv_pitcher_dic = dict()
        name_idx = 8
        for year in year_list:
            iv_array = np.zeros((1,14))
            for team_num in range(1,11):
                pitcher_array = self.pitcher_dic[year][team_num]
                names = np.unique(pitcher_array[:,name_idx])
                sum_by_name = np.array([np.sum(pitcher_array[pitcher_array[:,name_idx]==name],0) for name in names])
                game_by_name = np.array([len(pitcher_array[pitcher_array[:,name_idx]==name]) for name in names])
                new_iv_array = np.column_stack((names,sum_by_name[:,name_idx+1:],game_by_name))
                iv_array = np.vstack([iv_array,new_iv_array])
            iv_pitcher_dic[year] = iv_array[1:,:]
            
        self.iv_batter_dic = iv_batter_dic
        self.iv_pitcher_dic = iv_pitcher_dic
        
        
    
        
    def set_game_num_dic(self):
        '''
        Set game_num_dic
        
            Batter_array, Pitcher_array 의 게임번호별 시작 idx를 구하는 함수
            
            indexing time을 줄이기 위해 미리 인덱싱 번호를 구해둠
        '''
        def cre_game_num_dic(year_list, max_game_dic, data_dic):
            game_num_dic = dict()
            gn_idx = 5
            for year in year_list:
                team_list = [0]
                for team_num in range(1,11):
                    team_array = data_dic[year][team_num]
                    game_num_list = list(team_array[:,gn_idx])
                    game_num_idx_list = [0]
                    max_game_num = max_game_dic[year][team_num]
                    for game_num in range(1,max_game_num+1):
                        game_num_idx = game_num_list.index(game_num)
                        game_num_idx_list.append(game_num_idx)
                
                    team_list.append(game_num_idx_list)
                game_num_dic[year] = team_list
            return game_num_dic
        self.batter_game_num_dic = cre_game_num_dic(self.year_list,self.max_game_dic,self.batter_dic)
        self.pitcher_game_num_dic = cre_game_num_dic(self.year_list,self.max_game_dic,self.pitcher_dic)
        
        
        
    def set_lineup_dic(self):
        '''
        Set lineup_dic
        
            경기별 라인업을 미리 정리한 함수
            
            indexing time을 줄이기 위해 미리 구함 
        '''
        lineup_dic = dict()
        for year in self.year_list:
            team_list=[0]
            game_num_idx = 5
            bo_idx = 8
            b_name_idx = 10
            s_name_idx = 8
            
            for team_num in range(1,11):
                batter_array = self.batter_dic[year][team_num]
                pitcher_array = self.pitcher_dic[year][team_num]
                max_game_num = self.max_game_dic[year][team_num]
                
                name_array = np.zeros((1,10))
                for game_num in range(max_game_num):
                    this_batter_array = batter_array[batter_array[:,game_num_idx]==game_num+1,: ]
                    this_pitcher_array = pitcher_array[pitcher_array[:,game_num_idx]==game_num+1,: ]
                    name_list=list()
                    sp_name = this_pitcher_array[0,s_name_idx]
                    name_list.append(sp_name)
                    for i in range(1,10):
                    
                        name = this_batter_array[(this_batter_array[:,bo_idx] == i)][0,b_name_idx]
                        name_list.append(name)
                    name_array = np.vstack([name_array,name_list])
                
                team_list.append(name_array)
            lineup_dic[year] = team_list
        self.lineup_dic = lineup_dic
        
    
    ###################################### 분석용 function #####################################
    
    
    
    
        
    
    
    def xr_by_game(self, year, team_num, game_num):
        '''
        
        Get Sum of XR(float) : XR
            
            game_num 경기 '라인업 선발타자'의 Sum of XR 구하는 함수(-batter_range)
        
        Parameter
        ---------
        
            year : year
            
            team_num : number of team
            
            game_num : number of game (1-144 games)
            
            
        ETC
        ---
            
            XR : (0.49 x 1B) + (0.79 x 2B) + (1.06 x 3B) + (1.42 x HR) + (0.34 x (HP + BB - IBB))
                     + (0.25 x IBB) - (.090 x (AB - H - K)) - (0.098 x K) + (0.18 x SB) - (0.32 x CS)
                     - (0.37 x GIDP) + (0.37 x SB) + (0.04 x SH) 
        
        
        batter_dic[year][team] (n x 30) : 타자정보
        ----------------------------------    
            date:0, game_idx:1, year:2, team_num:3, foe_num:4, game_num:5, home&away:6, stadium: 7 
            
            bo:8, po:9, name:10, h1:11, h2:12, h3:13, hr:14, bb:15, hbp:16, ibb:17, sac:18
            
            sf:19, so:20, go:21, fo:22, gidp:23, etc:24, h:25, tbb:26, ab:27, pa:28, xr:29
        
        
        '''
        
        batter_array = self.batter_dic[year][team_num] # (n x 24)
        game_num_list = self.batter_game_num_dic[year][team_num]
        try:
            iv_array = self.iv_batter_dic[year-1]
        except:
            iv_array = np.zeros((1,1))
        
        br_range = self.br_range
        least_pa = self.least_pa
        start_pa = self.start_pa
        
        game_idx = 1 # 고유게임번호
        game_num_idx = 5 # 게임번호
        bo_idx = 8 # 타순
        po_idx = 9 # 선발/후보
        name_idx = 10 # 이름
        record_idx = 11 # 기록시작지점
        
        if self.is_new_game:
            end_idx = len(batter_array)
            lineup_list = list(self.today_array[self.today_array[:,3] == team_num,-1])
        else:
            end_idx = game_num_list[game_num]
            lineup_list = self.lineup_dic[year][team_num][game_num]
        
        start = game_num - br_range
        if start <= 0:
            start = 1

        total_range_array = batter_array[:end_idx]
        start_idx = game_num_list[start]
        
        range_array = batter_array[start_idx:end_idx]
        name_array = range_array[:,name_idx]
        
        
        # 선발타자 색출을 위한 for문(1-9번 타자)
        old_array = np.zeros((1,2))
        
        
        for i in range(1,10):
            
            name = lineup_list[i]
            
            range_by_batter = range_array[name_array == name]
            record_by_batter = range_by_batter[:,-2:]
            
            
            
            range_xr = record_by_batter[:,1].reshape(-1,1)
            range_sum_pa = np.sum(record_by_batter[:,0])
            
            
            if range_sum_pa < least_pa: # 주어진 구간(batter_range)의 기록(pa)이 적을 경우 전체구간으로 설정
                
                total_name_array = total_range_array[:,name_idx]
                
                record_by_batter = total_range_array[total_name_array == name][:,-2:]
                
                sum_record_array = np.sum(record_by_batter, axis = 0)
                
                total_pa = sum_record_array[-2]
                
                if iv_array[0,0] != 0 and total_pa < least_pa:
                    
                    new_iv_array = iv_array[iv_array[:,0] == name][:,-2:]
                    
                    sum_record_array = np.sum(np.vstack([sum_record_array,new_iv_array]),0)
            else:
                
                
                
                if self.is_park_factor: #park_factor 적
                    ground_array = range_by_batter[:,7]
                    pf_list = list()
                    for ground in ground_array:
                        try:
                            pf_list.append(self.park_factor_total[ground])
                        except:
                            pf_list.append(1)
                            
                    pf_array = np.divide(1,np.array(pf_list).reshape(-1,1)).reshape(-1,1)
                    range_sum_xr = np.dot(np.transpose(range_xr),pf_array)[0,0]
                    
                    
                else:
                    range_sum_xr = np.sum(range_xr)
                
                
                
                
                sum_record_array = np.array([range_sum_pa,range_sum_xr]).reshape(-1,2).astype(np.float)
                
            old_array = np.vstack([old_array,sum_record_array])
        
        team_record_array = old_array[1:]
        xr_array = team_record_array[:,-1]
        pa_array = team_record_array[:,-2]
        
        result_array = np.divide(xr_array, pa_array,out = np.zeros_like(xr_array),where = pa_array!=0) # 타수당 XR로 환산
        
        if self.is_pa:
            result_xr = np.round(np.dot(result_array,self.pa_params),3)
        else:
            result_xr = np.round(np.sum(result_array,axis = 0) * start_pa,3)
        
            
        return result_xr
    
    
    def xr_by_team(self, year, team_num):
        '''
        Get xr_array(n x 1) : xr
            
            주어진 year, team의 xr 가져오는 함수
            
        Parameter
        ---------
        
            year : year
            
            team_num : number of team
            
        '''
        xr_list = list()
        max_game_num = self.max_game_dic[year][team_num]
        for game_num in range(1,max_game_num+1):
            #try:
            xr_list.append(self.xr_by_game(year, team_num, game_num))
            #except:
            #    print('error xr_by_team!!' ,year,team_num,game_num)
        return  np.array(xr_list).reshape(max_game_num,-1)
    
    
    
    
    def run_by_team(self,year,team_num):
        '''
        Get run_array(n x 1) : run
        
            주어진 year, team의 득점(run) 가져오는 함수
            
        Parameter
        ---------
        
            year : year
            
            team_num : number of team
            
        '''
        run_array = self.score_dic[year][team_num][:,-4].reshape(-1,1)
        inn_array = self.score_dic[year][team_num][:,-16:-4]
        
        
        if self.is_park_factor:
            ground_array = self.score_dic[year][team_num][:,7]
            pf_list = list()
            for ground in ground_array:
                try:
                    pf_list.append(self.park_factor_total[ground])
                except:
                    pf_list.append(1)
            pf_array = np.divide(1,np.array(pf_list).reshape(-1,1)).reshape(-1,1)
            run_array = run_array *pf_array
        else:
            run_array = run_array
        
        inn_count_list = list()
        for inn in inn_array:
            inn_count = 12 - list(inn).count('-')
            inn_count_list.append(inn_count)
        inn_count_array = np.array(inn_count_list).reshape(-1,1)
        result_run_array = np.divide(run_array,inn_count_array)*9
        return result_run_array
    
    
    def sp_by_game(self, year, team_num, game_num):
        
        '''
        Get sp_array(1 x 5) : (sp_name, inn,fip)
        
            game_num 경기의 선발투수 기록 array 얻는 함수(-sp_range)
        
        
        
        Parameter
        ---------
        
            year : year
            
            team_num : number of team
            
            game_num : number of game(1-144 games)
        
        ETC
        ---
        
            fip = (((13 * HR) + (3 * TBB) - (2 * SO)) / IP) + 3.2
            
            era = (ER / IP)
            
            ra = R / IP
            
            inn = IP
        
        pitcher_dic[year][team] (n x 21) : 투수정보
        ----------------------------------
            date:0, game_idx:1, year:2, team_num:3, foe_num:4, game_num:5, home&away:6, stadium: 7
            
            name:8, po:9, ip:10, tbf:11, np:12, ab:13, h:14, hr:15, tbb:16, so:17, r:18, er:19, fip:20
            
        '''
        
        pitcher_array = self.pitcher_dic[year][team_num]
        game_num_list = self.pitcher_game_num_dic[year][team_num]
        try:
            iv_array = self.iv_pitcher_dic[year-1]
        except:
            iv_array = np.zeros((1,1))
            
        sp_range = self.sp_range
        least_inn = self.least_inn
        start_fip = self.start_fip
        start_inn = self.start_inn
            
        game_idx = 1 # 고유게임번호
        game_num_idx = 5 # 게임번호
        name_idx = 8 # 이름
        po_idx = 9 # 선발/계투
        inn_idx = 10 # 이닝
        r_idx = 18 # 실점
        er_idx = 19 # 자책점
        fip_idx = 20 # fip
        
        
        if self.is_new_game:
            end_idx = len(pitcher_array)
            lineup_list = list(self.today_array[self.today_array[:,3] == team_num,-1])
        else:
            end_idx = game_num_list[game_num]
            lineup_list = self.lineup_dic[year][team_num][game_num]
        
        
        
        
        
        name = lineup_list[0]
        
        total_range_array = pitcher_array[:end_idx]        
        name_array = total_range_array[:,name_idx]
        po_array = total_range_array[:,po_idx] # 1: 선발 2~: 계투
        
        record_by_sp = total_range_array[(name_array == name) & (po_array == 1),:] 
        
        len_sp = len(record_by_sp)
         
        start_idx = len_sp - sp_range
        if start_idx <= 0:
            start_idx = 0
         
        range_record_array = record_by_sp[start_idx:]
            
        sum_record_array = np.sum(range_record_array[:,(inn_idx,fip_idx)],axis = 0)
        
        if len_sp > sp_range:
            len_range = sp_range
        else:
            len_range = len_sp
        
        
        
        if self.is_park_factor:
            ground_array = range_record_array[:,7]
            pf_list = list()
            for ground in ground_array:
                try:
                    pf_list.append(self.park_factor_total[ground])
                except:
                    pf_list.append(1)
                    
            pf_array = np.divide(1,np.array(pf_list).reshape(-1,1)).reshape(-1,1)
            
            range_sum_fip = np.dot(np.transpose(range_record_array[:,fip_idx]),pf_array)[0]
        else:
            range_sum_fip = sum_record_array[1]
        
        range_sum_inn = sum_record_array[0]
        
        sum_list = [range_sum_inn,range_sum_fip,len_range] # #inn, fip, len_sp
        
        
        
        if range_sum_inn <= least_inn: # 주어진 구간의 기록(inn)이 적을 경우 전체구간으로 설정
            
            sum_record_array = np.sum(record_by_sp[:,(inn_idx,fip_idx)], axis = 0)
            sum_record_array = np.hstack([sum_record_array, len_sp])
            inn_sum_total = sum_record_array[0]
            
            if inn_sum_total <= least_inn and iv_array[0,0] != 0: # 전체구간의 기록(inn)이 적을 경우 작년 기록 가져옴
                
                new_iv_array = iv_array[iv_array[:,0] == name][:,(2,-2,-1)] # inn, fip, len_sp                
                sum_record_array = np.sum(np.vstack([sum_record_array,new_iv_array]),0)
                
                if len(new_iv_array) != 0:
                    new_len_sp = sum_record_array[-1]
                    len_sp+=new_len_sp
                    sp_range +=new_len_sp
                    
        
        
        
        
                
        inn_sum = sum_list[0]
        fip_sum = sum_list[1]
        

        if len_sp == 0: #  선발투수 등판 기록이 없을경우 초기값 사용
            fip = start_fip
            inn = start_inn
            
        elif inn_sum == 0: # 이닝이 0이닝일 경우 초기값 사용(에러방지)
            fip = start_fip
            inn = start_inn
            
        else:
            
            
             # fip = (((13 * HR) + (3 * TBB) - (2 * SO)) / IP) + 3.2
            fip = (fip_sum / inn_sum) + 3.2

            if len_sp <= sp_range:
                inn = inn_sum / len_sp
                
            else:
                inn = inn_sum / sp_range
                

        #보정
        
        if np.isnan(inn): inn = self.start_inn # 무한대 or None값이 있을경우 초기갑사용
        if np.isnan(fip): inn = self.start_fip
        
        sp_record_array = np.hstack([name,inn,fip]) 
        
        return sp_record_array # array : sp_name, inn, fip
    
    def sp_by_team(self, year, team_num):
        '''
        Get sp_array(n x 3) : (sp_name, inn, fip)
        
            team의 선발투수 기록 array 얻는 함수
            
        Parameter
        ---------
        
            year : year
            
            team_num : number of team
        
        '''
        
        
        
        max_game_num =self.max_game_dic[year][team_num]
        
        old_array = np.zeros((1,3))
        for game_num in range(1, max_game_num+1):
            try:
                sp_array = self.sp_by_game(year, team_num, game_num)
                old_array = np.vstack([old_array, sp_array])
            except:
                print('Error sp_by_team!!' ,year,team_num,game_num)
        
        total_sp_array = old_array[1:,:]
        
        return total_sp_array # array: name, inn, fip


    def rp_by_game(self,year,team_num,game_num):
        '''
        Get rp_array(1 x 3) : (fip, era, ra)
        
            game_num 경기의 선발투수 기록 array 얻는 함수
        
        ETC
        ---
        
            fip = (((13 * HR) + (3 * TBB) - (2 * SO)) / IP) + 3.2
            
            era = (ER / IP)
            
            
        pitcher_dic[year][team] (n x 21) : 투수정보
        ----------------------------------
            date:0, game_idx:1, year:2, team_num:3, foe_num:4, game_num:5, home&away:6, stadium: 7
            
            name:8, po:9, ip:10, tbf:11, np:12, ab:13, h:14, hr:15, tbb:16, so:17, r:18, er:19, fip:20
            
        '''
        
        
        pitcher_array = self.pitcher_dic[year][team_num]
        game_num_list = self.pitcher_game_num_dic[year][team_num]        

        
        rp_range = self.rp_range
        start_fip = self.start_fip
        
        
        game_idx = 1 # 고유게임번호
        game_num_idx = 5 # 게임번호
        name_idx = 8 # 이름
        po_idx = 9 # 선발/계투
        inn_idx = 10 # 이닝
        fip_idx = 20 # fip
        
        if self.is_new_game:
            end_idx = len(pitcher_array)
            
        else:
            end_idx = game_num_list[game_num]
        

        start_game_num = game_num - rp_range
        if start_game_num <= 0:
            start_game_num = 1
        
        start_idx = game_num_list[start_game_num]
        
        range_array = pitcher_array[start_idx:end_idx]
        po_array = range_array[:,po_idx]
        record_by_rp = range_array[po_array != 1,:]
        
        old_array = np.zeros((1,2)) # rp 경기별 기록 합치기(파크팩터처리)
        for gn in range(start_game_num,game_num):
            
            range_record = record_by_rp[record_by_rp[:,5]==gn,:]
            if len(range_record) ==0:
                continue
            if self.is_park_factor:
                ground = range_record[0,7]
                pf = self.park_factor_total.get(ground)
                if pf == None:
                    pf = 1
            else:
                pf = 1
            sum_record = np.sum(range_record[:,(inn_idx,fip_idx)],axis = 0)
            new_array = np.array([sum_record[0],sum_record[1]/pf]).reshape(1,2)
            old_array = np.vstack([old_array,new_array])
        
        
        sum_record_array = np.sum(old_array,axis=0)
 
        inn_sum = sum_record_array[0]
        fip_sum = sum_record_array[1]
            
                
         
        
        if inn_sum == 0: # 이닝이 0이닝일 경우 초기값 사용(에러방지)
            fip = start_fip

        else:
             # fip = (((13 * HR) + (3 * TBB) - (2 * SO)) / IP) + 3.2
            fip = (fip_sum / inn_sum) + 3.2
     
        fip_array = np.array([fip])
        fip_array[np.isnan(fip_array)] = start_fip # 무한대 값이 있을경우 start_fip로 초기화
        
        return fip_array
    
    def rp_by_team(self, year, team_num):
        '''
        Get rp_array(n x 3) : (fip, era, ra)
        
        team의 계투기록 array 얻는 함수
            
        
        '''
        max_game_num = self.max_game_dic[year][team_num]

        
        old_array = np.zeros((1,1))
        for game_num in range(1, max_game_num+1):            
            try:
                rp_array = self.rp_by_game(year, team_num, game_num)
                old_array = np.vstack([old_array, rp_array])
            except:
                print("error rp_by_team!!!",year,team_num,game_num)
        
        rp_array = old_array[1:,:]
        
        return rp_array # array(n x 1) : fip
    
    def set_xr_dic(self,dic_num):     
        
        if dic_num == 1:
            data_dic = self.xr_dic
            data_range = self.br_range
        elif dic_num == 2:
            data_dic = self.sp_dic
            data_range = self.sp_range
        
        elif dic_num ==3:
            data_dic = self.rp_dic
            data_range = self.rp_range
        else:
            print('plz correct dic_num!!!')
            
        data_dic[data_range] = dict()
        for year in self.year_list:
            team_list = [0]
            for team_num in range(1,11):
                if dic_num ==1:
                    team_list.append(self.xr_by_team(year,team_num))
                if dic_num ==2:
                    team_list.append(self.sp_by_team(year,team_num))
                if dic_num ==3:
                    team_list.append(self.rp_by_team(year,team_num))
            data_dic[data_range][year] = team_list
    
    def set_record_dic(self,br_range,sp_range,rp_range):
        '''
        Set record_dic[year][team_num](n x 3) : 
        
            10개팀의 모든 기록을 dictionary로 가져오기
            
        
        Parameter
        ---------
        
            mod_list : 분석에 사용할 변수들 데이터 리스트
                       ex) XR, Run, Pitcher 등등
        '''
        record_dic = dict()
        
        for year in self.year_list:
            team_list =[0]
            for team_num in range(1,11):
                max_game = self.max_game_dic[year][team_num]
                
                #분석에 사용할 변수 데이터 불러오기
                old_array = np.zeros((max_game,1))
                
                old_array = np.hstack([old_array,self.run_by_team(year,team_num)]) # run(n x 1) : run
                old_array = np.hstack([old_array,self.xr_dic[br_range][year][team_num]]) # xr(n x 1) : xr
                old_array = np.hstack([old_array,self.sp_dic[sp_range][year][team_num]])# sp(n x 3) : name, inn , sp_fip
                old_array = np.hstack([old_array,self.rp_dic[rp_range][year][team_num]])# rp_fip 
                total_record_array = old_array[:,1:] # total_record_array(n x 6): run, xr, name, inn, sp_fip, rp_fip       
                team_list.append(total_record_array)
                
            record_dic[year] = team_list
        
        self.record_dic = record_dic
    
    
    
    def set_record_total_dic(self,br_range,sp_range,rp_range):
        '''
        Set record_dic[year][team_num]
            
            10개 팀의 모든기록 + 해당경기 상대팀 기록 가져오기
        '''
        self.set_record_dic(br_range,sp_range,rp_range)
        info_dic = self.game_info_dic # info_array(n x 10) : (date,game_num,total_game_num,year,team_num,foe_num,game_num,home&away,stadium,result)  
        record_dic = self.record_dic # record_array(n x 6): run, xr, name, inn, sp_fip, rp_fip
        
        len_info = 10
        len_record = 12
        
        date_idx = 0
        foe_num_idx = 5
        
        record_total_dic = dict()
        for year in self.year_list:
            team_list = [0]
            for team_num in range(1,11):
                
                record_array = record_dic[year][team_num]
                info_array = info_dic[year][team_num]
                
                old_array = np.zeros((1,len_record))
                
                for info,record in zip(info_array,record_array):
                    
                    date = info[date_idx]
                    foe_num = info[foe_num_idx]
                    
                    info_by_foe = info_dic[year][foe_num]
                    record_by_foe = record_dic[year][foe_num]
                    record_by_foe = np.hstack([info_by_foe,record_by_foe])                    
                    record_by_foe = record_by_foe[record_by_foe[:,date_idx] == date][0,len_info:]
                    
                    
                    team_record_array = np.hstack([record,record_by_foe])
                    
                    old_array = np.vstack([old_array,team_record_array])
                old_array = old_array[1:,(0,6,2,8,1,3,4,5,7,9,10,11)] # record_array(n x 6): run, xr, name, inn, sp_fip, rp_fip
                
                total_array = np.hstack([info_array,old_array])
                
                team_list.append(total_array)
            record_total_dic[year] = team_list
            
        self.record_total_dic = record_total_dic
        
    def set_toto_dic(self):
        '''
        toto_dic setting 작업
        
        csv로 저장된 도박사 배당 데이터를 년도 / 팀별 딕셔너리로 만드는 함수
        '''
        
        
        # toto_Array = 0: 년도 / 1: 년월일 / 2: 홈팀 / 3: 원정팀 / 4: 홈점수 / 5: 원정점수 / 6: 홈배당 / 7: 원정배당 / 8: 홈팀번 / 9: 원정팀번 
        self.toto_array = self.load_csv(self.address,'normal_toto_array')
        toto_array = self.toto_array[:,(1,2,6,7,8,9,10,12,18,19)]
        
        toto_dic = dict()
        for year in range(2017,2021):
            team_list = [0]
            year_mask = (toto_array[:,0] == year)
            year_array = toto_array[year_mask,:]
            for team_num in range(1,11):
                
                
                # team_num에 맞는 array 추출(home & away)
                home_mask = (year_array[:,8] == team_num)
                away_mask = (year_array[:,9] == team_num)
                team_home_array = year_array[home_mask,:]
                team_away_array = year_array[away_mask,:]
                
                
                # 홈 / 원정 column 추가
                team_home_array = np.insert(team_home_array,2,'홈',axis = 1)
                team_away_array = np.insert(team_away_array,2,'원정',axis = 1)
                team_away_array = team_away_array[:,(0,1,2,4,3,6,5,8,7,10,9)] # team_away_array 순서변경(home과 합치기위해)
                
                team_array = np.vstack([team_home_array,team_away_array])
                
                date_array = team_array[:,1]
                sort_mask = np.lexsort([date_array])
                team_array = team_array[sort_mask,:]
                
                home_odds_array = team_array[:,7]
                away_odds_array = team_array[:,8]
                
                # odds를 통한 승률 및 수수료 계산
                commission_array = 1 / ((1 / home_odds_array) + (1 / away_odds_array)) 
                home_rate_array = np.multiply((1 / home_odds_array),commission_array)
                away_rate_array = np.multiply((1 / away_odds_array),commission_array)
                
                idx_array = np.array(np.arange(1,self.max_game_dic[year][team_num]+1)).reshape(-1,1)
                
                team_array[:,7] = home_rate_array
                team_array[:,8] = away_rate_array
                team_array = np.hstack([idx_array,team_array])
                team_list.append(team_array)
                   
                
            toto_dic[year] = team_list
        self.toto_dic = toto_dic

