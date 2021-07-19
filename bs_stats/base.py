#%%
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
#%%
class Baseball:
    '''
        공통적으로 쓰이는 값들을 선언한 부모 Class
    
    '''
    
    # 팀 이름 번호에 매칭하는 dictionary
    team_dic = {'LG':1,'롯데':2,'KIA':3,'삼성':4,'두산':5,'한화':6,'SK':7,'SSG':7,'키움':8,'넥센':8,'NC':9,'KT':10,'kt':10}
    team_list = [0,'LG','롯데','KIA','삼성','두산','한화','SSG','키움','NC','KT']
    # 팀 별 Stadium dictionary
    stadium_dic = {'LG':'잠실','롯데':'사직','KIA':'광주','삼성':'대구','두산':'잠실','한화':'대전','SK':'문학','SSG':'문학','키움':'고척','넥센':'고척','NC':'창원','KT':'수원','kt':'수원'}
    
    # 분석에 사용한 year_list
    year_list = [2017,2018,2019,2020,2021]
    
    # 구장, 연도별 Park Factor dictionary
    park_factor_2016 = {'잠실': 0.954,'사직': 1.030,'광주':0.997, '대구': 1.042, '대전':1.007,'문학':1.016,'고척':1.032,'마산':0.943,'수원':1.021}
    park_factor_total = {'잠실': 0.854,'사직': 1.099,'광주':1.003, '대구': 1.153, '대전': 0.977,'문학':1.046,'고척':0.931,'창원':1.051,'수원':1.032}
    park_factor_list = [0.   , 0.863, 1.099, 1.003, 1.153, 0.863, 0.977, 1.046, 0.931, 1.051, 1.032]
    
    #타순별 기여
    pa_params = [4.756, 4.625,4.558,4.467,4.354,4.227,4.110,3.976,3.829]                   
       
    
    # csv 데이터 저장 주소
    address = 'C:\\Users\\Chan\\Desktop\\BaseballProject\\data'
    
    def __init__(self):
        return

     
    def save_csv(self,data,address,name):
        '''
        Save data_array
        
            csv 데이터 저장하기
            
        '''
        data = pd.DataFrame(data)
        data.to_csv(address + '\\' + name + '.csv',encoding = 'cp949')
        
    
    def load_csv(self,address,name):
        '''
        Load data_array
        
            csv 데이터 불러오기
            
        '''
        
        data = pd.read_csv(address + '\\' + name + '.csv',encoding = 'cp949' )
        result = np.array(data)
        return result
    


    
    ################################    DB관련코드    #############################################
    
    
class Database(Baseball):
    '''
        db관련 코드를 저장하는 Class
    
    '''
    '''
    분석에 사용하는 db column
    
    game_info_columns = ['game_idx', 'home_name', 'away_name', 'stadium', 'end', 'etc']
    team_game_info_columns = ['game_idx', 'team_game_idx', 'year', 'team_num', 'foe_num', 'game_num', 'home_away']
    score_columns = ['team_game_idx', 'result', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6',
                         'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'r', 'h', 'e', 'b']
    batter_columns = ['team_game_idx','bo','po','name','b1','b2','b3','hr','bb',
                   'hbp','ibb','sac','sf','so','go','fo','gidp','etc','h','tbb','ab','pa','xr']
    pitcher_columns = ['team_game_idx','name', 'po', 'inn', 'tbf', 'np', 'ab', 'h', 'hr', 'tbb', 'so', 'r','er', 'fip']
    team_info_columns = ['year','team_num','team_name','stadium','total_game_num','win','lose','draw','win_rate']
    '''
    
    team_game_info_columns = ['date','game_idx', 'team_game_idx', 'year', 'team_num', 'foe_num', 'game_num', 'home_away','stadium']
    score_columns = ['result', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6',
                         'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'r', 'h', 'e', 'b']
    batter_columns = ['bo','po','name','b1','b2','b3','hr','bb',
                   'hbp','ibb','sac','sf','so','go','fo','gidp','etc','h','tbb','ab','pa','xr']
    pitcher_columns = ['name', 'po', 'inn', 'tbf', 'np', 'ab', 'h', 'hr', 'tbb', 'so', 'r','er', 'fip']
    team_info_columns = ['year','team_num','team_name','stadium','total_game_num','win','lose','draw','win_rate']
    last_game_num_list = list() # 팀 별 마지막 게임 번호 list
    def __init__(self):
        return
    
    def fetch_sql(self,sql,conn):
        
        cursor = conn.cursor()
        
        sql = sql
        cursor.execute(sql)
        result = cursor.fetchall()
        return result
    
    def delete_table_data(self,conn,table_name):
        cursor = conn.cursor()
        sql = 'delete from' + ' ' + table_name
        cursor.execute(sql)
        
    def set_last_game_num_list(self,year,conn):
        '''
        Get last_game_num_list
        
            DB에서 팀 별 마지막 게임번호 가져오기
            
        '''
        last_game_num_list = list(self.fetch_sql('select total_game_num from team_info ' + 'where year = ' + str(year),conn))
        new_list = list()
        for last_game_num in last_game_num_list:
            new_list.append(last_game_num[0])
        self.last_game_num_list = [0] + new_list
        
    def update_total_game_num(self, conn, year, update_game_num_list):

        cursor = conn.cursor()
        
        for team_num in range(1,11):
            update_game_num = update_game_num_list[team_num]
            sql = "update team_info set total_game_num =" + str(update_game_num) + " where year = " + str(year) + " and team_num = "+ str(team_num)
            cursor.execute(sql)
            
    def array_to_db(self,conn, data_array,table_name):
        '''
        파이썬 array DB에 저장하는 코드
        
        '''
        cursor = conn.cursor()
        
        for data in data_array:
            data_str = str(tuple(data))
            data_str = data_str.replace('None','Null')
            sql = 'insert into' +' '+ table_name + ' ' + 'values' + data_str
            cursor.execute(sql)


    def array_to_db_df(self,data_array,table_name, col_name):

        data_pd = pd.DataFrame(data_array,columns = col_name)
        engine = self.engine
        data_pd.to_sql(name=table_name,con = engine,if_exists ='append',index = False)
        
        
    def load_data_all(self,db_address, code, file_address):
        '''
        Load to Record of team_game_info / batter / pitcher / score by Mysql
        
        Set game_info_array / batter_array / pitcher_array / score_array
        
            
        '''
        
        engine = create_engine(db_address + code + file_address, encoding = 'utf-8')
        conn = engine.connect()
        self.team_info_array = np.array(pd.read_sql_table('team_info',conn))
        
        game_info_df = pd.read_sql_table(('game_info'),conn)[['game_idx','stadium']]
        team_game_info_df = pd.read_sql_table(('team_game_info'),conn)
        game_info = pd.merge(team_game_info_df,game_info_df,on = 'game_idx',how = 'left')
        self.game_info_array = np.array(game_info)
        
        
        score_df = pd.read_sql_table('score_record',conn)
        self.score_array = np.array(pd.merge(game_info , score_df, on = 'team_game_idx',how = 'left'))
        
        batter_df = pd.read_sql_table('batter_record',conn)
        self.batter_array = np.array(pd.merge(game_info , batter_df, on = 'team_game_idx',how = 'left'))
        
        pitcher_df = pd.read_sql_table('pitcher_record',conn)
        self.pitcher_array = np.array(pd.merge(game_info , pitcher_df, on = 'team_game_idx',how = 'left'))
        
        conn.close()
    
    def load_today_array(self,db_address, code, file_address):
        engine = create_engine(db_address + code + file_address ,encoding = 'utf-8')
        conn = engine.connect()
        
        game_info_df = pd.read_sql_table(('today_game_info'),conn)
        self.today_game_info = np.array(game_info_df)
        
        game_info_df = game_info_df[['game_idx','stadium']]
        team_game_info_df = pd.read_sql_table(('today_team_game_info'),conn)
        game_info = pd.merge(team_game_info_df,game_info_df,on = 'game_idx',how = 'left')
        
        
        
        today_lineup_df = pd.read_sql_table('today_lineup',conn)
        today_array = self.pitcher_array = np.array(pd.merge(game_info , today_lineup_df, on = 'team_game_idx',how = 'left'))
        self.today_array= today_array
        
        conn.close()