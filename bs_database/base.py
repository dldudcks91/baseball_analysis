#%%
#%%
import sys
sys.path.append('D:\\BaseballProject\\python')
#%%
import numpy as np
import pandas as pd
import pymysql
from datetime import datetime
from sqlalchemy import create_engine

import bs_stats.base as bs
#%%

class Database(bs.Baseball):
    '''
        db관련 코드를 저장하는 Class
    
    
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
        
        
        cursor.execute(sql)
        result = cursor.fetchall()
        return result
    
    def load_table(self, conn, table):
        
        cursor = conn.cursor()
        sql = 'select * from ' + table
        cursor.execute(sql)
        result_array = np.array(cursor.fetchall())
        
        return result_array
    
    def insert_table(self, conn, table, data_array):
        '''
        파이썬 array를 DB에 저장하는 코드
        array_to_db와 같은 코드 
        
        '''
        cursor = conn.cursor()
        
        for i,data in enumerate(data_array):
            data_str = str(tuple(data))
            data_str = data_str.replace('None','Null')
            sql = 'insert into '+ table + ' values ' + data_str
            cursor.execute(sql)
        

    
    def array_to_db(self,conn, data_array, table):
        '''
        파이썬 array를 DB에 저장하는 코드
        insert_table과 같은 코드 
        '''
        cursor = conn.cursor()
        
        
        for data in data_array:
            data_list = list()
            for d in data:
                data_list.append(str(d))
            data_str = str(tuple(data_list))
            data_str = data_str.replace('None','Null')
            sql = f'insert into {table} values {data_str}'
            
            cursor.execute(sql)
            
    def set_last_game_num_list(self,year,conn):
        '''
        
            DB에서 팀 별 게임번호 리스트 세팅
            
        '''
        last_game_num_list = list(self.fetch_sql('select total_game_num from team_info ' + 'where year = ' + str(year),conn))
        new_list = list()
        for last_game_num in last_game_num_list:
            new_list.append(last_game_num[0])
        self.last_game_num_list = [0] + new_list
        
        
    def update_team_info(self,conn, year, record_list, update_type):
        cursor = conn.cursor()
        
        if update_type == 'game_num':
            
            for team_num in range(1,11):
                update_game_num = record_list[team_num]
                sql = "update team_info set total_game_num =" + str(update_game_num) + " where year = " + str(year) + " and team_num = "+ str(team_num)
                cursor.execute(sql)
                
        elif update_type == 'record':
            cursor = conn.cursor()
            sql = ''' UPDATE team_info SET win= %s, lose= %s, draw= %s, win_rate=%s WHERE year= %s AND team_num= %s; ''' 
            cursor.executemany(sql,record_list)
        
        elif update_type == 'local_to_aws':
            cursor = conn.cursor()
            sql = ''' UPDATE team_info SET total_game_num = %s, win= %s, lose= %s, draw= %s, win_rate=%s WHERE year= %s AND team_num= %s; ''' 
            cursor.executemany(sql,record_list)
            
    
  
    def array_to_db_df(self,data_array,table_name, col_name):
        '''
        
            주어진 array를 data-frame 형태로 변환시켜 col_name을 통해 db에 넣기
            
        '''
        
        data_pd = pd.DataFrame(data_array,columns = col_name)
        engine = self.engine
        data_pd.to_sql(name=table_name,con = engine,if_exists ='append',index = False)
        
        
    def load_data_all(self,db_address, code, file_address,pyconn):
        '''
        
        baseball DataBase에 있는 모든 분석용 테이블 불러오기
        
        Load to Record of team_game_info / batter / pitcher / score by Mysql
        
        Set game_info_array / batter_array / pitcher_array / score_array
        
            
        '''
        
        engine = create_engine(db_address + code + file_address)#, encoding = 'utf-8')
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
        
    def load_data_this_year(self, db_address, code, file_address, year):
        '''
        
        baseball DataBase에 있는 해당 년도 테이블 불러오기
        
        Load to Record of team_game_info / batter / pitcher / score by Mysql
        
        Set game_info_array / batter_array / pitcher_array / score_array
        
            
        '''
        
        engine = create_engine(db_address + code + file_address)#, encoding = 'utf-8')
        conn = engine.connect()
        self.team_info_array = np.array(pd.read_sql_table('team_info',conn))
        
        start_game_idx = year * 10000000000
        start_team_game_idx = year * 100000
        
        
        
        
        game_info_query = f'SELECT game_idx, stadium from game_info where game_idx >= {start_game_idx}'
        game_info_df = pd.read_sql(game_info_query, conn)
#        game_info_df = pd.read_sql_table(('game_info'),conn)[['game_idx','stadium']]
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'success load game_info_df')
        
        team_game_info_query = f'SELECT * from team_game_info where team_game_idx >= {start_team_game_idx}'
        team_game_info_df = pd.read_sql(team_game_info_query, conn)
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'success load team_game_info_df')
        
        game_info = pd.merge(team_game_info_df, game_info_df,  on = 'game_idx',how = 'left')
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'success merge team_game_info_df & game_info_df')
        self.game_info_array = np.array(game_info)
        conn.close()
        
        conn = engine.connect()
        score_query = f'SELECT * from score_record where team_game_idx >= {start_team_game_idx}'
        score_df = pd.read_sql(score_query, conn)
        self.score_array = np.array(pd.merge(game_info , score_df, on = 'team_game_idx',how = 'left'))
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'success load & merge game_info_df & score_df')
        conn.close()
        
        
        conn = engine.connect()
        batter_query = f'SELECT * from batter_record where team_game_idx >= {start_team_game_idx}'
        batter_df = pd.read_sql(batter_query, conn)
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'success load batter_df')
        self.batter_array = np.array(pd.merge(game_info , batter_df, on = 'team_game_idx',how = 'left'))
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'success load & merge game_info_df & batter_df')
        conn.close()
        
        
        conn = engine.connect()
        pitcher_query = f'SELECT * from pitcher_record where team_game_idx >= {start_team_game_idx}'
        pitcher_df = pd.read_sql(pitcher_query, conn)
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'success load pitcher_df')
        self.pitcher_array = np.array(pd.merge(game_info , pitcher_df, on = 'team_game_idx',how = 'left'))
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'success merge game_info_df & pitcher_df')
        
        conn.close()
    
    def load_data_this_year_new(self, db_address, code, file_address, year):
        '''
        
        baseball DataBase에 있는 해당 년도 테이블 불러오기
        
        Load to Record of team_game_info / batter / pitcher / score by Mysql
        
        Set game_info_array / batter_array / pitcher_array / score_array
        
            
        '''
        def load_data(query, conn):
            """SQL 쿼리를 실행하고 결과를 DataFrame으로 반환하는 함수"""
            return pd.read_sql(query, conn)
        engine = create_engine(db_address + code + file_address)
        def load_data_fetch(query, conn):
            
            cursor = conn.cursor()
            
            cursor.execute(query)
            result_data = pd.DataFrame(cursor.fetchall())
            return result_data
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Start loading data')

        # 쿼리 시작 인덱스 설정
        start_game_idx = year * 10000000000
        start_team_game_idx = year * 100000

        # 필요한 모든 데이터 한 번에 로드
        game_info_query = f'SELECT game_idx, stadium FROM game_info WHERE game_idx >= {start_game_idx} order by game_idx'
        team_game_info_query = f'SELECT * FROM team_game_info WHERE team_game_idx >= {start_team_game_idx} order by team_game_idx'
        score_query = f'SELECT * FROM score_record WHERE team_game_idx >= {start_team_game_idx} order by team_game_idx'
        batter_query = f'SELECT * FROM batter_record WHERE team_game_idx >= {start_team_game_idx} order by team_game_idx'
        pitcher_query = f'SELECT * FROM pitcher_record WHERE team_game_idx >= {start_team_game_idx} order by team_game_idx'
        with engine.connect() as conn:
            # 데이터 로드
            game_info_df = load_data(game_info_query, conn)
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Success loading game_info_df')
        
            team_game_info_df = load_data(team_game_info_query, conn)
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Success loading team_game_info_df')
        
            score_df = load_data(score_query, conn)
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Success loading score_df')
            
            
        
            batter_df = load_data(batter_query, conn)  
            #batter_df = load_data_fetch(batter_query, conn)
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Success loading batter_df')
        
            pitcher_df = load_data(pitcher_query, conn)
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Success loading pitcher_df')
    
        #데이터 병합
        game_info = pd.merge(team_game_info_df, game_info_df, on='game_idx', how='left')
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Success merge team_game_info_df & game_info_df')
        
        self.game_info_array = np.array(game_info)
        
        self.score_array = np.array(pd.merge(game_info, score_df, on='team_game_idx', how='left'))
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Success merge game_info & score_df')
        
        self.batter_array = np.array(pd.merge(game_info, batter_df, on='team_game_idx', how='left'))
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Success merge game_info & batter_df')
        
        self.pitcher_array = np.array(pd.merge(game_info, pitcher_df, on='team_game_idx', how='left'))
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Success merge game_info & pitcher_df')
        
    
    def load_today_array(self,db_address, code, file_address):
        
        '''
        
            오늘 있는 경기 정보관련 테이블 가져오기
            
        '''
        
        
        engine = create_engine(db_address + code + file_address)
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
        
    def delete_table_data(self,conn,table_name):
        cursor = conn.cursor()
        sql = f'delete from {table_name}'
        cursor.execute(sql)