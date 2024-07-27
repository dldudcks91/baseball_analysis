#%%
'''
library 가져오기
'''
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from datetime import datetime, timedelta

import pymysql

from bs_personal import personal_code as cd
from bs_crawling import kbo_request as kr


#%%
host, user, password, db = cd.aws_host, cd.aws_user, cd.aws_code, cd.db #aws와 연결하는 계정 데이터 가져오기

conn = pymysql.connect(host = host, user = user, password= password, db= db, charset='utf8')

#%%
ck = kr.Crawling_kbo_request()

end_date_time = (datetime.today() + timedelta(hours = 9)).date()
year = end_date_time.year
ck.year = year

#%%

sql = 'select max(game_idx) from game_info'
last_game_idx_tuple = ck.fetch_sql(sql,conn)
last_game_idx = last_game_idx_tuple[0][0]
last_game_date_str = last_game_idx[:8]
#%%
print(datetime.now())
date_time = (datetime.strptime(last_game_date_str, "%Y%m%d") + timedelta(days = 1)).date()
#%%


ck.set_last_game_num_list(ck.year,conn) #ck.last_num_list = [0 for i in range(10)]#

while date_time < end_date_time:
    date_str = date_time.strftime("%Y%m%d")
    date = int(date_str)
    
    ck.year = int(date_str[:4])
    ck.craw_game_info(date)
    ck.craw_box_score(date)
    ck.craw_score_board(date)  
    if ck.game_dic[date]:
        ck.set_date_total(date)    
    
    date_time = datetime.strptime(date_str, "%Y%m%d").date()
    date_time+= timedelta(days = 1)
    
    print(date, ck.game_info_array.shape)
    
#%%

print(ck.game_info_array.shape)
print(ck.team_game_info_array.shape)
print(ck.batter_array.shape)
print(ck.pitcher_array.shape)
print(ck.score_array.shape)



#%%


#%%
try:
    ck.array_to_db(conn, ck.game_info_array, 'game_info')  
    ck.array_to_db(conn, ck.team_game_info_array, 'team_game_info')
    ck.array_to_db(conn, ck.batter_array, 'batter_record')
    ck.array_to_db(conn, ck.pitcher_array, 'pitcher_record')
    ck.array_to_db(conn, ck.score_array, 'score_record')
    ck.update_team_info(conn, year, ck.last_game_num_list, update_type = 'game_num')
    
    conn.commit()
except pymysql.InterfaceError as e:
    conn.rollback()
    print(f"InterfaceError 발생: {e}")
except Exception as e:
    print(f"다른예외처리: {e}")
finally:

    conn.close()


