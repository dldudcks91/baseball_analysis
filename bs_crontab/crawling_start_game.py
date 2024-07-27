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
'''
크롤링 전 세팅
'''


host, user, password, db = cd.aws_host, cd.aws_user, cd.aws_code, cd.db #aws와 연결하는 계정 데이터 가져오기

today = (datetime.today() + timedelta(hours = 9)).date()
date_str = str(today).replace("-","")
date = int(date_str)
year = int(date_str[:4])


ck = kr.Crawling_kbo_request()
ck.year = year

conn = pymysql.connect(host = host, user = user, password= password, db= db, charset='utf8')
ck.set_last_game_num_list(ck.year,conn) #팀별 게임 번호 가져오기
conn.close()

'''
진행예정인 경기정보 크롤링
'''

ck.craw_game_info(date) #game_info crawling
ck.craw_lineup(date) #lineup crawling
ck.set_date_start(date)
#%%
print(datetime.now())
print(ck.game_info_array.shape)
print(ck.team_game_info_array.shape)
print(ck.lineup_array.shape)
#%%
'''
today 테이블 데이터 삭제
'''
conn = pymysql.connect(host = host, user = user, password= password, db= db, charset='utf8')
try:
    ck.delete_table_data(conn, 'today_lineup')
    ck.delete_table_data(conn, 'today_team_game_info')
    ck.delete_table_data(conn, 'today_game_info')
    
    conn.commit()
    print(f"Success Delete start_game_data")
except pymysql.InterfaceError as e:
    conn.rollback()
    print(f"InterfaceError 발생: {e}")
except Exception as e:
    print(f"다른예외처리: {e}")
finally:
    
    conn.close()

#%%
'''
크롤링한 데이터 mysql에 저장하기
'''
conn = pymysql.connect(host = host, user = user, password= password, db= db, charset='utf8')
try:
    ck.array_to_db(conn, ck.game_info_array, 'today_game_info')  
    ck.array_to_db(conn, ck.team_game_info_array, 'today_team_game_info')
    ck.array_to_db(conn, ck.lineup_array, 'today_lineup')
    
    conn.commit()
    print(f"Success insert start_game_data")
except pymysql.InterfaceError as e:
    conn.rollback()
    print(f"InterfaceError 발생: {e}")
except Exception as e:
    print(f"다른예외처리: {e}")
    
finally:
    conn.close()









