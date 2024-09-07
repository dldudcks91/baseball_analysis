#%%
import sys

import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 불러오기 위치 설정
import sys
sys.path.append('D:\\BaseballProject\\python')

import numpy as np
import pandas as pd
import pymysql
from sqlalchemy import create_engine
#%%
from bs_database import base as db_base
from bs_database import update as db_update
from bs_personal import personal_code as cd

#%%
YEAR = 2024
d = db_update.Update()
conn  = pymysql.connect(host = cd.aws_host, user = cd.aws_user, password = cd.aws_code , db = cd.db)


d.load_data_this_year_new(cd.db_address, cd.aws_code, cd.file_aws_address, YEAR)

win_rate_list = d.get_new_win_rate(d.game_info_array, d.score_array)

d.update_team_info(conn, YEAR, win_rate_list, update_type = 'record')
conn.commit()
conn.close()