#%%



from bs_database import update as db_update

YEAR = 2024
#conn_local = pymysql.connect(host= cd.local_host, user=cd.local_user, password= cd.local_code, db= cd.db, charset='utf8')
#conn = conn_local

conn_aws = pymysql.connect(host = cd.aws_host, user=cd.aws_user, password= cd.aws_code, db= cd.db, charset='utf8')
conn = conn_aws
ck.load_data_all(db_address = cd.db_address ,code = cd.aws_code , file_address = cd.file_aws_address)
ck.set_last_game_num_list(YEAR,conn)

#%%


#%%
d = db_update.Update()
win_rate_list = d.get_new_win_rate(ck.game_info_array, ck.score_array)
#%%
ck.update_team_info(conn, YEAR, ck.last_game_num_list, update_type = 'game_num')
ck.update_team_info(conn, YEAR, win_rate_list, update_type = 'record')
conn.commit()
conn.close()
#%%
z = ck.score_array