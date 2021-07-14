#%%
import sys
sys.path.append('C:\\Users\\Chan\\Desktop\\BaseballProject\\python')

#%%


from baseball_2021.crawling import crawling
from baseball_2021.crawling import crawling_kbo
#from baseball_2021.crawling import crawling_today_toto
#%%

'''
Crawling_KBO

'''


c = crawling_kbo.Crawling_baseball()
#%%
c.driver_start()
c.date = 20210712
c.driver.get('https://www.koreabaseball.com/Schedule/GameCenter/Main.aspx?gameDate=' + str(c.date))
#c.end_game_crawling()
c.start_game_crawling()
c.driver.close()

#%%
'''
Crawling_toto
'''
c = crawling_today_toto.Crawling_toto()
#%%
c.craw_toto_all()
#c.update_craw_time()
print(c.toto_array)

#%%
