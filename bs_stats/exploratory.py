#%%
import sys
sys.path.append('D:\\BaseballProject\\python')

#%%
import numpy as np
import pandas as pd
import math
import scipy
import scipy.stats as stats


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from bs_database import base as bs
from bs_stats import preprocess as pr
from bs_stats import sample as sp

#%%        
# data dictionary 기본 세팅
d = bs.Database()
b = pr.Preprocess() 
s = sp.Sample()

#%%
# data dictionary 기본 세팅

b.game_info_array = d.game_info_array
b.batter_array = d.batter_array
b.pitcher_array = d.pitcher_array
b.score_array = d.score_array

b.set_dic_all()
b.set_toto_dic()

#%%
def kde_graph():

    '''
    1. 득점의 분포
    
    1-1. KDE로 추정한 득점의 분포와 MOM,MLE를 통해 추정한 모수 alpha, beta를 따르는 감마분포와 비교 
    
    '''

    old_run = [-1]
    for year in range(2017,2021):
        for team_num in range(1,11):
            new_run = b.toto_dic[year][team_num][:,6].reshape(-1,1)
            if old_run[0] == -1:
                old_run = new_run
            else:
                old_run = np.vstack([old_run,new_run])
    
    real_run = old_run.reshape(-1)
    
    # 그래프를 그리기 위한 MOM, MLE 모수 추정
    s.set_basic(max_loop = 1000000, size = 10000) # set basic parameter
    
    mod_run = s.mod_array_zero_to_num(real_run,0.1) # 0을 0.1로 변환 
    s.set_data(mod_run)
    #s.start_theta = s.mom_gamma_theta # MLE추정을 위한 theta시작점을 MOM추정치로 설정
    s.start_theta = [1,1] # MLE추정을 위한 theta 시작점을 임의의 점으로 설
    mle_theta = s.fit(dist = 'gamma')[-1,1:]
    mom_run = s.gamma_sample(theta = s.mom_gamma_theta)
    mle_run = s.gamma_sample(theta = mle_theta)
    norm_run = s.norm_sample(theta = [s.data_mean,s.data_var])
    print(mle_theta,s.mom_gamma_theta)
    # KDE 그리기
    plt.figure(figsize = (16,8))
    plt.ylim((0,0.2))
    label_patches = []
    run_list = [mod_run,mom_run,mle_run,norm_run]
    name_list = ['Run', 'MOM_Run','MLE_Run','Norm_run']
    color_list = ['Red','Blue','Green','Gray']
    for data, name, color in zip(run_list,name_list,color_list):
        
        sns.kdeplot(data,shade = True,label = name,color = color )
        label_patch = mpatches.Patch(label = name,color = color)
        label_patches.append(label_patch)
        
    plt.xlabel('Run',fontdict = {'fontsize':20})
    plt.ylabel('Density',fontdict = {'fontsize':20})
    plt.title('KDE of GammaDistritution',fontdict = {'fontsize':30})
    plt.legend(handles = label_patches)
    plt.xlim((-10,30))
    
kde_graph()
#%%

def compare_win_rate():
    '''
    1-2. KBO팀 승률 비교
    
    result : dic = 2017 : [MLE, MOM, pita(1.83), pita(2)]
                   2018 : [MLE, MOM, pita(1.83), pita(2)]
                   2019 : [MLE, MOM, pita(1.83), pita(2)]
    '''
     
    s.set_basic(max_loop = 10000, size = 100000) # set basic parameter
    s.start_theta = s.mom_gamma_theta # set start_theta to get mle
    compare_dic = {2017:[0]*11, 2018:[0]*11, 2019:[0]*11}
    for year in toto_dic:
        
        for i,record in enumerate(toto_dic[year]):
            if i == 0: 
                pass
            else:
                
                team_run = record[:,6]
                foe_run = record[:,7]
                '''
                team_mle_run = s.mod_array_zero_to_num(team_run,1) # 0을 1로 변환 
                s.set_data(team_mle_run)
                theta = s.fit(dist = 'gamma')[-1,1:]
                team_mle_sample = s.gamma_sample(theta)
                
                
                
                
                foe_mle_run = s.mod_array_zero_to_num(foe_run,1)
                s.set_data(foe_mle_run)
                theta = s.fit(dist = 'gamma')[-1,1:]
                foe_mle_sample = s.gamma_sample(theta)
                '''
                team_mean = np.mean(team_run)
                team_var = np.var(team_run)
                team_beta = team_var/team_mean
                team_alpha = team_mean / team_beta
                team_mom_sample = s.gamma_sample([team_alpha, team_beta])
                
                foe_mean = np.mean(foe_run)
                foe_var = np.var(foe_run)
                foe_beta = foe_var/foe_mean
                foe_alpha = foe_mean / foe_beta
                foe_mom_sample = s.gamma_sample([foe_alpha, foe_beta])
                
                '''
                team_mod_beta = 2.5
                team_mod_alpha = team_mean / team_mod_beta
                team_mod_sample = s.gamma_sample([team_mod_alpha,team_mod_beta])
                
                foe_mod_beta = 2.5
                foe_mod_alpha = foe_mean / foe_mod_beta
                foe_mod_sample = s.gamma_sample([foe_mod_alpha,foe_mod_beta])
                '''
                team_mod_sample = s.norm_sample(theta = [team_mean,np.sqrt(team_var)])
                foe_mod_sample = s.norm_sample(theta = [foe_mean,np.sqrt(foe_var)])
                real_win = sum(team_run > foe_run)
                total_game = 144 - sum(team_run == foe_run)
                
                mom_win = round((sum(team_mom_sample > foe_mom_sample) / s.size)*total_game)
                mod_win = round((sum(team_mod_sample > foe_mod_sample) / s.size)*total_game)
                
                pita_win = round((math.pow(sum(team_run),2) / (math.pow(sum(foe_run),2) + math.pow(sum(team_run),2)))*total_game)
                total_list = [real_win, mom_win, mod_win, pita_win,total_game,abs(real_win - mom_win),abs(real_win - mod_win), abs(real_win - pita_win)]
                compare_dic[year][i] = total_list
                
    return compare_dic

rate_dic = compare_win_rate()
#%%
old_array = np.zeros((1,8))
for i in range(1,11):
    new_array = rate_dic[2019][i]
    old_array = np.vstack([old_array,new_array])
old_array= old_array[1:]
sum_array = np.sum(old_array,axis = 0)
#%%
def flow_run_result(range_list):
    '''
    2. 타격의 흐름
    
    최근 N경기 득점과 승률의 흐름을 나타내는 함수
    
    range_list = 최근 N경기 list
    
    
    result : list(1 x 2) = [total_run_list, total_result_list] 
    '''
    
    range_list = range_list
    
    record_dic = toto_dic[2017]
    total_run_list = [0]
    total_result_list = [0]
    
    
    for team_num in range(1,11):
        record = record_dic[team_num]
        run_array = record[:,6]
        lose_array = record[:,7]
             
        team_run_list = list()
        team_result_list = list()
        for n in range_list:
            new_run = list()
            new_result = list()
            for j in range(144):
                if j < n-1: # 누적 경기수가 N경기 이하일 떄
                    new_run.append(np.sum(run_array[:j+1]) / (j+1))
                    new_result.append(len(record[:j+1,:][run_array[:j+1]>lose_array[:j+1],:]) / (j+1))
                else: # 누적 경기수가 N경기 이상일 떄
                    
                    new_run.append(np.sum(run_array[j-n+1:j+1]) / n)
                    '''
                    new_run.append(np.mean(b.epa(run_array[j-n+1:j+1])))
                    '''
                    new_result.append(len(record[j-n+1:j+1][run_array[j-n+1:j+1]>lose_array[j-n+1:j+1],:]) / n)
            team_run_list.append(new_run)
            team_result_list.append(new_result)
        total_run_list.append(team_run_list)
        total_result_list.append(team_result_list)
    return [total_run_list, total_result_list]

range_list = [1,5,15,30,100]
flow_run_result_list = flow_run_result(range_list) 
total_run_list = flow_run_result_list[0]
total_result_list = flow_run_result_list[1]

#%%
def show_flow_run(team_num):
    '''
    team의 최근 N경기 득점의 흐름을 나타내는 그래프
    
    result : plt.show()
    '''
    for i,flow_range in enumerate(range_list):
        plt.figure(figsize = (16,8))
        plt.ylim((0,10))
        plt.xlabel("Game" ,fontdict = {"fontsize": 30})
        plt.ylabel("Run" ,fontdict = {"fontsize": 30})
        plt.title("Recent {} Games".format(flow_range),fontdict = {"fontsize":40})
        plt.plot(total_run_list[team_num][i])
        '''
        plt.savefig(b.address + 'run' + str(flow_range) + '.png',format = 'png')
        '''
show_flow_run(1)
#%%
def show_flow_result(team_num):
    '''
    최근 N경기 승률의 흐름을 나타내는 그래프
    
    result : plt(show)
    '''
    for i,flow_range in enumerate(range_list):
        plt.figure(figsize = (16,8))
        plt.ylim((0,1))
        plt.xlabel("Game",fontdict = {"fontsize": 30})
        plt.ylabel("Win Rate",fontdict = {"fontsize": 30})
        plt.title("Recent {} Games".format(flow_range),fontdict = {"fontsize":40})
        plt.plot(total_result_list[team_num][i])
        plt.savefig(b.address + 'winrate' + str(flow_range) + '.png',format = 'png')
        
show_flow_result(3)
#%%


def home_away(year):
    '''
    3. 홈/원정 & 파크팩터
    3-1. 홈/원정
    
    result : array(10 x 6) = [홈 득점, 홈 실점, 원정 득점, 원정 실점, 홈승, 원정 승]
    
    '''
    toto_data = toto_dic[year]
    old_array = np.zeros((1,6))
    for i in range(1,11):
        team_data = toto_data[i]
        home_data = team_data[team_data[:,3]=='홈']
        away_data = data = team_data[team_data[:,3]=='원정']
        home_run = home_data[:,6]
        home_lose = home_data[:,7]
        away_run = away_data[:,6]
        away_lose = away_data[:,7]
        
        home_win = len(home_data[home_run>home_lose])
        away_win = len(away_data[away_run>away_lose])
        
        new_array = np.array([np.sum(home_run), np.sum(home_lose), np.sum(away_run), np.sum(away_lose),home_win,away_win]).reshape(1,-1)
        
            
        
        old_array = np.vstack([old_array,new_array])
        
    old_array = old_array[1:,:]
    return old_array

home_away_array = home_away(2017)
for year in [2018,2019]:
    home_away_array = home_away_array + home_away(year)
#%%
def win_by_enemy(toto_dic):
    '''
    3. 홈/원정 & 파크팩터
    3-1. 홈/원정
    
    result : array(10 x 6) = [홈 득점, 홈 실점, 원정 득점, 원정 실점, 홈승, 원정 승]
    
    '''
    new_dic = dict()
    for year in [2017,2018,2019,2020]:
        toto_data = toto_dic[year]
        
        for team_num in range(1,11):
            total_data = toto_data[team_num]
            
            
            win = len(total_data[total_data[:,6]>total_data[:,7]])
            old_array = np.array([0,round(win/144,3)]).reshape(1,-1)
            
            for foe_num in range(1,11):
                if team_num == foe_num:
                    continue
                team_data = total_data[total_data[:,11]==foe_num,:]
                
                run = np.sum(team_data[:,6])
                lose = np.sum(team_data[:,7])
                
                win = len(team_data[(team_data[:,6]>team_data[:,7]),:])
            
                new_array = np.array([foe_num, round(win/16,3)]).reshape(1,-1)
                old_array = np.vstack([old_array,new_array])
                
            try:
                new_dic[team_num] = np.hstack([new_dic[team_num],old_array])
                
            except:
                new_dic[team_num] = old_array
    
        
    
    return new_dic

enemy_array = win_by_enemy(b.toto_dic)

z = pd.DataFrame(enemy_array[1])
#%%
z = pd.DataFrame(b.toto_dic[2017][1])
#%%
ga = b.game_info_array
#%%
b.score_array[0]
#%%
stadium_list = [0, '잠실','사직','광주','대구','잠실','대전','문학','고척','창원','수원']
pf_list = [0]
for tn in range(1,11):
    
    stadium = stadium_list[tn]
    
    team_gn_home = ga[(ga[:,3]<2021)&(ga[:,4]==tn)&(ga[:,7]=='home')&(ga[:,8]==stadium),2]
    foe_gn_home = ga[(ga[:,3]<2021)&(ga[:,5]==tn)&(ga[:,7]=='away')&(ga[:,8]==stadium),2]
    team_gn_away = ga[(ga[:,3]<2021)&(ga[:,4]==tn)&(ga[:,7]=='away'),2]
    foe_gn_away = ga[(ga[:,3]<2021)&(ga[:,5]==tn)&(ga[:,7]=='home'),2]
    
    
    def get_score(team_game_idx):
        run_sum=0
        pa_sum=0
        for tgi in team_game_idx:
            
            sa = b.score_array[(b.score_array[:,1]==tgi),:][0]
            
            run = sa[-4]
            run_sum+=run
            
            pa = sa[9:21]
            pa = len(pa[(pa=='-')])
            pa_sum+=12 - pa
    
        return [run_sum, pa_sum]
    
    
    tgh = get_score(team_gn_home)
    fgh = get_score(foe_gn_home)
    tga = get_score(team_gn_away)
    fga = get_score(foe_gn_away)
    
    hr = (tgh[0] + fgh[0])/(tgh[1]+fgh[1]) 
    ar =(tga[0] + fga[0])/(tga[1]+fga[1])
    pf = hr / ((1/10)*hr + (9/10)*ar)
    pf_list.append(pf)
#%%
ipf = (sum(pf_list)/10)
lpf_list = np.round(np.divide(pf_list,ipf),3)
#%%
lpf_list

#%%
def park_factor():
    '''
    3. 홈/원정 & 파크팩터
    3-2. 파크팩터
    
    
    '''
    toto_data = toto_dic[2017] 
    
    return
#%%

#%%

def toto_basic(year_list):
    for year in year_list:
        toto_list = list()
        for team_num in range(1,11):
            data_array = toto_dic[year][team_num]
            win_rate_array = data_array[:,8]
            team_run = data_array[:,6]
            foe_run = data_array[:,7]
            game_length = len(data_array[(win_rate_array!=0.5),0])

            win_array = data_array[(win_rate_array>0.5) & (team_run>foe_run),0]
            loss_array = data_array[(win_rate_array<0.5) & (team_run<foe_run),0]
            toto_rate = round((len(win_array) + len(loss_array)) / game_length,3)
            
            toto_list.append(toto_rate)
    return toto_list

toto_basic([2017])
#%%

#%%
def toto_result(year_list,toto_dic):
    '''
    4. 도박사
    
    도박사들의 예측결과 나타내는 함수
    
    year_list = 년도 list
    
    result(list) = [win_array, game_array] - 승률구간별 실제 승수 / 게임수
    '''
    win_array = np.zeros((10,20))
    game_array = np.zeros((10,20))
    
    for year in year_list:
        for team_num in range(1,11):
            data_array = toto_dic[year][team_num]
            data_array = data_array[data_array[:,3] =="홈",:]
            win_rate_array = data_array[:,8]
            
            # 10개의 구간([0-10) ~ [90-100))
            for rate in range(20):
                rate_idx = rate * 0.05
                in_rate_array = data_array[(win_rate_array >= rate_idx) & (win_rate_array < rate_idx+0.05),:]
                game_num = len(in_rate_array)
                win_num = len(in_rate_array[(in_rate_array[:,6] > in_rate_array[:,7]),:])
                game_array[team_num-1,rate]+=game_num
                win_array[team_num-1,rate]+=win_num
                
    return [win_array,game_array]

t_array = np.zeros((1,20))
w_array = np.zeros((1,20))

win_rate_list_byteam =list()
for year in [2017,2018,2019]:
    toto_result_list = toto_result([year],b.toto_dic)
    win_array = toto_result_list[0]
    game_array = toto_result_list[1]
    win_rate_array = np.round(np.divide(win_array,game_array),3)
    
    '''
    win_total_array = np.sum(win_array,axis = 0)
    game_total_array = np.sum(game_array,axis = 0)
    rate_total_array = np.divide(win_total_array,game_total_array).reshape(1,-1)
    '''
    t_array = np.vstack([t_array,game_total_array])
    w_array = np.vstack([w_array,win_total_array])
    
#%%
def toto_rate(year_list,toto_dic):
    '''
    4. 도박사
    
    도박사들의 예측결과 나타내는 함수
    
    year_list = 년도 list
    
    result(list) = [win_array, game_array] - 승률구간별 실제 승수 / 게임수
    '''
    win_array = np.zeros((10,20))
    game_array = np.zeros((10,20))
    rate_list = [0]
    win_rate_list=[0]
    for year in year_list:
        for team_num in range(1,11):
            toto_array = toto_dic[year][team_num]
            '''
            toto_array = toto_array[toto_array[:,3] =="홈",:]
            '''
            win_rate_array = toto_array[:,8]
            result_array1 = np.where(toto_array[:,6] > toto_array[:,7],1,0)
            result_array2 = np.where(toto_array[:,6] == toto_array[:,7],0.5,0)
            result_array = result_array1 + result_array2
            rate_list.append(np.round(np.sum(result_array  - win_rate_array) /144,3))
            win_rate_list.append(np.round(np.sum(win_rate_array)))
            
            # 10개의 구간([0-10) ~ [90-100))
            for rate in range(20):
                rate_idx = rate * 0.05
                in_rate_array = toto_array[(win_rate_array >= rate_idx) & (win_rate_array < rate_idx+0.05),:]
                game_num = len(in_rate_array)
                win_num = len(in_rate_array[(in_rate_array[:,6] > in_rate_array[:,7]),:])
                game_array[team_num-1,rate]+=game_num
                win_array[team_num-1,rate]+=win_num
                
    return [win_array,game_array,rate_list,win_rate_list]

#%%
    
z = toto_rate([2019],b.toto_dic)
zz = pd.DataFrame(n_array)
#%%

#%%
win_array = np.zeros((10,20))
game_array = np.zeros((10,20))
rate_list = [0]
win_rate_list=[0]
year_list=[2017]
toto_dic = b.toto_dic
total_rate_list = [0]
for year in year_list:
    new_list = list()
    for team_num in range(1,11):
        toto_array = toto_dic[year][team_num]
        '''
        toto_array = toto_array[toto_array[:,3] =="홈",:]
        '''
        win_rate_array = toto_array[:,8]
        result_array1 = np.where(toto_array[:,6] > toto_array[:,7],1,0)
        result_array2 = np.where(toto_array[:,6] == toto_array[:,7],0.5,0)
        result_array = result_array1 + result_array2
        rate_list.append(np.round(np.sum(result_array  - win_rate_array) /144,3))
        total_rate_list.append(list(result_array  - win_rate_array))
        win_rate_list.append(np.round(np.sum(win_rate_array)))
        
new_list = [0]
for team_num in range(1,11):
    nn = total_rate_list[team_num]
    nnn = list()
    for i in range(144):
        new_sum = sum(nn[:i+1])
        nnn.append(new_sum)
        
    new_list.append(nnn)
    
#%%
plt.plot(new_list[7])
    
#%%    
from random import *
#%%
aa = list()
aaa = list()
for i in range(144):
    aa.append(uniform(-0.7,0.7))
    aaa.append(sum(aa))
plt.plot(aaa)
#%%