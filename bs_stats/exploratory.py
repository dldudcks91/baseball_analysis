#%%
import sys
sys.path.append('D:\\BaseballProject\\python')


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
from bs_personal import personal_code as cd
      
# data dictionary 기본 세팅
d = bs.Database()

b = pr.Preprocess() 
s = sp.Sample()


# data dictionary 기본 세팅
d.load_data_all(db_address = cd.db_address ,code = cd.local_code , file_address = cd.file_address)

b.game_info_array = d.game_info_array
b.batter_array = d.batter_array
b.pitcher_array = d.pitcher_array
b.score_array = d.score_array

b.set_dic_all()
#%%
b.set_toto_dic()
#%%


#%%
def kde_graph():

    '''
    1. 득점의 분포
    
    1-1. KDE로 추정한 득점의 분포와 MOM,MLE를 통해 추정한 모수 alpha, beta를 따르는 감마분포와 비교 
    
    '''

    old_run = [-1]
    for year in range(2017,2022):
        for team_num in range(1,11):
            new_run = b.toto_dic[year][team_num][:,6].reshape(-1,1)
            if old_run[0] == -1:
                old_run = new_run
            else:
                old_run = np.vstack([old_run,new_run])
    
    real_run = old_run.reshape(-1)
    
    # 그래프를 그리기 위한 MOM, MLE 모수 추정
    s.set_basic(max_loop = 1000000, size = 10000) # set basic parameter
    
    mod_run = s.mod_array_zero_to_num(real_run,0.5) # 0을 0.1로 변환 
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
        
        sns.kdeplot(data,shade = True,label = name,color = color)
        label_patch = mpatches.Patch(label = name,color = color)
        label_patches.append(label_patch)
        
    plt.xlabel('Run',fontdict = {'fontsize':20})
    plt.ylabel('Density',fontdict = {'fontsize':20})
    plt.title('KDE of GammaDistribution',fontdict = {'fontsize':30})
    plt.legend(handles = label_patches)
    plt.xlim((-10,30))
    
kde_graph()
#%%
# alpha, beta 에 따른 분포 비교
s.size = 100000
plt.figure(figsize = (16,8))
plt.ylim((0,0.2))
label_patches = []
home_run = s.gamma_sample(theta = [2,3])
away_run = s.gamma_sample(theta = [3,2])
run_list = [home_run,away_run]
name_list = ['Home', 'Away']
color_list = ['Blue','Green']
for data, name, color in zip(run_list,name_list,color_list):
    
    sns.kdeplot(data,shade = True,label = name,color = color )
    label_patch = mpatches.Patch(label = name,color = color)
    label_patches.append(label_patch)
    
plt.xlabel('Run',fontdict = {'fontsize':20})
plt.ylabel('Density',fontdict = {'fontsize':20})
plt.title('KDE of GammaDistribution',fontdict = {'fontsize':30})
plt.legend(handles = label_patches)
plt.xlim((-10,30))

#%%
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

team_num = 1
plt.figure(figsize = (16,8))
old_array = np.array(1)
for year in range(2017,2022):
    run = b.toto_dic[year][team_num][:,6].astype(np.float)
    old_array = np.hstack([old_array, run])
lose = b.toto_dic[year][team_num][:,7].astype(np.float)
run_array = old_array[1:]


beta = np.var(run_array) / np.mean(run_array) 
alpha = np.mean(run_array)/beta
s.size = 144*5
#run_array = s.gamma_sample([alpha,beta])

for i in range(0,6):
    plt.axvline(x = i*144, color = 'orange')
plt.title('Run of Games', fontdict = {'fontsize':24})
plt.xlabel('Games',fontdict = {'fontsize':20})
plt.ylabel('Run',fontdict = {'fontsize':20})
plt.plot(run_array)

#%%
s.size = 144
new_sample = s.norm_sample(theta=[0,1])
run = b.toto_dic[2017][1][:,6].astype(np.float)
x_list = list()
for i in range(144):
    x_list.append(run[i] - new_sample[i]) 
    
#%%

#%%
    run_array = b.toto_dic[2017][1][:,6]
plot_acf(run_array, lags=20, alpha=0.05)
plot_pacf(run_array, lags=20, alpha=0.05)
#%%
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.figure(figsize = (20,8))
for team_num in range(1,11):
    score_array = b.score_dic[2017][team_num]
    ground_array = score_array[:,7]
    run_array = score_array[:,-4]
    pf_list = list()
    for ground in ground_array:
        try:
            pf = b.park_factor_total[ground]
            pf_list.append(1/pf)
            
        except:
            pf_list.append(1)
    pf_run_array = run_array * pf_list
    
    
    fig = plot_pacf(run_array, lags=20, alpha=0.05)
    plt.show()
    
#%%
import statsmodels.api as sm
for year in range(2017,2022):
    for team_num in range(1,11):
        score_array = b.score_dic[year][team_num]
        ground_array = score_array[:,7]
        run_array = score_array[:,-4]
        pf_list = list()
        for ground in ground_array:
            try:
                pf = b.park_factor_total[ground]
                pf_list.append(1/pf)
                
            except:
                pf_list.append(1)
        pf_run_array = run_array * pf_list
        pf_run_array = pf_run_array.reshape(-1,1)
        if team_num ==1:
            old_array = pf_run_array
        else:
            old_array = np.hstack([old_array,pf_run_array])
            
    new_array = np.mean(old_array,axis = 1).reshape(-1,1)
    if year == 2017:
        result_array = new_array
    else:
        result_array = np.hstack([result_array,new_array])
result_array = np.mean(result_array,axis =1)    
plt.plot(result_array)
X = np.arange(1,145)
X = sm.add_constant(X)
models = sm.OLS(result_array.astype(np.float),X).fit()

#%%
models.summary()
    #%%
def compare_win_rate():
    '''
    1-2. KBO팀 승률 비교
    
    result : dic = 2017 : [MLE, MOM, pita(1.83), pita(2)]
                   2018 : [MLE, MOM, pita(1.83), pita(2)]
                   2019 : [MLE, MOM, pita(1.83), pita(2)]
    '''
    toto_dic = b.toto_dic
    s.set_basic(max_loop = 10000, size = 10000) # set basic parameter
    s.start_theta = s.mom_gamma_theta # set start_theta to get mle
    compare_dic = {2017:[0]*11, 2018:[0]*11, 2019:[0]*11, 2020:[0]*11, 2021:[0]*11,2022:[0]*11}
    for year in compare_dic:
        
        for i,record in enumerate(b.score_dic[year]):
            if i == 0: 
                pass
            else:
                if year !=2022:
                    game_num = 144
                else:
                    conn  = pymysql.connect(host = cd.local_host, user = cd.local_user, password = cd.local_code , db = cd.db, charset = cd.charset)    
                    b.set_last_game_num_list(2022,conn)
                    game_num = b.last_game_num_list[i]
                
                
                record = record[:game_num]
                team_run = record[:,-4]
                
                foe_run = b.score_array[(b.score_array[:,2]==year) & (b.score_array[:,4] == i),:]
                foe_run = foe_run[foe_run[:,0].argsort(),-4]
                
                
                #mom
                
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
                # mle
                team_mle_run = s.mod_array_zero_to_num(team_run,1) # 0을 1로 변환 
                s.set_data(team_mle_run)
                theta = s.fit(dist = 'gamma')[-1,1:]
                team_mle_sample = s.gamma_sample(theta)
                foe_mle_run = s.mod_array_zero_to_num(foe_run,1)
                s.set_data(foe_mle_run)
                theta = s.fit(dist = 'gamma')[-1,1:]
                foe_mle_sample = s.gamma_sample(theta)
                
                
                #map
                
                s.alpha_theta = [2,4]
                s.beta_theta = [2.5,4]
                
                team_mle_run = s.mod_array_zero_to_num(team_run,1)
                s.set_data(team_mle_run)
                s.start_theta = [team_alpha, team_beta]
                theta = s.fit(dist = 'map')[-1,1:]
                team_mle_sample = s.gamma_sample(theta)
                
                foe_mle_run = s.mod_array_zero_to_num(foe_run,1)
                s.set_data(foe_mle_run)
                s.start_theta = [foe_alpha, foe_beta]
                theta = s.fit(dist = 'map')[-1,1:]
                foe_mle_sample = s.gamma_sample(theta)
                '''
                
                
                '''
                team_mod_beta = 2.5
                team_mod_alpha = team_mean / team_mod_beta
                team_mod_sample = s.gamma_sample([team_mod_alpha,team_mod_beta])
                
                foe_mod_beta = 2.5
                foe_mod_alpha = foe_mean / foe_mod_beta
                foe_mod_sample = s.gamma_sample([foe_mod_alpha,foe_mod_beta])
                
                team_mod_sample = s.norm_sample(theta = [team_mean,np.sqrt(team_var)])
                foe_mod_sample = s.norm_sample(theta = [foe_mean,np.sqrt(foe_var)])
                '''
                real_win = sum(team_run > foe_run)
                total_game_num = game_num - sum(team_run == foe_run)
                print(game_num, sum(team_run == foe_run))
                
                mom_win = round((sum(team_mom_sample > foe_mom_sample) / s.size)*total_game_num)
                #mle_win = round((sum(team_mle_sample > foe_mle_sample) / s.size)*total_game_num)
                #mod_win = round((sum(team_mod_sample > foe_mod_sample) / s.size)*total_game)
                pita_win = round((math.pow(sum(team_run),1.8) / (math.pow(sum(foe_run),1.8) + math.pow(sum(team_run),1.8)))*total_game_num)
                
                mom_mse = round(np.sqrt((real_win - mom_win)**2),2)
                #mle_mse = round(np.sqrt((real_win - mle_win)**2),2)
                pita_mse = round(np.sqrt((real_win - pita_win)**2),2)
                total_list = [real_win, mom_win, pita_win,total_game_num, mom_mse,pita_mse, round(team_alpha,2), round(team_beta,2)]
                compare_dic[year][i] = total_list
                
    return compare_dic

rate_dic = compare_win_rate()
#%%
year_array = np.zeros((1,10))
for year in range(2017,2023):
    print(year)
    old_array = np.zeros((1,8))
    for i in range(1,11):
        new_array = rate_dic[year][i]
        old_array = np.vstack([old_array,new_array])
    
    
#    old_array= old_array[1:]
       
    sum_array = np.sum(old_array,axis = 0)
    #year_array = np.vstack([year_array,old_array])
    year_array = old_array
    for team_num in range(1,11):
        
        print( year_array[team_num], b.team_list[team_num])
    year_array = year_array[1:]
    print(np.round(np.mean(year_array,axis = 0),1))
    print(np.round(np.std(year_array,axis=0),1))
    print("")
'''
year_array = year_array[1:]

'''
#%%
'''
pita예측 수렴속도 알아보는 차트
'''
year = 2022
team_num = 3

total_team_run = b.score_dic[year][team_num][:,-4]
foe_run = b.score_array[(b.score_array[:,2]==year) & (b.score_array[:,4] == team_num),:]
total_foe_run = foe_run[foe_run[:,0].argsort(),-4]

real_list = list()
pita_list = list()
max_i = len(total_team_run)
for i in range(10,max_i+1):
    game_num = i
    start_idx = 0
    if i >=10:
        start_idx = i - 10
    team_run = total_team_run[start_idx:i]
    foe_run = total_foe_run[start_idx:i]
    
    
    
    real_win = sum(team_run > foe_run)
    
    total_game_num = game_num - start_idx - sum(team_run == foe_run)
    
    pita_win = round((math.pow(sum(team_run),1.8) / (math.pow(sum(foe_run),1.8) + math.pow(sum(team_run),1.8)))*total_game_num)
    #real_list.append(np.round(real_win/total_game_num,3) - np.round(pita_win/total_game_num,3))
    real_list.append(np.round(real_win/total_game_num,3))
    pita_list.append(np.round(pita_win/total_game_num,3))

plt.plot(real_list)
plt.plot(pita_list)

                
                
#%%

#%%
'''
연승확률
'''
k_list = list()
for k in range(10,91):
    max_list =list()
    for i in range(10000):
        max_num = 0
        last_num = -1
        new_max = 0
        for j in range(144):
            new = np.random.rand(1)
            if new <=k*0.01:
                new_num = 0
            else:
                new_num = 1
                
            if new_num == last_num:
                    new_max +=1
                    
                
                    
            else:
                
                new_max = 0
            if max_num < new_max:
                if new_num ==0:
                    max_num = new_max    
                
            last_num = new_num
        max_list.append(max_num)
    result = np.round(np.mean(max_list),1)

    k_list.append(result)
    print(result,k)
#%%
    plt.figure(figsize = (9,6))
plt.title('Max Continuous number by Probability',fontdict = {'fontsize':15})
plt.plot([i*0.01 for i in range(10,91)],k_list)
    
    #%%
k = 50 
max_list =list()
for i in range(10000):
    max_num = 0
    last_num = -1
    new_max = 0
    for j in range(144):
        new = np.random.rand(1)
        if new <=k*0.01:
            new_num = 0
        else:
            new_num = 1
            
        if new_num == last_num:
                new_max +=1
                
            
                
        else:
            
            new_max = 0
        if max_num < new_max:
            if new_num ==0:
                max_num = new_max    
            
        last_num = new_num
    max_list.append(max_num)
    #%%
plt.figure(figsize = (9,6))
plt.xlabel('Count',fontdict = {'fontsize':10})
    
plt.title('Max Continuous number of Heads',fontdict = {'fontsize':15})
plt.xticks(ticks=[i for i in range(16)])
plt.xlim((0,15))
sns.histplot(max_list,stat = 'probability',binwidth=1)



#%%
'''
map 계
'''
team_run = b.score_dic[2017][1][:30,-4]
team_run = s.mod_array_zero_to_num(team_run,1)
s.set_data(team_run)

team_mean = np.mean(team_run)
team_var = np.var(team_run)
team_beta = team_var / team_mean
team_alpha = team_mean / team_beta
team_theta = [team_alpha, team_beta]
print(team_theta, team_alpha*team_beta)
s.alpha_theta = [2,4]
s.beta_theta = [2.5,4]


s.start_theta = team_theta
theta = s.fit(dist = 'map')
            
theta = theta[-1,1:]
print(theta, theta[0]* theta[1])


#%%
b.park_factor_list
#%%
label_patches = [0,0]
label_patches[0] = mpatches.Patch(label = 'alpha',color = 'blue')
label_patches[1] = mpatches.Patch(label = 'beta',color = 'orange')
plt.legend(handles = label_patches)
sns.kdeplot(year_array[:,-2])
sns.kdeplot(year_array[:,-1])




#%%
def flow_run_result(range_list):
    '''
    2. 타격의 흐름
    
    최근 N경기 득점과 승률의 흐름을 나타내는 함수
    
    range_list = 최근 N경기 list
    
    
    result : list(1 x 2) = [total_run_list, total_result_list] 
    '''
    toto_dic = b.toto_dic
    range_list = range_list
    
    
    result_dic = dict()
    result_dic['run'] = dict()
    result_dic['result'] = dict()
    
    for year in range(2017,2022):
        record_dic = toto_dic[year]
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
            
       
        
        
        result_dic['run'][year] = total_run_list
        result_dic['result'][year] = total_result_list
       
        
    return result_dic
range_list = [5,20,30,100]
flow_dic = flow_run_result(range_list)

#%%
'''
team의 최근 N경기 득점 및 승률 그래프 그리기
'''
team_list = [0, 'LG', 'LOTTE', 'KIA', 'SAMSUNG', 'DOOSAN', 'HANHWA', 'SSG', 'KIWOOM', 'NC', 'KT']
year = 2021
for team_num in range(1,11):
    
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (8, 4)
    plt.rcParams['font.size'] = 12
    
    # 2. 데이터 준비
    x = np.arange(11,145)
    
    y1 = flow_dic['run'][year][team_num][1][10:]
    y2 = flow_dic['result'][year][team_num][1][10:]
    
    # 3. 그래프 그리기
    patch_list = list()
    fig, ax1 = plt.subplots()
    
    ax1_color = 'green'
    ax1.plot(x, y1,color=ax1_color, alpha=0.5, label='Price')
    ax1.set_title(str(team_list[team_num]))
    ax1.set_ylim(0, 10)
    ax1.set_xlabel('games')
    ax1.set_ylabel('Run')
    ax1.tick_params(axis='both', direction='in')
    patch_list.append(mpatches.Patch(label = 'Run',color = ax1_color, alpha = 0.5))
    
    ax2 = ax1.twinx()
    ax2_color = 'blue'
    ax2.plot(x, y2, color=ax2_color, label='Rate', alpha=0.5)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Rate')
    ax2.tick_params(axis='y', direction='in')
    patch_list.append(mpatches.Patch(label = 'Rate',color = ax2_color,alpha = 0.5))
    
    plt.legend(handles = patch_list)
    save_address = b.address + team_list[team_num] + '.png'
    plt.savefig(save_address)
    plt.show()
#%%


#%%
toto_dic = b.toto_dic
def home_away(year):
    '''
    3. 홈/원정 & 파크팩터
    3-1. 파크팩터 구하기 최신 220725
    
    result : array(10 x 6) = [홈 득점, 홈 실점, 원정 득점, 원정 실점, 홈승, 원정 승]
    
    '''
    toto_data = toto_dic[year]
    old_array = np.zeros((1,8))
    for i in range(1,11):
        team_data = toto_data[i]
        home_data = team_data[team_data[:,3]=='홈']
        away_data = data = team_data[team_data[:,3]=='원정']
        home_run = home_data[:,6]
        home_lose = home_data[:,7]
        away_run = away_data[:,6]
        away_lose = away_data[:,7]
        
        home_win = len(home_data[home_run>home_lose])
        home_len = 72 - len(home_data[home_run==home_lose])
        away_win = len(away_data[away_run>away_lose])
        away_len = 72 - len(away_data[away_run==away_lose])
        
        new_array = np.array([np.sum(home_run), np.sum(home_lose), np.sum(away_run), np.sum(away_lose),home_win,home_len,away_win,away_len]).reshape(1,-1)
        
            
        
        old_array = np.vstack([old_array,new_array])
        
    old_array = old_array[1:,:]
    return old_array
#%%
'''
파크팩터 년도별 구하기
'''
old_array = np.zeros((10,1))
total_array = np.zeros((10,4))
for year in range(2017,2022):
    new_array = home_away(year)
    total_array+=new_array[:,:4]
    def get_park_factor(new_array):
        home = (new_array[:,0] + new_array[:,1])
        away = (new_array[:,2] + new_array[:,3])
        new_array = np.round(home / (1/8*home + 7/8*away),3).reshape(10,1)
        return new_array
    new_array = get_park_factor(new_array)
    old_array = np.hstack([old_array,new_array])
total_array = get_park_factor(total_array)
park_factor_array = np.hstack([old_array,total_array])[:,1:]

park_factor_array *=1000

pf_pd = pd.DataFrame(park_factor_array)
pf_pd.columns = [i for i in range(2017,2022)] + ['Total']

df = pf_pd
plt.figure(figsize=(10, 7))
plt.rcParams['font.family'] = 'NanumGothic'

plt.yticks(rotation = 0)

ax = sns.heatmap(df, annot=True,  cmap='Blues',linewidths=.5, fmt='.0f')
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

ax.xaxis.tick_top()
plt.show()
#%%
'''
승리 오즈비 년도별 구하기
'''
old_array = np.zeros((10,1))
total_array = np.zeros((10,4))
for year in range(2017,2022):
    new_array = home_away(year)
    total_array+=new_array[:,-4:]
    def get_win_odds(new_array):
        win = (new_array[:,-4] / new_array[:,-3])
        lose = (new_array[:,-2] / new_array[:,-1])
        result_array = win / lose
        
        
        return result_array.reshape(10,1)
    new_array = get_win_odds(new_array)
    old_array = np.hstack([old_array,new_array])
total_array = get_win_odds(total_array)
win_odds_array = np.hstack([old_array,total_array])[:,1:]


#%%
def win_by_enemy(toto_dic):
    '''
    3. 홈/원정 & 파크팩터
    3-1. 홈/원정
    
    result : array(10 x 6) = [홈 득점, 홈 실점, 원정 득점, 원정 실점, 홈승, 원정 승]
    
    '''
    new_dic = dict()
    for year in [2017,2018,2019,2020,2021]:
        toto_data = toto_dic[year]
        
        for team_num in range(1,11):
            total_data = toto_data[team_num]
            
            
            win = len(total_data[total_data[:,6]>total_data[:,7]])
            len_game = 144 - len(total_data[total_data[:,6]==total_data[:,7]])
            old_array = np.array([0,round(win/144,3)]).reshape(1,-1)
            
            for foe_num in range(1,11):
                if team_num == foe_num:
                    continue
                team_data = total_data[total_data[:,11]==foe_num,:]
                
                foe_run = np.sum(team_data[:,6])
                foe_lose = np.sum(team_data[:,7])
                
                foe_win = len(team_data[(team_data[:,6]>team_data[:,7]),:])
            
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
    pf = 1-(1 - pf)/2
    pf_list.append(pf)
#%%

score_array = b.score_array
run_array = score_array[:,-4]
ground_array = score_array[:,7]
pf_list = list()
for ground in ground_array:
    pf = b.park_factor_total.get(ground)
    if pf == None:
        pf = 1
    pf_list.append(pf)
s.size = len(run_array)
run_array += s.norm_sample(theta = [0,0.01])
plt.figure(figsize = (16,8))
plt.scatter(pf_list,run_array)
#%%
b.batter_array[0]


batter_array = b.batter_array
old_array = np.zeros((1,22))

for year in range(2017,2022):
    year_array = batter_array[batter_array[:,2]==year]
    for team_num in range(1,11):
        team_array = year_array[year_array[:,3] == team_num]
        for game_num in range(1,145):
            game_array = team_array[team_array[:,5]  == game_num,11:]
            
            new_array = np.sum(game_array,axis = 0)
            new_array = np.hstack([[year],[team_num],[game_num],new_array])
            old_array = np.vstack([old_array,new_array])
            #%%
old_array = old_array[1:]
#%%
b.score_array[0]
#%%
import statsmodels.api as sm
Y = b.score_array[b.score_array[:,2]<2022,-4].astype(np.float)
X = old_array[:,3:-5].astype(np.float)
#X = np.hstack([np.ones(len(X)).reshape(-1,1),X])
models = sm.OLS(Y,X).fit()

models.summary()
            #%%
            len(b.batter_array[0])
#%%
def park_factor():
    '''
    3. 홈/원정 & 파크팩터
    3-2. 파크팩터
    
    
    '''
    toto_data = toto_dic[2017] 
    
    return
#%%
x = [i for i in range(5,55,5)]
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
for year in [2017,2018,2019,2020,2021]:
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
'''
경기 수 에 따른 비교 그래프 그리기
'''
x_list = [i for i in range(1,21)]#[i for i in range(5,101,5)]
y1_list = list()
for new_score in new_score_dic.values():
    y1_list.append(round(new_score[0]/10,4))
    
#%%
y_list = np.hstack([np.array(y1_list).reshape(-1,1),np.array(y2_list).reshape(-1,1)])
#%%
from matplotlib.pyplot import style

style.use('seaborn-whitegrid')
plt.figure(figsize = (6,4))

#plt.ylim((3.66,3.68))
#plt.yticks([i/10000 for i in range(36600,36800,40)])
plt.xticks([i for i in range(0,21,5)])
label_patches = []


name_list = ['Uniform', 'Epa']
color_list = ['C0','C2']
plt.grid(True)
i = 0 
for name, color in zip(name_list,color_list):
    
    y = y_list[:,i]
    label_patch = mpatches.Patch(label = name,color = color)
    label_patches.append(label_patch)
    plt.plot(x_list,y,color = color)
    i+=1
plt.xlabel('recent games N',fontdict = {'fontsize':12})
plt.ylabel('RMSE',fontdict = {'fontsize':12})
plt.title('RMSE of recent Games N',fontdict = {'fontsize':15})
plt.legend(handles = label_patches)

#%%
plt.figure(figsize = (6,4))
y_list = list(total_data_len_dic.values())
x_list = [i for i in range(1,21)]

plt.xlabel('recent games N',fontdict = {'fontsize':12})
plt.ylabel('DataNumber',fontdict = {'fontsize':12})
plt.title('DataNumber of recent Games N',fontdict = {'fontsize':15})

plt.plot(x_list,y_list)
plt.xticks([i for i in range(0,25,5)])