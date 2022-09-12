#%%

import pandas as pd
import numpy as np

#%%

data_df = pd.read_csv('C:\\Users\\Chan\\Desktop\\KTX-이음\\People-data.csv',encoding = 'cp949').astype('str')

#%%
data_df['출발일'] = pd.to_datetime(data_df['출발일'])
data_df['출발요일']=data_df['출발일'].dt.day_name()
#%%
new_data_df = data_df.drop_duplicates()

#%%
na = np.array(new_data_df)
#%%
new_array = na[(na[:,6]=='영주')|(na[:,6]=='풍기')|(na[:,6]=='안동')|(na[:,9]=='영주')|(na[:,9]=='풍기')|(na[:,9]=='안동')]

#%%


for j in ['F','M']:
    count = 0
    for i in range(1,8):
        age = i*10
        last_age = i*10 - 10
    
        if i < 7:
            age_mask = (new_array[:,1].astype(np.int) < age) & (new_array[:,1].astype(np.int)>= last_age)
        else: 
            age_mask = (new_array[:,1].astype(np.int) >= last_age)    
        if j == 'F':
            new_count = round(len(new_array[age_mask&(new_array[:,2]==j),:])/16146,3)
        elif j =='M':
            new_count = round(len(new_array[age_mask&(new_array[:,2]==j),:])/16192,3)
        
        count+= new_count
        print(new_count,i,j)
        
#%%
        16192+16146
        #%%
32401 - 16192
sum(new_array[:,2]=='F')

#%%
new_data_df.describe()
        