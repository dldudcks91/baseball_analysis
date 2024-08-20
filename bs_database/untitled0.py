
"""
입력 형식
학습과 검색에 사용될 중복 없는 단어 N개가 주어진다.
모든 단어는 알파벳 소문자로 구성되며 단어의 수 N과 단어들의 길이의 총합 L의 범위는 다음과 같다.

2 <= N <= 100,000
2 <= L <= 1,000,000
"""

# 1. trie 방식
#시간복잡도 = log(NM) , N = 단어 갯수, M = 단어 평균 길이

# 2. 
#%%
words, result = ["go", "gone", "guild"],7
#words, result = ["abc", "def", "ghi", "jklm"], 4
#words, result = ["word", "war", "warrior", "world"], 15

#%%
# trie 방식
index_dic = dict()

for word in words:

    current_dic = index_dic
    for w in word:
        
        current_dic.setdefault(w, [0, {}])
        current_dic[w][0]+=1
        current_dic = current_dic[w][1]
        

total_count = 0 
for word in words:
    current_dic = index_dic
    count = 0 
    for w in word:
        count+=1
        idx_count = current_dic[w][0]
        if idx_count == 1:
            break
        else:
            current_dic = current_dic[w][1]
        
    total_count += count





#%%

words.sort()
total_count = 0
last_count = 0
last_same = True
is_same = True
for i in range(len(words)-1):

    left, right = words[i], words[i+1]
    count = 0
    is_same = True
    for l, r in zip(left, right):
        
        count+=1
        if l == r:
            pass
            
        else:
            is_same =  False
            break
    
    
    if count <= last_count:
        
        total_count += last_count+1 if last_same else last_count
    else:
        total_count += count
        
    last_same = is_same
    
    last_count = count
    print(left, count, total_count, is_same)
total_count += last_count+1 if last_same else last_count