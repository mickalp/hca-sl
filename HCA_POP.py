# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 02:30:10 2022

@author: User
"""
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np

# import test data
wine = load_wine()
data1 = pd.DataFrame(data=np.c_[wine['data'], wine['target']], columns=wine['feature_names']+['target'])
data = data1.iloc[:5]
data = data.drop('target', axis=1)

# scaling of data

mean_df = data.mean(axis=0)
std_df = data.std(axis=0)

df1 = []
i = 0

for dot in data.columns:
    ascaled = (data[dot]-mean_df[i])/std_df[i]
    i += 1
    df1.append(ascaled)

df2 = pd.concat(df1, axis=1)

# [optional: adding letter index]

# kal = list(string.ascii_uppercase)
# ind=[]

# for i in range(len(data)):
#     s = kal[i]
#     ind+=s    

# df2.index = ind
df = df2

# macierz odległosci euklidesowych

def df_odl_eukl(df):
    
    n_df=(df.values)
    n_df
    
    (df.values).shape
    
    matrix=np.zeros(((df.values).shape[0],(df.values).shape[0]))
    matrix
    
    
    for i in range((df.values).shape[0]):
        for j in range((df.values).shape[0]):
            matrix[i,j]=np.sqrt(np.sum((n_df[i]-n_df[j])**2))
            #print('i',i,'j',j)


    df_odl = pd.DataFrame(matrix)
    # df_odl.index = ind
    df_odl_eukl = df_odl
    return df_odl_eukl

df_oe = df_odl_eukl(df)
# ind_li = []
# for i in range(len(data)): ind_li.append(i) 
# df_oe.index = ind_li
# df_oe.set_axis(ind_li, inplace = True, axis=1)
df_odl = np.tril(df_oe)
X = df_oe.copy()


#%%


Z = X.copy()

X = df_oe.copy()
nazwy = iter(range(len(X),200))

full_dict = {}

lista_tych_coordow = []
min_value_lits = []
list_coord_name = []
empty_list = []
coord_list = []
while len(X) >= 0:

    triangle = pd.DataFrame(np.tril(X), columns = X.columns, index = X.index)
    # triangle = np.tril(X)
    triangle_flat = np.array(triangle).flatten()
    
    # print(triangle)
    
    
    min1 = min(i for i in triangle_flat if i > 0)
    coord = (np.where(triangle == min1))
    lista_tych_coordow.append(coord)
    
    coord1 = max(coord)[0]
    coord2 = min(coord)[0]
    
    
    lista_cor = X.iloc[coord1].name, X.iloc[coord2].name
    coord_list.append(lista_cor)
    
    new_name = next(nazwy)
    
    #liczenie iosci obiektow w klastrze

    nazwy_col = []

   
        

    for i in range(0, len(Z)): 
        nazwy_col.append(Z.columns[i])
    #trzeba bedzie zmienic zeby to były tylko liczby od 0 do len(data)
    #bo inaczej zawsze te wartosci beda przy zmianie indeksow w macierzy
    if X.iloc[coord1].name in nazwy_col and X.iloc[coord2].name in nazwy_col:
        
        # new_dict = {new_name: 2}
        full_dict[new_name] = 2

    
        
        
    elif (X.iloc[coord1].name in nazwy_col and X.iloc[coord2].name not in nazwy_col):
        full_dict[new_name] =  1 + full_dict.get(X.iloc[coord2].name)
    elif (X.iloc[coord1].name not in nazwy_col and X.iloc[coord2].name in nazwy_col):
        full_dict[new_name] = 1 + full_dict.get(X.iloc[coord1].name)
    else:
        full_dict[new_name] = full_dict.get(X.iloc[coord1].name) + full_dict.get(X.iloc[coord2].name)
                                           
     
    
    min_value_lits.append(min1)

    
    wektor = []
    

    
    for i in range(len(X)):
        if X.iloc[coord1,i] < X.iloc[coord2,i]:
            wektor.append(X.iloc[coord1, i])

        else:
            wektor.append(X.iloc[coord2, i])

          
    X.iloc[coord2] = wektor
    X.iloc[:, coord2] = wektor
    
    
    X.drop(X.columns[coord1], axis=1, inplace = True)
    X.drop(X.index[coord1], axis=0, inplace = True)


    

    # X = X.set_axis(list(range(len(X.index))), axis = 0)
    # X = X.set_axis(list(range(len(X.index))), axis = 1)
   
    
    X.rename(columns={X.columns[coord2]: new_name}, inplace=True)
    X.rename(index={X.index[coord2]: new_name}, inplace=True)
    

    # input('kjas')
    # input()
    
print(full_dict)



# it works!
