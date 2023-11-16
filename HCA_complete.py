from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
import doctest
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import complete
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt


wine = load_wine()
data1 = pd.DataFrame(data=np.c_[wine['data'], wine['target']], columns=wine['feature_names']+['target'])
data = data1.iloc[:26].drop('target', axis = 1)



#autoscailing data

def autoscale(data:pd.DataFrame) -> pd.DataFrame:

    """ Autoscale the data by substraction mean value and dividing by standard deviation
    >>> import pandas as pd
    >>> df_t = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> autoscale(df_t)
         A    B
    0 -1.0 -1.0
    1  0.0  0.0
    2  1.0  1.0
    """
    mean_df = data.mean(axis=0)
    std_df = data.std(axis=0)
    
    df1 = []
    i = 0
    
    for dot in data.columns:
        ascaled = (data[dot]-mean_df[i])/std_df[i]
        i+=1
        df1.append(ascaled)
    
    df = pd.concat(df1, axis=1)
    return df

df = autoscale(data)



#Euclidean Distance Matrix

def df_odl_eukl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the squared Euclidean distance matrix for a DataFrame.

    This function takes a DataFrame as input and returns a DataFrame representing
    the squared Euclidean distance between rows.

    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> df_odl_eukl(df)
              0         1         2
    0  0.000000  1.414214  2.828427
    1  1.414214  0.000000  1.414214
    2  2.828427  1.414214  0.000000
    """

    n_df=(df.values)
    n_df
    
    (df.values).shape
    
    matrix=np.zeros(((df.values).shape[0],(df.values).shape[0]))
    matrix
    
    
    for i in range((df.values).shape[0]):
        for j in range((df.values).shape[0]):
            matrix[i,j]=np.sqrt(np.sum((n_df[i]-n_df[j])**2))



    df_odl = pd.DataFrame(matrix)
    # df_odl.index = ind  #uncomment if letter index needed
    df_odl_eukl = df_odl
    return df_odl_eukl

df_oe = df_odl_eukl(df)

X = df_oe.copy()




def HCA(distance_matrix: pd.DataFrame, method = 'complete') -> pd.DataFrame:
    
    """

    >>> import pandas as pd
    >>> df = pd.DataFrame.from_dict({0: [0, 1.5, 2], 1: [1.5, 0, 3], 2: [2, 3, 0]})
    >>> HCA(df)
    [[1, 0, 1.5, 2], [2, 3, 3.0, 3]]


    """
    
    X = distance_matrix
    Z = X.copy()
    full_dict = {}
    lista_tych_coordow = []
    min_value = []
    list_cord = []
    names = iter(range(len(X),2000))


    while X.size >= 2:
    
        triangle = pd.DataFrame(np.tril(X), columns = X.columns, index = X.index)
        triangle_flat = np.array(triangle).flatten()
        
        
        min1 = min(i for i in triangle_flat if i > 0)
        coord = tuple(np.where(triangle == min1))
        lista_tych_coordow.append(list(coord))
        
        
        coord1 = max(coord)[0]
        coord2 = min(coord)[0]
        
        min1 = round(min1, 3)
        min_value.append(min1)
        
        lista_cor = X.iloc[coord1].name, X.iloc[coord2].name
        list_cord.append(lista_cor)
        
        new_name = next(names)
        
        #Couting number of object in cluster
    
        names_col = []
    

        for i in range(0, len(Z)): 
            names_col.append(Z.columns[i])
    
        if X.iloc[coord1].name in names_col and X.iloc[coord2].name in names_col:
            full_dict[new_name] = 2
            
        elif (X.iloc[coord1].name in names_col and X.iloc[coord2].name not in names_col):
            full_dict[new_name] =  1 + full_dict.get(X.iloc[coord2].name)
            
        elif (X.iloc[coord1].name not in names_col and X.iloc[coord2].name in names_col):
            full_dict[new_name] = 1 + full_dict.get(X.iloc[coord1].name)
            
        else:
            full_dict[new_name] = full_dict.get(X.iloc[coord1].name) + full_dict.get(X.iloc[coord2].name)
                        
                        

        
        wektor = []
        
        if method == "complete":
            for i in range(len(X)):
                if X.iloc[coord1,i] == 0:
                    wektor.append(X.iloc[coord1,i])
                
                elif X.iloc[coord2,i] == 0:
                    wektor.append(X.iloc[coord2,i])
                
                elif X.iloc[coord1,i] > X.iloc[coord2,i]:
                    wektor.append(X.iloc[coord1, i])
        
                else:
                    wektor.append(X.iloc[coord2, i])
                    
                    
        if method == "single":
            
            for i in range(len(X)):
                if X.iloc[coord1,i] > X.iloc[coord2,i]:
                    wektor.append(X.iloc[coord2, i])
                    
                else:
                    wektor.append(X.iloc[coord1, i])


              
        X.iloc[coord2] = wektor
        X.iloc[:, coord2] = wektor
        
        
        X.drop(X.columns[coord1], axis=1, inplace = True)
        X.drop(X.index[coord1], axis=0, inplace = True)
    
    
       
        
        X.rename(columns={X.columns[coord2]: new_name}, inplace=True)
        X.rename(index={X.index[coord2]: new_name}, inplace=True)
        
    last = []

    k =  list(full_dict.values())

    for i in range(len(min_value)):
        last.append([list_cord[i][0], list_cord[i][1], 
                      min_value[i], k[i]])

    last 

        
    return last
        
        

HCA_complete = HCA(X, method = 'complete')



# =============================================================================
# Takes final matrix from HCA, and generate dendrogram by use of libraries
# =============================================================================

fig = plt.figure(figsize=(12, 7))
dendrogram = sch.dendrogram(HCA_complete)
plt.title('Dendrogram')
plt.xlabel('Obiekty')
plt.ylabel('Euclidean distance')
plt.show()

test_mat = complete(pdist(df)) #if this is the same as matrix last
#then you do a good job!


# Add this line to run the doctests when the script is executed directly.
if __name__ == "__main__":
    doctest.testmod()




