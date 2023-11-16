from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
import doctest
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import complete
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

class ClusteringAnalysis:
    def __init__(self):
        self.data = None
        self.df = None
        self.df_oe = None
        self.X = None

    def load_wine_data(self):
        wine = load_wine()
        data1 = pd.DataFrame(data=np.c_[wine['data'], wine['target']], columns=wine['feature_names']+['target'])
        self.data = data1.iloc[:26].drop('target', axis=1)

    def autoscale(self, data):
        mean_df = data.mean(axis=0)
        std_df = data.std(axis=0)
        df1 = []

        for dot in data.columns:
            ascaled = (data[dot] - mean_df[dot]) / std_df[dot]
            df1.append(ascaled)

        self.df = pd.concat(df1, axis=1)

    def df_odl_eukl(self, df):
        n_df = df.values
        matrix = np.zeros((n_df.shape[0], n_df.shape[0]))

        for i in range(n_df.shape[0]):
            for j in range(n_df.shape[0]):
                matrix[i, j] = np.sqrt(np.sum((n_df[i] - n_df[j]) ** 2))

        self.df_oe = pd.DataFrame(matrix)

    def HCA(self, distance_matrix, method='complete'):
        X = distance_matrix.copy()
        Z = X.copy()
        full_dict = {}
        lista_tych_coordow = []
        min_value = []
        list_cord = []
        names = iter(range(len(X), 2000))

        while X.size >= 2:
            triangle = pd.DataFrame(np.tril(X), columns=X.columns, index=X.index)
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

            names_col = []
            for i in range(len(Z.columns)):
                names_col.append(Z.columns[i])

            if X.iloc[coord1].name in names_col and X.iloc[coord2].name in names_col:
                full_dict[new_name] = 2

            elif X.iloc[coord1].name in names_col and X.iloc[coord2].name not in names_col:
                full_dict[new_name] = 1 + full_dict.get(X.iloc[coord2].name)

            elif X.iloc[coord1].name not in names_col and X.iloc[coord2].name in names_col:
                full_dict[new_name] = 1 + full_dict.get(X.iloc[coord1].name)

            else:
                full_dict[new_name] = full_dict.get(X.iloc[coord1].name) + full_dict.get(X.iloc[coord2].name)

            wektor = []
            if method == "complete":
                for i in range(len(X)):
                    if X.iloc[coord1, i] == 0:
                        wektor.append(X.iloc[coord1, i])
                    elif X.iloc[coord2, i] == 0:
                        wektor.append(X.iloc[coord2, i])
                    elif X.iloc[coord1, i] > X.iloc[coord2, i]:
                        wektor.append(X.iloc[coord1, i])
                    else:
                        wektor.append(X.iloc[coord2, i])

            if method == "single":
                for i in range(len(X)):
                    if X.iloc[coord1, i] > X.iloc[coord2, i]:
                        wektor.append(X.iloc[coord2, i])
                    else:
                        wektor.append(X.iloc[coord1, i])

            X.iloc[coord2] = wektor
            X.iloc[:, coord2] = wektor
            X.drop(X.columns[coord1], axis=1, inplace=True)
            X.drop(X.index[coord1], axis=0, inplace=True)

            X.rename(columns={X.columns[coord2]: new_name}, inplace=True)
            X.rename(index={X.index[coord2]: new_name}, inplace=True)

        last = []
        k = list(full_dict.values())

        for i in range(len(min_value)):
            last.append([list_cord[i][0], list_cord[i][1],
                         min_value[i], k[i]])

        return last

    def plot_dendrogram(self, hca_result):
        fig = plt.figure(figsize=(12, 7))
        dendrogram = sch.dendrogram(hca_result)
        plt.title('Dendrogram')
        plt.xlabel('Obiekty')
        plt.ylabel('Euclidean distance')
        plt.show()

    def run_doctests(self):
        doctest.testmod()


if __name__ == "__main__":
    clustering_analysis = ClusteringAnalysis()
    clustering_analysis.load_wine_data()
    clustering_analysis.autoscale(clustering_analysis.data)
    clustering_analysis.df_odl_eukl(clustering_analysis.df)
    HCA_result = clustering_analysis.HCA(clustering_analysis.df_oe, method='complete')
    clustering_analysis.plot_dendrogram(HCA_result)
    clustering_analysis.run_doctests()

