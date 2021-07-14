import pandas as pd


class preprocessing():
    def extract_data(file, data):
        df = pd.read_csv(f"./external/csvs/{file}", sep = ';')
        df_res = pd.DataFrame()
        for e in data:
            if e['action'] == "sum":
                df[e['name']] = df[e["data"]].sum(axis=1)
                df_res[e['name']] = df[e["data"]].sum(axis=1)

            elif e['action'] == "mean":
                df[e['name']] = df[e["data"]].mean(axis=1)
                df_res[e['name']] = df[e["data"]].mean(axis=1)

            elif e['action'] == "div":
                res = df[e['data'][0]]
                for d_index in range(1, len(e['data'])):
                    res = res / df[e['data'][d_index]]
                res.fillna(value=0, inplace=True)
                df[e['name']] = res
                df_res[e['name']] = res

        return df_res
        
    # A function that returns a dataframe from a csv file containing only the specified columns
    def extract_columns(file, columns):
        return pd.read_csv(f"./external/csvs/{file}", sep = ';')[columns]

    # A function that returns a dataframe containing only rows with more than half of the values missing
    def drop_missing_values(df):
        return df.dropna(thresh=len(df.columns) / 2)

    # A function that performs principal component analysis on a dataframe
    def pca(df, n_components):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        pca.fit(df)
        return pca.transform(df)

    # A function that performs a PCA on a dataframe
    def add_pca_features(df, n_components):
        df_pca = preprocessing.pca(df, n_components)
        df_pca = pd.DataFrame(df_pca)
        df_pca.columns = ['pca_' + str(i) for i in range(n_components)]
        df = pd.concat([df, df_pca], axis=1)
        return df

    # A function that returns a dataframe with filled in predicts the missing values using a specified method
    def fill_missing_values_by(df, method):
        return df.fillna(method=method)