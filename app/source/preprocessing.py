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
        

    def select_data(file, data):
        df = pd.read_csv(f"./external/csvs/{file}", sep = ';')
        return df[data]