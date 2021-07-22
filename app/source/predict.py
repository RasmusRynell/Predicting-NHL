import pandas as pd


def predict_game(data_in, config):
    for target in config['columns']['targets']:
        data = data_cleanup(data_in.copy(), config)

        # Save data to csv
        data.to_csv('data_new.csv', sep=";", index=False)


        # Onehot encode categorical variables
        #data2 = pd.get_dummies(data)
        #print(data2.columns)


        # Logistic regression
        log_reg = create_logistic_model()
        


        # SVM
        svm = create_svm_model()
        

        # Naive Bayes
        naive_bayes = create_naive_bayes_model()



        # Random forest
        random_forest = create_random_forest_model()


        # Neural network




def search_model(data, grid_params):
    pass
    


def data_cleanup(data, config):
    '''
    This function cleans up the data by removing the columns that are not needed
    '''

    # Drop all columns with only Nan values
    data.dropna(axis=1, how='all', inplace=True)

    # Remove columns with all same values
    data.drop(data.columns[data.nunique() == 1], axis=1, inplace=True)

    # Remove rows containing NaN values
    data.dropna(inplace=True)

    # Turn all times into a timedelta objects
    for col in config['columns']['types']['time']:
        data[col] = pd.to_timedelta(data[col])

        # Turn timedelta objects into seconds
        data[col] = data[col].dt.total_seconds().astype(int)

    # Drop all columns in config
    data.drop(config['columns']['drop'], axis=1, inplace=True, errors='ignore')

    return data






def create_logistic_model():
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    return model


def create_svm_model():
    from sklearn.svm import SVC
    model = SVC()
    return model

def create_naive_bayes_model():
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    return model

def create_random_forest_model():
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    return model