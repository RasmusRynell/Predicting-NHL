from numpy import e, float32
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
import pandas as pd



# A function that takes in a configuration dict and runs a scikitlearn pipeline with its settings
def run_eval_pipeline(config):
    # Load the data
    data = pd.read_csv("./external/csvs/" + config['data_path'], sep=";")

    pre_fix = data.shape

    # Turn false and true into 0 and 1
    if config['replace_missing_values_with_zero']:
        print("Replacing true/false with 1/0")
        data.replace(['false', 'true'], [0, 1], inplace=True)
        data.replace(['False', 'True'], [0, 1], inplace=True)

    # Drop columns with all missing values
    if config['drop_all_missing_columns']:
        print("Dropping all columns with missing values")
        data.dropna(axis=1, how='all', inplace=True)

    # Drop rows with missing values
    if config['drop_all_missing_rows']:
        print("Dropping rows with missing values")
        data = data.dropna(axis=0, how='any')

    print("Data shape before removing missing values: " + str(pre_fix))
    print("Data shape after removing missing values: " + str(data.shape))

    # save to csv
    if config['save_cleaned_csv']:
        data.to_csv("./external/csvs/cleaned_" + config['data_path'], sep=";", index=False)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(data.drop(config['remove_cols'], axis=1, errors='ignore'),
                                                        data[config['train_test_split']['target_col']],
                                                        **config['train_test_split']['settings'])


    # A function call to create the right scaler
    scaler = None
    if 'scaler' in config:
        scaler = construct_scaler(config['scaler'])


    # A function call to create the right column tranformer
    col_transformer = None
    if 'col_transformer' in config:
        col_transformer = construct_col_transformer(config['col_transformer'])


    # A function call to create the right matrix decomposition
    decomposition = None
    if 'decomposition' in config:
        decomposition = construct_matrix_decomposition(config['decomposition'])


    # A function call to create the right model
    model = construct_model(config['model'])
    if model == None:
        return None
    

    # Create the pipeline
    pipeline = construct_pipeline(scaler, col_transformer, decomposition, model)

    score = None
    # Cross validate the pipeline
    if 'cross_validate' in config:
        from sklearn.model_selection import cross_val_score
        print("Cross validating the pipeline")
        print(config['cross_validate']['settings'])
        scores = cross_val_score(pipeline, X_train, y_train, **config['cross_validate']['settings']).mean()
        print("Cross validation score:", scores)
        print("Cross validation std:", scores.std())
        print(y_train.value_counts(normalize=True))

        pipeline.fit(X_train, y_train)
        score = pipeline.score(X_test, y_test)
        
        # print confusion matrix
        y_pred = pipeline.predict(X_test)
        print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
        X_test["pred"] = y_pred

        print("Score on test: " + str(score))

    else:
        # Fit the pipeline
        print("Fitting pipeline without cross validation: " + str(pipeline))
        pipeline.fit(X_train, y_train)

        # Calculate accuracy
        accuracy = pipeline.score(X_test, y_test)
        print("Accuracy:", accuracy)
        X_test["pred"] = pipeline.predict(X_test)


    return X_test







def construct_pipeline(scaler, col_transformer, decomposition, model):
    pipeline = None
    if scaler != None and col_transformer != None and decomposition != None:
        pipeline = Pipeline([('col_transformer', col_transformer), ('scaler', scaler), ('decomposition', decomposition), ('model', model)])
    elif scaler != None and col_transformer != None:
        pipeline = Pipeline([('col_transformer', col_transformer), ('scaler', scaler), ('model', model)])
    elif scaler != None and decomposition != None:
        pipeline = Pipeline([('scaler', scaler), ('decomposition', decomposition), ('model', model)])
    elif scaler != None and model != None:
        pipeline = Pipeline([('scaler', scaler), ('model', model)])
    elif col_transformer != None and decomposition != None:
        pipeline = Pipeline([('col_transformer', col_transformer), ('decomposition', decomposition), ('model', model)])
    elif col_transformer != None and model != None:
        pipeline = Pipeline([('col_transformer', col_transformer), ('model', model)])
    elif decomposition != None and model != None:
        pipeline = Pipeline([('decomposition', decomposition), ('model', model)])
    elif scaler != None:
        pipeline = Pipeline([('scaler', scaler), ('model', model)])
    elif col_transformer != None:
        pipeline = Pipeline([('col_transformer', col_transformer), ('model', model)])
    elif decomposition != None:
        pipeline = Pipeline([('decomposition', decomposition), ('model', model)])
    elif model != None:
        pipeline = Pipeline([('model', model)])
    else:
        pipeline = Pipeline([('model', model)])
    
    return pipeline



def construct_scaler(scaler_config):
    scaler = None
    if scaler_config['name'] == 'standard':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler(**scaler_config['settings'])
    elif scaler_config['name'] == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(**scaler_config['settings'])
    elif scaler_config['name'] == 'maxabs':
        from sklearn.preprocessing import MaxAbsScaler
        scaler = MaxAbsScaler(**scaler_config['settings'])
    elif scaler_config['name'] == 'robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler(**scaler_config['settings'])
    elif scaler_config['name'] == 'quantile':
        from sklearn.preprocessing import QuantileTransformer
        scaler = QuantileTransformer(**scaler_config['settings'])
    elif scaler_config['name'] == 'power':
        from sklearn.preprocessing import PowerTransformer
        scaler = PowerTransformer(**scaler_config['settings'])
    elif scaler_config['name'] == 'normalize':
        from sklearn.preprocessing import Normalizer
        scaler = Normalizer(**scaler_config['settings'])
    
    if scaler != None:
        return scaler
    raise Exception("No scaler found in scaler config")


def construct_col_transformer(col_transformer_config):
    from sklearn.compose import make_column_transformer

    encoders = []
    for encoder_config in col_transformer_config['encoders']:
        encoder = None
        if encoder_config['name'] == 'onehot':
            from sklearn.preprocessing import OneHotEncoder
            encoder = OneHotEncoder(**encoder_config['settings'])
        elif encoder_config['name'] == 'label':
            from sklearn.preprocessing import LabelEncoder
            encoder = LabelEncoder(**encoder_config['settings'])
        elif encoder_config['name'] == 'Ordinal':
            from sklearn.preprocessing import OrdinalEncoder
            encoder = OrdinalEncoder(**encoder_config['settings'])

        encoders.append((encoder, encoder_config['targets']))

    if len(encoders) > 0:
        print(encoders)
        return make_column_transformer(*encoders, **col_transformer_config['settings'])
    raise Exception("No encoders found in col_transformer config")


def construct_matrix_decomposition(decomposition_config):
    decomposition = None
    if decomposition_config['name'] == 'pca':
        from sklearn.decomposition import PCA
        decomposition = PCA(**decomposition_config['settings'])
    elif decomposition_config['name'] == 'kernelpca':
        from sklearn.decomposition import KernelPCA
        decomposition = KernelPCA(**decomposition_config['settings'])
    elif decomposition_config['name'] == 'face':
        from sklearn.decomposition import FactorAnalysis
        decomposition = FactorAnalysis(**decomposition_config['settings'])
    elif decomposition_config['name'] == 'nmf':
        from sklearn.decomposition import NMF
        decomposition = NMF(**decomposition_config['settings'])
    elif decomposition_config['name'] == 'sparsepca':
        from sklearn.decomposition import SparsePCA
        decomposition = SparsePCA(**decomposition_config['settings'])
    elif decomposition_config['name'] == 'factoranalysis':
        from sklearn.decomposition import FactorAnalysis
        decomposition = FactorAnalysis(**decomposition_config['settings'])
    elif decomposition_config['name'] == 'truncatedsvd':
        from sklearn.decomposition import TruncatedSVD
        decomposition = TruncatedSVD(**decomposition_config['settings'])
    elif decomposition_config['name'] == 'pca_sparse':
        from sklearn.decomposition import PCA
        print(*decomposition_config['settings'])
        decomposition = PCA(**decomposition_config['settings'])

    if decomposition != None:
        return decomposition
    raise Exception("No decomposition found in decomposition config")


def construct_model(model_config):
    model = None
    if model_config['name'] == 'logistic':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(**model_config['settings'])
    elif model_config['name'] == 'svm':
        from sklearn.svm import SVC
        model = SVC(**model_config['settings'])
    elif model_config['name'] == 'knn':
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(**model_config['settings'])
    elif model_config['name'] == 'gaussian':
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB(**model_config['settings'])
    elif model_config['name'] == 'tree':
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(**model_config['settings'])
    elif model_config['name'] == 'forest':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(**model_config['settings'])
    elif model_config['name'] == 'gradient':
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(**model_config['settings'])
    elif model_config['name'] == 'adaboost':
        from sklearn.ensemble import AdaBoostClassifier
        model = AdaBoostClassifier(**model_config['settings'])
    elif model_config['name'] == 'extra':
        from sklearn.ensemble import ExtraTreesClassifier
        model = ExtraTreesClassifier(**model_config['settings'])
    elif model_config['name'] == 'bagging':
        from sklearn.ensemble import BaggingClassifier
        model = BaggingClassifier(**model_config['settings'])
    elif model_config['name'] == 'gradientboosting':
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(**model_config['settings'])
    elif model_config['name'] == 'voting':
        from sklearn.ensemble import VotingClassifier
        model = VotingClassifier(**model_config['settings'])
    elif model_config['name'] == 'mlp':
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(**model_config['settings'])
    elif model_config['name'] == 'sgd':
        from sklearn.linear_model import SGDClassifier
        model = SGDClassifier(**model_config['settings'])
    
    if model != None:
        return model
    raise Exception("No model found in model config")