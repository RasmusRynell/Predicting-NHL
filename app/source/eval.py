from numpy import e, float32
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
import pandas as pd



# A function that takes in a configuration dict and runs a scikitlearn pipeline with its settings
def run_pipeline(config):
    # Load the data
    data = pd.read_csv("./external/csvs/" + config['data_path'], sep=";")

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(data.drop(config['remove_cols'], axis=1),
                                                        data[config['train_test_split']['target_col']],
                                                        test_size=config['train_test_split']['test_percent'],
                                                        random_state=config['train_test_split']['random_state'],
                                                        shuffle=config['train_test_split']['shuffle'])


    # A function call to create the right scaler
    scaler = None
    if 'scaler' in config:
        scaler = construct_scaler(config['scaler'])


    # A function call to create the right encoder
    encoder = None
    if 'encoder' in config:
        encoder = construct_encoder(config['encoder'])


    # A function call to create the right matrix decomposition
    decomposition = None
    if 'decomposition' in config:
        decomposition = construct_matrix_decomposition(config['decomposition'])


    # A function call to create the right model
    model = construct_model(config['model'])
    
    if model == None:
        return None
    

    # Create the pipeline
    pipeline = None
    if scaler != None and encoder != None and decomposition != None:
        pipeline = Pipeline([('encoder', encoder), ('scaler', scaler), ('decomposition', decomposition), ('model', model)])
    elif scaler != None and encoder != None:
        pipeline = Pipeline([('encoder', encoder), ('scaler', scaler), ('model', model)])
    elif scaler != None and decomposition != None:
        pipeline = Pipeline([('scaler', scaler), ('decomposition', decomposition), ('model', model)])
    elif scaler != None and model != None:
        pipeline = Pipeline([('scaler', scaler), ('model', model)])
    elif encoder != None and decomposition != None:
        pipeline = Pipeline([('encoder', encoder), ('decomposition', decomposition), ('model', model)])
    elif encoder != None and model != None:
        pipeline = Pipeline([('encoder', encoder), ('model', model)])
    elif decomposition != None and model != None:
        pipeline = Pipeline([('decomposition', decomposition), ('model', model)])
    elif scaler != None:
        pipeline = Pipeline([('scaler', scaler), ('model', model)])
    elif encoder != None:
        pipeline = Pipeline([('encoder', encoder), ('model', model)])
    elif decomposition != None:
        pipeline = Pipeline([('decomposition', decomposition), ('model', model)])
    elif model != None:
        pipeline = Pipeline([('model', model)])
    else:
        pipeline = Pipeline([('model', model)])

    print("Running: " + str(pipeline))
    
    # Fit the pipeline
    pipeline.fit(X_train, y_train)


    # Calculate the accuracy
    accuracy = pipeline.score(X_test, y_test)
    print("Accuracy:", accuracy)
    X_test["pred"] = pipeline.predict(X_test)


    return X_test







def construct_scaler(scaler_config):
    scaler = None
    if scaler_config['name'] == 'standard':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler(*scaler_config['settings'])
    elif scaler_config['name'] == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(*scaler_config['settings'])
    elif scaler_config['name'] == 'maxabs':
        from sklearn.preprocessing import MaxAbsScaler
        scaler = MaxAbsScaler(*scaler_config['settings'])
    elif scaler_config['name'] == 'robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler(*scaler_config['settings'])
    elif scaler_config['name'] == 'quantile':
        from sklearn.preprocessing import QuantileTransformer
        scaler = QuantileTransformer(*scaler_config['settings'])
    elif scaler_config['name'] == 'power':
        from sklearn.preprocessing import PowerTransformer
        scaler = PowerTransformer(*scaler_config['settings'])
    elif scaler_config['name'] == 'normalize':
        from sklearn.preprocessing import Normalizer
        scaler = Normalizer(*scaler_config['settings'])
    return scaler

def construct_encoder(encoder_config):
    encoder = NotImplementedError
    if encoder_config['name'] == 'onehot':
        from sklearn.preprocessing import OneHotEncoder
        encoder = OneHotEncoder(*encoder_config['settings'])
    elif encoder_config['name'] == 'label':
        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder(*encoder_config['settings'])
    elif encoder_config['name'] == 'Ordinal':
        from sklearn.preprocessing import OrdinalEncoder
        encoder = OrdinalEncoder(*encoder_config['settings'])
    return encoder

def construct_matrix_decomposition(decomposition_config):
    decomposition = None
    if decomposition_config['name'] == 'pca':
        from sklearn.decomposition import PCA
        decomposition = PCA(*decomposition_config['settings'])
    elif decomposition_config['name'] == 'kernelpca':
        from sklearn.decomposition import KernelPCA
        decomposition = KernelPCA(*decomposition_config['settings'])
    elif decomposition_config['name'] == 'face':
        from sklearn.decomposition import FactorAnalysis
        decomposition = FactorAnalysis(*decomposition_config['settings'])
    elif decomposition_config['name'] == 'nmf':
        from sklearn.decomposition import NMF
        decomposition = NMF(*decomposition_config['settings'])
    elif decomposition_config['name'] == 'sparsepca':
        from sklearn.decomposition import SparsePCA
        decomposition = SparsePCA(*decomposition_config['settings'])
    elif decomposition_config['name'] == 'factoranalysis':
        from sklearn.decomposition import FactorAnalysis
        decomposition = FactorAnalysis(*decomposition_config['settings'])
    elif decomposition_config['name'] == 'truncatedsvd':
        from sklearn.decomposition import TruncatedSVD
        decomposition = TruncatedSVD(*decomposition_config['settings'])
    elif decomposition_config['name'] == 'pca_sparse':
        from sklearn.decomposition import PCA
        decomposition = PCA(*decomposition_config['settings'])
    return decomposition





def construct_model(model_config):
    model = None
    if model_config['name'] == 'logistic':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(*model_config['settings'])
    elif model_config['name'] == 'svm':
        from sklearn.svm import SVC
        model = SVC(*model_config['settings'])
    elif model_config['name'] == 'knn':
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(*model_config['settings'])
    elif model_config['name'] == 'gaussian':
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB(*model_config['settings'])
    elif model_config['name'] == 'tree':
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(*model_config['settings'])
    elif model_config['name'] == 'forest':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(*model_config['settings'])
    elif model_config['name'] == 'gradient':
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(*model_config['settings'])
    elif model_config['name'] == 'adaboost':
        from sklearn.ensemble import AdaBoostClassifier
        model = AdaBoostClassifier(*model_config['settings'])
    elif model_config['name'] == 'extra':
        from sklearn.ensemble import ExtraTreesClassifier
        model = ExtraTreesClassifier(*model_config['settings'])
    elif model_config['name'] == 'bagging':
        from sklearn.ensemble import BaggingClassifier
        model = BaggingClassifier(*model_config['settings'])
    elif model_config['name'] == 'gradientboosting':
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(*model_config['settings'])
    elif model_config['name'] == 'voting':
        from sklearn.ensemble import VotingClassifier
        model = VotingClassifier(*model_config['settings'])
    elif model_config['name'] == 'mlp':
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(*model_config['settings'])
    elif model_config['name'] == 'sgd':
        from sklearn.linear_model import SGDClassifier
        model = SGDClassifier(*model_config['settings'])
    return model