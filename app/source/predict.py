from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix

from tqdm import tqdm
import json
import numpy as np
import pandas as pd
import os
import glob



def predict_game(org_data, org_config, game_id, player_id, targets):
    results = {}

    for target in targets:
        # Clean data
        config = org_config.copy()
        data = data_cleanup(org_data.copy(), config)

        # Check if game that is to be predicted is in data
        if int(game_id) not in data['gamePk_SkaterStats'].values:
            print(f"Game {game_id} not found in data")
            return None

        # Remove identifier columns
        data.drop(org_config['columns']['types']['identifier'], axis=1, inplace=True)

        # Load model config and information
        models_config = load_model_config(player_id, data, config, target)


def load_model_config(player_id, data, config, target):
    '''
    Loads the model config for a given player
    '''
    model_config = None

    # Check if config file exists
    oldpwd=os.getcwd()
    os.chdir("./external/model_configs/")
    for file in glob.glob("*"):
        if str(player_id)+"_BAJS"+str(target) == str(file):
            os.chdir(oldpwd)
            with open(f'./external/model_configs/{str(player_id)+"_"+str(target)}.cfg', 'r') as f:
                model_config = json.load(f)
            return model_config

    os.chdir(oldpwd)
    print("No sufficient file found, generating one")

    # Generate a model_config file
    return generate_models_config(player_id, data, config, target)


def generate_models_config(player_id, data, config, target):
    '''
    Generates a model config file for a given player
    '''
    # Split data into train and test using sci-kit learn
    X_train, X_test, y_train, y_test = train_test_split(data.drop(config['columns']['targets']['all'], axis=1), data["O_" + str(target)], test_size=0.2, random_state=42)

    # Create column transformer
    col_transformer = ColumnTransformer([
        ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'), config['columns']['types']['categorical'])
    ],
        remainder='passthrough',
        n_jobs=-1
    )

    # Create pipeline
    pipeline = Pipeline(steps=[
        ('col_transformer', col_transformer),
        ('standard_scaler', StandardScaler()),
        ('pca', PCA(n_components=10)),
        ('classifier', None),
    ])

    # Create grid parameters
    grid = [
            {"name": "LogisticRegression",
                "parameters": {
                    'classifier': [LogisticRegression()],
                    'classifier__penalty': ['l1', 'l2'],
                    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'classifier__solver': ['liblinear'],
                    'classifier__class_weight': ['balanced', None],
                    'classifier__max_iter': [100, 150, 300],
                    'classifier__tol': [0.01, 0.001, 0.0001, 0.00001]
            }},{
                "name": "SVC",
                "parameters": {
                    'classifier': [SVC()],
                    'classifier__kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
                    'classifier__C': [0.01, 0.1, 1, 10, 100],
                    'classifier__gamma': ['scale', 'auto'],
                    'classifier__class_weight': ['balanced', None],
                    'classifier__probability': [True]
            }},{
                "name": "GaussianNB",
                "parameters": {
                    'classifier': [GaussianNB()],
                    'pca__n_components': [10, 50, 100, 200],
                    'classifier__var_smoothing': np.logspace(0,-9, num=10)
            }},{
                 "name": "SGDClassifier",
                 "parameters": {
                     'classifier': [SGDClassifier()],
                     'classifier__loss': ['log', 'modified_huber'],
                     'classifier__penalty': ['l1', 'l2', 'elasticnet'],
                     'classifier__alpha': [0.001, 0.0001, 0.00001],
                     'classifier__l1_ratio': [0.10, 0.15, 0.20],
                     'classifier__n_jobs': [-1]
            }},{
                 "name": "RandomForestClassifier",
                 "parameters": {
                     'classifier': [RandomForestClassifier()],
                     'classifier__n_estimators': [10, 20, 50, 100],
                     'classifier__criterion': ['gini', 'entropy'],
                     'classifier__max_depth': [None, 5, 10, 20],
                     'classifier__min_samples_split': [2, 4, 8],
                     'classifier__min_samples_leaf': [1, 2, 4],
                     'classifier__class_weight': ['balanced', None],
                     'classifier__n_jobs': [-1]
            }}
    ]

    results = {}

    # Create grid search
    for d in tqdm(grid):
        name, parameters = d['name'], d['parameters']
        # grid search
        clf = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1, scoring='roc_auc')
        clf.fit(X_train, y_train)

        results[name] = {}
        # Save results
        results[name]['train_best_score'] = (clf.best_score_)
        results[name]['train_best_params'] = (clf.best_params_['classifier'].get_params())
        results[name]['test_ROC_AUC'] = (roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))
        results[name]['test_classification_report'] = classification_report(y_test, clf.predict(X_test), output_dict=True)


    # Save results
    with open(f'./external/model_configs/{str(player_id)+"_"+str(target)}.cfg', 'w') as f:
        json.dump(results, f, indent=4)

    return results






















# def predict_game(org_data, org_config, player_id, game_id, results):
#     for target in tqdm(org_config['columns']['targets']['active']):
#         config = org_config.copy()
#         data = data_cleanup(org_data.copy(), config)

#         # Split data into train and test using sci-kit learn
#         X_train, X_test, y_train, y_test = train_test_split(data.drop(config['columns']['targets']['all'], axis=1), data[target], test_size=0.2, random_state=42)

#         # Create column transformer
#         col_transformer = ColumnTransformer([
#             ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'), config['columns']['types']['categorical'])
#         ],
#             remainder='passthrough',
#             n_jobs=-1
#         )

#         # Logistic regression
#         grid_params = None#\
#         # {
#         #     'pca__n_components': [10, 50, 100, 200],
#         #     'classifier__penalty': ['l1', 'l2'],
#         #     'classifier__solver': ['liblinear'],
#         #     'classifier__C': [0.001, 0.01, 0.1, 1, 10],
#         # }
#         log_reg_pipeline, log_reg_res = run_pipeline(X_train, X_test, y_train, y_test, col_transformer, target, LogisticRegression(C=0.1, penalty='l1', solver='liblinear', class_weight='balanced'), grid_params)

#         # SVM
#         grid_params = None#\
#         # {
#         #     'pca__n_components': [7, 8, 9, 10, 11, 12, 13],
#         #     'classifier__C': [0.001, 0.01, 0.1, 1, 10],
#         #     'classifier__kernel': ['rbf'],
#         #     'classifier__gamma': [0.001, 0.01, 0.1, 1, 10]
#         # }
#         svm_pipeline, svm_res = run_pipeline(X_train, X_test, y_train, y_test, col_transformer, target, SVC(C=1, gamma=0.01, probability=True), grid_params)
        

#         # Naive Bayes
#         grid_params = None#\
#             # {
#             #     'pca__n_components': [10, 50, 100, 200],
#             #     'classifier__var_smoothing': np.logspace(0,-9, num=10)
#             # }
#         bayes_pipeline, bayes_res = run_pipeline(X_train, X_test, y_train, y_test, col_transformer, target, GaussianNB(var_smoothing=0.3511191734215131), grid_params)


#         # Random forest
#         grid_params = None#\
#             # {
#             #     'pca__n_components': [10],
#             #     'classifier__n_estimators': [10, 50, 100, 200],
#             #     'classifier__max_features': ['auto', 'sqrt', 'log2'],
#             #     'classifier__min_samples_split': [2, 4, 8, 9, 10],
#             #     'classifier__n_jobs': [-1],
#             # }
#         forest_pipeline, forest_res = run_pipeline( \
#             X_train, X_test, y_train, y_test, col_transformer, target, \
#                 RandomForestClassifier(max_depth=None, max_features='auto', min_samples_split=4, n_estimators=1000, class_weight="balanced"), grid_params)


#         # # Stacking
#         # estimators = [
#         #     ('log_reg', log_reg_pipeline),
#         #     ('svm', svm_pipeline),
#         #     ('bayes', bayes_pipeline),
#         #     ('forest', forest_pipeline)
#         # ]

#         # # Create meta-pipeline
#         # clf = StackingClassifier(
#         #     estimators=estimators,
#         #     final_estimator=LogisticRegression(),
#         #     cv=5,
#         #     verbose=1,
#         #     n_jobs=-1,
#         #     stack_method='predict_proba'
#         # )

#         # clf.fit(X_train, y_train)
#         # y_pred = clf.predict(X_test)
#         #print(f"\n\nStacking report: ")
#         #print("{classification_report(y_test, y_pred)}")
#         #print(f"Stacking acc: {accuracy_score(y_test, y_pred):.4%}")
#         #print(f"{y_train.value_counts(normalize=True)}")
        
#         # Print ROC AUC
#         #print(f"ROC AUC: {roc_auc_score(y_test, y_pred):.4%}")


#         # Neural network
        

#         #print("---")
#         # Add new game to results
#         new = {
#             'player_id': player_id,
#             'target': target,
#             **log_reg_res,
#             **svm_res,
#             **bayes_res,
#             **forest_res
#         }
#         # Append dict to results df with keys as columns
#         results = results.append(pd.DataFrame([new], columns=results.columns))

#     return results

# def run_pipeline(X_train, X_test, y_train, y_test, col_transformer, target, model, grid_params=None):
#     # Create pipeline with column transformer, standard scaler, and logistic regression
#     pipeline = Pipeline(steps=[
#         ('col_transformer', col_transformer),
#         ('standard_scaler', StandardScaler()),
#         ('pca', PCA(n_components=10)),
#         ('classifier', model),
#     ])
#     #print("\n")
    
#     results = {}
#     name = model.__class__.__name__

#     # Create grid search object
#     if grid_params:
#         grid_search = GridSearchCV(pipeline, grid_params, cv=5, n_jobs=-1)
#         grid_search.fit(X_train, y_train)
        
#         # Print ROC AUC score
#         #print(f"ROC AUC: {roc_auc_score(y_test, grid_search.predict_proba(X_test)[:, 1]):.4%}")
#         results[name + 'ROC AUC'] = roc_auc_score(y_test, grid_search.predict_proba(X_test)[:, 1])

#         #print("Best score:", grid_search.best_score_)
#         results[name + 'Best score'] = grid_search.best_score_
#         #print("Best params:", grid_search.best_params_)
#         results[name + 'Best params'] = grid_search.best_params_
#         #print("Best estimator:", grid_search.best_estimator_)
#         results[name + 'Best estimator'] = grid_search.best_estimator_

#         # fit best model to train data
#         y_pred = grid_search.predict(X_test)
#         #print(f"Accuracy on test data: {accuracy_score(y_test, y_pred):.4%}")
#         results[name + 'Accuracy on test data'] = accuracy_score(y_test, y_pred)
    
#     else:
#         # Cross validate pipeline
#         scores = cross_val_score(
#             pipeline, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0)
#         #print(f'Crossvaledating {target} using {str(model)}')
#         #print(f'Accuracy: {scores.mean():.4%}')
#         results[name + '_acc'] = scores.mean()
#         #print(f'Std: {scores.std():.4%}')
#         results[name + '_std'] = scores.std()

#         # Run the pipeline on test data
#         pipeline.fit(X_train, y_train)
#         y_pred = pipeline.predict(X_test)
#         #print(f'Accuracy on test data: {accuracy_score(y_test, y_pred):.4%}')
#         results[name + '_acc_test_data'] = accuracy_score(y_test, y_pred)

#         # Print ROC AUC score
#         #print(f"ROC AUC: {roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1]):.4%}")
#         results[name + '_roc_auc'] = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])

#     return (pipeline, results)
    


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

    # Turn all times into a timedelta objects and then to seconds
    for col in config['columns']['types']['time']:
        data[col] = pd.to_timedelta(data[col])

        # Turn timedelta objects into seconds
        data[col] = data[col].dt.total_seconds().astype(int)

    # Drop all columns in specified in the config
    data.drop(config['columns']['drop'], axis=1, inplace=True, errors='ignore')

    # Get first date in first row as datetime object
    first_date = pd.to_datetime(data.iloc[0]['gameDate_Game'])

    # Change all dates to seconds since first date
    for col in config['columns']['types']['date']:
        # Create new column with date difference
        data[col] = data[col].apply(lambda x: pd.to_datetime(x) - first_date)

        # Change col column from datetime object to total seconds
        data[col] = data[col].dt.total_seconds().astype(int)

    # Fix config to work with removed columns
    tmp = config['columns']['types']['categorical'].copy()
    for name in tmp:
        if name not in data.columns:
            config['columns']['types']['categorical'].remove(name)

    return data
