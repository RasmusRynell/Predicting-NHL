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

import numpy as np
import pandas as pd

def predict_game(org_data, config):
    for target in config['columns']['targets']['active']:
        data = data_cleanup(org_data.copy(), config)

        # Split data into train and test using sci-kit learn
        X_train, X_test, y_train, y_test = train_test_split(data.drop(config['columns']['targets']['all'], axis=1), data[target], test_size=0.2, random_state=42)

        # Create column transformer
        col_transformer = ColumnTransformer([
            ('one_hot_encoder', OneHotEncoder(), config['columns']['types']['categorical'])
        ],
            remainder='passthrough',
            n_jobs=-1
        )

        # Logistic regression
        grid_params = None#\
        # {
        #     'pca__n_components': [10, 50, 100, 200],
        #     'classifier__penalty': ['l1', 'l2'],
        #     'classifier__solver': ['liblinear'],
        #     'classifier__C': [0.001, 0.01, 0.1, 1, 10],
        # }
        log_reg_pipeline = run_pipeline(X_train, X_test, y_train, y_test, col_transformer, target, LogisticRegression(C=0.1, penalty='l1', solver='liblinear'), grid_params)

        # SVM
        grid_params = None#\
        # {
        #     'pca__n_components': [7, 8, 9, 10, 11, 12, 13],
        #     'classifier__C': [0.001, 0.01, 0.1, 1, 10],
        #     'classifier__kernel': ['rbf'],
        #     'classifier__gamma': [0.001, 0.01, 0.1, 1, 10]
        # }
        svm_pipeline = run_pipeline(X_train, X_test, y_train, y_test, col_transformer, target, SVC(C=1, gamma=0.01,probability=True), grid_params)
        

        # Naive Bayes
        grid_params = None#\
            # {
            #     'pca__n_components': [10, 50, 100, 200],
            #     'classifier__var_smoothing': np.logspace(0,-9, num=10)
            # }
        bayes_pipeline = run_pipeline(X_train, X_test, y_train, y_test, col_transformer, target, GaussianNB(var_smoothing=0.3511191734215131), grid_params)


        # Random forest
        grid_params = None#\
            # {
            #     'pca__n_components': [10],
            #     'classifier__n_estimators': [10, 50, 100, 200],
            #     'classifier__max_features': ['auto', 'sqrt', 'log2'],
            #     'classifier__min_samples_split': [2, 4, 8, 9, 10],
            #     'classifier__n_jobs': [-1],
            # }
        forest_pipeline = run_pipeline(X_train, X_test, y_train, y_test, col_transformer, target, RandomForestClassifier(max_depth=None, max_features='auto', min_samples_split=4, n_estimators=200), grid_params)


        # Stacking
        estimators = [
            ('log_reg', log_reg_pipeline),
            ('svm', svm_pipeline),
            ('bayes', bayes_pipeline),
            ('forest', forest_pipeline)
        ]

        # Create meta-pipeline
        clf = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(),
            cv=5,
            verbose=1,
            n_jobs=-1,
            stack_method='predict_proba'
        )

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f"\n\nStacking report: ")
        print("{classification_report(y_test, y_pred)}")
        print(f"Stacking acc: {accuracy_score(y_test, y_pred)}")
        print(f"{y_train.value_counts(normalize=True)}")
        
        # Print ROC AUC
        print(f"ROC AUC: {roc_auc_score(y_test, y_pred)}")


        # Neural network

        print("---")


def run_pipeline(X_train, X_test, y_train, y_test, col_transformer, target, model, grid_params=None):
    # Create pipeline with column transformer, standard scaler, and logistic regression
    pipeline = Pipeline(steps=[
        ('col_transformer', col_transformer),
        ('standard_scaler', StandardScaler()),
        ('pca', PCA(n_components=10)),
        ('classifier', model),
    ])
    print("\n")

    # Create grid search object
    if grid_params:
        grid_search = GridSearchCV(pipeline, grid_params, cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Print ROC AUC score
        print(f"ROC AUC: {roc_auc_score(y_test, grid_search.predict_proba(X_test)[:, 1])}")

        print("Best score:", grid_search.best_score_)
        print("Best params:", grid_search.best_params_)
        print("Best estimator:", grid_search.best_estimator_)

        # fit best model to train data
        y_pred = grid_search.predict(X_test)
        print(f"Accuracy on test data: {accuracy_score(y_test, y_pred):.2%}")
    
    else:
        # Cross validate pipeline
        scores = cross_val_score(
            pipeline, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0)
        print(f'Crossvaledating {target} using {str(model)}')
        print(f'Accuracy: {scores.mean():.2%}')
        print(f'Std: {scores.std()}')

        # Run the pipeline on test data
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        print(f'Accuracy on test data: {accuracy_score(y_test, y_pred):.2%}')

        # Print ROC AUC score
        print(f"ROC AUC: {roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])}")

    return pipeline
    


def data_cleanup(data, config):
    '''
    This function cleans up the data by removing the columns that are not needed
    '''

    # Remove data before season "2015"
    data = data[data['season_Game'] >= 2019]

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

    # Get first date in first row as datetime object
    first_date = pd.to_datetime(data.iloc[0]['gameDate_Game'])

    for col in config['columns']['types']['date']:
        # Create new column with date difference
        data[col] = data[col].apply(lambda x: pd.to_datetime(x) - first_date)

        # Change col column from datetime object to total seconds
        data[col] = data[col].dt.total_seconds().astype(int)

    return data
