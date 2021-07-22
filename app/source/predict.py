from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
import sklearn.svm as svm
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def predict_game(org_data, config):
    for target in config['columns']['targets']['active']:
        data = data_cleanup(org_data.copy(), config)

        # Split data into train and test using sci-kit learn
        X_train, X_test, y_train, y_test = train_test_split(data.drop(config['columns']['targets']['all'], axis=1), data[target], test_size=0.2, random_state=42)

        # Create column transformer
        col_transformer = ColumnTransformer([
            ('one_hot_encoder', OneHotEncoder(), config['columns']['types']['categorical'])
        ])


        # Logistic regression
        run_pipeline(X_train, X_test, y_train, y_test, col_transformer, target, LogisticRegression(), 'Logistic Regression')

        # SVM
        run_pipeline(X_train, X_test, y_train, y_test, col_transformer, target, svm.SVC(), 'SVM')
        

        # Naive Bayes
        run_pipeline(X_train, X_test, y_train, y_test, col_transformer, target, GaussianNB(), 'Naive Bayes')


        # Random forest
        run_pipeline(X_train, X_test, y_train, y_test, col_transformer, target, RandomForestClassifier(), 'Random Forest')


        # Neural network

        print("---")


def run_pipeline(X_train, X_test, y_train, y_test, col_transformer, target, model, name):
    # Create pipeline with column transformer, standard scaler, and logistic regression
    pipeline = Pipeline([
        ('col_transformer', col_transformer),
        ('standard_scaler', StandardScaler()),
        ('classifier', model)
    ])

    # Cross validate pipeline
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    print(f'\nCrossvaledating {target} using {name}')
    print(f'target: {target}')
    print(f'Accuracy: {scores.mean()}')
    print(f'Std: {scores.std()}')

    # Run the pipeline on test data
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(f'Accuracy on test data: {accuracy_score(y_test, y_pred)}')


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

    # Get first date in first row as datetime object
    first_date = pd.to_datetime(data.iloc[0]['gameDate_Game'])

    for col in config['columns']['types']['date']:
        # Create new column with date difference
        data[col] = data[col].apply(lambda x: pd.to_datetime(x) - first_date)

        # Change col column from datetime object to total seconds
        data[col] = data[col].dt.total_seconds().astype(int)

    return data