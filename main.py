__author__ = 'G'

import csv as csv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from scipy.stats import mode


"""
This code was develop for the Kaggle competition where the goal is to predict
the survivor based on a data sample.

Here are the data given:

Data fields

     0 'PassengerId',
     1 'Survived',
     2 'Pclass',
     3 'Name',
     4 'Sex',
     5 'Age',
     6 'SibSp',
     7 'Parch',
     8 'Ticket',
     9 'Fare',
     10 'Cabin',
     11 'Embarked'
"""

parameter_grid = {
    'max_features': [0.5,1],  #0.5, 1
    'max_depth': [5,None],    # 5, None
    'min_samples_split': [2, 4]  #2, 4
    }


def preprocess_df(df, name_list):

    #Remove NaN or empty entries
    #df.dropna()
    df = df.drop(['Ticket', 'Cabin'], axis=1)

    #Replace missing value by mean
    # For the age, make sense to replace the value using the title (Mr, Miss, Mrs,..)
    df['Title'] = df['Name'].str.split(',').apply(lambda x: x[1])
    df['Title'] = df['Title'].str.split('.').apply(lambda x: x[0])
    age_mean = df.pivot_table('Age', index='Title', aggfunc='mean')
    df['Age'] = df[['Age', 'Title']].apply(lambda x: age_mean[x['Title']] if pd.isnull(x['Age']) else x['Age'], axis=1)
    #For people with missing title:
    age_mean = df['Age'].mean()
    df['Age'] = df['Age'].fillna(age_mean)


    #Replace missing value by most common
    embarked_mode = mode(df['Embarked'])[0][0]
    df['Embarked'] = df['Embarked'].fillna(embarked_mode)


    # For 'Fare' makes sense to replace with average price for each category PClass
    fare_mean = df.pivot_table('Fare', index='Pclass', aggfunc='mean')
    df['Fare'] = df[['Fare', 'Pclass']].apply(lambda x: fare_mean[x['Pclass']] if pd.isnull(x['Fare']) else x['Fare'], axis=1)


    #Replace string by value
    df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
    #df['Port'] = df['Embarked'].map({'C': 1, 'S': 2, 'Q': 3}).astype(int)
    #Use dummy variables
    df = pd.concat([df, pd.get_dummies(df['Embarked'], prefix='Embarked')], axis=1)

    #Use the names: keep only family name
    df['Family'] = df['Name'].str.split(',').apply(lambda x: x[0])

    for name in name_list:
        key = "Family_" + name
        df[key] = df['Family'].apply(lambda x: 1 if x == name else 0)

    #df['Family'] = df['Family'].apply(lambda x: x if x in name_count  and name_count[x] > 3 else np.nan)
    #df['Family'] = df['Family'].apply(lambda x: x if x in name_count else np.nan)
    #df = pd.concat([df, pd.get_dummies(df['Family']+list(name_count.index), prefix='Family')], axis=1)
    #df = pd.concat([df, pd.get_dummies(df['Family']+list(name_count.index), prefix='Family')], axis=1)
    #Drop old categories
    df = df.drop(['Sex', 'Embarked', 'Name', 'Title', 'Family'], axis=1)

    return df


def main():
    #Read file
    df = pd.read_csv("data/train.csv")
    df['Family'] = df['Name'].str.split(',').apply(lambda x: x[0])
    name_count = df['Family'].value_counts()
    name_list = [name for name in name_count.keys() if name_count[name] > 1]

    df = preprocess_df(df, name_list)
    cols = df.columns.tolist()
    #Move survived to 1st column
    cols = [cols[1]] + cols[0:1] + cols[2:]
    df = df[cols]
    train_data = df.values

    ### SciKit Random Forest

    #grid search for optimizing parameters
    grid_search = GridSearchCV(GradientBoostingClassifier(n_estimators=50), parameter_grid,
                            cv=5, verbose=3)
    grid_search.fit(train_data[0:,2:], train_data[0:,0])
    sorted(grid_search.grid_scores_, key=lambda x: x.mean_validation_score)
    print str(grid_search.best_score_)
    print str(grid_search.best_params_)


    # Train randowm forest classifier
    rfc_model = RandomForestClassifier(n_estimators = 100, max_features=grid_search.best_params_['max_features'], max_depth=grid_search.best_params_['max_depth'],
                                        min_samples_split=grid_search.best_params_['min_samples_split'])
    #rfc_model = GradientBoostingClassifier(n_estimators = 50, max_features=grid_search.best_params_['max_features'], max_depth=grid_search.best_params_['max_depth'],
    #                                    min_samples_split=grid_search.best_params_['min_samples_split'])
    # train model on data
    # column 0 is the Results, we don't need column 1 (the index number of the passenger)
    #rfc_model.fit(train_data[:, 2:], train_data[:,0])

    ### Cross validation
    # pair number of values for convenience
    train_data = df.values[:891]
    X = train_data[:, 2:]
    y = train_data[:, 0]
    n = len(df)/2

    # Train and test datasets
    X_train = X[:n, :]
    y_train = y[:n]
    X_test = X[n:, :]
    y_test = y[n:]

    rfc_model = rfc_model.fit(X_train, y_train)
    y_prediction = rfc_model.predict(X_test)
    print "prediction accuracy:", np.sum(y_test == y_prediction)*1./len(y_test)


    rfc_model = RandomForestClassifier(n_estimators = 100, max_features=grid_search.best_params_['max_features'], max_depth=grid_search.best_params_['max_depth'],
                                        min_samples_split=grid_search.best_params_['min_samples_split'])

    #rfc_model = GradientBoostingClassifier(n_estimators = 50, max_features=grid_search.best_params_['max_features'], max_depth=grid_search.best_params_['max_depth'],
    #                                    min_samples_split=grid_search.best_params_['min_samples_split'])

    X_train, X_test = X_test, X_train
    y_train, y_test = y_test, y_train
    rfc_model = rfc_model.fit(X_train, y_train)
    y_prediction = rfc_model.predict(X_test)

    print "prediction accuracy:", np.sum(y_test == y_prediction)*1./len(y_test)

    rfc_model = RandomForestClassifier(n_estimators = 100, max_features=grid_search.best_params_['max_features'], max_depth=grid_search.best_params_['max_depth'],
                                        min_samples_split=grid_search.best_params_['min_samples_split'])
    #rfc_model = GradientBoostingClassifier(n_estimators = 50, max_features=grid_search.best_params_['max_features'], max_depth=grid_search.best_params_['max_depth'],
    #                                min_samples_split=grid_search.best_params_['min_samples_split'])
    rfc_model = rfc_model.fit(X, y)

    # Load test dataset
    df_test = pd.read_csv("data/test.csv")

    #Do same processing as above
    df_test = preprocess_df(df_test, name_list)
    data_test = df_test.values

    #Output results from test data
    output = rfc_model.predict(data_test[:, 1:])

    #Save results in csv file
    results = np.c_[data_test[:,0].astype(int), output.astype(int)]
    df_results = pd.DataFrame(results[:,0:2], columns=['PassengerId', 'Survived'])
    df_results.to_csv("data/results.csv", index=False)



if __name__ == "__main__":
    main()
