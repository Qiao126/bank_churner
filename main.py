from mpl_toolkits.mplot3d import Axes3D
import os
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, StratifiedShuffleSplit, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
import seaborn as sns
import numpy as np
import pandas as pd
import argparse
from utils import feature_discretizaiton
from plot import plot_learning_curve, plot_feat_distribution, plot_PCA_analysis, plot_curve, plot_feature_importance


def preprocessing(data_file, feature_discret=False, plot_feat_dist=False, plot_PCA=False, plot_corr=False):
    data = pd.read_csv(data_file)
    cols = [
        #'CLIENTNUM',
        'Customer_Age',
        'Gender',
        'Dependent_count',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Customer_Churn']

    data = data[cols]
    
    """
    # Data visualization
    fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    fig.suptitle('Data Exploration1')
    sns.countplot(ax=axes[0, 0],x=data['Gender'],hue=data['Customer_Churn'], palette="muted")
    sns.countplot(ax=axes[0, 1], x=data['Marital_Status'],hue=data['Customer_Churn'], palette="muted")
    sns.countplot(ax=axes[1, 0], x=data['Education_Level'],hue=data['Customer_Churn'], palette="muted")
    sns.countplot(ax=axes[1, 1], x=data['Income_Category'],hue=data['Customer_Churn'], palette="muted")
    sns.countplot(ax=axes[2, 0], x=data['Card_Category'],hue=data['Customer_Churn'], palette="muted")
    sns.countplot(ax=axes[2, 1], x=data['Customer_Churn'],hue=data['Customer_Churn'], palette="muted")

    fig, axes = plt.subplots(3, 4, figsize=(10, 6))
    fig.suptitle('Data Exploration2')
    sns.displot(data, ax=axes[0, 0], x="Months_on_book", hue="Customer_Churn", kde=True)
    sns.displot(data, ax=axes[0, 1], x="Total_Relationship_Count", hue="Customer_Churn",  kde=True)
    sns.displot(data, ax=axes[0, 2], x="Months_Inactive_12_mon", hue="Customer_Churn",  kde=True)
    sns.displot(data, ax=axes[0, 3], x="Contacts_Count_12_mon", hue="Customer_Churn",  kde=True)
    sns.displot(data, ax=axes[1, 0], x="Credit_Limit", hue="Customer_Churn",  kde=True)
    sns.displot(data, ax=axes[1, 1], x="Total_Revolving_Bal", hue="Customer_Churn",  kde=True)
    sns.displot(data, ax=axes[1, 2], x="Avg_Open_To_Buy", hue="Customer_Churn",  kde=True)
    sns.displot(data, ax=axes[1, 3], x="Total_Amt_Chng_Q4_Q1", hue="Customer_Churn",  kde=True)
    sns.displot(data, ax=axes[2, 0], x="Total_Trans_Amt", hue="Customer_Churn",  kde=True)
    sns.displot(data, ax=axes[2, 1], x="Total_Trans_Ct", hue="Customer_Churn",  kde=True)
    sns.displot(data, ax=axes[2, 2], x="Total_Ct_Chng_Q4_Q1", hue="Customer_Churn",  kde=True)
    
    plt.figure()
    focus = data[['Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Amt_Chng_Q4_Q1', 'Total_Ct_Chng_Q4_Q1', 'Customer_Churn']]
    g = sns.PairGrid(focus, hue="Customer_Churn")
    g.map_diag(sns.histplot,kde=True)
    g.map_offdiag(sns.scatterplot)
    g.add_legend()
    
    plt.figure()
    sns.jointplot(data=data, x="Total_Trans_Amt", y="Total_Trans_Ct", hue="Customer_Churn")
    """
    print("Data shape:", data.shape)
    print(data.describe(include='all'))
    print("Missing values:")
    print(data.isnull().sum())
    
    y = data['Customer_Churn']
    data = data.drop(['Customer_Churn'], 1)
    X = data
    le = LabelEncoder()
    X['Gender'] = le.fit_transform(X['Gender'])
    X['Education_Level'] = le.fit_transform(X['Education_Level'])
    X['Marital_Status'] = le.fit_transform(X['Marital_Status'])
    X['Income_Category'] = le.fit_transform(X['Income_Category'])
    X['Card_Category'] = le.fit_transform(X['Card_Category'])
    print(X.head(10))
    
    if plot_PCA:
        plot_PCA_analysis(X, y, 3)

    if plot_feat_dist:
        plot_feat_distribution(X, X.columns)    
        
    if plot_corr:
        corr = X.corr(method='pearson')
        hm = sns.heatmap(corr, annot=True, annot_kws={"size": 8})
        hm.set_xticklabels(hm.get_xticklabels(), rotation=90)
        hm.set_yticklabels(hm.get_yticklabels(), rotation=0) 

    if feature_discret:
        bins = [0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 100]
        X = feature_discretizaiton(X, 'Customer_Age', bins)
        
    return data, X, y


def train_gbdt_classifier(X, y):
    print('start training gbdt classifier...')
    clf = HistGradientBoostingClassifier()
    clf.fit(X, y)
    return clf


def train_random_forest_classifier(class_weight, X, y):
    print('start training random forest classifier...')
    """
    #grid search for best parameters
    param_grid = { 
      'n_estimators': [10,20,30],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth' : [5,10,15]
    }
    rf=RandomForestClassifier(random_state=0)
    clf = GridSearchCV(estimator=rf, param_grid=param_grid, cv= 5)
    """
    clf = RandomForestClassifier(
        n_estimators=20, max_depth=10,max_features=None, class_weight=class_weight)
    clf.fit(X, y)
    return clf


def train_logit_classifier(class_weight, X, y):
    print('strat training logistic classifier...')
    clf = SGDClassifier(loss='log', class_weight=class_weight)
    clf.fit(X, y)
    return clf


def train_MLP_classifier(X, y):
    print('strat training neural network...')
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(40, 20, 2),max_iter=500, verbose=0, random_state=0)
    clf.fit(X, y)
    return clf


def main():
    parser = argparse.ArgumentParser(description='Classification task')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--plot-pca', action='store_true')
    parser.add_argument('--plot-corr', action='store_true')
    parser.add_argument('--plot-feat-dist', action='store_true')
    parser.add_argument('--plot-curve', action='store_true')
    parser.add_argument('--classifier')
    parser.add_argument('datafile')

    args = parser.parse_args()

    data, X, y = preprocessing(
        args.datafile,
        #feature_discret=True,
        plot_feat_dist=args.plot_feat_dist,
        plot_corr=args.plot_corr,
        plot_PCA=args.plot_pca)    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        stratify=y, 
                                                        test_size=0.4,
                                                        random_state=0)
    X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test,
                                                        stratify=y_test, 
                                                        test_size=0.5,
                                                        random_state=0)
    if args.normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_dev = scaler.transform(X_dev)
        X_test = scaler.transform(X_test)
        
        """
        #for neural network
        scaler = MinMaxScaler((-1, 1)) 
        X_train = scaler.fit_transform(X_train)
        X_dev = scaler.transform(X_dev)
        X_test = scaler.transform(X_test)
        """
        
    print("DataSet summary:")
    print("train X: ", X_train.shape, "y: ", y_train.shape)
    print("dev X: ", X_dev.shape, "y: ", y_dev.shape)
    print("test X:", X_test.shape, "y: ", y_test.shape)

    positive_weight = 1 - sum(y_train) / y_train.count()
    class_weight = {1: positive_weight, 0: 1-positive_weight}
    print("Class weights: ", {1: positive_weight, 0: 1-positive_weight})

    if args.classifier == 'gbdt':
        clf = train_gbdt_classifier(X_train, y_train)
    elif args.classifier == 'rf':
        clf = train_random_forest_classifier(class_weight, X_train, y_train)
    elif args.classifier == 'nn':
        clf = train_MLP_classifier(X_train, y_train)
    else:
        clf = train_logit_classifier(class_weight, X_train, y_train)

    title = "Learning Curve"
    cv = StratifiedShuffleSplit(n_splits=10, test_size=0.33, random_state=0)
    X = pd.concat([X_train, X_dev])
    y = pd.concat([y_train, y_dev])
    plot_learning_curve(clf, title, X, y, cv=cv, n_jobs=2)
    plt.show()

    plot_feature_importance(clf, X_train, y_train, data.columns)

    y_train_pred = clf.predict(X_train)
    y_train_prop = clf.predict_proba(X_train)
    y_dev_pred = clf.predict(X_dev)
    y_dev_prop = clf.predict_proba(X_dev)
    y_test_pred = clf.predict(X_test)
    y_test_prop = clf.predict_proba(X_test)
    y_train_score = y_train_prop[:, 1]
    y_dev_score = y_dev_prop[:, 1]
    y_test_score = y_test_prop[:, 1]

    print('Training data report')
    print(classification_report(y_train, y_train_pred))
    print('Dev data report')
    print(classification_report(y_dev, y_dev_pred))
    #print('Test data report')
    #print(classification_report(y_test, y_test_pred))
    
    if args.plot_curve:
        opthd = plot_curve(y_dev, y_dev_score, min_p=None, min_r=None)
        print("optimal threshold: ", opthd)
        
        y_test_pred = np.array(y_test_score > opthd)
        y_test_pred = y_test_pred.astype(int)
        print('Test data report (with optimal theshold)')
        print(classification_report(y_test, y_test_pred))

if __name__ == '__main__':
    main()
