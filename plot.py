import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, classification_report,
    precision_recall_fscore_support,
    average_precision_score, roc_auc_score,
    roc_curve, auc, precision_recall_curve)
from sklearn.inspection import permutation_importance


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.
    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
    title : string
        Title for the chart.
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.
    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.
        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.
    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='f1_macro')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def plot_feat_distribution(X, keys):
    plt.title('Freature distribution')
    rows = int(len(keys) / 3 + 1)
    for idx, key in enumerate(keys):
        plt.subplot(rows, 3, idx+1)
        plt.title(key)
        X[key].plot(kind='hist', bins=20,  color ='teal')


def plot_feature_importance(clf, X, y, cols):
    result = permutation_importance(clf, X, y, n_repeats=10,
                                    random_state=0, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()

    fig, ax = plt.subplots()
    ax.boxplot(result.importances[sorted_idx].T,
               vert=False, labels=cols[sorted_idx])
    ax.set_title("Permutation Importances (train set)")
    fig.tight_layout()
    plt.show()


def plot_PCA_analysis(X, y, n_components=2, size=5000):
    X = pd.DataFrame(X)
    if X.shape[0] > size:  # reduce number of samples to plot
        # reset index to match is important!
        X.reset_index(inplace=True, drop=True)
        y.reset_index(inplace=True, drop=True)
        X_sample = X.sample(n=size, random_state=0)
        y_sample = y[X_sample.index]
    else:
        X_sample = X
        y_sample = y

    pca = PCA(n_components)
    X_sample = pca.fit_transform(X_sample)

    if(n_components == 2):
        plt.figure()
        plt.legend()
        plt.scatter(X_sample[:, 0], X_sample[:, 1], c=y_sample, cmap='cool')
        plt.legend()
        plt.show()
    else:
        cdict = {0: 'green', 1: 'red'}
        labl = {0: 'Not churn', 1: 'Churn'}
        marker = {0: 'o', 1: 'o'}
        alpha = {0: .3, 1: .7}

        fig = plt.figure(figsize=(7, 5))
        plt.title('PCA')
        ax = fig.add_subplot(111, projection='3d')

        fig.patch.set_facecolor('white')
        for l in np.unique(y_sample):
            ix = np.where(y_sample == l)
            ax.scatter(X_sample[:, 0][ix], X_sample[:, 1][ix], X_sample[:, 2][ix], c=cdict[l], s=40,
                       label=labl[l], alpha=alpha[l])

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")

        ax.legend()
        plt.show()


def plot_curve(y_dev, y_dev_score, min_p=None, min_r=None):
    # roc, optimal threshold
    fpr, tpr, thd_ss = roc_curve(y_dev, y_dev_score)
    tmp1 = tpr - fpr
    ind1 = np.argmax(tmp1, axis=0)
    # pr, optimal threshold
    precision, recall, thd_pr = precision_recall_curve(y_dev, y_dev_score)
    tmp2 = precision + recall
    ind2 = np.argmax(tmp2, axis=0)
    # find optimal threshold
    max_item = 0
    max_idx = 0
    if min_p is None and min_r is None:
        pass
    else:
        if min_p and (min_r is None):
            # (precision min at 0.5, the highest recall)
            for i, p in enumerate(precision):
                if p > min_p and i < len(thd_pr):
                    if recall[i] > max_item:
                        max_item = recall[i]
                        max_idx = i
        elif min_r and (min_p is None):
            # (recall min at 0.8, the highest precision)
            for i, r in enumerate(recall):
                if r > min_r and i < len(thd_pr):
                    if precision[i] > max_item:
                        max_item = precision[i]
                        max_idx = i
        ind2 = max_idx

    opthd = thd_pr[ind2]

    rocauc = auc(fpr, tpr)
    prauc = average_precision_score(y_dev, y_dev_score)

    # plot curve
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot(fpr, tpr, color='darkorange',
             label='ROC curve (area = %0.2f)' % rocauc)

    section = int(len(thd_ss)/10) if len(thd_ss) > 10 else 1
    for i, txt in enumerate(thd_ss):
        if(i % section == 0):
            ax1.annotate('%0.2f' % txt, (fpr[i], tpr[i]))

    ax1.plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax1.vlines(fpr[ind1], plt.ylim()[0], tpr[ind1], color='k',
               linewidth=3, label='Optimum threshold: %0.2f' % thd_ss[ind1])
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC curve')
    ax1.legend(loc="lower right")

    ax2.plot(recall, precision, color='darkorange',
             label='PR curve (area = %0.2f)' % prauc)
    section = int(len(thd_pr)/10) if len(thd_pr) > 10 else 1
    for i, txt in enumerate(thd_pr):
        if(i % section == 0):
            ax2.annotate('%0.2f' % txt, (recall[i], precision[i]))
    ax2.plot([0, 1], [1, 0], color='navy', linestyle='--')
    ax2.vlines(recall[ind2], plt.ylim()[0], precision[ind2], color='k',
               linewidth=3, label='Optimum threshold: %0.2f' % thd_pr[ind2])
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall curve ')
    ax2.legend(loc="lower right")
    plt.show()

    return opthd
