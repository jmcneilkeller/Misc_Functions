from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import itertools
import matplotlib.pyplot as plt
from matplotlib.ticker import IndexLocator
import pprint
from sklearn.svm import SVC

def plot_cm(y_test, y_pred_class,classes=['NON-default','DEFAULT']):
    # plots confusion matrix
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_test, y_pred_class)

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    plt.title("Confusion Matrix")
    ax.set(yticks=[-0.5, 1.5],
           xticks=[0, 1],
           yticklabels=classes,
           xticklabels=classes)
    ax.yaxis.set_major_locator(IndexLocator(base=1, offset=0.5))
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def scores(y_test, y_pred_class):
    # Prints formatted classification metrics.
    print('Classification Accuracy: ', format(accuracy_score(y_test, y_pred_class), '.3f'))
    print('Precision score: ', format(precision_score(y_test, y_pred_class), '.3f'))
    print('Recall score: ', format(recall_score(y_test, y_pred_class), '.3f'))
    print('F1 score: ', format(f1_score(y_test, y_pred_class), '.3f'))

def svmClass(X_train, y_train, X_test, y_test, **kwargs):
    # Instantiate model. Use kwargs to pass parameters.
    # Pass GridSearch best params with ** to unpack.
    svm = SVC(random_state=1,**kwargs)
    # Fit to training data.
    svm.fit(X_train, y_train)
    # Class predictions
    y_pred_class = svm.predict(X_test)
    # Scoring metrics
    scores(y_test, y_pred_class)
    # Plot confusion matrix
    plot_cm(y_test,y_pred_class)

def logiRegr(X_train, y_train, X_test, y_test,**kwargs):
    # Instantiate model. Use kwargs to pass parameters.
    # Pass GridSearch best params with ** to unpack.
    logreg = LogisticRegression(random_state=1,**kwargs)
    # Fit to training data.
    logreg.fit(X_train, y_train)
    # Examine coefficients
    pprint.pprint(list(zip(X_train.columns,logreg.coef_[0])))
    # Class predictions (not predicted probabilities)
    y_pred_class = logreg.predict(X_test)
    # Scoring metrics
    scores(y_test, y_pred_class)
    # Plot confusion matrix
    plot_cm(y_test,y_pred_class)


def deciTree(X_train, y_train, X_test, y_test,**kwargs):
    # Instantiate model. Use kwargs to pass parameters.
    # Pass GridSearch best params with ** to unpack.
    dt = DecisionTreeClassifier(random_state=1, **kwargs)
    # Fit to training data.
    dt.fit(X_train,y_train)
    # Class predictions
    y_pred_class = dt.predict(X_test)
    # Scoring metrics
    scores(y_test, y_pred_class)
    # Confusion matrix
    plot_cm(y_test,y_pred_class)

def randomForest(X_train, y_train, X_test, y_test,**kwargs):
    # Instantiate model. Use kwargs to pass parameters.
    # Pass GridSearch best params with ** to unpack.
    rf = RandomForestClassifier(random_state=1, **kwargs)
    # Fit to training data.
    rf.fit(X_train,y_train)
    # Class predictions
    y_pred_class = rf.predict(X_test)
    # Scoring metrics
    scores(y_test, y_pred_class)
    # Confusion matrix
    plot_cm(y_test,y_pred_class)

def xgbClass(X_train, y_train, X_test, y_test,**kwargs):
    # Instantiate model. Use kwargs to pass parameters.
    # Pass GridSearch best params with ** to unpack.
    xg = xgb.XGBClassifier(seed=1,**kwargs)
    # Fit to training data.
    xg.fit(X_train,y_train)
    # Class predictions
    y_pred_class = xg.predict(X_test)
    # Scoring metrics
    scores(y_test, y_pred_class)
    # Confusion matrix
    plot_cm(y_test,y_pred_class)
# For removing outliers.
# For each column, first it computes the Z-score of each value in the column, relative to the column mean and standard deviation.
# Then is takes the absolute of Z-score because the direction does not matter, only if it is below the threshold.
# all(axis=1) ensures that for each row, all column satisfy the constraint.
# Finally, result of this condition is used to index the dataframe.
from scipy import stats
df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
