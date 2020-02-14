import pandas as pd
from sklearn.feature_selection import RFECV
import pprint

def rfe(estimator,step,cv,scoring,X_train,y_train):
    # Instantiate Recursive Feature Elimination
    selector = RFECV(estimator=estimator, step=step, cv=cv, scoring=scoring)
    selector.fit(X_train,y_train)
    
    selected_columns = X_train.columns[selector.support_]
    removed_columns = X_train.columns[~selector.support_]
    print('*'*20+'SELECTED'+'*'*19)
    pprint.pprint(list(selected_columns))
    print('\n'+'*'*20+'REMOVED'+'*'*20)
    pprint.pprint(list(removed_columns))
    return selected_columns

# Normalizing data for SelectKBest "chi2".

normed_data= (data - data.min(0)) / data.ptp(0)