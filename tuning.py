from sklearn.model_selection import GridSearchCV
import pandas as pd
from operator import itemgetter


def grid_tune(estimator, params, cv, scoring, X_train, y_train, X_test, y_test):
    #create the instance of GridSearchCV
    gs = GridSearchCV(estimator, params, cv=cv, scoring=scoring)
    #fit the Gridsearch to our data
    gs.fit(X_train,y_train)
    # examine the best model
    print('Training Best Score: ', gs.best_score_, '\n')
    print('Training Best Params:  \n', gs.best_params_, '\n\n')
    print('Training Best Estimator:  \n', gs.best_estimator_, '\n\n')

    return gs.best_params_

def evaluate_param(parameter, num_range, index):
    grid_search = GridSearchCV(clf, param_grid = {parameter: num_range})
    grid_search.fit(X_train[features], y_train)

    df = {}
    for i, score in enumerate(grid_search.grid_scores_):
        df[score[0][parameter]] = score[1]


    df = pd.DataFrame.from_dict(df, orient='index')
    df.reset_index(level=0, inplace=True)
    df = df.sort_values(by='index')

    plt.subplot(3,2,index)
    plot = plt.plot(df['index'], df[0])
    plt.title(parameter)
    return plot, df

index = 1
plt.figure(figsize=(16,12))
for parameter, param_range in dict.items(param_grid):
    evaluate_param(parameter, param_range, index)
    index += 1

def report(grid_scores, n_top):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.4f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")
