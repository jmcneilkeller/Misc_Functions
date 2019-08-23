#
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


# Create train, test, split
# Random state is our choice. Defaults to 20% for test, but adjustable.
X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=34,test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Instantiate scaler object.
scalar = StandardScaler()
# Scalar learns the std, etc.. for all features. No need to fit your test data.
scalar.fit(X_train)
# This does the scaling.
X_train_scaled  = scalar.transform(X_train)
X_test_scaled = scalar.transform(X_test)

# Run your regression.
lm = LinearRegression()
lm.fit(X_train_scaled,y_train)
y_train_pred = lm.predict(X_train_scaled)

# Check your RMSE
train_rmse = np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))

print('Root Mean Squared Error:' , train_rmse)

# Scale your test
X_test_scaled = scalar.transform(X_test)
# Predict and get your test RMSE.
y_test_pred = lm.predict(X_test_scaled)

test_rmse = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
print('Root Mean Squared Error:' + str(np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))))

# THEN PLOT.

from sklearn.preprocessing import PolynomialFeatures

# Create polynomials first, then scale.
poly_2 = PolynomialFeatures(degree=2, include_bias=False)
poly_2.fit(X_train)
X_train_2= pd.DataFrame(poly_2.transform(X_train), columns=poly_2.get_feature_names(columns))

columns_2  = poly_2.get_feature_names(columns)

scalar_2 = StandardScaler()

scalar_2.fit(X_train_2)
X_train_2_scaled  = scalar_2.transform(X_train_2)

lm2 = LinearRegression()
model2 = lm2.fit(X_train_2_scaled, y_train)
y_train_2_pred = lm2.predict(X_train_2_scaled)

train_2_rmse = np.sqrt(metrics.mean_squared_error(y_train, y_train_2_pred))

X_test_2= pd.DataFrame(poly_2.transform(X_test), columns=poly_2.get_feature_names(columns))
X_test_2_scaled = scalar_2.transform(X_test_2)

y_test_pred_2 = lm2.predict(X_test_2_scaled)

test_2_rmse = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred_2))

# test2_mae = metrics.mean_absolute_error(y_test2, y_pred2)

print(train_2_rmse, test_2_rmse)

## RIDGE
# Alpha is our hyperparameter. Adjust to test.
ridgeReg = Ridge(alpha=0.1, normalize=True)

ridgeReg.fit(X_train_2_scaled, y_train)

y_pred_ridge = ridgeReg.predict(X_test_2_scaled)

#calculating rmse
RMSE_R01 =np.sqrt(metrics.mean_squared_error(y_test, y_pred_ridge))
print('Test RMSE:', RMSE_R01)

# LASSO
from sklearn.linear_model import Lasso
# Adjust alpha to change how you regularize.
lasso = Lasso(alpha=0.1, normalize=False)

lasso.fit(X_train,y_train)

y_train_pred = lasso.predict(X_train)
y_pred = lasso.predict(X_test)

train_rmse = metrics.mean_absolute_error(y_train, y_train_pred)
test_rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
print('Training Error: '+ str(train_rmse) )
print('Testing Error: '+ str(test_rmse) )

lasso_coef01 = pd.DataFrame(data=lasso.coef_).T
lasso_coef01.columns = X_train.columns
lasso_coef01 = lasso_coef01.T.sort_values(by=0).T
lasso_coef01.plot(kind='bar', title='Modal Coefficients', legend=True, figsize=(16,8))

# Logistic regression model
def logiRegr(X_train, y_train, X_test, y_test, params):
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    # examine coefficients
    # print(zip(X_train.columns, logreg.coef_[0]))
    # print(logreg.coef_)
    # class predictions (not predicted probabilities)
    y_pred_class = logreg.predict(X_test)
    print('Classification Accuracy: ', metrics.accuracy_score(y_test, y_pred_class))
    print('Test F1 score: ', f1_score(y_test, y_pred_class))
    # plot confusion matrix
    cm = confusion_matrix(y_test,y_pred_class)
    classes = ['DEFAULT', 'NON-default']
    plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues)

# Decision tree classifier
def deciTree(X_train, y_train, X_test, y_test):
    clf = DecisionTreeClassifier() # Create Decision Tree classifer object
    clf = clf.fit(X_train,y_train) # Train Decision Tree Classifer
    y_pred_train = clf.predict(X_train) #predict the training set
    y_pred_test = clf.predict(X_test) #Predict the response for test dataset

    # Model Accuracy, how often is the classifier correct?
    print("Training F1 Score:",metrics.f1_score(y_train, y_pred_train))
    print("Testing F1 Score:",metrics.f1_score(y_test, y_pred_test))


def knn(k_number, X_train, y_train, X_test, y_test):
    knn = KNeighborsClassifier(n_neighbors=k_number)
    # train
    knn.fit(X_train, y_train)
    # test
    y_pred_class = knn.predict(X_test)


    # check Accuracy
    print('Test Accuracy score: ', accuracy_score(y_test, y_pred_class))
    # check F1 score
    print('Test F1 score: ', f1_score(y_test, y_pred_class))
    return y_pred_class

# For removing outliers.
# For each column, first it computes the Z-score of each value in the column, relative to the column mean and standard deviation.
# Then is takes the absolute of Z-score because the direction does not matter, only if it is below the threshold.
# all(axis=1) ensures that for each row, all column satisfy the constraint.
# Finally, result of this condition is used to index the dataframe.
from scipy import stats
df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
