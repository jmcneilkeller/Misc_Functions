import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_style('darkgrid',{'axes.edgecolor': '.9'})

def kde_hist(dataframe):
    # Creates histograms/KDE of DataFrame columns.
    for column in dataframe:
        dataframe[column].plot.hist(normed=True, label = column+' histogram')
        dataframe[column].plot.kde(label =column+' kde')
        plt.legend()
        plt.show()

def

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16,3))

for xcol, ax in zip(['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot'], axes):
    data.plot(kind='scatter', x=xcol, y='price', ax=ax, alpha=0.4, color='b')

def heatmap_corr(dataframe):
    # Plots a heatmap of correlation between features with masking.
    fig, ax = plt.subplots(figsize=(20,20))
    mask=np.zeros_like(dataframe.corr(), dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    color_map = sns.color_palette("hot_r")
    ax = sns.heatmap(dataframe.corr(), cmap = color_map, mask=mask, square=True, annot=True)


# Seaborn kde
sns.kdeplot(values)

# matplotlib uniform distribution with normalization.
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
values = np.random.uniform(-10.0, 10.0, 100000)
plt.hist(values, 50, normed=True)

plt.show();

# Poisson distribution
from scipy.stats import poisson
mu = 500
x = np.arange(400, 600, 0.5)
plt.plot(x, poisson.pmf(x, mu))
plt.axvline(550, color= 'g')

# Residuals plot

sns.set(style="whitegrid")

#residual plot - Seaborn

sns.residplot(y_test_pred, y_test, lowess=True, color="g")

# Confusion matrix
def plot_cm(y_test, y_pred_class,classes):
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

def distplot(data,bins=10,color='indigo'):
    plt.rcParams["patch.force_edgecolor"] = True
    for column in data:
        sns.distplot(data[column], bins=bins, color=color,
                          hist_kws=dict(edgecolor="k", linewidth=2))
        plt.show()

def visualize_chisq(chisq_stat, df, alpha):
    # initialize a matplotlib "figure"
    fig = plt.figure(figsize=(16,10))
    # get the current "axis" out of the figure
    ax = fig.gca()

    # X-values will be adjusted for each graph.
    xs = np.linspace(0, 7, 50)
    ys = stats.chi2.pdf(xs, df)
    ax.plot(xs, ys, 'r-', lw=5, alpha=alpha, label='chi2 pdf')
    # plot the lines using matplotlib's plot function:
    ax.plot(xs, ys, linewidth=2, color='darkblue')

    ax.xlabel = 'Chi-Sqaured Values'
    # plot a vertical line for our measured difference in rates t-statistic
    ax.axvline(chisq_stat, color='red', linestyle='--', lw=5,label='chi-sq-statistic')
    chi_sq_crit = stats.chi2.ppf(1-alpha, df)
    ax.plot(xs, ys, linewidth=1, color='darkblue')
    ax.axvline(chi_sq_crit,color='green',linestyle='--',lw=4,label='crit chi-sq-value')
    ax.fill_betweenx(ys,xs,chi_sq_crit, where= xs > chi_sq_crit)

    ax.legend()
    plt.show()
    return None

def plotSVC(title):
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() — 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() — 1, X[:, 1].max() + 1
    h = (x_max / x_min)/100
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h))
    plt.subplot(1, 1, 1)
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel(‘Sepal length’)
    plt.ylabel(‘Sepal width’)
    plt.xlim(xx.min(), xx.max())
    plt.title(title)
    plt.show()


fig= go.Figure(data=[
    go.Bar(name=name, x=time_var, y=target),
    go.Bar(name=name, x=time_var, y=target)
    ])
fig.add_trace(go.Scatter(name=name, x=time_var, y=average, mode='lines'))
fig.add_trace(go.Scatter(name=name, x=time_var, y=finalaverage, mode='lines'))
# Change the bar mode
fig.update_layout(barmode='group')
fig.update_layout(title_text=title)
fig.show()
