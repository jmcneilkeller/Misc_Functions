def overlap_superiority(group1, group2, n=1000):
    """Estimates overlap and superiority based on a sample.

    group1: scipy.stats rv object
    group2: scipy.stats rv object
    n: sample size
    """

    # Get a sample of size n from both groups
    group1_sample = group1.rvs(n)
    group2_sample = group2.rvs(n)

    # Identify the threshold between samples
    thresh = (group1.mean() + group2.mean()) / 2
    print(thresh)

    # Calculate no. of values above and below for group 1 and group 2 respectively
    above = sum(group1_sample < thresh)
    below = sum(group2_sample > thresh)

    # Calculate the overlap
    overlap = (above + below) / n

    # Calculate probability of superiority
    superiority = sum(x > y for x, y in zip(group1_sample, group2_sample)) / n

    return overlap, superiority


def Cohen_d(group1, group2):

    # Compute Cohen's d.

    # group1: Series or NumPy array
    # group2: Series or NumPy array

    # returns a floating point number

    diff = group1.mean() - group2.mean()

    n1, n2 = len(group1), len(group2)
    var1 = group1.var()
    var2 = group2.var()

    # Calculate the pooled threshold as shown earlier
    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)

    # Calculate Cohen's d statistic
    d = diff / np.sqrt(pooled_var)

    return d

def exp_pdf(mu, x):
    decay_rate = 1 / mu
    return decay_rate * np.exp(-decay_rate * x)


def exp_cdf(mu, x):
    decay_rate = 1 / 4
    return 1 - np.exp(-decay_rate * x)

def calculate_cdf(lst, X):
    # lst = List of all possible values.
    # X = Value for which we want to calculate the CDF.
    count = 0
    for i in lst:
        if i <= X:
            count += 1
        else:
            pass
    cum_prob = count / len(lst)
    return round(cum_prob, 3)

def factorial(n):
    prod = 1
    while n >= 1:
        prod = prod * n
        n = n - 1
    return prod

def binom_distr(n,p,k):
    # (number of trials / num_trials - num_desired_outcomes)
    # *(probability_of_success)**num_desired_outcomes
    # *(1-probability of success)**num_trials - num_desired_outcomes
    binom = (factorial(n) /(factorial(n-k)*factorial(k)))*p**k*(1-p)**(n-k)
    return binom

z score = raw_score - population_mean / population_standard deviation

def sample_means(sample_size, data):

    """
    This function takes in population data as a dictionary along with a chosen sample size
    to generate all possible combinations of given sample size.
    The function calculates the mean of each sample and returns:
    a) a list of all combinations ( as tuples )
    b) a list of means for all sample
    """

    n = sample_size

    # Calculate the mean of population
    mu = calculate_mu(data)
    #print ("Mean of population is:", mu)
    print("Mean of population is:", mu)

    # Generate all possible combinations using given sample size
    combs = list(itertools.combinations(data.keys(), n))

    # Calculate the mean weight (x_bar) for all the combinations (samples) using the given data
    x_bar_list = []

    # Calculate sample mean for all combinations and append to x_bar_list
    for i in range(len(combs)):
        sum = 0

        for j in range(n):
            key = combs[i][j]
            val =data[str(combs[i][j])]
            sum += val

        x_bar = sum/n
        x_bar_list.append(x_bar)


    return combs, x_bar_list

def sample_variance(sample):
    var = sum((sample - sample.mean())**2) / (len(sample) - 1)
    return var

def pooled_variance(sample1, sample2):
    numer = (len(sample1) - 1)*sample_variance(sample1) + (len(sample2) - 1)*sample_variance(sample2)
    denom = (len(sample1) + len(sample2)) - 2
    return numer / denom

def twosample_tstatistic(expr, ctrl):
    numer = expr.mean() - ctrl.mean()
    pool_var = pooled_variance(expr,ctrl)
    n_exp = 1/len(expr)
    n_ctrl = 1/len(ctrl)
    denom = (pool_var*(n_exp+n_ctrl))**0.5
    return numer / denom


def calc_slope(xs,ys):
    return (np.mean(xs) * np.mean(ys) - np.mean(xs*ys)) / (np.mean(xs)**2 - np.mean(xs**2))

def best_fit(xs,ys):
    slope = calc_slope(xs,ys)
    intercept = np.mean(ys) - slope*np.mean(xs)
    return (slope, intercept)

def reg_line (m, c, xs):
    y = []
    for x in xs:
         y.append(m*x+c)

    return y

# Solves for power, effect size, alpha or sample size, given exactly three of those values.
sample_power = TTestPower()
sample_power.solve_power(effect_size=, nobs1=, alpha=, power=)
