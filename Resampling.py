def resamplingDataPrep(X_train, y_train, target_var): # ! should be be call upSamplingDataPrep?
    # concatenate our training data back together
    resampling = X_train.copy()
    resampling[target_var] = y_train.values
    # separate minority and majority classes
    majority_class = resampling[resampling[target_var]==0]
    minority_class = resampling[resampling[target_var]==1]
    # Get a class count to understand the class imbalance.
    print('majority_class: '+ str(len(majority_class)))
    print('minority_class: '+ str(len(minority_class)))
    return majority_class, minority_class

def upSampleMinority(target_var, minority_class, majority_class):  # ! double check params needed
    # upsample minority
    minority_upsampled = resample(minority_class,
                          replace=True, # sample with replacement
                          n_samples=len(majority_class), # match number in majority class
                          random_state=23) # reproducible results
    # combine majority and upsampled minority
    upsampled = pd.concat([majority_class, minority_upsampled])
    # check new class counts
    print(upsampled[target_var].value_counts())
    # return new upsampled X_train, y_train
    X_train_upsampled = upsampled.drop(target_var, axis=1)
    y_train_upsampled = upsampled[target_var]
    return X_train_upsampled, y_train_upsampled

def upSampleMinoritySMOTE(X_train, y_train):
    sm = SMOTE(random_state=23, ratio=1.0)
    X_train_sm, y_train_sm = sm.fit_sample(X_train, y_train)
    print(len(X_train_sm), len(y_train_sm))
    return X_train_sm, y_train_sm

def downSampleMajority(target_var, minority_class, majority_class):
    # downsample majority
    majority_downsampled = resample(majority_class,
                                    replace = False, # sample without replacement
                                    n_samples = len(defaulted), # match minority n
                                    random_state = 23) # reproducible results
    # combine majority and upsampled minority
    downsampled = pd.concat([majority_downsampled, minority_class])
    # check new class counts
    print(downsampled[target_var].value_counts())
    # return new downsampled X_train, y_train
    X_train_downsampled = downsampled.drop(target_var, axis=1)
    y_train_downsampled = downsampled[target_var]
    return X_train_downsampled, y_train_downsampled

def downSampleMajorityTomekLinks(X_train, y_train):
    tl = TomekLinks()
    X_train_tl, y_train_tl = tl.fit_sample(X_train, y_train)
    print(X_train_tl.count(), len(y_train_tl))
    return X_train_tl, y_train_tl
