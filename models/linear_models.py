from sklearn.linear_model import ElasticNetCV, HuberRegressor, LinearRegression

def train_elastic_net(X_train, y_train):

    # using cross-valiadtion to find the optimal L1/L2 balance

    model = ElasticNetCV(
        l1_ratio = [.1, .5, .7, .9, .95, .99, 1],
        cv = 5,
        random_state = 42
    )
    model.fit(X_train, y_train)

    return model

def train_ols(X_train, y_train):

    # Ordinary Leasst Squares(OLS) - baseline linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model

def train_huber(X_train, y_train):

    # Huber regression - robust to outliers 
    model = HuberRegressor(epsilon = 1.35, max_iter = 200)
    model.fit(X_train, y_train)

    return  model

def train_wls(X_train, y_train, sample_weights):

    # weighted Least Squares - addresses hetroscedasticity
    model = LinearRegression()
    model.fit(X_train, y_train, sample_weight=sample_weights)

    return model

    
