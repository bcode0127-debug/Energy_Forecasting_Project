from sklearn.linear_model import ElasticNetCV

def train_elastic_net(X_train, y_train):

    # using cross-valiadtion to find the optimal L1/L2 balance

    model = ElasticNetCV(
        l1_ratio = [.1, .5, .7, .9, .95, .99, 1],
        cv = 5,
        random_state = 42
    )
    model.fit(X_train, y_train)

    return model

