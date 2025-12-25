from sklearn.ensemble import HistGradientBoostingRegressor

def train_gbr_model(X_train, y_train):

    # non-linear ensemble phase
    print("Initiating Phase 2: Gradient Boosting Regressor...")
    
    gbr = HistGradientBoostingRegressor(
        max_iter=100,          # Number of trees
        learning_rate=0.1,     # How much each tree corrects the previous one
        max_depth=5,           # Complexity of each tree
        random_state=42,
        scoring='neg_root_mean_squared_error'
    )
    
    gbr.fit(X_train, y_train)
    return gbr