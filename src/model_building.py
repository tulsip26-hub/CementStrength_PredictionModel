from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def model_build(X_train, X_test, y_train, y_test):
    # Creating the Random Forest Regressor Model

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    score = r2_score(y_test, y_pred)

    return score