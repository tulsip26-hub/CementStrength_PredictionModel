# importing necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler,LabelEncoder,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer



def preprocessing(df):

    # Dropping the Duplicate Values
    df.drop_duplicates(inplace=True)

    # Sagregateing numerical and categorical columns
    numerical_data = df.select_dtypes(exclude = 'object').columns
    categorical_data = df.select_dtypes(include = 'object').columns

    # Split the data into X & y
    X = df.drop(columns=['Concrete compressive strength(MPa, megapascals) '])
    y = df['Concrete compressive strength(MPa, megapascals) ']

    # Using Train & Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=0.2, random_state=42)

    # Using Scaling Techniques
    sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train) # Seen Data
    X_test = sc.transform(X_test) # Unseen Data

    return X_train, X_test, y_train, y_test