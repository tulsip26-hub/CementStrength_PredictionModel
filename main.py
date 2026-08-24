from src.data_ingestion import data_loader
from src.data_preprocessing import preprocessing
from src.model_building import model_build

def main():

    # Step1: Data Ingestion

    df = data_loader()
    print(df.shape)

    # Step2 : Data Preprocessing
    X_train, X_test, y_train, y_test = preprocessing(df)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    # Step3: model building
    score = model_build(X_train, X_test, y_train, y_test)
    print(f"Model R2 Score: {score}")
main()