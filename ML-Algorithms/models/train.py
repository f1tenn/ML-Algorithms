import pandas as pd
from src.models.model import create_model
from src.models.model_utils import save_model
from src.models.preprocessing import preprocess_data

def train_and_save_model(X, y, model_filename):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Предобработка
    X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)

    model = create_model()
    model.fit(X_train_scaled, y_train)

    save_model(model, model_filename)

if __name__ == '__main__':
    df = pd.read_csv('data/dataset.csv')
    X = df.drop('label', axis=1)
    y = df['label']
    train_and_save_model(X, y, 'model/rf_model.pkl')
