import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml

# Carregar o dataset Boston Housing
boston = fetch_openml(name="boston", version=1, as_frame=True)
df = boston.frame

# Selecionar as variáveis de interesse
X = df[["RM", "PTRATIO", "LSTAT"]]
y = df["MEDV"]

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Padronizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Treinar o modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Salvar o modelo treinado
with open("modelo_regressao.pkl", "wb") as file:
    pickle.dump(modelo, file)

# Salvar o scaler também
with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

print("✅ Modelo treinado e salvo como 'modelo_regressao.pkl'!")
