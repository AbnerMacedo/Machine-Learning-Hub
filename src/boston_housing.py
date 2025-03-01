import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml

# Carregando o DataSet 
boston = fetch_openml(name="boston", version=1, as_frame=True)

# Criando o Dataframe com Pandas
df = boston.data 

# Variável alvo / preço das casas
df["MEDV"] = boston.target

# Exibindo as primeiras linhas do Dataset 
#print(df.head()) 

# Informações gerais do dataset
#print(df.info())

# Informações estatística
#print(df.describe())

# Identificando valores ausentes
#print(df.isnull().sum)

# Preenchendo os valores nulos com a média, por se tratar de um dataset com poucos dados, trabalhando com colunas numéricas e categóricas.
df["CHAS"] = df["CHAS"].astype(int)
df["RAD"] = df["RAD"].astype(int)
df.fillna(df.mean(), inplace=True)

#Plotando a figura para entender outliers
#plt.figure(figsize=(10,6))
#sns.boxplot(data=df)
#plt.xticks(rotation=90)
#plt.show()

#Tratando os outliers que nesse datasete vamos excluir os que estejam al[em de 1,5x o intervalo de interquartil (IQR).
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1 

df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

from sklearn.model_selection import train_test_split

# separar variável independente (X) e variável alvo (Y)
X = df.drop(columns=["MEDV"]) # todas as culunas, menos o preço da casa
y = df["MEDV"] # variável alvo

# dividir os dados em 80% treino e 20% teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# exibir os tamanhos dos conjuntos de treino e teste
#   print(f"Tamanho do treino: {X_train.shape}, Tamanho do testo: {X_test.shape}")

from sklearn.linear_model import LinearRegression

# criando o modelo
modelo = LinearRegression()

# Treinando o Modelo nos dados de treino 
modelo.fit(X_train, y_train)

# Exibir os coeficientes do modelo
#   print("coeficientes:", modelo.coef_)
#   print("intercepto:", modelo.intercepto_)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Fazer previsões nos dados de teste
y_pred = modelo.predict(X_test)

# Calcular métricas
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Exibir os resultados
print(f"r²: {r2:.4f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")