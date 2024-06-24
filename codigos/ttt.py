# Importando as bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import silhouette_score
import os

# Configurando os ambientes do TensorFlow e desativando avisos
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Carregando os dados de treinamento e teste
train_data = pd.read_csv('/kaggle/input/playground-series-s4e2/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s4e2/test.csv')

# Mapeando as variáveis categóricas para valores numéricos
train_data['family_history_with_overweight'] = train_data['family_history_with_overweight'].map({'yes': 1, 'no': 0})
train_data['FAVC'] = train_data['FAVC'].map({'yes': 1, 'no': 0})
train_data['CAEC'] = train_data['CAEC'].map({'Always': 4, 'Frequently': 3, 'Sometimes': 2, 'no': 1})
train_data['SMOKE'] = train_data['SMOKE'].map({'yes': 1, 'no': 0})
train_data['SCC'] = train_data['SCC'].map({'yes': 1, 'no': 0})
train_data['Gender'] = train_data['Gender'].map({'Male': 1, 'Female': 0})
train_data['CALC'] = train_data['CALC'].map({'Frequently': 3, 'Sometimes': 2, 'no': 1})
train_data['MTRANS'] = train_data['MTRANS'].map({'Automobile': 5, 'Bike': 4, 'Motorbike': 3, 'Public_Transportation': 2, 'Walking': 1})
train_data['NObeyesdad'] = train_data['NObeyesdad'].map({'Obesity_Type_III': 6, 'Obesity_Type_II': 5, 'Obesity_Type_I': 4, 'Overweight_Level_II': 3, 'Overweight_Level_I': 2, 'Normal_Weight': 1, 'Insufficient_Weight': 0})

# Selecionando as colunas para o clustering
cluster_cols = ['Age', 'FAVC', 'CAEC', 'Gender', 'CALC', 'MTRANS', 'Height', 'Weight']
cluster_data = train_data[cluster_cols]

# Padronizando os dados
scaler_cluster = StandardScaler()
cluster_data_scaled = scaler_cluster.fit_transform(cluster_data)

# Validando o número ideal de clusters usando validação cruzada e silhouette score
silhouette_scores = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    fold_scores = []
    for train_index, val_index in kf.split(cluster_data_scaled):
        X_train, X_val = cluster_data_scaled[train_index], cluster_data_scaled[val_index]
        kmeans.fit(X_train)
        val_labels = kmeans.predict(X_val)
        fold_score = silhouette_score(X_val, val_labels)
        fold_scores.append(fold_score)
    silhouette_score_mean = np.mean(fold_scores)
    silhouette_scores.append(silhouette_score_mean)

# Encontrando o melhor número de clusters com base no silhouette score
best_k = np.argmax(silhouette_scores) + 2

# Imprimindo o melhor número de clusters
print("Melhor número de clusters:", best_k)

# Aplicando KMeans com o melhor número de clusters encontrado
kmeans = KMeans(n_clusters=best_k, random_state=42)
kmeans.fit(cluster_data_scaled)

# Adicionando rótulos de cluster aos dados de treinamento
train_data['Cluster_Labels'] = kmeans.labels_

# Criando uma tabela de ID e rótulos de cluster
id_cluster_list = []
for index, row in train_data.iterrows():
    id_cluster_list.append({'ID': int(row['id']), 'Cluster_Label': int(row['Cluster_Labels'])})
id_cluster_table = pd.DataFrame(id_cluster_list)
print(id_cluster_table)

# Calculando as médias de cada cluster para as variáveis selecionadas
cluster_means = train_data.groupby('Cluster_Labels')[cluster_cols].mean()
print(cluster_means)

# Visualizando a distribuição dos clusters
plt.figure(figsize=(8, 5))
sns.countplot(x='Cluster_Labels', data=train_data, palette='viridis')
plt.title('Distribuição dos Clusters')
plt.xlabel('Clusters')
plt.ylabel('Quantidade')
plt.show()

# Visualizando a distribuição de peso por cluster
plt.figure(figsize=(10, 8))
sns.boxplot(x='Cluster_Labels', y='Weight', data=train_data, palette='viridis')
plt.title('Distribuição de Peso por Cluster')
plt.xlabel('Cluster')
plt.ylabel('Peso (kg)')
plt.show()

# Visualizando a distribuição de peso em histogramas por cluster
plt.figure(figsize=(15, 10))
num_clusters = train_data['Cluster_Labels'].nunique()
cluster_labels_sorted = sorted(train_data['Cluster_Labels'].unique())

for i, cluster_label in enumerate(cluster_labels_sorted, 1):
    plt.subplot(2, num_clusters // 2 + num_clusters % 2, i)
    sns.histplot(train_data[train_data['Cluster_Labels'] == cluster_label]['Weight'], 
                 bins=20, kde=True, color='skyblue', label=f'Cluster {cluster_label}')
    plt.title(f'Distribuição de Peso - Cluster {cluster_label}')
    plt.xlabel('Peso (kg)')
    plt.ylabel('Frequência')

plt.tight_layout()
plt.show()

# Pré-processamento dos dados para a rede neural
X = train_data.drop(columns=['id', 'NObeyesdad', 'Cluster_Labels'])
y = train_data['NObeyesdad']
X = pd.get_dummies(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Construindo e treinando o autoencoder
encoder = Sequential([
    Dense(32, activation='relu', input_shape=(X.shape[1],)),
    Dense(X.shape[1], activation='relu')
])
encoder.compile(optimizer='adam', loss='mse')

encoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_val, X_val))

# Codificando os dados de treinamento e validação
X_train_encoded = encoder.predict(X_train)
X_val_encoded = encoder.predict(X_val)

# Construindo e treinando o modelo de classificação
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train_encoded.shape[1],)),
    Dense(7, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train_encoded, y_train, epochs=50, batch_size=32, validation_data=(X_val_encoded, y_val))

# Avaliando a acurácia do modelo
_, accuracy = model.evaluate(X_val_encoded, y_val)
print(f'Acurácia do Modelo: {accuracy * 100:.2f}%')

# Obtendo os pesos das camadas da rede neural para calcular a importância das características
layer_weights = model.layers[0].get_weights()[0]
feature_importance = np.mean(np.abs(layer_weights), axis=0)

# Mapeando os nomes das colunas originais
original_columns = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS']

# Agrupando os pesos das características por categoria
grouped_weights = []
for col in original_columns:
    idx = original_columns_onehot.str.startswith(col)
    col_weights = feature_importance[idx]
    grouped_weights.append(np.mean(col_weights))
feature_importance_no_onehot = np.array(grouped_weights)

# Criando um DataFrame com a importância das características
importance_df = pd.DataFrame({'Atributos': original_columns, 'Importância': feature_importance_no_onehot})

# Ordenando as características por importância
importance_df = importance_df.sort_values(by='Importância', ascending=False)

# Mapeando os nomes das características para tornar mais legíveis
meaning = {
    'Gender': 'Gênero',
    'Age': 'Idade',
    'Height': 'Altura',
    'Weight': 'Peso',
    'family_history_with_overweight': 'Histórico familiar de sobrepeso',
    'FAVC': 'Consumo de alimentos com alto teor calórico',
    'FCVC': 'Consumo de vegetais',
    'NCP': 'Número de refeições principais por dia',
    'CAEC': 'Consumo de alimentos entre refeições',
    'SMOKE': 'Fumante',
    'CH2O': 'Consumo diário de água',
    'SCC': 'Prática de atividades físicas',
    'FAF': 'Frequência de atividade física',
    'TUE': 'Tempo gasto assistindo TV ou usando computador',
    'CALC': 'Consumo de bebidas alcoólicas',
    'MTRANS': 'Meio de transporte'
}

# Aplicando os nomes legíveis às características
importance_df['Atributos'] = importance_df['Atributos'].map(meaning)

# Visualizando a importância das características
plt.figure(figsize=(10, 8))
sns.barplot(x='Importância', y='Atributos', data=importance_df, palette='viridis')
plt.title('Importância dos Atributos')
plt.xlabel('Importância')
plt.ylabel('Atributos')
plt.show()

# Pré-processamento dos dados de teste e realização de previsões
test_data_processed = pd.get_dummies(test_data.drop(columns=['id']))
test_data_processed = test_data_processed.reindex(columns=X.columns, fill_value=0)
X_test_scaled = scaler.transform(test_data_processed)
X_test_encoded = encoder.predict(X_test_scaled)
predictions = model.predict(X_test_encoded)
predicted_labels = np.argmax(predictions, axis=1)

# Mapeando as previsões de volta para as categorias originais
category_map = {
    0: 'Insufficient_Weight',
    1: 'Normal_Weight',
    2: 'Overweight_Level_I',
    3: 'Overweight_Level_II',
    4: 'Obesity_Type_I',
    5: 'Obesity_Type_II',
    6: 'Obesity_Type_III'
}

# Criando o arquivo de saída com as previsões
output = pd.DataFrame({'id': test_data['id'], 'NObeyesdad': predicted_labels})
output['NObeyesdad'] = output['NObeyesdad'].map(category_map)
output.to_csv('submission.csv', index=False)
