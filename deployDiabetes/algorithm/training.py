#importando as bibliotecas

import pandas as pd
from sklearn.model_selection import train_test_split   # função que realiza a divisão do dataset
from sklearn.preprocessing import MinMaxScaler # função para normalização do dataset
from sklearn.metrics import classification_report, confusion_matrix #importação para construção de matrix
from mlxtend.plotting import plot_confusion_matrix # importação para plot de matrix confusão
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import joblib

#importando csv
df = pd.read_csv('C:/deployDiabetes/pima-indians-diabetes.csv', header=None)
df.columns = ['NUM_GRAV', 'CONCENTRACAO_GLICOSE', 'PRESSSAO_DIASTOLICA', 'ESPESSURA_TRICEPS', 'INSULINA', 'IMC',
                  'HISTORICO_FAMILIAR', 'IDADE', 'CLASSIFICACAO']

#transforma os dados em array
entradas = df.iloc[:, :-1].values  #dados de entrada
saida = df.iloc[:, 8].values  # saídas ou target

# realiza o processo de normalização dos dados
normaliza = MinMaxScaler() #objeto para a normalização
entradas_normalizadas=normaliza.fit_transform(entradas)

# realiza a visisão dos dados entre treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(entradas_normalizadas, saida,
test_size=0.30,random_state=42)

#Algoritmo Rede MLP

#treinando o modelo
classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,
10), random_state=1)
classifier.fit(X_train, y_train) # aplica a classificação

#realiza a previsão
y_pred = classifier.predict(X_test)

#constroi a matriz de confusão para comparar o modelo criado
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#realiza o plot da matriz de confusão
matriz_confusao = confusion_matrix(y_test, y_pred)
fig, ax = plot_confusion_matrix(conf_mat=matriz_confusao)
plt.show()

#importando modelo
melhor_modelo = classifier

#salvando o modelo no disco
nome_do_arquivo = 'C:/deployDiabetes/melhor_modelo.sav' #observem a extensão ".sav"
joblib.dump(melhor_modelo, nome_do_arquivo) # melhor_modelo = modelo com maior acurácia
                                            # nome_do_arquivo = caminho do local onde deve ser salvo o modelo
#carregando o modelo
modelo_salvo = joblib.load(nome_do_arquivo) #realiza a carga do modelo salvo