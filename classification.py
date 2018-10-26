#Script para classificação de dados de vinho com kNN e MLP
''' Tiago Aro'''

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from dados import retornaDadosNormalizados


#Declara bases de treino e teste para atributos e marcações
xTreino = []
xTeste = []
yTreino = []
yTeste = []
#retorna bases de treino e teste
xTreino, xTeste, yTreino, yTeste = retornaDadosNormalizados()

def classificacaoKNN():
	resultado = []
	for k in [3, 5, 9]:
		acerto = 0
		#Instância kNN
		knn = KNeighborsClassifier(n_neighbors=k)
		for i in range(0, 10):
			#Treinamento dos dados
			knn.fit((xTreino[i]), (yTreino[i]))
			#Teste dos dados
			classificacao = knn.predict(xTeste[i])
			#Iteração para verificar se acertou ou não o resultado
			for res in (classificacao - yTeste[i]):
				if res == 0:
					acerto = acerto + 1
		#Armazena resultado da classificação
		resultado.append( "K: " + str(k) + " | Media acertos: " + str(round((acerto/(len(yTeste[0]) * len(yTeste))) * 100, 2)) + " %")
	print("Classificacao kNN: ")
	print(resultado)

def classificacaoMLP():
	resultado = []
	quantidadeEntradas = len(xTeste[0].columns)
	#Parâmetros do MLP
	activation = 'logistic' 
	solver = 'sgd' 
	learning_rate =  0.03
	batch_size=50
	max_iter = 100000

	for k in [1, 2, 3, 4]:
		acerto = 0
		# Tamanhos das camadas ocultas multiplicado pela quantidade de entradas
		hidden_layer_sizes = [k * quantidadeEntradas]
		mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, 
						learning_rate_init=learning_rate, batch_size=batch_size, max_iter=max_iter)

		for i in range(0, 10):
			#Treinamento dos dados
			mlp.fit((xTreino[i]), (yTreino[i]))
			#Teste dos dados
			classificacao = mlp.predict(xTeste[i])
			#Iteração para verificar se acertou ou não o resultado
			for res in (classificacao - yTeste[i]):
				if res == 0:
					acerto = acerto + 1
		#Armazena resultado da classificação
		resultado.append("K: " + str(k * quantidadeEntradas) + " | Media acertos: " + str(round((acerto/(len(yTeste[0]) * len(yTeste))) * 100, 2)) + " %")
	print("Classificacao MLP:")
	print(resultado)

#Chama funções
classificacaoKNN()
classificacaoMLP()







