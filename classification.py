from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from dados import retornaDados


#Declara bases de treino e teste para atributos e marcações
xTreino = []
xTeste = []
yTreino = []
yTeste = []
#retorna bases de treino e teste
xTreino, xTeste, yTreino, yTeste = retornaDados()

def classificacaoKNN():

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
		resultado = "K: " + str(k) + " | Media acertos: " + str(((acerto/(len(yTeste[0]) * len(yTeste))) * 100))
	print("Classificacao kNN: ")
	print(resultado)

def classificacaoMLP():
	#Parâmetros do MLP
	activation = 'logistic' 
	solver = 'sgd' 
	learning_rate =  0.03
	batch_size=50
	max_iter = 100000

	for k in [1, 2, 3, 4]:
		acerto = 0
		# Tamanhos das camadas ocultas
		hidden_layer_sizes = [k]
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
		resultado = "K: " + str(k) + " | Media acertos: " + str(((acerto/(len(yTeste[0]) * len(yTeste))) * 100))
	print("Classificacao MLP:")
	print(resultado)

#Chama funções
classificacaoKNN()
classificacaoMLP()







