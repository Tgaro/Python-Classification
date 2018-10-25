from sklearn.neighbors import KNeighborsClassifier
from dados import retornaDados


#Declara bases de treino e teste para atributos e marcações
xTreino = []
xTeste = []
yTreino = []
yTeste = []
#retorna bases de treino e teste
xTreino, xTeste, yTreino, yTeste = retornaDados()

def classificacaoKNN():
	resultado = []
	for k in [3, 5, 9]:
		for i in range(0, 9):
			acerto = 0
			knn = KNeighborsClassifier(n_neighbors=k)
			knn.fit((xTreino[i]), (yTreino[i]).astype('long'))
			
			classificacao = knn.predict(xTeste[i])
			
			for res in (classificacao - yTeste[i].astype('long')):
				if res == 0:
					acerto = acerto + 1
					
			resultado.append([k, i, ((acerto/len(yTeste[i]))* 100)])

	print(resultado)


classificacaoKNN()








