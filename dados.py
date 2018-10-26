#Script para realizar a coleta dos dados e normalização
''' Tiago Aro'''


from sklearn.model_selection import train_test_split as splitter
import pandas as pd

xTreino = []
xTeste = []
yTreino = []
yTeste = []

def renomeiaColunas(dataframe):
	dataframe.rename(columns={
	0: 'Class',
	1: 'Alcohol',
 	2: 'Malic acid',
 	3: 'Ash',
	4: 'Alcalinity of ash',
 	5: 'Magnesium',
	6: 'Total phenols',
 	7: 'Flavanoids',
 	8: 'Nonflavanoid phenols',
 	9: 'Proanthocyanins',
	10:'Color intensity',
 	11:'Hue',
 	12:'OD280/OD315 of diluted wines',
 	13:'Proline'
	}, inplace = True)

def retornaDadosNormalizados():
	#Define URL do dataframe
	url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
	#Lê o arquivo para dentro de um dataframe
	dataframe = pd.read_csv(url, header=None)
	#Renomeia as colunas
	renomeiaColunas(dataframe)


	for i in range(0, 10):
		#Separa 80% para treino e 20% para teste
		treino, teste = splitter(dataframe, test_size = 0.2)
		
		#Seleciona somente colunas que serão usadas como atributo
		atrTreino = treino[['Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium']]
		atrTeste = teste[['Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium']]
		#Separa somente coluna de marcação
		marcTreino = treino.loc[:,'Class']
		marcTeste = teste.loc[:,'Class']

		#normalizacao dos dados
		atrTreino_norm = (atrTreino - atrTreino.mean())/atrTreino.std()
		atrTeste_norm = (atrTeste - atrTreino.mean())/atrTreino.std()
		#carrega listas
		yTreino.append(marcTreino)
		yTeste.append(marcTeste)
		xTreino.append(atrTreino_norm)
		xTeste.append(atrTeste_norm)

	return xTreino, xTeste, yTreino, yTeste 