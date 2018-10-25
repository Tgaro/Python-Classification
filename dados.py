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

def retornaDados():
	#Define URL do dataframe
	url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
	#Lê o arquivo para dentro de um dataframe
	dataframe = pd.read_csv(url, header=None)
	#Renomeia as colunas
	renomeiaColunas(dataframe)
	#Separa somente atributos
	subdfX = dataframe.drop(columns=['Class'])
	#Separa somente marcação
	subdfY = dataframe.loc[:,'Class']

	for i in range(0, 10):
		atrTreino, atrTeste = splitter(subdfX, test_size = 0.2)
		marcTreino, marcTeste = splitter(subdfY, test_size = 0.2)
		
		#normalizacao dos dados
		atrTreino_norm = (atrTreino - atrTreino.mean())/atrTreino.std()
		atrTeste_norm = (atrTeste - atrTreino.mean())/atrTreino.std()
		marcTreino_norm = (marcTreino - marcTreino.mean())/marcTreino.std()
		marcTeste_norm = (marcTeste - marcTreino.mean())/marcTreino.std()

		yTreino.append(marcTreino_norm)
		yTeste.append(marcTeste_norm)
		xTreino.append(atrTreino_norm)
		xTeste.append(atrTeste_norm)
		'''
		yTreino.append(marcTreino)
		yTeste.append(marcTeste)
		xTreino.append(atrTreino)
		xTeste.append(atrTeste)
		'''
	return xTreino, xTeste, yTreino, yTeste 