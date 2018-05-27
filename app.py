from flask import Flask, render_template
from single_perceptron import Perceptron
import numpy as np #Libreria para manejo avanzado de arrays
from colorama import Fore, Back, Style, init

app = Flask(__name__)


@app.route(r'/', methods=['GET'])
def home():
	return render_template('home.html')

@app.route(r'/and_gate', methods=['GET'])
def and_gate():
	X = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])
	d = np.array([0, 0, 0, 1])
	perceptron = Perceptron(input_size=2) # Creación del esqueleto del perceptrón
	perceptron.fit(X, d) #Entrenamiento del perceptrón
	pesos_finales = perceptron.W

	"""La predicción se va a realizar en una nueva ventana"""
	prediccion = perceptron.predict(np.array([1,1]))
	print(Fore. CYAN + "El Resultado es: {}".format(prediccion))
	#Encontrar una mejor forma para mostrar los logs
	log_reverse = perceptron.log[::-1]
	last_log_reverse = log_reverse[:10]
	last_log = last_log_reverse[::-1]
	learning_lap = perceptron.learning_lap
	return render_template('and_gate.html', pesos_finales=pesos_finales,
											last_log=last_log, 
											learning_lap=learning_lap)

if __name__ == '__main__':
	app.run()