from flask import Flask, render_template, request
from single_perceptron import Perceptron
import numpy as np #Libreria para manejo avanzado de arrays
from colorama import Fore, Back, Style, init
from multi_layer_perceptron import NeuralNetwork
from flask_pure import Pure

app = Flask(__name__)
app.config['PURECSS_RESPONSIVE_GRIDS'] = True
app.config['PURECSS_USE_CDN'] = True
app.config['PURECSS_USE_MINIFIED'] = True
Pure(app)


@app.route(r'/', methods=['GET'])
def home():
	return render_template('home.html')

@app.route(r'/and_gate', methods=['GET', 'POST'])
def and_gate():
	X = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])
	d = np.array([0, 0, 0, 1])
	perceptron = Perceptron(input_size=2) # Creación del esqueleto del perceptrón
	perceptron.fit(X, d) #Entrenamiento del perceptrón
	pesos_finales = perceptron.W

	#Encontrar una mejor forma para mostrar los logs
	log_reverse = perceptron.log[::-1]
	last_log_reverse = log_reverse[:10]
	last_log = last_log_reverse[::-1]
	learning_lap = perceptron.learning_lap

	if request.form:
		first_input = int(request.form.get('first_input'))
		second_input = int(request.form.get('second_input'))
		prediction = perceptron.predict(np.array([first_input,second_input]))
		return render_template('predict.html', first_input=first_input,
											   second_input=second_input,
											   prediction = prediction,
											   gate = 'AND',
											   prev_url='and_gate')


	return render_template('and_gate.html', pesos_finales=pesos_finales,
											last_log=last_log, 
											learning_lap=learning_lap)

@app.route(r'/or_gate', methods=['GET', 'POST'])
def or_gate():
	X = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])
	d = np.array([0, 1, 1, 1])
	perceptron = Perceptron(input_size=2) # Creación del esqueleto del perceptrón
	perceptron.fit(X, d) #Entrenamiento del perceptrón
	pesos_finales = perceptron.W

	#Encontrar una mejor forma para mostrar los logs
	log_reverse = perceptron.log[::-1]
	last_log_reverse = log_reverse[:10]
	last_log = last_log_reverse[::-1]
	learning_lap = perceptron.learning_lap

	if request.form:
		first_input = int(request.form.get('first_input'))
		second_input = int(request.form.get('second_input'))
		prediction = perceptron.predict(np.array([first_input,second_input]))
		return render_template('predict.html', first_input=first_input,
											   second_input=second_input,
											   prediction = prediction,
											   gate = 'OR',
											   prev_url = '/or_gate')


	return render_template('or_gate.html', pesos_finales=pesos_finales,
											last_log=last_log, 
											learning_lap=learning_lap)


@app.route(r'/predict', methods=['GET'])
def predict():
	return render_template('predict.html')

@app.route(r'/xor_gate', methods=['GET', 'POST'])
def xor_gate():
    nn = NeuralNetwork([2,2,1])
    X = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])
    Z = np.array([0, 1])
    y = np.array([0, 1, 1, 0])
    nn.fit(X, y)
    pesos_finales = nn.weights

    for e in X:
        print(e,nn.predict(e))

    if request.form:
    	first_input = int(request.form.get('first_input'))
    	second_input = int(request.form.get('second_input'))
    	prediction = nn.predict(np.array([first_input,second_input]))
    	return render_template('predict.html', first_input=first_input,
    										   second_input=second_input,
    										   prediction = prediction,
    										   gate = 'XOR',
    										   prev_url = '/xor_gate')

    return render_template('xor_gate.html', pesos_finales=pesos_finales)

if __name__ == '__main__':
	app.run()