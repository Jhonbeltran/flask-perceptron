import numpy as np #Libreria para manejo avanzado de arrays
 
class Perceptron(object):
    """Implements a perceptron network"""
    def __init__(self, input_size, lr=1, epochs=True):
        self.W = np.zeros(input_size+1)
        self.log = []
        self.log.append("Matriz inicial vacia para los pesos: {} ".format(self.W))
        print("Matriz inicial vacia para los pesos: {} ".format(self.W))
        # add one for bias
        self.epochs = epochs
        self.lr = lr
        self.learning_lap = 0
        
    def activation_fn(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        self.log.append("Datos de entrada {}".format(x))
        print("Datos de entrada {}".format(x))
        x = np.insert(x, 0, 1)
        self.log.append("Producto punto entre las columnas de la matriz W y la x")
        print("Producto punto entre las columnas de la matriz W y la x")
        #Se realiza producto punto .T es para convertir filas en columnas
        self.log.append("Matriz W transformada: {}".format(self.W.T))
        print("Matriz W transformada: {}".format(self.W.T))
        self.log.append("Matriz X: {}".format(x))
        print("Matriz X: {}".format(x))
        z = self.W.T.dot(x)
        self.log.append("Resultado del producto punto z = {}".format(z))
        print("Resultado del producto punto z = {}".format(z))
        a = self.activation_fn(z)
        self.log.append("Valor dado por la función de activación: {}".format(a))
        print("Valor dado por la función de activación: {}".format(a))
        return a

    def fit(self, X, d):
        learn_loop = 1
        while self.epochs==True:
            verification = []
            self.log.append("CICLO DE APRENDIZAJE: {}".format(learn_loop))
            print("CICLO DE APRENDIZAJE: {}".format(learn_loop))
            for i in range(d.shape[0]):
                self.log.append("Generación del resultado número {} esperado".format(i+1))
                self.log.append("Estado actual de la matriz de pesos {} ".format(self.W))
                print("Generación del resultado número {} esperado".format(i+1))
                print("Estado actual de la matriz de pesos {} ".format(self.W))
                y = self.predict(X[i])
                if d[i] == y:
                    self.log.append("El resultado es Correcto")
                    print("El resultado es Correcto")
                    verification.append(y)
                    self.log.append("{}".format(verification))
                    print("{}".format(verification))
                    if len(verification)==4:
                        self.epochs = False
                else: 
                    verification = []

                self.log.append("Para los dato de entrada {} el resultado obtenido es {} ".format(X[i],y))
                print("Para los dato de entrada {} el resultado obtenido es {} ".format(X[i],y))
                e = d[i] - y
                self.log.append("Valor de error actual: {}".format(e))
                print("Valor de error actual: {}".format(e))
                self.W = self.W + self.lr * e * np.insert(X[i], 0, 1)
                self.log.append("Ajuste de pesos {}".format(self.W))
                print("Ajuste de pesos {}".format(self.W))

                learn_loop += 1
                self.learning_lap = learn_loop


#if __name__ == '__main__':