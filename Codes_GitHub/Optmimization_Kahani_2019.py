'''
This script is related to the optimiaztion process for the problems and the
comparison to RSM
'''
import talos 
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from talos.model.early_stopper import early_stopper
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras import metrics
from keras.metrics import mse
import keras


dataset = np.loadtxt("tops2_csv.csv", delimiter=",")
 
scalerX = MinMaxScaler()
scalerY = MinMaxScaler()

path = r'C:\Users\Usuario\Google Drive\Python\Mor data\RESULTS\tops_nural'
# split into input (X) and output (Y) variables   
X = dataset[:,0:4]
Y = np.expand_dims(dataset[:,4],axis=1)
#
scaledX = scalerX.fit_transform(X)
scaledY = scalerY.fit_transform(Y)
#X_train, X_test, y_train, y_test = train_test_split(scaledX, scaledY, 
 #                                                   test_size = 0.15,
  #                                                  random_state = 20)

X_train, X_test, y_train, y_test = train_test_split(scaledX, scaledY, 
                                                    test_size = 0.15,
                                                    random_state = 10)

#sgd = keras.optimizers.SGD(learning_rate=0.00001, momentum=0.8, nesterov=False)
#rms = keras.optimizers.RMSprop(learning_rate=0.0001, rho=0.9)

p = {'activation':['relu', 'tanh', 'sigmoid'],
     'first_neuron':[8,10,11],
     'optimizer': ['Nadam', 'Adam','SGD','RMSprop'],
     'losses': ['mean_squared_error'],
     'activation2':['relu', 'tanh', 'sigmoid'],
     #'batch_size': [20,30,40],
     'epochs': [500,1000]}

# NAO TREINOU COM OS VALORES NORMALIZADOS
def iris_model(x_train, y_train, x_val, y_val, params):

    model = Sequential()
    model.add(Dense(params['first_neuron'], input_dim=4, 
                    init='normal', activation=params['activation']))
    model.add(Dense(1, init='normal', activation=params['activation2']))
    model.compile(optimizer=params['optimizer'], loss=params['losses'],
                  metrics=[metrics.mse,'acc'])
    
    
    out = model.fit(x_train, y_train,
                     #batch_size=params['batch_size'],
                     epochs=params['epochs'],
                     validation_split = 0.15,
                     #validation_data=[x_val, y_val],
                     verbose=0,
                     callbacks=[early_stopper(params['epochs'], monitor = 'val_mse',mode='moderate')])

    return out, model

scan_object = talos.Scan(scaledX, scaledY, model=iris_model, params=p, experiment_name='/RESULTS/tops_nural',
                         print_params=True)

def evaluating(file_scan):
    #file_scan é o arquivo .csv gerado pela otimizacao
    from talos import Reporting
    r = Reporting(file_scan)
    dataset = r.data
    
    best = r.best_params("mean_squared_error",["loss"],ascending=True)
    r.plot_corr("mean_squared_error",["val_mean_squared_error",'loss','val_loss','acc','val_acc'])
    
    from talos.utils.best_model import activate_model, best_model
    Talos_best = best_model(scan_object,metric="mean_squared_error",asc=True)
    Talos_model = activate_model(scan_object, Talos_best)
    
    previsao = Talos_model.predict(scaledX)
    
    real_prev = scalerY.inverse_transform(previsao)
    
    from sklearn.metrics import r2_score
    
    r2 = r2_score(Y,real_prev)
    
    print(r2)
    
    return dataset,r2_score, best
