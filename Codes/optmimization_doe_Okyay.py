import talos 
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from talos.model.early_stopper import early_stopper
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras import metrics
from keras.metrics import mse




dataset = np.loadtxt("mico_doe.csv", delimiter=",")
 
scalerX = MinMaxScaler()
scalerY = MinMaxScaler()

path = r'C:\Users\Usuario\Google Drive\Python\Mor data\RESULTS\micp_doe'
# split into input (X) and output (Y) variables   
X = dataset[:,0:3]
Y = np.expand_dims(dataset[:,3],axis=1)
#
scaledX = scalerX.fit_transform(X)
scaledY = scalerY.fit_transform(Y)
#X_train, X_test, y_train, y_test = train_test_split(scaledX, scaledY, 
 #                                                   test_size = 0.15,
  #                                                  random_state = 20)

X_train, X_test, y_train, y_test = train_test_split(scaledX, scaledY, 
                                                    test_size = 0.15,
                                                    random_state = 10)

p = {'activation':['relu', 'elu', 'softmax', 'tanh', 'sigmoid'],
     'first_neuron':[2,5,7,10,20],
     'optimizer': ['Nadam', 'Adam','SGD','RMSprop'],
     'losses': ['mean_absolute_error','mean_squared_error'],
     #'batch_size': [20,30,40],
     'epochs': [1000]}

# NAO TREINOU COM OS VALORES NORMALIZADOS
def iris_model(x_train, y_train, x_val, y_val, params):

    model = Sequential()
    model.add(Dense(params['first_neuron'], input_dim=3, 
                    init='normal', activation=params['activation']))
    model.add(Dense(1, init='normal', activation='relu'))
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

scan_object = talos.Scan(X, Y, model=iris_model, params=p, experiment_name='/RESULTS/doe_nural')

