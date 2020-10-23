import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import metrics
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


#dataset = np.loadtxt(r"C:\Users\Xeon\Google Drive\Python\Mor data\mico_doe.csv", delimiter=",") # MUDAR PARA O TOPS 
dataset = np.loadtxt(r"C:\Users\Usuario\Google Drive\Python\Mor data\tops2_csv_mean_0.csv", delimiter=",")

scalerX = MinMaxScaler()
scalerY = MinMaxScaler()


path = r'C:\Users\Usuario\Google Drive\Python\Mor data\RESULTS\mico_Doe2'
# split into input (X) and output (Y) variables   
X = dataset[:,0:4]
Y = np.expand_dims(dataset[:,4],axis=1)
#
scalerX.fit(X)
scalerY.fit(Y)

#scaledX = scalerX.fit_transform(X)

#scaledY = scalerY.fit_transform(Y)

model = Sequential()
model.add(Dense(11, input_dim=4, activation="relu", 
                kernel_initializer="normal"))
model.add(Dense(1, activation="relu", 
                kernel_initializer="normal"))
model.compile(optimizer='adam', loss='mean_squared_error',
              metrics=[metrics.mse,'acc'])

model.load_weights(r"C:\Users\Usuario\Google Drive\Python\Mor data\RESULTS\tops_nural\testing random datasets\36_04_05.hdf5")

# X1 -> Yeast extract (g/L)
# X2 -> Whey Powder (g/L)
# X3 -> Heating temperature (ºC)
# X4 -> Sewater (%)


# matrix = np.zeros((100,100,100,100))
# x1 = np.reshape(np.linspace(3.43,20,100),(100,1))
# x2 = np.reshape(np.linspace(3.43,20,100),(100,1))
# x3 = np.reshape(np.linspace(78.6,120,100),(100,1))
# x4 = np.reshape(np.linspace(14.64,85.36,100),(100,1))

matrix = np.zeros((10,10,10,10))
x1 = np.reshape(np.linspace(0,0.2,10),(10,1))
x2 = np.reshape(np.linspace(0,23.43,10),(10,1))
x3 = np.reshape(np.linspace(70,128.6,10),(10,1))
x4 = np.reshape(np.linspace(0,1,10),(10,1))


X_app = np.concatenate((x1,x2,x3,x4),axis=1)

scaledX_app = scalerX.transform(X_app)

value = 0

for i in range(len(x1)):
    for j in range(len(x2)):
        for k in range(len(x3)):
            for l in range(len(x4)):
                arr = np.reshape(np.array([scaledX_app[i,0],scaledX_app[j,1],scaledX_app[k,2],scaledX_app[l,3]]),(1,4))
                value_new = model.predict(arr) #mudar dps esse 1 para o valor max do fatorial
                matrix[i,j,k,l] = scalerY.inverse_transform(value_new)
                if matrix[i,j,k,l] > value:
                    value = matrix[i,j,k,l]
                    print(value)
                    X1 = x1[i]
                    X2 = x2[j]
                    X3 = x3[k]
                    X4 = x4[l]

                    print(X1, X2, X3, X4)
 
# matrix = matrix. T

# xx, yy = np.meshgrid(x1,x2)

# plt.rcParams.update({'font.size': 9})
# plt.rcParams.update({'font.family':'sans-serif'})
# plt.rcParams.update({'font.sans-serif':'Helvetica'})
# plt.rcParams.update({'mathtext.default':  'regular' })

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.view_init(40, 240)
# #ax.set_zlim(0, 0.05)

# surf = ax.plot_surface(xx, yy, matrix, cmap='RdYlGn_r',
#                        linewidth=0, antialiased=False)

# locs, labels = plt.xticks()  # Get the current locations and labels.
# #plt.xticks(np.arange(0.005, 0.035, step=0.007))  # Set label locations.
# plt.xticks(np.arange(4, 20, step=5)) 

# locs, labels = plt.yticks()  # Get the current locations and labels.
# plt.yticks(np.arange(4, 20, step=5))  # Set label locations.


# #ax.set_xticklabels(x1)
# #ax.set_yticklabels(x2)

# ax.set_xlabel('Yeast Extract (g/L)')
# ax.set_ylabel("Whey Powder (g/L)")
# ax.set_zlabel('SUA',rotation='90',labelpad=0.5)

# fig.colorbar(surf, shrink=0.7, aspect=10, pad = 0.00)


# plt.savefig(r'C:\Users\Usuario\Google Drive\Python\Artigo nural biocementation\Figures\top_surfaceX1X2_nural.png', dpi=600, facecolor='w', edgecolor='w',
#         orientation='portrait', papertype=None, format='png',
#         transparent=False, bbox_inches=None, pad_inches=0.1,
#         frameon=None, metadata=None)

# plt.show()

# fig = plt.figure()
# ax = fig.gca()

# cs = plt.contourf(xx, yy, matrix,levels=10,cmap='RdYlGn_r')
# plt.contour(cs, colors='k',linewidths=0.5)

# locs, labels = plt.xticks()  # Get the current locations and labels.
# plt.xticks(np.arange(4, 20, step=5))  # Set label locations.

# locs, labels = plt.yticks()  # Get the current locations and labels.
# plt.yticks(np.arange(4, 20, step=5))  # Set label locations.

# fig.colorbar(surf, shrink=1, aspect=15, pad = 0.07)


# ax.set_xlabel('Yeast Extract (g/L)')
# ax.set_ylabel("Whey Powder (g/L)")

# plt.savefig(r'C:\Users\Usuario\Google Drive\Python\Artigo nural biocementation\Figures\top_countourX1X2_nural.png', dpi=600, facecolor='w', edgecolor='w',
#         orientation='portrait', papertype=None, format='png',
#         transparent=False, bbox_inches=None, pad_inches=0.1,
#         frameon=None, metadata=None)

# plt.show()