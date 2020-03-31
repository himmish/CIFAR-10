import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
import sys
print(sys.argv)

X_train=pd.read_csv(sys.argv[0],delimiter=' ',header=None)
X_test=pd.read_csv(sys.argv[1],delimiter=' ',header=None)

X_train=X_train.values
X_test=X_test.values
Y_train=X_train[:,-1:]
X_train=X_train[:,0:X_train.shape[1]-1]
X_test=X_test[:,0:X_test.shape[1]-1]

X_validation=X_train[0:10000,:]
Y_validation=Y_train[0:10000,:]

X_train=X_train[10000:X_train.shape[0],:]
Y_train=Y_train[10000:Y_train.shape[0],:]

X_train=X_train.astype('float64')
X_test=X_test.astype('float64')
X_validation=X_validation/255.0
X_validation=X_validation.astype('float64')

print(X_train[0:10,:],X_test[0:10,:])
print(Y_train[0:10,:])
def onehotencode(Y):
    Y_return = np.zeros([len(Y), 10])
    for i in range(0, len(Y)):
        Y_return[i, int(Y[i])] = 1

    return Y_return
Y_train=onehotencode(Y_train)
Y_validation=onehotencode(Y_validation)
print(X_train.shape,X_test.shape)
print(Y_train.shape)
print(Y_train[0:10,:])

X_train=np.reshape(X_train, (X_train.shape[0],3,32,32)).transpose(0, 2, 3, 1)
X_test=np.reshape(X_test, (X_test.shape[0],3,32,32)).transpose(0, 2, 3, 1)
X_validation=np.reshape(X_validation, (X_validation.shape[0],3,32,32)).transpose(0, 2, 3, 1)

avg = np.mean(X_train,axis=(0,1,2,3))
sigm = np.std(X_train,axis=(0,1,2,3))
X_train=(X_train-avg)/sigm
avg = np.mean(X_test,axis=(0,1,2,3))
sigm = np.std(X_test,axis=(0,1,2,3))
X_test=(X_test-avg)/sigm
avg = np.mean(X_validation,axis=(0,1,2,3))
sigm = np.std(X_validation,axis=(0,1,2,3))
X_validation=(X_validation-avg)/sigm

print(X_train.shape,X_test.shape,X_validation.shape)
plt.imshow(X_train[50])

datagen = keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False,
    samplewise_std_normalization=False, zca_whitening=False, zca_epsilon=1e-06, rotation_range=0,
    width_shift_range=0.1, height_shift_range=0.1, shear_range=0., zoom_range=0., channel_shift_range=0.,
    fill_mode='nearest', cval=0., horizontal_flip=True, vertical_flip=False, rescale=None,
    preprocessing_function=None, data_format=None, validation_split=0.0)

datagen.fit(X_train)
new_data = datagen.flow(X_train, Y_train, batch_size=50)
print(len(new_data))

from keras.models import Sequential
from keras.layers import Dense, Activation, MaxPooling2D, Conv2D, BatchNormalization, Flatten, Dropout

model=Sequential([])

model.add(Conv2D(32, (3, 3), activation='elu', padding='same', strides=1, input_shape=(32,32,3)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='elu', padding='same', strides=1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), activation='elu', padding='same', strides=1))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='elu', padding='same', strides=1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), activation='elu', padding='same', strides=1))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='elu', padding='same', strides=1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='elu'))
model.add(BatchNormalization())
model.add(Dense(10,activation='softmax'))

lr=0.0005
epoch=20
dec=lr/epoch
opt=keras.optimizers.rmsprop(lr=0.005,decay=1e-3)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit_generator(datagen.flow(X_train, Y_train, batch_size=50), epochs=epoch, validation_data=(X_validation, Y_validation), workers=4)


def reversehotencode(Y_predicted):
    Y_p = np.argmax(Y_predicted, axis=1)
    print(Y_predicted, Y_p)
    return Y_p
Y_P1=model.predict(X_test)
Y_p=reversehotencode(Y_P1)
Y_p=Y_p.astype('int')
np.savetxt(sys.argv[2],Y_p)

scores = model.evaluate(X_test, Y_P1, verbose=1)
print('Loss and Accuracy',scores)