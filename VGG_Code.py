
"""
                                    Processing Data
"""


import cv2
import numpy as np
import json
################################ Loading Data
# Loading Train set
with open('Train_labels.json') as file:
    train_data = json.load(file)
x_train = np.zeros(shape=(len(train_data), 224, 224, 3), dtype=np.float16)
y_train = np.zeros(shape=(len(train_data), ))

for i, name in enumerate(train_data):
    print(i, name)
    img = cv2.imread('Train/'+name)
    img = cv2.resize(img, dsize=(224, 224))
    lbl = int(train_data[name])
    x_train[i] = np.asanyarray(img, np.float16)
    y_train[i] = (lbl-1)
print("Loading Train set Completed")

# Loading Test1 set
with open('Test1.json') as file:
    test1_data = json.load(file)
x_test1 = np.zeros(shape=(len(test1_data), 224, 224, 3), dtype=np.float16)
y_test1 = np.zeros(shape=(len(test1_data), ))

for i, name in enumerate(test1_data):
    print(i, name)
    img = cv2.imread('Test1/'+name)
    img = cv2.resize(img, dsize=(224, 224))
    lbl = int(test1_data[name])
    x_test1[i] = np.asanyarray(img, np.float16)
    y_test1[i] = (lbl-1)
print("Loading Test1 set Completed")

# Loading Test2 set
with open('Test2.json') as file:
    test2_data = json.load(file)
x_test2 = np.zeros(shape=(len(test2_data), 224, 224, 3), dtype=np.float16)
y_test2 = np.zeros(shape=(len(test2_data), ))

for i, name in enumerate(test2_data):
    if(test2_data[name] == "-1"):
        print(i, name," deleted")
        continue
    print(i, name)
    img = cv2.imread('Test2/'+name)
    img = cv2.resize(img, dsize=(224, 224))
    lbl = int(test2_data[name])
    x_test2[i] = np.asanyarray(img, np.float16)
    y_test2[i] = (lbl-1)
print("Loading Test2 set Completed")

################################ Shuffling Data And Labels
from sklearn.utils import shuffle
print("Shuffling Datas ...")
x_train, y_train = shuffle(x_train, y_train)
x_test1, y_test1 = shuffle(x_test1, y_test1)
x_test2, y_test2 = shuffle(x_test2, y_test2)
print("Shuffling Data Completed")

############################### VGG16 pre-processing
from keras.applications.vgg16 import preprocess_input
print("Processing Data For VGG16 ...")
x_train = preprocess_input(x_train)
x_test1 = preprocess_input(x_test1)
x_test2 = preprocess_input(x_test2)
print("Processing Data For VGG16 Done!")

################################ Checking Shapes of the parameters
print(f'X_train Shape: {x_train.shape}')
print(f'X_test1 Shape: {x_test1.shape}')
print(f'X_test2 Shape: {x_test2.shape}')
print(f'Y_train lenghs: {len(y_train)}')
print(f'Y_test1 lenghs: {len(y_test1)}')
print(f'Y_test2 lenghs: {len(y_test2)}')

"""
                                  Using VGG16 Pre-trained Neural Network
"""
# imporing Packages
from keras.layers import Dropout, Dense, GlobalAveragePooling2D, GlobalMaxPool2D
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from keras.models import Model, Sequential
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam, SGD


################################## Creating Layers
base_model = VGG16(include_top=False, weights='imagenet')

out_vgg = base_model.output

lyr = GlobalAveragePooling2D()(out_vgg)
lyr = Dense(256, activation='relu')(lyr)
lyr = Dense(512, activation='relu')(lyr)
lyr_out = Dense(5, activation='softmax')(lyr)
M1 = Model(base_model.input, lyr_out)

for L in base_model.layers:
  L.trainable = False

# Printing model
print(M1.summary())

################################### Compiling Model
from keras.utils import np_utils
y_train_cat = np_utils.to_categorical(y=y_train, num_classes=5)

M1.compile(optimizer="Adam", loss=categorical_crossentropy, metrics=['accuracy'])
from keras.callbacks import EarlyStopping
early_S = EarlyStopping(monitor='val_loss', patience=10)
detail_train = M1.fit(x=x_train, y=y_train_cat, batch_size=32, epochs=10, validation_split=0.15, callbacks=[early_S])

############################################ Predicting results
predict_train = M1.predict(x=x_train)
predict_train = np.argmax(predict_train, axis=1)
predict_test1 = M1.predict(x=x_test1)
predict_test1 = np.argmax(predict_test1, axis=1)
predict_test2 = M1.predict(x=x_test2)
predict_test2 = np.argmax(predict_test2, axis=1)

############################################ Classification Report
from sklearn.metrics import classification_report
clf_report_train = classification_report(y_true=y_train, y_pred=predict_train, digits=4)
print(clf_report_train)
clf_report_test1 = classification_report(y_true=y_test1, y_pred=predict_test1, digits=4)
print(clf_report_test1)
clf_report_test2 = classification_report(y_true=y_test2, y_pred=predict_test2, digits=4)
print(clf_report_test2)

############################################ Val-Train Accuracy Plot
train_acc = detail_train.history['accuracy']
val_acc = detail_train.history['val_accuracy']
train_loss = detail_train.history['loss']
val_loss = detail_train.history['val_loss']

import matplotlib.pyplot as plt
plt.plot(train_acc, label='train_acc')
plt.plot(val_acc, label='val_acc')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()
plt.savefig('train_val_acc.jpg')

################################################### Confusion Matrix
from sklearn.metrics import confusion_matrix

cm_train = confusion_matrix(y_true=y_train, y_pred=predict_train)
print(cm_train)
fig = plt.figure()
plt.matshow(cm_train)
plt.title('Confusion Matrix for train data')
plt.colorbar()
plt.ylabel('True Label')
plt.xlabel('Predicated Label')
plt.savefig('confusion_matrix_train.jpg')

cm_test1 = confusion_matrix(y_true=y_test1, y_pred=predict_test1)
print(cm_test1)
fig = plt.figure()
plt.matshow(cm_test1)
plt.title('Confusion Matrix for test1 data')
plt.colorbar()
plt.ylabel('True Label')
plt.xlabel('Predicated Label')
plt.savefig('confusion_matrix_test1.jpg')

cm_test2 = confusion_matrix(y_true=y_test2, y_pred=predict_test2)
print(cm_test2)
fig = plt.figure()
plt.matshow(cm_test2)
plt.title('Confusion Matrix for test2 data')
plt.colorbar()
plt.ylabel('True Label')
plt.xlabel('Predicated Label')
plt.savefig('confusion_matrix_test2.jpg')

###################################################### Saving Model
import joblib
joblib.dump(M1, filename='VGG16_Model.joblib')

