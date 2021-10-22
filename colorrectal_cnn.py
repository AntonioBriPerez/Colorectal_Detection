# -*- coding: utf-8 -*-
"""colorrectal_cnn.ipynb

## Colorectal histology

Methods to diagnose colorectal using histology images (<https://zenodo.org/record/53169#.XGZemKwzbmG>, <https://www.tensorflow.org/datasets/catalog/colorectal_histology>)

In this case, the purpose is to classify the type of histology in a given image in the following categories:

- 0: TUMOR
- 1: STROMA
- 2: COMPLEX
- 3: LYMPHO
- 4: DEBRIS
- 5: MUCOSA
- 6: ADIPOSE
- 7: EMPTY

## Local instalation (option 1)

Install the following Python packages to run this notebook

`pip install pip -U`

`pip install tensorflow jupyter`

## Google Colab (option 2)

[Google Colab](https://colab.research.google.com/) is a research project created to help disseminate machine learning education and research. It's a `Jupyter notebook` environment that requires no setup to use and runs entirely in the cloud.

Colaboratory notebooks are stored in [Google Drive](https://drive.google.com) and can be shared just as you would with Google Docs or Sheets. Colaboratory is free to use.

For more information, see our [FAQ](https://research.google.com/colaboratory/faq.html).

### How install extra packages
Google Colab installs a series of basic packages if we need any additional package just install it.
"""

!pip install -q keras sklearn
!pip install shap

"""## Import packages"""

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from scipy import interp
import collections
from itertools import cycle

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow import keras
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

"""## Define global constants

Lets start with a few epochs to test learning network parameters
"""

batch_size = 32 # cada 32 imagenes hace un resumen y actualiza neuronas. Hace sondeos.
nb_classes = 8  #or 8
epochs = 20 # las vueltas que da la red 
#se pueden poner un parametro de "paciencia" para que si ha dado p.ej 10 vueltas
# y la red no mejora que corte. Se usa con epocas muy grandes

# Scaling input image to theses dimensions
img_rows, img_cols = 32, 32 #importante que sean pequeñas

"""# Plot learning curves funcion:"""

def plot_learning_curves(hist):
  plt.plot(hist.history['loss'])
  plt.plot(hist.history['val_loss'])
  plt.title('Curvas de aprendizaje')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Conjunto de entrenamiento', 'Conjunto de validación'], loc='upper right')
  plt.show()

"""## Load image database"""

def format_example(image): #podemos utilizarla tal cual para cargar imagen
    image = tf.cast(image, tf.float32)
    # Normalize the pixel values
    image = image / 255.0
    # Resize the image
    image = tf.image.resize(image, (img_rows, img_cols))
    return image


def load_data(name="colorectal_histology"): #la utilizamos tal cual para cargar los datos
  train_ds = tfds.load(name, split=tfds.Split.TRAIN, batch_size=-1)
  train_ds['image'] = tf.map_fn(format_example, train_ds['image'], dtype=tf.float32)
  numpy_ds = tfds.as_numpy(train_ds)
  X, y = numpy_ds['image'], numpy_ds['label']
  #X imagenes en rgb
  #y es la clasificación

  input_shape = X.shape[1:] #formato de la matriz

  return np.array(X), np.array(y), input_shape

"""## Plot images"""

def plot_symbols(X,y,n=15):
    index = np.random.randint(len(y), size=n)
    plt.figure(figsize=(25, 2))
    for i in np.arange(n):
        ax = plt.subplot(1,n,i+1)
        plt.imshow(X[index[i],:,:,:])
        plt.gray()
        ax.set_title(f'{y[index[i]]}-{index[i]}')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()



"""## Build LeNet5 structure

<center><img src="https://www.dlsi.ua.es/~juanra/UA/curso_verano_DL/images/LeNet5.jpg"></center>

#Modelo 1
"""

#X_train.shape[1:]

def cnn_model1(input_shape):
  model = keras.Input(shape=(input_shape), name='img')
  x = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(model)
  x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
  x = layers.MaxPooling2D(pool_size=(2,2))(x)
  x = layers.Dropout(0.2)(x)

  x = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(x)
  x = layers.Conv2D(64, (3, 3), activation='relu')(x)
  x = layers.MaxPooling2D(pool_size=(2,2))(x)
  x = layers.Dropout(0.2)(x)
  x = layers.Flatten()(x)
  x = layers.Dense(512, activation='relu')(x)
  x = layers.Dropout(0.2)(x)
  outputs = layers.Dense(nb_classes, activation='softmax')(x)

  model = keras.Model(model, outputs, name='model')


  model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

  return model

"""#Modelo 2

"""

def cnn_model2(input_shape):
  model2_input = keras.Input(shape=(input_shape), name='img')
  x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(model2_input)
  x = layers.MaxPooling2D(2, 2)(x)

  x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
  x = layers.MaxPooling2D(2,2)(x)
  x = layers.Dropout(0.2)(x)

  x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
  x = layers.MaxPooling2D(2,2)(x)
  x = layers.Dropout(0.2)(x)

  x = layers.Flatten()(x)
  x = layers.Dropout(0.2)(x)
  outputs = layers.Dense(nb_classes, activation='softmax')(x)

  model2 = keras.Model(model2_input, outputs, name='model2')


  model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )


  return model2

"""### Load data"""

##################################################################################
# Main program
 
X, y, input_shape = load_data() #cargamos la bbdd

print(X.shape, 'train samples')
print(img_rows,'x', img_cols, 'image size')
print(input_shape,'input_shape')
print(epochs,'epochs')

"""Only for binary classification. All number of classes greater than 0 will be set to 1."""

if nb_classes==2:
  y[y>0] = 1

"""### Let to see examples of the dataset"""

plot_symbols(X, y, 15)

"""## Number of examples per class"""

collections.Counter(y)

"""## 10-CV Modelo 1

This section is introductory to serve as a simple example. To test the model created in different situations, a 10 cross validation (10-CV) strategy should be used.
"""

cvscores1 = []
skf = StratifiedKFold(n_splits=10,shuffle=True, random_state=42)
for train_index, test_index in skf.split(X, y):
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  y_train_nn = keras.utils.to_categorical(y_train, nb_classes)
  y_test_nn = keras.utils.to_categorical(y_test, nb_classes)

  model = cnn_model1(X_train.shape[1:])
  history = model.fit(X_train, y_train_nn, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=2)
  scores = model.evaluate(X_test, y_test_nn, verbose=1)

  #Resultados del primer clasificador

  print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
  cvscores1.append(scores[1] * 100)

print('Mostramos las curvas de aprendizaje')
plot_learning_curves(history)

print("%.2f%% (+/-%.2f%%)" % (np.mean(cvscores1), np.std(cvscores1)))

ResultadosPrimerClasificador = cvscores1

loss, acc = model.evaluate(X_test, y_test_nn, batch_size=batch_size)
print(f'loss: {loss:.2f} acc: {acc:.2f}')

"""# SHAPLEY"""

import shap

background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]

e = shap.DeepExplainer(model, background)

shap_values = e.shap_values(X_test[1:6])

shap.image_plot(shap_values, -X_test[1:6])

"""### Testing AUC result for two and multiple classes"""

y_pred = model.predict(X_test)


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
lw = 2
y_pred_hashable = y_pred.argmax(axis=1)
classes = 0
if y_test_nn.shape[1] == 2:
  classes = nb_classes
else:
  classes = nb_classes

for i in range(classes):
    fpr[i], tpr[i], _ = metrics.roc_curve(y_test_nn[:,i], y_pred[:, i])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    print('AUC class {} {:.4f} '.format(i,roc_auc[i]))

print('AUC mean {:.4f} '.format(np.array(list(roc_auc.values())).mean()))

fpr["micro"], tpr["micro"], _ = roc_curve(y_test_nn.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])



# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nb_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(nb_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= nb_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(nb_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
from sklearn.metrics import accuracy_score
acc= accuracy_score(y_test, y_pred_hashable)
print('Acc: {:.4f}'.format(acc))

"""## More metrics about results

We can find more information about `precision`, `recall` and `f1` metrics in <https://en.wikipedia.org/wiki/Precision_and_recall>.
"""

print('Predictions')
print(collections.Counter(y_pred_hashable))

print('Confusion matrix')
print(metrics.confusion_matrix(y_test,y_pred_hashable))

target_names = ['TUMOR', 'HEALTHY'] if nb_classes ==  2 else ['TUMOR','STROMA','COMPLEX','LYMPHO','DEBRIS','MUCOSA','ADIPOSE','EMPTY']

print(metrics.classification_report(y_test, y_pred_hashable, target_names=target_names))

"""# 10-CV del Segundo Modelo


"""

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, GlobalAveragePooling2D




#Resultados del segundo clasificador
cvscores2 = []
skf = StratifiedKFold(n_splits=10,shuffle=True, random_state=42)
for train_index, test_index in skf.split(X, y):
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]
  y_train_nn = keras.utils.to_categorical(y_train, nb_classes)
  y_test_nn = keras.utils.to_categorical(y_test, nb_classes)

  model2 = cnn_model2(X_train.shape[1:])
  history = model2.fit(X_train, y_train_nn, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=2)
  scores2 = model2.evaluate(X_test, y_test_nn, verbose=1)

  

  print("%s: %.2f%%" % (model2.metrics_names[1], scores2[1] * 100))
  cvscores2.append(scores2[1] * 100)

print('Mostramos las curvas de aprendizaje')
plot_learning_curves(history)

print("%.2f%% (+/-%.2f%%)" % (np.mean(cvscores2), np.std(cvscores2)))
#Resultados del segundo clasificador

ResultadosSegundoClasificador = cvscores2

"""# WILCOXON RANKED TEST

"""

from scipy.stats import wilcoxon
import warnings
warnings.filterwarnings('ignore')

wilcox_V, p_value =  wilcoxon(ResultadosPrimerClasificador, ResultadosSegundoClasificador, alternative='greater', zero_method='wilcox', correction=False)

print('Resultado completo del test de Wilcoxon')
print(f'Wilcox V: {wilcox_V}, p-value: {p_value:.2f}')

"""# AUMENTADO DE DATOS

"""

"""
Aumentado de datos
"""
from keras.preprocessing.image import ImageDataGenerator



datagen2 = ImageDataGenerator(rotation_range=20, 
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             horizontal_flip=True)

datagen2.fit(X_train)



history = model.fit_generator(datagen2.flow(X_train, y_train_nn, batch_size=32),
                               steps_per_epoch=len(X_train) / 32, 
                               epochs=25,
                               validation_data=datagen2.flow(X_test, y_test_nn))

print('Mostramos las curvas de aprendizaje')
plot_learning_curves(history)


# Evaluamos usando el test set
score = model.evaluate(X_test, y_test_nn, verbose=0)

print('Resultado en el test set:')
print('Test loss: {:0.4f}'.format(score[0]))
print('Test accuracy: {:0.2f}%'.format(score[1] * 100))

"""# PREPROCESADO DE IMAGENES"""

from keras.preprocessing.image import ImageDataGenerator
from skimage import data, img_as_float
from skimage import exposure


for i in range(0, X_train.shape[0]):
  X_train[i] = exposure.equalize_adapthist(X_train[i], clip_limit=0.03)


for i in range(0, X_train.shape[0]):
  X_train[i] = exposure.equalize_hist(X_train[i], mask=None)




datagen2 = ImageDataGenerator(brightness_range=(1.0, 1.0))
datagen2.fit(X_train)


history = model.fit(datagen2.flow(X_train, y_train_nn, batch_size=32),
                               steps_per_epoch=len(X_train) / 32, 
                               epochs=25,
                               validation_data=(X_test, y_test_nn))

print('Mostramos las curvas de aprendizaje')
plot_learning_curves(history)


# Evaluamos usando el test set
score = model.evaluate(X_test, y_test_nn, verbose=0)

print('Resultado en el test set:')
print('Test loss: {:0.4f}'.format(score[0]))
print('Test accuracy: {:0.2f}%'.format(score[1] * 100))