```python
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow import keras
tf.get_logger().setLevel('ERROR')
print("tensorflow version", tf.__version__)
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import sys
```

    tensorflow version 2.13.0-rc0



```python
## Define image properties:
import random
imgDir = "./languages"
targetWidth, targetHeight = 50, 50
imageSize = (targetWidth, targetHeight)
channels = 1  # color channels

## define other constants, including command line argument defaults
epochs = 10
plot = True  # show plots?
```


```python
import __main__ as main
if hasattr(main, "__file__"):
    # run as file
    print("parsing command line arguments")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d",
                        help = "directory to read images from",
                        default = imgDir)
    parser.add_argument("--epochs", "-e",
                        help = "how many epochs",
                        default= epochs)
    parser.add_argument("--plot", "-p",
                        action = "store_true",
                        help = "plot a few wrong/correct results")
    args = parser.parse_args()
    imgDir = args.dir
    epochs = int(args.epochs)
    plot = args.plot
else:
    # run as notebook
    print("run interactively from", os.getcwd())
    imageDir = os.path.join(os.path.expanduser("~"),
                            "data", "images", "text", "language-text-images")
print("Load images from", imgDir)
print("epochs:", epochs)

```

    run interactively from /Users/andrewjumanca/GitHub/MLProjects/Neural Networks
    Load images from ./languages
    epochs: 10



```python
def find_abbrev(filenames):
    abbrev = []
    for file in filenames:
        abbrev.append(file.rpartition("_")[2])
    return abbrev
```


```python
## Prepare dataset for training model:

# random sample of 1000 images
# (1.2.3)
filenames = os.listdir("languages/train")

# extract language (EN, ZN, TH, etc.)
abbrev = find_abbrev(filenames)


print(len(filenames), "images found")
trainingResults = pd.DataFrame({
    'filename':filenames,
    'category':pd.Series(abbrev).str[:2]
})
print("data files:")
print(trainingResults.sample(5))
nCategories = trainingResults.category.nunique()
print("categories:\n", trainingResults.category.value_counts())
```

    31869 images found
    data files:
                                   filename category
    12675  tolstoy-voina-i-mir-1_RU-apg.jpg       RU
    27626            novel_00002_TH-aau.jpg       TH
    17782  tolstoy-voina-i-mir-3_RU-cta.jpg       RU
    5919            chinese-laws_ZN-caa.jpg       ZN
    20546            novel_00003_TH-abe.jpg       TH
    categories:
     category
    EN    8442
    TH    7425
    RU    6511
    ZN    5396
    DA    4095
    Name: count, dtype: int64



```python
from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,\
    MaxPooling2D, AveragePooling2D,\
    Dropout,Flatten,Dense,Activation,\
    BatchNormalization
```


```python
model=Sequential()

model.add(Conv2D(32, kernel_size=3, strides=(2,2), padding='valid', activation='relu', input_shape=(targetWidth, targetHeight, channels)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(200,
                activation='relu'))
#model.add(BatchNormalization())
#model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.add(Dense(nCategories,
                activation='softmax'))

model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

## Training and validation data generator:
trainingGenerator = ImageDataGenerator(
    # rotation_range=15,
    rescale=1./255#,
    # shear_range=0.1,
    # zoom_range=0.2,
    # horizontal_flip=True,
    # width_shift_range=0.1,
    # height_shift_range=0.1
).\
    flow_from_dataframe(trainingResults,
                        os.path.join(imgDir, "train"),
                        x_col='filename', y_col='category',
                        target_size=imageSize,
                        class_mode='categorical',
                        color_mode="grayscale",
                        shuffle=True)
label_map = trainingGenerator.class_indices
## Model Training:
history = model.fit(
    trainingGenerator,
    epochs=8
)

## Validation data preparation:
validationDir = "languages/validation"

fNames = (os.listdir(validationDir))

abbrev = find_abbrev(fNames)

print(len(fNames), "validation images")
validationResults = pd.DataFrame({
    'filename': fNames,
    'category': pd.Series(abbrev).str[:2]
})
print(validationResults.shape[0], "validation files read from", validationDir)
validationGenerator = ImageDataGenerator(rescale=1./255).\
    flow_from_dataframe(validationResults,
                        os.path.join(imgDir, "validation"),
                        x_col='filename',
                        class_mode = None,
                        target_size = imageSize,
                        shuffle = False,
                        # do not randomize the order!
                        # this would clash with the file name order!
                        color_mode="grayscale"
    ) 

## Make categorical prediction:
print(" --- Predicting on validation data ---")
phat = model.predict(validationGenerator)
print("Predicted probability array shape:", phat.shape)
print("Example:\n", phat[:5])


## Convert labels to categories:
validationResults['predicted'] = pd.Series(np.argmax(phat, axis=-1), index=validationResults.index)
print(validationResults.head())
labelMap = {v: k for k, v in label_map.items()}
validationResults["predicted"] = validationResults.predicted.replace(labelMap)
print("confusion matrix (validation)")
print(pd.crosstab(validationResults.category, validationResults.predicted))
print("Validation accuracy", np.mean(validationResults.category == validationResults.predicted))
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d (Conv2D)             (None, 24, 24, 32)        320       
                                                                     
     max_pooling2d (MaxPooling2  (None, 12, 12, 32)        0         
     D)                                                              
                                                                     
     flatten (Flatten)           (None, 4608)              0         
                                                                     
     dense (Dense)               (None, 200)               921800    
                                                                     
     dense_1 (Dense)             (None, 10)                2010      
                                                                     
     dense_2 (Dense)             (None, 5)                 55        
                                                                     
    =================================================================
    Total params: 924185 (3.53 MB)
    Trainable params: 924185 (3.53 MB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________
    Found 31869 validated image filenames belonging to 5 classes.
    Epoch 1/8


    2023-05-22 16:29:29.803578: I tensorflow/core/common_runtime/executor.cc:1210] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype int32
    	 [[{{node Placeholder/_0}}]]


    996/996 [==============================] - 23s 23ms/step - loss: 0.9933 - accuracy: 0.7345
    Epoch 2/8
    996/996 [==============================] - 23s 23ms/step - loss: 0.4640 - accuracy: 0.8951
    Epoch 3/8
    996/996 [==============================] - 23s 23ms/step - loss: 0.3167 - accuracy: 0.9088
    Epoch 4/8
    996/996 [==============================] - 23s 23ms/step - loss: 0.2621 - accuracy: 0.9201
    Epoch 5/8
    996/996 [==============================] - 23s 23ms/step - loss: 0.2264 - accuracy: 0.9318
    Epoch 6/8
    996/996 [==============================] - 23s 23ms/step - loss: 0.1942 - accuracy: 0.9411
    Epoch 7/8
    996/996 [==============================] - 23s 23ms/step - loss: 0.1691 - accuracy: 0.9500
    Epoch 8/8
    996/996 [==============================] - 23s 23ms/step - loss: 0.1463 - accuracy: 0.9575
    7967 validation images
    7967 validation files read from languages/validation
    Found 7967 validated image filenames.
     --- Predicting on validation data ---
      5/249 [..............................] - ETA: 6s

    2023-05-22 16:32:33.713487: I tensorflow/core/common_runtime/executor.cc:1210] [/device:CPU:0] (DEBUG INFO) Executor start aborting (this does not indicate an error and you can ignore this message): INVALID_ARGUMENT: You must feed a value for placeholder tensor 'Placeholder/_0' with dtype int32
    	 [[{{node Placeholder/_0}}]]


    249/249 [==============================] - 6s 23ms/step
    Predicted probability array shape: (7967, 5)
    Example:
     [[3.9779223e-04 6.3853798e-04 3.8464044e-04 6.4583134e-04 9.9793315e-01]
     [3.9778696e-04 6.3852948e-04 3.8463555e-04 6.4581423e-04 9.9793327e-01]
     [7.2483224e-04 1.1772875e-03 6.9707131e-04 2.0173532e-03 9.9538344e-01]
     [3.9778373e-04 6.3852401e-04 3.8463241e-04 6.4580404e-04 9.9793327e-01]
     [9.5724022e-01 1.9010577e-02 2.1389645e-02 1.3208371e-03 1.0387985e-03]]
                                   filename category  predicted
    0               chinese-laws_ZN-frt.jpg       ZN          4
    1               chinese-laws_ZN-dik.jpg       ZN          4
    2  lin-huang-tien-fei-hsieng_ZN-aac.jpg       ZN          4
    3  lin-huang-tien-fei-hsieng_ZN-abx.jpg       ZN          4
    4   aakjaer-samlede-verker-2_DA-blt.jpg       DA          0
    confusion matrix (validation)
    predicted   DA    EN    RU    TH    ZN
    category                              
    DA         670   271    70     6     2
    EN          69  1938    32     4     7
    RU          40   105  1542     2     4
    TH           5     7     4  1874     9
    ZN           7     3     2     8  1286
    Validation accuracy 0.9175348311786118

