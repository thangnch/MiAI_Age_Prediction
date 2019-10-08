import scipy.io
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tensorflow as tf
import keras
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.layers import Dense, Activation, Dropout, Flatten, Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Activation
from keras.layers import Conv2D, AveragePooling2D
from keras.models import Model, Sequential
from sklearn.model_selection import train_test_split
from keras import metrics
from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model


classes = 101 # Tu 0 den 100 tuoi
target_size = (224, 224)

# Ham load anh
def getImagePixels(image_path):
    img = image.load_img("data/%s" % image_path[0], grayscale=False, target_size=target_size)
    x = image.img_to_array(img).reshape(1, -1)[0]
    #x = preprocess_input(x)
    return x

# Ham chuyen doi datenum thanh date year
def datenum_to_datetime(datenum):
    days = datenum % 1
    hours = days % 1 * 24
    minutes = hours % 1 * 60
    seconds = minutes % 1 * 60
    exact_date = datetime.fromordinal(int(datenum)) \
                 + timedelta(days=int(days)) \
                 + timedelta(hours=int(hours)) \
                 + timedelta(minutes=int(minutes)) \
                 + timedelta(seconds=round(seconds)) \
                 - timedelta(days=366)

    return exact_date.year


# Ham load anh de test
def loadImage(filepath):
    test_img = image.load_img(filepath, target_size=(224, 224))
    test_img = image.img_to_array(test_img)
    test_img = np.expand_dims(test_img, axis = 0)
    test_img /= 255
    return test_img

# Ham load du lieu
def load_data():
    # Load du lieu tu file wiki.mat gom cac khuon mat va thong tin lay tu wiki
    mat = scipy.io.loadmat('data/wiki.mat')
    columns = ["dob", "photo_taken", "full_path", "gender", "name", "face_location", "face_score", "second_face_score"]

    instances = mat['wiki'][0][0][0].shape[1]
    df = pd.DataFrame(index = range(0,instances), columns = columns)

    for i in mat:
        if i == "wiki":
            current_array = mat[i][0][0]
            for j in range(len(current_array)):
                #print(columns[j],": ",current_array[j])
                df[columns[j]] = pd.DataFrame(current_array[j][0])



    # Chuyen truong DOB thanh nam sinh
    df['date_of_birth'] = df['dob'].apply(datenum_to_datetime)

    # Tinh tuoi cho tung face
    df['age'] = df['photo_taken'] - df['date_of_birth']

    # Xu ly du lieu : xoa cac anh khong co face, xoa cac anh co tu 2 mat tro len, xoa cac cot khong can thiet
    # xoa cac mat co tuoi > 100 hoac < 0
    df = df[df['face_score'] != -np.inf]
    df = df[df['second_face_score'].isna()]
    df = df[df['face_score'] >= 3]
    df = df.drop(columns = ['name','face_score','second_face_score','date_of_birth','face_location'])
    df = df[df['age'] <= 100]
    df = df[df['age'] > 0]

    # Chuyen age thanh one hot vector
    df['pixels'] = df['full_path'].apply(getImagePixels)
    target = df['age'].values
    target_classes = keras.utils.to_categorical(target, classes)

    # Them cac vector anh input vao list features
    features = []
    for i in range(0, df.shape[0]):
        features.append(df['pixels'].values[i])

    features = np.array(features)
    features = features.reshape(features.shape[0], 224, 224, 3)
    features /= 255 #normalize in [0, 1]

    # Phan chia train, test
    train_x, test_x, train_y, test_y = train_test_split(features, target_classes
                                            , test_size=0.30)#, random_state=42), stratify=target_classes)
    return train_x, test_x, train_y, test_y

def get_model():
    # Khoi tao model
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))

    model.load_weights('models/vgg_face_weights.h5')

    # Dong bang cac layer ko can train
    for layer in model.layers[:-7]:
        layer.trainable = False

    base_model_output = Sequential()
    base_model_output = Convolution2D(classes, (1, 1), name='predictions')(model.layers[-4].output)
    base_model_output = Flatten()(base_model_output)
    base_model_output = Activation('softmax')(base_model_output)

    age_model = Model(inputs=model.input, outputs=base_model_output)

    #sgd = keras.optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)

    age_model.compile(loss='categorical_crossentropy'
                      , optimizer=keras.optimizers.Adam()
                      # , optimizer = sgd
                      , metrics=['accuracy']
                      )

    return age_model



# Load du lieu
train_x, test_x, train_y, test_y = load_data()
#Load model
age_model =get_model()

# So epoch va batch_size
epochs = 250
batch_size = 256

check_point = ModelCheckpoint(
    filepath='models/classification_age_model.hdf5'
    , monitor="val_loss"
    , verbose=1
    , save_best_only=True
    , mode='auto'
)

# Bat dau train
for i in range(epochs):
    print ("Train epoch: ", i)
    ix_train = np.random.choice(train_x.shape[0], size=batch_size)
    age_model.fit(
        train_x[ix_train], train_y[ix_train]
        , epochs=1
        , validation_data=(test_x, test_y)
        , callbacks=[check_point]
    )

# Luu model
age_model = load_model("models/classification_age_model.hdf5")
age_model.save_weights('models/age_model_weights.h5')

