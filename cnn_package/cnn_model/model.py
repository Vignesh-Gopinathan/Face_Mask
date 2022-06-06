import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Input, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from config import core


def cnn_model():
    vgg16_model = tensorflow.keras.applications.vgg16.VGG16()

    classifier = Sequential()
    for layer in vgg16_model.layers[:-1]:
        classifier.add(layer)
    for layer in classifier.layers:
        layer.trainable = False
    classifier.add(Dense(units=2, activation='softmax'))

    classifier.compile(Adam(lr=core.config.model_config.learning_rate),
                       loss='binary_crossentropy',
                       metrics=['accuracy'])

    classifier.summary()
    return classifier


checkpoint = ModelCheckpoint(core.MODEL_PATH,
                             monitor='acc',
                             verbose=1,
                             save_best_only=True,
                             mode='max')

reduce_lr = ReduceLROnPlateau(monitor='acc',
                              factor=0.5,
                              patience=2,
                              verbose=1,
                              mode='max',
                              min_lr=0.00001)

callbacks_list = [checkpoint, reduce_lr]

cnn_clf = KerasClassifier(build_fn=cnn_model,
                          batch_size=core.config.model_config.batch_size,
                          validation_split=0.1,
                          epochs=core.config.model_config.epochs,
                          verbose=1,  # progress bar - required for CI job
                          callbacks=callbacks_list,
                          )

if __name__ == '__main__':
    model = cnn_model()
    model.summary()
