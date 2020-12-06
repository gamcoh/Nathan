from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet

from lib.custom_callbacks import HistoryGraph

WIDTH = 200
HEIGHT = 200
BATCH = 64

def get_generators(target_size: tuple = (135, 180), batch_size: int = 32) -> tuple:
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory('./datasets/data/train', target_size=target_size, batch_size=batch_size, class_mode='categorical', color_mode='rgb', shuffle=True)
    val_generator = train_datagen.flow_from_directory('./datasets/data/val', target_size=target_size, batch_size=batch_size, class_mode='categorical', color_mode='rgb', shuffle=True)
    test_generator = train_datagen.flow_from_directory('./datasets/data/test', target_size=target_size, batch_size=batch_size, class_mode='categorical', color_mode='rgb', shuffle=True)
    return train_generator, val_generator, test_generator

train_generator, val_generator, test_generator = get_generators(target_size=(WIDTH, HEIGHT), batch_size=BATCH)

net = MobileNet(include_top=False, input_shape=(WIDTH, HEIGHT, 3), weights=None)
x = GlobalAveragePooling2D()(net.output)
x = Dense(128)(x)
x = Dropout(.5)(x)
x = Dense(8, activation='softmax')(x)

model = Model(inputs=net.input, outputs=x)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

callbacks = [
    HistoryGraph(model_path_name='./models/graphs'),
    ModelCheckpoint('./models/mobilenet_128_img_200_b64.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True),
    EarlyStopping(patience=10, monitor='val_accuracy')
]

model.fit(
    train_generator,
    epochs=50,
    steps_per_epoch=34841//BATCH,
    validation_steps=4200//BATCH,
    validation_data=val_generator,
    callbacks=callbacks
)
