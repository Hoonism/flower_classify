import tensorflow as tf
from functools import partial
from tensorflow.keras.utils import normalize
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten
from tensorflow.keras.models import Sequential, Model, load_model
import numpy as np
from tensorflow.keras.applications import DenseNet121
from colorama import init, Fore, Back, Style
init(autoreset=True)

vgg_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in vgg_model.layers[:7]:
    layer.trainable = False

vgg_model.trainable = True
for layer in vgg_model.layers[:54]:
    layer.trainable = False

TRAINING_FILENAMES = tf.io.gfile.glob("dataset/train/*.tfrec")
TEST_FILENAMES = tf.io.gfile.glob("dataset/test/*.tfrec")
VALID_FILENAMES = tf.io.gfile.glob("dataset/val/*.tfrec")

print(len(TRAINING_FILENAMES))
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 64
IMAGE_SIZE = [224, 224]
CLASSES = ['pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea', 'wild geranium', 'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood', 'globe thistle',
           'snapdragon', "colt's foot", 'king protea', 'spear thistle', 'yellow iris', 'globe-flower', 'purple coneflower', 'peruvian lily', 'balloon flower', 'giant white arum lily',
           'fire lily', 'pincushion flower', 'fritillary', 'red ginger', 'grape hyacinth', 'corn poppy', 'prince of wales feathers', 'stemless gentian', 'artichoke', 'sweet william',
           'carnation', 'garden phlox', 'love in the mist', 'cosmos', 'alpine sea holly', 'ruby-lipped cattleya', 'cape flower', 'great masterwort', 'siam tulip', 'lenten rose',
           'barberton daisy', 'daffodil', 'sword lily', 'poinsettia', 'bolero deep blue', 'wallflower', 'marigold', 'buttercup', 'daisy', 'common dandelion',
           'petunia', 'wild pansy', 'primula', 'sunflower', 'lilac hibiscus', 'bishop of llandaff', 'gaura', 'geranium', 'orange dahlia', 'pink-yellow dahlia',
           'cautleya spicata', 'japanese anemone', 'black-eyed susan', 'silverbush', 'californian poppy', 'osteospermum', 'spring crocus', 'iris', 'windflower', 'tree poppy',
           'gazania', 'azalea', 'water lily', 'rose', 'thorn apple', 'morning glory', 'passion flower', 'lotus', 'toad lily', 'anthurium',
           'frangipani', 'clematis', 'hibiscus', 'columbine', 'desert-rose', 'tree mallow', 'magnolia', 'cyclamen ', 'watercress', 'canna lily',
           'hippeastrum ', 'bee balm', 'pink quill', 'foxglove', 'bougainvillea', 'camellia', 'mallow', 'mexican petunia', 'bromelia', 'blanket flower',
           'trumpet creeper', 'blackberry lily', 'common tulip', 'wild rose']


def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image


def read_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string means bytestring
        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    label = tf.cast(example['class'], tf.int32)
    return image, label  # returns a dataset of (image, label) pairs


def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string means bytestring
        "id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
        # class is missing, this competitions's challenge is to predict flower classes for the test dataset
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    idnum = example['id']
    return image, idnum  # returns a dataset of image(s)


def load_dataset(filenames, labeled=True):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(
        filenames

    )
    dataset = dataset.with_options(
        ignore_order
    )
    dataset = dataset.map(read_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTOTUNE)

    return dataset


def get_dataset(filenames):
    dataset = load_dataset(filenames)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset


def get_validation_dataset(ordered=False):
    dataset = load_dataset(VALID_FILENAMES)
    dataset = dataset.batch(500)
    dataset = dataset.cache()
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset


def get_test_dataset(ordered=False):
    dataset = load_dataset(TEST_FILENAMES, labeled=False)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset


def exponential_lr(epoch, start_lr=0.00001, min_lr=0.00001, max_lr=0.000075, rampup_epochs=5, sustain_epochs=1, exp_decay=0.8):
    def lr(epoch, start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay):
        if epoch < rampup_epochs:
            lr = ((max_lr - start_lr) / rampup_epochs * epoch + start_lr)

        elif epoch < rampup_epochs + sustain_epochs:
            lr = max_lr

        else:
            lr = ((max_lr - min_lr) * exp_decay**(epoch - rampup_epochs - sustain_epochs) + min_lr)
        return lr
    return lr(epoch, start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay)


lr_callback = tf.keras.callbacks.LearningRateScheduler(exponential_lr, verbose=True)

# train_dataset = get_dataset(TRAINING_FILENAMES)
valid_dataset = get_validation_dataset(VALID_FILENAMES)
# test_dataset = get_test_dataset(TEST_FILENAMES)

new_model = load_model("flower_classification_test")
prediction = new_model.predict(valid_dataset)

val_x, val_y = next(iter(valid_dataset))
j = 0
for i in range(500):
    if CLASSES[np.where(prediction[i] == prediction[i].max())[0][0]] == CLASSES[val_y[i]]:
        print(Fore.GREEN + f"{CLASSES[np.where(prediction[i] == prediction[i].max())[0][0]]} {CLASSES[val_y[i]]}")
        j += 1
    else:
        print(Fore.RED + f"{CLASSES[np.where(prediction[i] == prediction[i].max())[0][0]]} {CLASSES[val_y[i]]}")
print(f"\n\n {j} correct in 500 samples")
# model = vgg_model.output
# # model = Conv2D(128, (3, 3), input_shape=vgg_model.output.shape[1:], activation='relu')(model)
# # model = MaxPooling2D(2, 2)(model)
# # model = Dropout(0.4)(model)
# # model = BatchNormalization()(model)

# # model = Conv2D(64, (3, 3), activation='relu')(model)
# # model = MaxPooling2D(2, 2)(model)
# # model = Dropout(0.4)(model)
# # model = BatchNormalization()(model)


# # model.add(Dense(512, activation='relu'))

# model = tf.keras.layers.GlobalAveragePooling2D()(model)
# model = BatchNormalization()(model)

# # model = Dense(1024, activation='relu')(model)
# model = Dense(104, activation='softmax')(model)
c
# finetuned_model = Model(inputs=vgg_model.input, outputs=model)
# print(finetuned_model.summary())
# finetuned_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# finetuned_model.fit(train_dataset, validation_data=(valid_dataset), epochs=24, steps_per_epoch=(12753 // 64), callbacks=[model_checkpoint_callback, lr_callbacks])

# finetuned_model.save("flower_classification_test")
