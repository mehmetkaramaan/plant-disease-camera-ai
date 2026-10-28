import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("plant_disease.training")

"""English export of the original Kaggle training workflow. See README before running."""

import os

def inspect_dataset(base_path, max_depth=3, max_items=15):
    print(f'\n{"="*80}')
    logger.info(f'{base_path}')
    print(f'{"="*80}')

    if not os.path.exists(base_path):
        logger.warning('Path not found!')
        return

    for root, dirs, files in os.walk(base_path):
        depth = root[len(base_path):].count(os.sep)
        if depth > max_depth:
            dirs[:] = []
            continue

        indent = '  ' * depth
        folder_name = os.path.basename(root) if root != base_path else os.path.basename(base_path)
        logger.info(f'{indent} {folder_name}/  ({len(dirs)} directories, {len(files)} files)')


        if dirs:
            shown_dirs = sorted(dirs)[:max_items]
            for d in shown_dirs:
                print(f'{indent}    └── {d}')
            if len(dirs) > max_items:
                print(f'{indent}    └── ... and {len(dirs) - max_items} more directories')


        if files:
            extensions = {}
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                extensions[ext] = extensions.get(ext, 0) + 1
            ext_summary = ', '.join([f'{ext}: {count}' for ext, count in extensions.items()])
            logger.info(f'{indent}     File types: {ext_summary}')
            sample_files = sorted(files)[:3]
            logger.info(f'{indent}     Example files: {sample_files}')


DATASET_PATHS = {
    'PlantVillage': '/kaggle/input/datasets/abdallahalidev/plantvillage-dataset',
    'PlantDoc': '/kaggle/input/datasets/bloox2/fieldplant',
    'FieldPlant': '/kaggle/input/datasets/bloox2/fieldplant',
}

for name, path in DATASET_PATHS.items():
    inspect_dataset(path)

print(f'\n{"="*80}')
logger.info('Dataset inspection complete.')
print(f'{"="*80}')

import pandas as pd
import os

CSV_PATH = None


CANDIDATE_ROOTS = [
    '/kaggle/input/fieldplant',
    '/kaggle/input/datasets/bloox2/fieldplant',
]


for root_dir in CANDIDATE_ROOTS:
    if not os.path.exists(root_dir):
        continue
    for r, dirs, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith('.csv'):
                CSV_PATH = os.path.join(r, f)
                break
        if CSV_PATH:
            break
    if CSV_PATH:
        break

if CSV_PATH is None:
    logger.warning('CSV not found. Check these candidate paths:')
    for root_dir in CANDIDATE_ROOTS:
        print(f'  {root_dir} -> exists: {os.path.exists(root_dir)}')
    print('\nCheck the mounted FieldPlant dataset path in Kaggle')
    print('and add it to CANDIDATE_ROOTS.')
else:
    logger.info(f'CSV found: {CSV_PATH}\n')

if CSV_PATH:
    df = pd.read_csv(CSV_PATH)
    print('Columns:', list(df.columns))
    print(f'\nTotal rows: {len(df)}')
    print(f'\nFirst five rows:')
    print(df.head())
    print(f'\nUnique values in candidate label/name columns:')
    for col in df.columns:
        if 'class' in col.lower() or 'label' in col.lower() or 'name' in col.lower():
            print(f'{col}: {df[col].nunique()} unique values')
            print(df[col].unique()[:30])

import pandas as pd
import os
from PIL import Image


CSV_PATH = '/kaggle/input/datasets/bloox2/fieldplant/train/_annotations.csv'
IMAGES_DIR = '/kaggle/input/datasets/bloox2/fieldplant/train'
OUTPUT_DIR = '/kaggle/working/fieldplant_classification'

MIN_CROP_SIZE = 20   
PADDING_RATIO = 0.05  


def sanitize_class_name(name):

    return name.strip().replace(' ', '_').replace('/', '_')


def convert_fieldplant_to_classification():
    df = pd.read_csv(CSV_PATH)
    print(f'Total annotations: {len(df)}')
    print(f'Unique source images: {df["filename"].nunique()}')
    print(f'Total classes: {df["class"].nunique()}\n')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    class_counts = {}
    skipped_missing = 0
    skipped_small = 0
    skipped_error = 0
    saved = 0


    crop_counter = {}

    for idx, row in df.iterrows():
        filename = row['filename']
        class_name = sanitize_class_name(row['class'])
        xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])

        img_path = os.path.join(IMAGES_DIR, filename)
        if not os.path.exists(img_path):
            skipped_missing += 1
            continue


        box_w = xmax - xmin
        box_h = ymax - ymin
        if box_w < MIN_CROP_SIZE or box_h < MIN_CROP_SIZE:
            skipped_small += 1
            continue

        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                img_w, img_h = img.size


                pad_x = int(box_w * PADDING_RATIO)
                pad_y = int(box_h * PADDING_RATIO)

                crop_xmin = max(0, xmin - pad_x)
                crop_ymin = max(0, ymin - pad_y)
                crop_xmax = min(img_w, xmax + pad_x)
                crop_ymax = min(img_h, ymax + pad_y)

                cropped = img.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))


                class_dir = os.path.join(OUTPUT_DIR, class_name)
                os.makedirs(class_dir, exist_ok=True)


                base_name = os.path.splitext(filename)[0]
                crop_counter[base_name] = crop_counter.get(base_name, 0) + 1
                out_name = f'{base_name}_crop{crop_counter[base_name]}.jpg'
                out_path = os.path.join(class_dir, out_name)

                cropped.save(out_path, 'JPEG', quality=95)

                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                saved += 1

        except Exception as e:
            skipped_error += 1
            continue

        if (idx + 1) % 1000 == 0:
            print(f'Processed: {idx + 1}/{len(df)}')


    print(f'\n{"="*60}')
    logger.info('Conversion complete')
    print(f'{"="*60}')
    print(f'Crops saved: {saved}')
    print(f'Skipped missing images: {skipped_missing}')
    print(f'Skipped small bounding boxes: {skipped_small}')
    print(f'Skipped errors: {skipped_error}')
    print(f'\nImages per class:')
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f'  {cls}: {count}')

    logger.info(f'Output directory: {OUTPUT_DIR}')
    print('Use this directory in the same way as PlantVillage or PlantDoc')
    print('by passing it as base_path to collect_images().')


if __name__ == '__main__':
    convert_fieldplant_to_classification()

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

print(f'TensorFlow: {tf.__version__}')
print(f'GPU: {tf.config.list_physical_devices("GPU")}')


PLANTVILLAGE_PATH = '/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color'
FIELDPLANT_PATH = '/kaggle/working/fieldplant_classification'
PLANTDOC_TRAIN_PATH = '/kaggle/input/datasets/nirmalsankalana/plantdoc-dataset/train'
PLANTDOC_TEST_PATH = '/kaggle/input/datasets/nirmalsankalana/plantdoc-dataset/test'

WORK_DIR = '/kaggle/working'
MODEL_PATH = os.path.join(WORK_DIR, 'best_model.h5')
TFLITE_PATH = os.path.join(WORK_DIR, 'plant_disease_model.tflite')
LABELS_PATH = os.path.join(WORK_DIR, 'labels.json')
CONFIG_PATH = os.path.join(WORK_DIR, 'training_config.json')

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_1 = 10
EPOCHS_2 = 20
LR_1 = 1e-3
LR_2 = 1e-5
VAL_SPLIT = 0.15
TEST_SPLIT = 0.10
SEED = 42
CONFIDENCE_THRESHOLD = 0.65


CLASS_NAME_MAP = {

    'Corn_(maize)___Northern_Leaf_Blight': 'corn_leaf_blight',
    'Corn_(maize)___healthy': 'corn_healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'corn_gray_leaf_spot',
    'Corn_(maize)___Common_rust_': 'corn_rust',
    'Tomato___Late_blight': 'tomato_blight_leaf',
    'Tomato___healthy': 'tomato_healthy',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'tomato_yellow_leaf_virus',
    'Tomato___Tomato_mosaic_virus': 'tomato_leaf_mosaic_virus',
    'Tomato___Early_blight': 'tomato_early_blight',
    'Tomato___Septoria_leaf_spot': 'tomato_septoria_leaf_spot',
    'Tomato___Bacterial_spot': 'tomato_bacterial_spot',
    'Tomato___Leaf_Mold': 'tomato_leaf_mold',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'tomato_spider_mites',
    'Apple___Apple_scab': 'apple_scab',
    'Apple___healthy': 'apple_healthy',
    'Apple___Cedar_apple_rust': 'apple_rust',
    'Apple___Black_rot': 'apple_black_rot',
    'Pepper,_bell___healthy': 'pepper_healthy',
    'Pepper,_bell___Bacterial_spot': 'pepper_bacterial_spot',
    'Blueberry___healthy': 'blueberry_healthy',
    'Cherry_(including_sour)___healthy': 'cherry_healthy',
    'Cherry_(including_sour)___Powdery_mildew': 'cherry_powdery_mildew',
    'Peach___healthy': 'peach_healthy',
    'Peach___Bacterial_spot': 'peach_bacterial_spot',
    'Potato___Early_blight': 'potato_early_blight',
    'Potato___Late_blight': 'potato_late_blight',
    'Potato___healthy': 'potato_healthy',
    'Raspberry___healthy': 'raspberry_healthy',
    'Soybean___healthy': 'soybean_healthy',
    'Squash___Powdery_mildew': 'squash_powdery_mildew',
    'Strawberry___healthy': 'strawberry_healthy',
    'Strawberry___Leaf_scorch': 'strawberry_leaf_scorch',
    'Grape___healthy': 'grape_healthy',
    'Grape___Black_rot': 'grape_black_rot',
    'Grape___Esca_(Black_Measles)': 'grape_esca',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'grape_leaf_blight',


    'Corn_leaf_blight': 'corn_leaf_blight',
    'Corn_Healthy': 'corn_healthy',
    'Corn_Gray_leaf_spot': 'corn_gray_leaf_spot',
    'Corn_rust_leaf': 'corn_rust',
    'Tomato_blight_leaf': 'tomato_blight_leaf',
    'Tomato_healthy': 'tomato_healthy',
    'Tomato_leaf_yellow_virus': 'tomato_yellow_leaf_virus',
    'Tomato_leaf_mosaic_virus': 'tomato_leaf_mosaic_virus',
    'Tomato_Brown_Spots': 'tomato_brown_spots',
    'Corn_Brown_Spots': 'corn_brown_spots',
    'Corn_Chlorotic_Leaf_Spot': 'corn_chlorotic_leaf_spot',
    'Corn_Yellowing': 'corn_yellowing',
    'Corn_Streak': 'corn_streak',
    'Corn_Stripe': 'corn_stripe',
    'Corn_Mildew': 'corn_mildew',
    'Cassava_Mosaic': 'cassava_mosaic',
    'Cassava_Healthy': 'cassava_healthy',
    'Cassava_Brown_Leaf_Spot': 'cassava_brown_leaf_spot',
    'Cassava_Root_Rot': 'cassava_root_rot',
    'Cassava_Bacterial_Blight': 'cassava_bacterial_blight',


    'Apple_Scab_Leaf': 'apple_scab',
    'Apple_leaf': 'apple_healthy',
    'Apple_rust_leaf': 'apple_rust',
    'Bell_pepper_leaf': 'pepper_healthy',
    'Bell_pepper_leaf_spot': 'pepper_bacterial_spot',
    'Blueberry_leaf': 'blueberry_healthy',
    'Cherry_leaf': 'cherry_healthy',
    'Peach_leaf': 'peach_healthy',
    'Potato_leaf_early_blight': 'potato_early_blight',
    'Potato_leaf_late_blight': 'potato_late_blight',
    'Raspberry_leaf': 'raspberry_healthy',
    'Soyabean_leaf': 'soybean_healthy',
    'Squash_Powdery_mildew_leaf': 'squash_powdery_mildew',
    'Strawberry_leaf': 'strawberry_healthy',
    'Tomato_Early_blight_leaf': 'tomato_early_blight',
    'Tomato_Septoria_leaf_spot': 'tomato_septoria_leaf_spot',
    'Tomato_leaf': 'tomato_healthy',
    'Tomato_leaf_bacterial_spot': 'tomato_bacterial_spot',
    'Tomato_leaf_late_blight': 'tomato_blight_leaf',
    'Tomato_mold_leaf': 'tomato_leaf_mold',
    'Tomato_two_spotted_spider_mites_leaf': 'tomato_spider_mites',
    'grape_leaf': 'grape_healthy',
    'grape_leaf_black_rot': 'grape_black_rot',
}


EXCLUDE_CLASSES = {
    'Corn_Charcoal',
    'Corn_Violet_Decoloration',
    'Corn_Purple_Discoloration',
    'Corn_Smut',
    'Corn_Yellow_Spots',
    'Corn_Insects_Damages',
    'Tomato_bacterial_wilt',
}


def collect_images(base_path, dataset_name, class_name_map=None, exclude=None):
    images, labels = [], []
    exclude = exclude or set()
    if not os.path.exists(base_path):
        logger.warning(f'{dataset_name} not found: {base_path}')
        return images, labels

    for class_name in sorted(os.listdir(base_path)):
        class_path = os.path.join(base_path, class_name)
        if not os.path.isdir(class_path):
            continue
        if class_name in exclude:
            continue

        final_label = class_name_map.get(class_name, class_name) if class_name_map else class_name

        for img_file in os.listdir(class_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                images.append(os.path.join(class_path, img_file))
                labels.append(final_label)

    logger.info(f'{dataset_name}: {len(images)} images, {len(set(labels))} classes')
    return images, labels


all_images, all_labels = [], []
for imgs, labs in [
    collect_images(PLANTVILLAGE_PATH, 'PlantVillage', CLASS_NAME_MAP, EXCLUDE_CLASSES),
    collect_images(FIELDPLANT_PATH, 'FieldPlant', CLASS_NAME_MAP, EXCLUDE_CLASSES),
    collect_images(PLANTDOC_TRAIN_PATH, 'PlantDoc-train', CLASS_NAME_MAP, EXCLUDE_CLASSES),
    collect_images(PLANTDOC_TEST_PATH, 'PlantDoc-test', CLASS_NAME_MAP, EXCLUDE_CLASSES),
]:
    all_images.extend(imgs)
    all_labels.extend(labs)

logger.info(f'Total: {len(all_images)} images')

unique_classes = sorted(list(set(all_labels)))
NUM_CLASSES = len(unique_classes)
logger.info(f'Total class count: {NUM_CLASSES}')

counts = Counter(all_labels)
print('\nClass distribution (10 least represented classes):')
for cls, count in sorted(counts.items(), key=lambda x: x[1])[:10]:
    print(f'  {cls}: {count}')


df = pd.DataFrame({'filepath': all_images, 'label': all_labels})

train_val_df, test_df = train_test_split(
    df, test_size=TEST_SPLIT, random_state=SEED, stratify=df['label']
)
train_df, val_df = train_test_split(
    train_val_df,
    test_size=VAL_SPLIT / (1 - TEST_SPLIT),
    random_state=SEED,
    stratify=train_val_df['label']
)

print(f'\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}')


train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.20,
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

def make_generator(datagen, dataframe, shuffle=True):
    return datagen.flow_from_dataframe(
        dataframe=dataframe,
        x_col='filepath',
        y_col='label',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=shuffle,
        seed=SEED
    )

train_gen = make_generator(train_datagen, train_df, shuffle=True)
val_gen = make_generator(val_test_datagen, val_df, shuffle=False)
test_gen = make_generator(val_test_datagen, test_df, shuffle=False)

class_indices = train_gen.class_indices
idx_to_class = {v: k for k, v in class_indices.items()}


class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array(list(class_indices.values())),
    y=[class_indices[l] for l in train_df['label']]
)
class_weight_dict = dict(enumerate(class_weights))


def build_model(num_classes, trainable_base=False):
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = trainable_base

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=trainable_base)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return keras.Model(inputs, outputs), base_model

model, base_model = build_model(NUM_CLASSES, trainable_base=False)
model.compile(
    optimizer=keras.optimizers.Adam(LR_1),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
)
model.summary()


callbacks_1 = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
]

history_1 = model.fit(
    train_gen,
    epochs=EPOCHS_1,
    validation_data=val_gen,
    callbacks=callbacks_1,
    class_weight=class_weight_dict,
    verbose=1
)


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix


LR_2 = 1e-5
EPOCHS_2 = 20
CONFIDENCE_THRESHOLD = 0.65

base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(LR_2),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
)

callbacks_2 = [
    EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-8, verbose=1),
    ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
]

history_2 = model.fit(
    train_gen,
    epochs=EPOCHS_2,
    validation_data=val_gen,
    callbacks=callbacks_2,
    class_weight=class_weight_dict,
    verbose=1
)


def plot_history(h1, h2):
    acc = h1.history['accuracy'] + h2.history['accuracy']
    val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    loss = h1.history['loss'] + h2.history['loss']
    val_loss = h1.history['val_loss'] + h2.history['val_loss']
    epochs = range(1, len(acc) + 1)
    split = len(h1.history['accuracy'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, acc, 'b-o', label='Train Acc', markersize=4)
    ax1.plot(epochs, val_acc, 'r-o', label='Val Acc', markersize=4)
    ax1.axvline(split, color='gray', linestyle='--', label='Fine-tuning begins')
    ax1.set_title('Accuracy'); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, loss, 'b-o', label='Train Loss', markersize=4)
    ax2.plot(epochs, val_loss, 'r-o', label='Val Loss', markersize=4)
    ax2.axvline(split, color='gray', linestyle='--', label='Fine-tuning begins')
    ax2.set_title('Loss'); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(WORK_DIR, 'training_history.png'), dpi=150, bbox_inches='tight')
    plt.show()

plot_history(history_1, history_2)


model.load_weights(MODEL_PATH)
test_gen.reset()
y_pred_probs = model.predict(test_gen, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_gen.classes

test_loss, test_acc, test_top3 = model.evaluate(test_gen, verbose=0)
logger.info(f'Internal Test Accuracy: {test_acc*100:.2f}%')
logger.info(f'Internal Test Top-3 Accuracy: {test_top3*100:.2f}%')
logger.warning('This is an internal test split from the combined dataset.')
print('   Evaluate separately on independently captured camera images before deployment.')

class_names = [idx_to_class[i] for i in range(NUM_CLASSES)]
print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

cm = confusion_matrix(y_true, y_pred)
n_show = min(NUM_CLASSES, 20)
plt.figure(figsize=(16, 14))
sns.heatmap(cm[:n_show, :n_show], annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names[:n_show], yticklabels=class_names[:n_show],
            annot_kws={'size': 8})
plt.title(f'Confusion Matrix (first {n_show} classes)')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(WORK_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.show()


def representative_dataset():
    val_gen.reset()
    for _ in range(100):
        batch = next(val_gen)[0]
        for img in batch:
            yield [np.expand_dims(img, axis=0).astype(np.float32)]

TFLITE_PATH = os.path.join(WORK_DIR, 'plant_disease_model.tflite')
LABELS_PATH = os.path.join(WORK_DIR, 'labels.json')
CONFIG_PATH = os.path.join(WORK_DIR, 'training_config.json')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
tflite_model = converter.convert()

with open(TFLITE_PATH, 'wb') as f:
    f.write(tflite_model)

logger.info(f'TFLite saved: {TFLITE_PATH}')


with open(LABELS_PATH, 'w', encoding='utf-8') as f:
    json.dump(idx_to_class, f, ensure_ascii=False, indent=2)

training_config = {
    'datasets_used': ['PlantVillage', 'FieldPlant', 'PlantDoc'],
    'num_classes': NUM_CLASSES,
    'confidence_threshold': CONFIDENCE_THRESHOLD,
    'model_architecture': 'EfficientNetB0',
    'test_accuracy': float(test_acc),
    'test_top3_accuracy': float(test_top3),
}

with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
    json.dump(training_config, f, ensure_ascii=False, indent=2)

logger.info(f'Labels saved: {LABELS_PATH}')
logger.info(f'Config saved: {CONFIG_PATH}')
logger.warning('MOBILE INTEGRATION NOTE:')
print('   EfficientNet preprocess_input is a pass-through function.')
print('   The model expects raw RGB pixel values in the 0-255 range.')
print('   Do not divide by 255 or normalize to [-1, 1]; resize and supply RGB')
print('   as float32 values in [0, 255].')
logger.warning('DATA ATTRIBUTION: cite PlantDoc (Singh et al., 2020) and')
print('   FieldPlant (Moupojou et al., 2023); verify terms for each dataset copy.')