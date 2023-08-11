import argparse
import json 

from sklearn.preprocessing import MultiLabelBinarizer
# from sklearn.model_selection import train_test_split

from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

from tensorflow.keras.applications import EfficientNetB3, EfficientNetB2, EfficientNetB1, EfficientNetB0
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152
from tensorflow.keras.applications import MobileNet, DenseNet169, InceptionV3, Xception

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator


from dataset import Dataset, Labels

def run(
    weight = 'EfficientNetB0',
    save_freq = 5,
    learning_rate = 0.0001,
    epoch = 20,
    batch_size = 16,
    img_size = 224
):  
    # Input shape for ANN and also to resize images 
    input_shape = (img_size, img_size, 3)
    dataset = Dataset('./dataset/', 'train')
    train_x, train_y = dataset.dataset_collect(input_shape)
    # Binarize labels
    mlb = MultiLabelBinarizer()
    train_y = mlb.fit_transform(train_y)
    class_name = mlb.classes_
    print('Class name: ')
    print(class_name)
    print()
    labels = Labels('./working/class_name.pickle')
    labels.write_pickle(class_name)

    train_x, train_y, val_x, val_y, test_x, test_y = dataset.dataset_split(train_x, train_y, train_size=0.8, test_size=0.5)

    print('Train: ' + str(train_x.shape) + str(train_y.shape))
    print('Val  : ' + str(val_x.shape) +  str(val_y.shape))
    print('Test : ' + str(test_x.shape) +  str(test_y.shape))
    print()

    datagen = ImageDataGenerator(rotation_range = 20,
                                width_shift_range = 0.1,
                                height_shift_range = 0.1,
                                zoom_range = 0.2,
                                horizontal_flip = True)
    # Define weight
    weight = weight[0]
    if weight == 'EfficientNetB0':
        base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)
    elif weight == 'EfficientNetB1':
        base_model = EfficientNetB1(include_top=False, weights='imagenet', input_shape=input_shape)
    elif weight == 'EfficientNetB2':
        base_model = EfficientNetB2(include_top=False, weights='imagenet', input_shape=input_shape)
    elif weight == 'EfficientNetB3':
        base_model = EfficientNetB3(include_top=False, weights='imagenet', input_shape=input_shape)
    elif weight == 'ResNet50':
        base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    elif weight == 'ResNet101':
        base_model = ResNet101(include_top=False, weights='imagenet', input_shape=input_shape)
    elif weight == 'ResNet152':
        base_model = ResNet152(include_top=False, weights='imagenet', input_shape=input_shape)
    elif weight == 'MobileNet':
        base_model = MobileNet(include_top=False, weights='imagenet', input_shape=input_shape)
    elif weight == 'DenseNet169':
        base_model = DenseNet169(include_top=False, weights='imagenet', input_shape=input_shape)    
    elif weight == 'InceptionV3':
        base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=input_shape)       
    elif weight == 'Xception':
        base_model = Xception(include_top=False, weights='imagenet', input_shape=input_shape)     
    # Creating model architecture
    x = GlobalAveragePooling2D()(base_model.output)
    output_layer = Dense(len(mlb.classes_), activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output_layer)
    checkpoint = ModelCheckpoint('./working/' + weight + '_last.h5', save_freq='epoch', monitor = 'val_loss', verbose = 1, period=save_freq)
    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    loss = BinaryCrossentropy()
    metric = [BinaryAccuracy(), Precision(), Recall()]
    model.compile(optimizer=optimizer, loss=loss, metrics=metric)
    history = model.fit_generator(
        datagen.flow(train_x, train_y, batch_size = batch_size), 
        validation_data = (val_x, val_y), 
        epochs = epoch, 
        verbose = 1, 
        callbacks = [checkpoint])
    with open('./working/training_history.json', 'w') as file:
        json.dump(history.history, file)
    print('Model evaluate:')
    model.evaluate(test_x, test_y)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', '--w', nargs='+', type=str, default='EfficientNetB1')
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--learning_rate', '--lr', nargs='+', type=float, default=1e-4)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch_size', '--bs', type=int, default=16)
    parser.add_argument('--img_size', '--imgsz', type=int, default=224)
    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)