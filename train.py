import pandas as pd
import numpy as np
#import keras
#from keras import backend as K

import tensorflow as tf
tf.config.run_functions_eagerly(True)
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.applications.mobilenet import preprocess_input
from itertools import cycle

import os

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
import shutil
import matplotlib.pyplot as plt 
from tqdm import tqdm
import shutil
import glob


def readData(input_dir):
    
    lesion_id = [] # unique identifier for each skin lesion
    image_id = [] # unique identifire for the corresponding image
    dx = [] # diagnosis = label
    '''
    nv: Melanocytic nevus (common mole)
    mel: Melanoma
    bkl: Benign keratosis-like lesion
    bcc: Basal cell carcinoma
    akiec: Actinic keratosis / Bowen's disease (intraepithelial carcinoma)
    vasc: Vascular lesion
    df: Dermatofibroma
    '''

    dx_type = [] # type of diagnoisis

    '''
    histo: Histopathology (microscopic examination of tissue)
    follow_up: Follow-up examination
    consensus: Consensus diagnosis by multiple experts
    confocal: Confocal microscopy imaging
    '''

    age = []
    sex	= []
    localization = []
    duplicates = []
    
    with open(input_dir+'/df_meta.csv', 'r') as f:
        lines = f.read().splitlines()
        
        for line in tqdm(lines):
            lesion_id_val, image_id_val, dx_val, dx_type_val, age_val, sex_val, localization_val = line.split(',')
            lesion_id.append(lesion_id_val)
            image_id.append(image_id_val)
            dx.append(dx_val)
            dx_type.append(dx_type_val)
            age.append(age_val)
            sex.append(sex_val)
            localization.append(localization_val)
            
    # Filter out lesion_id's with only one image
    unique_lesion_ids = set(lesion_id)
    filtered_lesion_ids = [id for id in unique_lesion_ids if lesion_id.count(id) == 1]
    
    # check lesion_id's that have only one image associated with it
    ''' 
    for id in filtered_lesion_ids:
        if id == "HAM_0000003":
            print("Yes")
    
    '''
    
    # filter out lesion id that only have 1 image or duplicates 
    # append more column name duplicates
    
    for id in lesion_id :
        if id in filtered_lesion_ids:
            duplicates.append("no_duplicates")
        else:
            duplicates.append("has_duplicates")
    
    # check duplicates val        
    '''
    for idx, id in enumerate(lesion_id):
        if id == "HAM_0002730":
            print(duplicates[idx])
            
    '''
    
    # count num of imgae that no_duplicates or has_duplicates:
    no_count = 0
    has_count = 0
    for dup_val in duplicates:
        if dup_val == "no_duplicates":
            no_count += 1
        else:
            has_count += 1
    
    #print(no_count)
    #print(has_count)
    #print(len(lesion_id))
    
    
    # filter data have no duplicates
    # make modified data
    mdf_data = []
    with open(input_dir + "/modified_data.csv", "w") as w:
        for index in tqdm(range(len(lesion_id))):
            if duplicates[index] == "no_duplicates":
                mdf_data.append(','.join([lesion_id[index], 
                                        image_id[index],
                                        dx[index],
                                        dx_type[index],
                                        age[index],
                                        sex[index],
                                        localization[index],
                                        duplicates[index]]))
        
        for line in tqdm(mdf_data):
                w.write(line + '\n')
    
    print(len(mdf_data))            
    # get label of image which does not have duplicates
    identical_label = []
    with open(input_dir + "/modified_data.csv", "r") as file:
        lines = file.read().splitlines()
        
        for line in lines:
            _, _, label, _, _, _, _, _ = line.split(',')
            identical_label.append(label)
        
                
    return identical_label, mdf_data, lesion_id, image_id, dx, dx_type, age, sex, localization 

                
def splitData(label, mdf_data, lesion_id, 
              image_id, dx, dx_type, 
              age, sex, localization, data_dir):
    _, val_set = train_test_split(mdf_data, 
                                  test_size = 0.17, 
                                  random_state=101, 
                                  stratify=label)
    
    # make a test_meta csv to contains information of valid dataset
    with open(data_dir+'/test_meta.csv', 'w') as vw:
        for line in val_set:
            vw.write(line + '\n')
            
            
    # make a train_meta csv file to contains information of train dataset 
    # train dataset excludes images in val/test set
    # train dataset will be taken in the raw dataset
    # modified dataset is different to raw dataset
    val_id = []
    with open(data_dir+'/test_meta.csv', 'r') as vr:
        lines = vr.read().splitlines()
        
        for line in lines:
            _, iden_id, _, _, _, _, _, _ = line.split(",")
            val_id.append(iden_id)
                           
    
    # check from raw datas
    # if image_id == val => val_set
    # else => train_set
    train_set = []
    with open(data_dir+'/train_meta.csv', 'w') as tw:
        for index in tqdm(range(len(image_id))):
            if image_id[index] not in val_id:
                train_set.append(','.join([lesion_id[index], 
                                           image_id[index],
                                           dx[index],
                                           dx_type[index],
                                           age[index],
                                           sex[index],
                                           localization[index]]))
            
                
        
        tw.writelines('\n'.join(train_set))
        
    print(len(train_set))
    print(len(val_set))
    return train_set, val_set
    
    

def createDataFolder(input_dir, data_dir, label, image_id):
    
    t_name = []
    with open(data_dir+'/train_meta.csv', 'r') as r1:
        lines = r1.read().splitlines()
        
        for line in lines:
            _, tname, _, _, _, _, _ = line.split(",")
            t_name.append(tname)
    
    v_name =[]
    with open(data_dir+'test_meta.csv', 'r') as r2:
        lines = r2.read().splitlines()
            
        for line in lines:
            _, vname, _, _, _, _, _, _ = line.split(",")
            v_name.append(vname)
    
        
    # print(len(img))
  
    '''    
    print(files)
    for path in files:
        if os.path.isfile(path) == True: 
            img_files.append(path)
    '''
    
    # create train folder contains image
    for index in tqdm(range(len(image_id))):
        if image_id[index] in v_name:
            val_img = image_id[index] + '.jpg'
            vimg_l = os.path.join(input_dir, 'HAM10000_images_part_1', val_img)
            # copy image into test folder
            shutil.move(vimg_l, (os.path.join(data_dir, 'test', label[index])))
        else:
            train_img = image_id[index] + '.jpg'
            timg_l = os.path.join(input_dir, 'HAM10000_images_part_1', train_img)
            # copy image into test folder
            shutil.move(timg_l, (os.path.join(data_dir, 'train', label[index])))
            
    

        

class TrainModel():
    def setGenerators(self):
        data_dir = "/home/phuonganh/speechbrain/shecodes/Skin-Lesion-Analyzer-master/data/"
        train_path = data_dir + 'train'
        test_path = data_dir + 'test'

        t_samples = glob.glob(train_path+'/*/*.jpg')
        
        v_samples = glob.glob(test_path+'/*/*.jpg')
        
                
        num_train_samples = len(t_samples)
        num_test_samples = len(v_samples)
        train_batch_size = 10
        val_batch_size = 10
        image_size = 224

        train_steps = np.ceil(num_train_samples / train_batch_size)
        test_steps = np.ceil(num_test_samples / val_batch_size)

        train_datagen = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.mobilenet.preprocess_input,
            rescale=1.0 / 255.0,  # Rescale pixel values to [0, 1]
            validation_split=0.2  # Split the data into train and validation sets
        )
        
        '''
        train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=(image_size, image_size),
            batch_size=train_batch_size,
            class_mode='categorical',
            subset='training'
        )

        train_dataset_repeated = tf.data.Dataset.from_generator(
            lambda: itertools.islice(cycle(train_generator), epochs * train_steps),
            output_signature=(
                tf.TensorSpec(shape=(None, image_size, image_size, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 7), dtype=tf.float32)
            )
        )
    '''

        train_batches = train_datagen.flow_from_directory(
            train_path,
            target_size=(image_size, image_size),
            batch_size=train_batch_size,
            class_mode='categorical',  # Encode the labels in one-hot format
            subset='training'  # Specify that this is the training set
        )

        val_batches = train_datagen.flow_from_directory(
            train_path,
            target_size=(image_size, image_size),
            batch_size=val_batch_size,
            class_mode='categorical',  # Encode the labels in one-hot format
            subset='validation'  # Specify that this is the validation set
        )

        test_datagen = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.mobilenet.preprocess_input,
            rescale=1.0 / 255.0  # Rescale pixel values to [0, 1]
        )

        test_batches = test_datagen.flow_from_directory(
            test_path,
            target_size=(image_size, image_size),
            batch_size=1,  # Set batch size to 1 for evaluation
            shuffle=False,  # Do not shuffle the test dataset
            class_mode='categorical'  # Encode the labels in one-hot format
        )
        
        


        return train_batches, train_steps, val_batches, test_batches, test_steps
        
    def createArchitecture(self):
        mobile = tf.keras.applications.mobilenet.MobileNet(input_shape=(224, 224, 3), include_top=False)
        x = mobile.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.25)(x)
        predictions = Dense(7, activation='softmax')(x)

        model = Model(inputs=mobile.input, outputs=predictions)

        for layer in model.layers[:-23]:
            layer.trainable = False

        return model
        
    @staticmethod            
    def top_3_accuracy(y_true, y_pred):
        top3 = top_k_categorical_accuracy(y_true, y_pred, k=3)
        return top3

    @staticmethod
    def top_2_accuracy(y_true, y_pred):
        top2 = top_k_categorical_accuracy(y_true, y_pred, k=2)
        return top2
        
    def modelComplile(self, model, top2, top3):
        model_com = model.compile(Adam(lr=0.01), loss='categorical_crossentropy', 
                          metrics=[categorical_accuracy, top2, top3],
                          run_eagerly=True)
        return model_com
    
    def addWeight(self):
        class_weights={
            0: 1.0, # akiec
            1: 1.0, # bcc
            2: 1.0, # bkl
            3: 1.0, # df
            4: 3.0, # mel # Try to make the model more sensitive to Melanoma.
            5: 1.0, # nv
            6: 1.0, # vasc
            }
        return class_weights
    
    import numpy as np

    def output(self, model, train_batches, test_batches, test_steps, train_steps):
        filepath = "/home/phuonganh/speechbrain/shecodes/Skin-Lesion-Analyzer-master/model2.h5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_top_3_accuracy', verbose=1, save_best_only=True, mode='max')
        reduce_lr = ReduceLROnPlateau(monitor='val_top_3_accuracy', factor=0.5, patience=2, verbose=1, mode='max', min_lr=0.00001)
        callbacks_list = [checkpoint, reduce_lr]

        #train_data = train_dataset_repeated.batch(train_batch_size)
        
        # Calculate class weights manually
        

        history = model.fit(train_batches, steps_per_epoch=train_steps, 
                                class_weight=class_weights,
                                validation_data=test_batches,
                                validation_steps=test_steps,
                                epochs=10, verbose=1,
                                callbacks=callbacks_list)

        return history
        
  
if __name__ == '__main__':
    input_dir = "/home/phuonganh/speechbrain/shecodes/Skin-Lesion-Analyzer-master/input_ham10000/"
    data_dir = "/home/phuonganh/speechbrain/shecodes/Skin-Lesion-Analyzer-master/data/"
    
    # read dataframe
    df = pd.read_csv(input_dir+'/HAM10000_metadata.csv')
    df.to_csv(input_dir+'/df_meta.csv', index = False, header = False)
    iden_lb, mdf_data, lesion_id, image_id, dx, dx_type, age, sex, localization  = readData(input_dir)
    train_set, val_set = splitData(label = iden_lb, mdf_data = mdf_data, lesion_id = lesion_id,
              image_id = image_id, dx = dx, dx_type = dx_type,
              age = age, sex = sex, localization=localization,
              data_dir = data_dir)
    # createDataFolder(input_dir, data_dir, label = dx, image_id=image_id)
    
    
    # instantiate the TrainModel class
    model_trainer = TrainModel()

    # call the setGenerators method
    train_batches, train_steps, val_batches, test_batches, test_steps = model_trainer.setGenerators()

    # call the createArchitecture method
    model = model_trainer.createArchitecture()

    # call the top_2_accuracy and top_3_accuracy methods
    top2 = model_trainer.top_2_accuracy
    top3 = model_trainer.top_3_accuracy

    # call the modelComplile method
    model_com = model_trainer.modelComplile(model, top2, top3)

    # call the addWeight method
    class_weights = model_trainer.addWeight()

    # call the output method
    history = model_trainer.output(model, train_batches, test_batches,test_steps, train_steps)
    
    #with open(data_dir+'/training_result.csv', 'w') as w:
    #    w.write(history)