import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy, categorical_accuracy, top_k_categorical_accuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications.mobilenet import MobileNet
import os
from sklearn.model_selection import train_test_split
import shutil
import glob

# Function to read data from the provided directory
def readData(input_dir):
    # Initialize lists to store data
    lesion_id = []
    image_id = []
    dx = []
    dx_type = []
    age = []
    sex = []
    localization = []
    duplicates = []

    # Read metadata from a CSV file
    with open(input_dir+'/df_meta.csv', 'r') as f:
        lines = f.read().splitlines()
        
        # Loop through each line in the CSV file
        for line in lines:
            # Split the line and extract information
            lesion_id_val, image_id_val, dx_val, dx_type_val, age_val, sex_val, localization_val = line.split(',')
            # Append extracted information to respective lists
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

    # Determine if each lesion_id has duplicates
    for id in lesion_id:
        if id in filtered_lesion_ids:
            duplicates.append("no_duplicates")
        else:
            duplicates.append("has_duplicates")
    
    # Filter data that have no duplicates and save it to a new CSV file
    mdf_data = []
    with open(input_dir + "/modified_data.csv", "w") as w:
        for index in range(len(lesion_id)):
            if duplicates[index] == "no_duplicates":
                mdf_data.append(','.join([lesion_id[index], image_id[index], dx[index], dx_type[index], age[index], sex[index], localization[index], duplicates[index]]))
        
        for line in mdf_data:
                w.write(line + '\n')

    # Get labels for images that do not have duplicates
    identical_label = []
    with open(input_dir + "/modified_data.csv", "r") as file:
        lines = file.read().splitlines()
        
        for line in lines:
            _, _, label, _, _, _, _, _ = line.split(',')
            identical_label.append(label)
                
    return identical_label, mdf_data, lesion_id, image_id, dx, dx_type, age, sex, localization 

# Function to split data into training and validation sets
def splitData(label, mdf_data, lesion_id, image_id, dx, dx_type, age, sex, localization, data_dir):
    _, val_set = train_test_split(mdf_data, test_size = 0.17, random_state=101, stratify=label)
    
    # Create a CSV file to store information of the validation dataset
    with open(data_dir+'/test_meta.csv', 'w') as vw:
        for line in val_set:
            vw.write(line + '\n')
    
    # Make a CSV file to store information of the training dataset
    val_id = []
    with open(data_dir+'/test_meta.csv', 'r') as vr:
        lines = vr.read().splitlines()
        
        for line in lines:
            _, iden_id, _, _, _, _, _, _ = line.split(",")
            val_id.append(iden_id)
    
    # Filter out images that are not in the validation set and save the training dataset to a CSV file
    train_set = []
    with open(data_dir+'/train_meta.csv', 'w') as tw:
        for index in range(len(image_id)):
            if image_id[index] not in val_id:
                train_set.append(','.join([lesion_id[index], image_id[index], dx[index], dx_type[index], age[index], sex[index], localization[index]]))
            
        tw.writelines('\n'.join(train_set))
        
    return train_set, val_set
    
# Function to create train and test folders and move images accordingly
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
    
    # Loop through all images and move them to appropriate folders based on the train/validation split
    for index in range(len(image_id)):
        if image_id[index] in v_name:
            val_img = image_id[index] + '.jpg'
            vimg_l = os.path.join(input_dir, 'HAM10000_images_part_1', val_img)
            shutil.move(vimg_l, (os.path.join(data_dir, 'test', label[index])))
        else:
            train_img = image_id[index] + '.jpg'
            timg_l = os.path.join(input_dir, 'HAM10000_images_part_1', train_img)
            shutil.move(timg_l, (os.path.join(data_dir, 'train', label[index])))
            
# Class to train the model
class TrainModel():
    # Method to set up data generators
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
            rescale=1.0 / 255.0,
            validation_split=0.2
        )

        train_batches = train_datagen.flow_from_directory(
            train_path,
            target_size=(image_size, image_size),
            batch_size=train_batch_size,
            class_mode='categorical',
            subset='training'
        )

        val_batches = train_datagen.flow_from_directory(
            train_path,
            target_size=(image_size, image_size),
            batch_size=val_batch_size,
            class_mode='categorical',
            subset='validation'
        )

        test_datagen = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.mobilenet.preprocess_input,
            rescale=1.0 / 255.0
        )

        test_batches = test_datagen.flow_from_directory(
            test_path,
            target_size=(image_size, image_size),
            batch_size=1,
            shuffle=False,
            class_mode='categorical'
        )
        
        return train_batches, train_steps, val_batches, test_batches, test_steps
        
    # Method to create the model architecture
    def createArchitecture(self):
        mobile = MobileNet(input_shape=(224, 224, 3), include_top=False)
        x = mobile.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.25)(x)
        predictions = Dense(7, activation='softmax')(x)

        model = Model(inputs=mobile.input, outputs=predictions)

        for layer in model.layers[:-23]:
            layer.trainable = False

        return model
        
    # Custom metrics: top_2_accuracy and top_3_accuracy
    @staticmethod            
    def top_3_accuracy(y_true, y_pred):
        top3 = top_k_categorical_accuracy(y_true, y_pred, k=3)
        return top3

    @staticmethod
    def top_2_accuracy(y_true, y_pred):
        top2 = top_k_categorical_accuracy(y_true, y_pred, k=2)
        return top2
        
    # Method to compile the model
    def modelComplile(self, model, top2, top3):
        model_com = model.compile(Adam(lr=0.01), loss='categorical_crossentropy', 
                          metrics=[categorical_accuracy, top2, top3],
                          run_eagerly=True)
        return model_com
    
    # Method to assign class weights
    def addWeight(self):
        class_weights={
            0: 1.0, # akiec
            1: 1.0, # bcc
            2: 1.0, # bkl
            3: 1.0, # df
            4: 3.0, # mel
            5: 1.0, # nv
            6: 1.0, # vasc
            }
        return class_weights

    # Method to train the model and output history
    def output(self, model, train_batches, test_batches, test_steps, train_steps):
        filepath = "/home/phuonganh/speechbrain/shecodes/Skin-Lesion-Analyzer-master/model2.h5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_top_3_accuracy', verbose=1, save_best_only=True, mode='max')
        reduce_lr = ReduceLROnPlateau(monitor='val_top_3_accuracy', factor=0.5, patience=2, verbose=1, mode='max', min_lr=0.00001)
        callbacks_list = [checkpoint, reduce_lr]

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
    
    # Read metadata and preprocess data
    df = pd.read_csv(input_dir+'/HAM10000_metadata.csv')
    df.to_csv(input_dir+'/df_meta.csv', index = False, header = False)
    iden_lb, mdf_data, lesion_id, image_id, dx, dx_type, age, sex, localization  = readData(input_dir)
    train_set, val_set = splitData(label = iden_lb, mdf_data = mdf_data, lesion_id = lesion_id,
              image_id = image_id, dx = dx, dx_type = dx_type,
              age = age, sex = sex, localization=localization,
              data_dir = data_dir)
    
    # Instantiate the TrainModel class
    model_trainer = TrainModel()

    # Set up data generators
    train_batches, train_steps, val_batches, test_batches, test_steps = model_trainer.setGenerators()

    # Create the model architecture
    model = model_trainer.createArchitecture()

    # Compile the model
    top2 = model_trainer.top_2_accuracy
    top3 = model_trainer.top_3_accuracy
    model_com = model_trainer.modelComplile(model, top2, top3)

    # Assign class weights
    class_weights = model_trainer.addWeight()

    # Train the model and output history
    history = model_trainer.output(model, train_batches, test_batches,test_steps, train_steps)
