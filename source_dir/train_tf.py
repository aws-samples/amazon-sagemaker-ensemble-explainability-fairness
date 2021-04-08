import argparse
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def parse_args():
    
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    
    # data directories
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    
    # model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
    return parser.parse_known_args()

def get_model2():
    
    output_bias = tf.keras.initializers.Constant(-0.84729786)
    inputs = tf.keras.Input(shape=(50,))
    hidden_1 = tf.keras.layers.Dense(64, activation='relu')(inputs)
    dropout_1 = tf.keras.layers.Dropout(0.1)(hidden_1)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid',bias_initializer=output_bias)(dropout_1)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def get_model():
    output_bias = tf.keras.initializers.Constant(-0.84729786)
    model = keras.Sequential([
      keras.layers.Dense(
          64, activation='relu',
          input_shape=(50,)),
      keras.layers.Dropout(0.1),
      keras.layers.Dense(1, activation='sigmoid',
                         bias_initializer=output_bias),])
    return model

def get_train_data(train_files_path,validation_files_path):
    
    
    train_features_path = os.path.join(train_files_path, 'train_features.csv')
    train_labels_path = os.path.join(train_files_path, 'train_labels.csv')
    
    val_features_path = os.path.join(validation_files_path, 'val_features.csv')
    val_labels_path = os.path.join(validation_files_path, 'val_labels.csv')
    
    print('Loading training dataframes...')
    df_train_features = pd.read_csv(train_features_path)
    df_train_labels = pd.read_csv(train_labels_path)
    
    print('Loading validation dataframes...')
    df_val_features = pd.read_csv(val_features_path)
    df_val_labels = pd.read_csv(val_labels_path)
    
    X = df_train_features.values
    y = df_train_labels.values
    
    val_X = df_val_features.values
    val_y = df_val_labels.values
    
    
    
    
    
    print('x train', X.shape,'y train', y.shape)
    print('x val', val_X.shape,'y val', val_y.shape)
    print(X)
    #print(y)
    return X, y, val_X, val_y


if __name__ == "__main__":
        
    args, _ = parse_args()
    
    x_train, y_train, x_val, y_val = get_train_data(args.train,args.validation)
        
    device = '/cpu:0' 
    print(device)
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    print('batch_size = {}, epochs = {}, learning rate = {}'.format(batch_size, epochs, learning_rate))

    with tf.device(device):
        
        model = get_model()
        optimizer=keras.optimizers.Adam(learning_rate)
        METRICS = [
          keras.metrics.TruePositives(name='tp'),
          keras.metrics.FalsePositives(name='fp'),
          keras.metrics.TrueNegatives(name='tn'),
          keras.metrics.FalseNegatives(name='fn'), 
          keras.metrics.BinaryAccuracy(name='accuracy'),
          keras.metrics.Precision(name='precision'),
          keras.metrics.Recall(name='recall'),
          keras.metrics.AUC(name='auc'),
        ]
        model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(),metrics=METRICS)  
        
        #model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
        #          validation_data=(x_val,y_val),class_weight=class_weight)
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                 validation_data=(x_val,y_val))
        # evaluate on train set
        scores = model.evaluate(x_train, y_train, None, verbose=1)
        print("\ntrain bce :", scores)
                
        # evaluate on val set
        scores = model.evaluate(x_val, y_val, None, verbose=1)
        print("\nval bce :", scores)
        
        #print val set predictions
        print(model.predict(x_val))
        
        # save model
        model.save(args.model_dir + '/1')