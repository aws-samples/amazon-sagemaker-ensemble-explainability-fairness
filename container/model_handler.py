"""
ModelHandler defines an example model handler for load and inference requests 
"""
from collections import namedtuple
import glob
import json
import logging
import os
import re
import tarfile
import pickle as pkl
import csv
import io
import subprocess
import pandas as pd
import xgboost as xgb
import tensorflow as tf
from sklearn.externals import joblib
from io import StringIO

import numpy as np
from six import BytesIO, StringIO

from sagemaker_inference import content_types, decoder, default_inference_handler, encoder, errors

class ModelHandler(object):     
    
    
    def __init__(self):
        self.initialized = False
        self.preprocessor = None
        self.model1 = None
        self.model2 = None
        self.shapes = None
        self.feature_columns_names = ["status","duration","credit_history","purpose","amount","savings","employment_duration","installment_rate","personal_status_sex","other_debtors","present_residence","property","age","other_installment_plans","housing","number_credits","job","people_liable","telephone","foreign_worker"]
        
    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self.initialized = True
        properties = context.system_properties
        # Contains the url parameter passed to the load request
        model_dir = properties.get("model_dir") 
        print(model_dir)
        print('***********************')
        result = subprocess.run(['ls', model_dir], stdout=subprocess.PIPE)
        print(result.stdout)
        result = subprocess.run(['pwd'], stdout=subprocess.PIPE)
        print(result.stdout)
        tar = tarfile.open( model_dir + '/models/xgb.tar.gz', "r:gz")
        tar.extractall('/home/model-server/')
        tar.close()
        tar = tarfile.open( model_dir + '/models/tf.tar.gz', "r:gz")
        tar.extractall('/home/model-server/')
        tar.close()
        
        self.preprocessor = joblib.load(model_dir + "/models/sklearn.joblib")
        
        #model_file1 = '/home/model-server/xgboost-model'
        model_file1 = '/home/model-server/model.bin'
        self.model1 = pkl.load(open(model_file1, 'rb'))   
        
        model_file2 = '/home/model-server/1/'
        self.model2 = tf.keras.models.load_model(model_file2)
        print(self.model2.summary())
        

    
    def preprocess(self, request):
        """
        Transform raw input into model input data.
        :param request: list of raw requests
        :return: list of preprocessed model input data
        """
        
        #model_input = decoder.decode(request[0].get('body').decode(),content_types.CSV)
        model_input = request[0].get('body').decode()
        print(model_input)
        print(type(model_input))
        print('preprocess')
        model_input = StringIO(model_input)
        print(type(model_input))
        model_input = pd.read_csv(model_input,header=None,sep=',')
        print(model_input)
        if len(model_input.columns) == len(self.feature_columns_names):
            model_input.columns = self.feature_columns_names 
        print(model_input)
        features = self.preprocessor.transform(model_input)
        print('features')
        print(features)
        if features.ndim == 1:
            features = np.expand_dims(features, axis=0)
        #print(model_input)     
        return features

    def inference(self, model_input):
        
        # Do some inference call to engine here and return output
        print('inference')
        output1 = np.reshape(self.model1.predict(xgb.DMatrix(model_input), validate_features=False),(-1,1))
        output2 = self.model2.predict(model_input)
        #print('tf2 output')
        #print(output2)
        avg = np.mean((output1,output2),axis=0)
        output = np.concatenate((output1,output2,avg),axis=1)
        print('inference')
        #print(output)
        return output

    def postprocess(self, inference_output):
        """
        Return predict result in as list.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        #print (inference_output)
        print('postprocess with encoder custom')
        out1 = ''
               
        for x in inference_output:
            #print(x)
            stream = StringIO()
            np.savetxt(stream, x, delimiter=',', fmt='%.2f', newline=',')
            out1+=str('"[' + str(stream.getvalue()) + ']",')
            #print(out1)
            
        print('output')
        out1 = out1[:-1]
        out = out1.encode("utf-8")
        print(out)
        return [out]
        
    def handle(self, data, context):
        """
        Call preprocess, inference and post-process functions
        :param data: input data
        :param context: mms context
        """
        
        model_input = self.preprocess(data)
        model_out = self.inference(model_input)
        return self.postprocess(model_out)

_service = ModelHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)