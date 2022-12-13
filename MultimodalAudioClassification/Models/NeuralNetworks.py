"""
Repo:       MultimodalAudioClassification
Solution:   MultimodalAudioClassification
Project:    Models
File:       NeuralNetworks.py

Author:     Landon Buell
Date:       November 2022
"""

    #### IMPORTS ####

import os
import sys

import tensorflow as tf

    #### CLASS DEFINITIONS ####

class NullLayer(tf.keras.layers.Layer):
    """ Null Layer returns Inputs as Outputs """

    def __init__(self,name):
        """ Constructor """
        super().__init__(trainable=False,name=name)

    def __del__(self):
        """ Destructor """
        super().__del__()

    def call(self,inputs):
        """ Define Layers forward pass """
        return inputs

class NeuralNetworkBuilders:
    """ Class to Build TF Neural Networks """

    def __init__(self,denseLayersA,denseLayersB,denseLayersC,
                    filterSizes,kernelSizes,poolSizes):
        """ Constructor """
        self._denseLayersA  = denseLayersA
        self._denseLayersB  = denseLayersB
        self._denseLayersC  = denseLayersC
        self._filterSizes   = filterSizes
        self._kernelSizes   = kernelSizes
        self._poolSizes     = poolSizes
        self._optimizer     = tf.keras.optimizers.Adam(learning_rate=0.01,beta_1=0.9,beta_2=0.999,epsilon=1e-8)
        self._objective     = tf.keras.losses.CategoricalCrossentroy()
        self._metrics       = [tf.keras.Accuracy(), tf.keras.Precision(), tf.keras.Recall()]

    @staticmethod
    def getMultiLayerPerceptron(shapeInput,layerWidths,shapeOutput=None):
        """ Generate Multilayer Perceptron """
        modelInput = tf.keras.layers.Input(shape=shapeInput,name="inputMLP")
        x = NullLayer(name="N1A")(modelInput)

        # Add Dense Layers
        for i,neurons in enumerate(layerWidths):
            layerName = "D{0}A".format(i + 1)
            x = tf.keras.layers.Dense(units=neurons,activation='relu',
                                      name=layerName)(x)
        # If Num Outputs specified
        if (shapeOutput is not None):
            x = tf.keras.layers.Dense(units=shapeOutput,activations='softmax',name='outputMLP')

        # Build the Model
        x = tf.keras.Model(inputs=modelInput,outputs=x,name="MultilayerPerceptron")
        return x

    @staticmethod
    def getConvolutional2D(shapeInput,filterSizes,kernelSizes,poolSizes,layerWidths,shapeOutput=None):
        """ Generator 2D Convolutional Neural Network """
        modelInput = tf.keras.layers.Input(shape=shapeInput,name="inputCNN")
        x = NullLayer(name="N1B")
        
        # Add Layer Groups
        for i,(filters,kernel,pool) in enumerate(zip(filterSizes,kernelSizes,poolSizes)):
            x = tf.keras.layers.Conv2D(filters=filters,kernel_size=kernel,activation='relu',name="C{0}A".format(i+1))(x)
            x = tf.keras.layers.Conv2D(filters=filters,kernel_size=kernel,activation='relu',name="C{0}B".format(i+1))(x)
            x = tf.keras.layers.MaxPool2D(pool_size=pool,name="P{0}".format(i+1))(x)
        x = tf.keras.layers.Flatten(name='F1')(x)

        # Add Dense Layers
        for i,neurons in enumerate(layerWidths):
            layerName = "D{0}B".format(i + 1)
            x = tf.keras.layers.Dense(units=neurons,activation='relu',
                                      name=layerName)(x)
        # If Num Outputs specified
        if (shapeOutput is not None):
            x = tf.keras.layers.Dense(units=shapeOutput,activations='softmax',name='outputCNN')

        # Build the Model
        x = tf.keras.Model(inputs=modelInput,outputs=x,name="ConvolutionalNeuralNetwork")
        return x

    @staticmethod
    def getDefaultHybridModel(shapeInputA,shapeInputB,numClasses,name):
        """ Get the Default Hybrid Model for Training/Testing """
        denseLayersA    = [64,128,128,64]
        denseLayersB    = [128,128,64,64]
        denseLayersC    = [128,64]
        filterSizes     = [64,64,64]
        kernelSizes     = [(3,3),(3,3),(3,3)]
        poolSizes       = [(3,3),(3,3),(3,3)]
        optimizer       = tf.keras.optimizers.Adam(learning_rate=0.01,beta_1=0.9,beta_2=0.999,epsilon=1e-8)
        objective       = tf.keras.losses.CategoricalCrossentroy()
        metrics         = [tf.keras.Accuracy(), tf.keras.Precision(), tf.keras.Recall()]
        # Get the builder
        #builder = NeuralNetworkBuilders(denseLayersA,denseLayersB,denseLayersC,
         #                               filterSizes,kernelSizes,poolSizes)

        # Get Models
        modelMLP = NeuralNetworkBuilders.getMultiLayerPerceptron(shapeInputA,denseLayersA,None)
        modelCNN = NeuralNetworkBuilders.getConvolutional2D(shapeInputB,filterSizes,kernelSizes,poolSizes,denseLayersB,None)

        # Models
        x = tf.keras.layers.concatenate([modelMLP.output,modelCNN.output])
        for i,neurons in enumerate(denseLayersC):
            layerName = "D{0}C".format(i + 1)
            x = tf.keras.layers.Dense(units=neurons,activation='relu',
                                      name=layerName)(x)
        x = tf.keras.layers.Dense(units=numClasses,activation='softmax',name='output')(x)

        # Aggregate Models
        modelHNN = tf.keras.Model(name=name,inputs=[modelMLP.input,modelCNN.input],outputs=x)
        modelHNN.compile(optimizer=optimizer,loss=objective,metrics=metrics)
        return modelHNN
        