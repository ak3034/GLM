'''
Created on Sep 21, 2013

@author: Anuj
'''
from __future__ import print_function
from Scheme import *
from FileIterators import *
from Feature import *
from FileWriter import *
import pickle

class Perceptron:
    """
    Perceptron class
    
    This class is responsible for performing Perceptron classification in the context of Part-Of-Speech Tagging. The
    classification algorithm makes its predictions based on a linear predictor function by combining a set of weights
    with the feature vector describing a given input.
    
    The PoS Perceptron classifier requires a scheme. A scheme is responsible for making the PoS problem tractable if such a scheme
    exists for the feature set. 
    
    
    On a side note The perceptron algorithm was invented in 1957 at the Cornell Aeronautical Laboratory by Frank Rosenblatt.
    """
    
    def __init__(self,scheme,feature_set):
        
        assert(scheme), \
        "Supplied scheme is None "
        assert isinstance(feature_set, Feature_Set), \
        "Feature Set must be an instance of the Feature_Set class"
        
        self.scheme = scheme
        self.feature_model = feature_set
          
    def train(self, iterations, examples):
        
        assert isinstance(examples, KeyFileIterator), \
        "Examples must be an extension of the KeyFileIterator class"
        
        def argmax(ls): return self.scheme.dec(ls)
        
        for i in range(iterations):
            print('I =' + str(i))
            for sentence, gold_tags in examples:
                max_candidate = argmax({c:w for (c,w) in [self.f(sentence,candidate) for candidate in self.scheme.gen(sentence)]})
                self.update(sentence, gold_tags, max_candidate) 
                
    
    def update(self,example, gold_tags, max_candidate):
        if(gold_tags != max_candidate):
            z = self.scheme.enc(gold_tags)
            y = self.scheme.enc(max_candidate)
            self.feature_model.update_weight_vector(example,z,y)
        
    def f(self,sentence,candidate):
        weight = sum(self.feature_model.compute_feature_vector(sentence,candidate).values())
        return (candidate,weight)
    
    def test(self, iterator, writer):
        """
        Classifies each sentence in a development file in the 
        """
        
        assert isinstance(iterator, DevFileIterator), \
        "Supplied iterator must be an extension of the DevFileIterator class"
        assert isinstance(writer, DevFileWriter), \
        "Supplied writer must be an extension of the DevFileWriter class"
        
        def argmax(ls): return self.scheme.dec(ls)
        
        with writer:
            for sentence in iterator:
                max_candidate = argmax({c:w for (c,w) in [self.f(sentence,candidate) for candidate in self.scheme.gen(sentence)]})
                writer.write(sentence,max_candidate)
             
    def save(self,file_name):
        """
        Serializes the weights of the Perceptron model to a file.
        """
        try:
            with open(file_name, "w+") as serial_file:
                pickle.dump(self.feature_model.features,serial_file)
        except(RuntimeError, IOError):
            print("There was an error saving the file")
    
    def load(self,file_name):
        """
        Loads the weights of the Perceptron model to a file.
        """
        try:
            with open(file_name, "r") as serial_file:
                features = pickle.load(serial_file)
                self.feature_model.features = features
        except (RuntimeError, IOError):
            print("There was an error loading the model.")