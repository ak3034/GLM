'''
Created on Sep 25, 2013

@author: Anuj
'''

from Feature import *
from FileIterators import *
from FileWriter import *
from Perceptron import *
from Scheme import *

if __name__ == '__main__':
    
    #Development File
    file1 = open("tag_train.dat")
    #Training File
    file2 = open("tag_dev.dat")
    
    #Create some features
    unigram = N_Gram_Feature_Template(1)
    bigram = N_Gram_Feature_Template(2)
    trigram = N_Gram_Feature_Template(3)
    fourgram = N_Gram_Feature_Template(4)
    tagm2 = Tag_Displacement_Feature_Template(-2)
    tagm1 = Tag_Displacement_Feature_Template(-1)
    tag0 = Tag_Displacement_Feature_Template(-0)
    tag1 = Tag_Displacement_Feature_Template(1)
    tag2 = Tag_Displacement_Feature_Template(2)
    suffix2 = Suffix_Feature_Template(2)
    suffix3 = Suffix_Feature_Template(3)
    suffix4 = Suffix_Feature_Template(4)
    suffix5 = Suffix_Feature_Template(5)
    
    #Load them all into a feature set
    feature_set = Feature_Set([bigram])
    
    
    #Create the iterators and writer
    keyfile = MyKeyFileIterator(file1)
    devfile = MyDevFileIterator(file2)
    devwriter = DevFileWriter("eval_tester")
    
    #Create the scheme
    gen = NGramHistoryGenerator(2)
    enc = NGramHistoryEncoder(2)
    dec = NGramHistoryDecoder(2)
    scheme = NGramHistoryScheme(gen,enc,dec)
    
    #Create the percepton and train.
    p = Perceptron(scheme,feature_set)
    p.train(5,keyfile)
    p.save("2G.model")
    p.load("2G.model")
    p.test(devfile,devwriter)
   