'''
Created on Sep 23, 2013

@author: Anuj
'''
from itertools import *
import tagger_config
from collections import defaultdict    
import tagger_config as baseline
        
class Scheme:
    """
    A Scheme class
    
    The scheme class is responsible for the following:
         Gen:    A generator that produces a list of candidate tag sequences that is defined over the entire sentence structure.
         Dec:    A search algorithm that finds the highest scoring tag sequence from the list provided by the generator.
         Enc:    An encoder which is responsible for converting a tag sequence into a series of histories/contexts under a
                 user-defined stipulation of whether or not the problem is tractable. 
         
    Note: A scheme of size 1 is still a bit buggy, we want to avoid that for the time being

    """
    def __init__(self):
        pass
    
class Generator:
    """
    A Generator class for future abstractions.
    """
    def __init__(self):
        pass

class Encoder:
    """
    An Encoder class for future abstractions.
    """
    def _init__(self):
        pass
    
class Decoder:
    """
    A Decoder class for future abstractions.
    """
    def _init__(self):
        pass


class NGramHistoryScheme(Scheme):
    """
    An NGramHistorySchene Class
    
    This class is responsible for ensuring that the tractability of the history based model of the Global Linear Tagger is consolidated and enforced.
    """
    
    def __init__(self,gen,enc,dec):
        
        Scheme.__init__(self)
        
        assert(isinstance(gen,NGramHistoryGenerator)),\
        "Supplied gen must extend the NGramHistoryGenerator Class"
        assert(isinstance(enc,NGramHistoryEncoder)),\
        "Supplied enc must extend the NGramHistoryEncoder Class"
        assert(isinstance(dec,NGramHistoryDecoder)),\
        "Supplied dec must extend the NGramHistoryDecoder Class"
        
        assert(gen.N == enc.N == dec.N),\
        "NGramHistoryScheme members must share the same N"
        
        
        self.generator = gen
        self.encoder = enc
        self.decoder = dec
        
    def gen(self,sentence):
        return self.generator(sentence)
        
    def enc(self,tag_sequence):
        return self.encoder(tag_sequence)
    
    def dec(self,candidate):
        return self.decoder(candidate)
    
class NGramHistoryGenerator(Generator):
    """
    This class is responsible for generating all possible contexts for a tag_sequence over the entire structure.
    """
    
    def __init__(self,N):
        assert(N & N > 0),\
        "Supplied n must be an integer greater than 0"
        
        self.N = N
        
    def __call__(self,sentence):
        return self.gen(sentence)

    def gen(self,sentence):
        tagsets = [baseline.dict.get(word, baseline.tags) for word in sentence]
        for i in range(1,len(sentence)+1):
            sets = [tagsets[j] if j >= 0 else ["*"] for j in range(i-self.N,i)]
            for enum in product(*sets):
                yield (i, enum)
                
class NGramHistoryEncoder:
    """
    This class is responsible for breaking a tag_sequence down into its contexts.
    """
    
    def __init__(self,N):
        self.N = N
        
    def __call__(self,tag_sequence):
        return self.enc(tag_sequence)
    
    def enc(self,tag_sequence):
        return [(i,tuple([tag_sequence[j] if j >= 0 else "*" for j in range(i-self.N,i)])) for i in range(1,len(tag_sequence)+1)] 

class NGramHistoryDecoder:
    """
    This class is responsible for decoding NGram history scores. 
    
    Note: This class needs further testing.
    """
    
    def __init__(self,N):
        self.N = N
        
    def __call__(self,x):
        return self.decode(x)
    
    
    def reconfigure(self,scores):
        """
        Reconfigues the scores file to be a dictionary of dictionaries so reference by word indexing is faster.
        Not a very clean implementation, will try to fix.
        """
        d = defaultdict(dict)
        for a,b in scores.items():
            position, ngram = a
            d[position].update({ngram:b})
        return d
        
    def decode(self,scores):
        """
        Implementation of the Viterbi Decoding algorithm.
        """      
        scores  = self.reconfigure(scores)
        
        n = max(scores.keys())
        N = self.N
        T = baseline.tags + baseline.terminals
        y = [""] * (n + 1)
        
        def q(k, ngram): return scores[k].get(ngram,-1e10)
        def argmax(ls): return max(ls, key = lambda x: x[1])
        # Decoding with 1-gram histories is solvable using a greedy strategy at
        #each possible history at K.
        if N == 1:
            for K in range(1, n+1):
                (tag,), score = argmax(scores[K].items())
                y[K] = tag
            y[0] = '*'
            return y[1:n + 1]
                
        # The Viterbi algorithm.
        # Create and initialize the chart.
        pi = defaultdict(lambda : -1e10)
        pi[(0,(N-1)*tuple("*"))] = 1.0
        bp = defaultdict(lambda : ('*',))
        
        # Run the main loop. 
        for K in range(1, n + 1):
            for ngram in scores[K].keys():
                quvw = ngram[1:]
                uvwx = ngram[:-1]
                bp[K,quvw], pi[K,quvw] = (uvwx[0],pi[K-1,uvwx] + q(K,ngram)) if pi[K-1,uvwx] + q(K,ngram) > pi[K,quvw]  else (bp[K,quvw],pi[K,quvw])
        #Compute back pointers
        
        y[-(N-1):], score  = argmax([(ngram[1:], pi[(n, ngram[1:])] + q(n + 1, ngram[1:] + ("STOP",))) for ngram in scores[n].keys()])
        for k in range(n - (N-1), 0, -1):
            y[k] = bp[(k+(N-1) ,tuple([y[k+i] for i in range(1,N)]))]    
        y[0] = "*"
        
        return y[1:n + 1]
            
            
                