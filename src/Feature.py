'''
Created on Sep 21, 2013

@author: Anuj
'''
from collections import defaultdict

class Feature_Set:
    """
    Feature_Set class.
 
    This class acts as an aggregation for all the feature templates comprising the learning model. This
    class handles the task of scoring a candidate tag_sequence by managing a dictionary of scores (f x W)
    for all features associated with a candidate tag sequence. 
    
    Note: All features must be extensions of the Feature_Template class and the set of features supplied to the 
    constructor must be iterable.
    """
    
    def __init__(self,features):
        
        assert(features), \
        "Supplied feature set is None"
        assert(hasattr(features, '__iter__')), \
        "Supplied feature set must be iterable"
        
        self.features = features
        self.weights = defaultdict(lambda: 0)
        
    def compute_feature_vector(self,sentence, candidate):
        return {feature.hash(sentence, candidate) : feature.get_feature_weight(sentence, candidate) for feature in self.features}
    
    def update_weight_vector(self, example, gold_tags, max_candidate):
        for feature in self.features:
            feature.update_weight_vector(example,gold_tags, max_candidate)        


def lambda0(): return 0

class Feature_Template():
    """
    Feature_Template class.
 
    This class provides an abstraction for any user-specified feature template denoting a particular
    local feature in the learning model. A feature template is fundamentally a dictionary
    of weights associated with a particular feature. For example the weigh_vector property stipulated
    in the children of the Feature_Template class might look something like this:
    
        {
            (2-gram,(*,DET) : 4
            (2-gram,(*,ADV) : 2
            (2-gram,(ADV,VERB) : -1
            .
            .
        }
        
    To implement a feature, the Feature_Template class must be extended and implementation be provided for
    the hash method.
        
    """
    def __init__(self):
        #Needed for pickling        
        self.weight_vector = dict()
    
    def hash(self, sentence,context):
        "Abstract method for hash functionality"
        raise NotImplementedError
    
    def get_feature_weight(self, sentence, context):
        """
        Returns the sum of all features for a particular context. Note that since each context has only one
        assignable feature for each feature template the sum is simply the weight of the feature for that
        context.
        """
        key = self.hash(sentence, context)
        return self.weight_vector.get(key,0)
    
    def update_weight_vector(self, sentence, gold_contexts,max_contexts):
        """
        A sparse hashmap implementation for updating the weight vector. Updates are based
        on the formula:
        
        W = W + f(sentence, gold_tag) - f(sentence, candidate)
        
        where f(x,y) is a binary function representing the existence of a particular feature.
        """
        
        w_z = defaultdict(lambda : 0)
        w_y = defaultdict(lambda : 0)
        for context in gold_contexts:w_y[self.hash(sentence, context)] += 1
        for context in max_contexts: w_z[self.hash(sentence, context)] += 1
        
        for k in set(w_y) | set(w_z): self.weight_vector[k] = self.weight_vector.get(k,0) + w_y.get(k,0) - w_z.get(k,0)      
    
        del(w_z)
        del(w_y)
    
class N_Gram_Feature_Template(Feature_Template):
    """
    An N_Gram_Feature_Template class
    
    This class extends Feature_Template and provides the necessary functionality to ascertain the relevant context weight
    for the respective N-Gram Feature that represents it. For example:
    
    Sentence = [The quick brown fox jumped over the lazy dog]
    Context = [4, (ADV,ADJ,NOUN)]
    
    N_Gram_Feature_Template(3)'s dictionary would consist of a mapping of the key (3-gram, (ADV,ADJ,NOUN)) which associates the trigram
    (ADV,ADJ,NOUN) with a history generated at position 4.
    """
    def __init__(self,n):
        Feature_Template.__init__(self)
        self.n = n
        
    def __str__(self):
        return str(self.n) + "-gram"
    
    def hash(self, sentence, context):
        i , ngram = context
        
        assert(len(ngram) >= self.n), \
        "Context encoding does not support this feature. Context must be a %5s" % self
        
        return (str(self),ngram)    
        
class Tag_Displacement_Feature_Template(Feature_Template):
    """
    An Tag_Displacement_Feature_Template class
    
    This class extends Feature_Template and provides the necessary functionality to compare the word associated at a displacement d 
    from the contextual tag. For example:
    
    Sentence = [The quick brown fox jumped over the lazy dog]
    Context = [4, (NOUN)]
    
    Tag_Displacment_Feature_Template(-2)'s dictionary would consist of a mapping of the key (quick, Noun) which associates the word at
    position 2 (quick) with the tag at position 4 (NOUN). 
    
    """
    def __init__(self,d):
        Feature_Template.__init__(self)
        self.d = d
        
    def __str__(self):
        return str(self.d) + "-tag"
    
    def hash(self, sentence, context):
        i, ngram = context
        word = sentence[i-1+self.d] if i-1+self.d >= 0 and i-1+self.d < len(sentence) else "*" if i-1+self.d <= 0 else "."
        tag = ngram[-1]
        f_key = (word,tag)
        return (str(self),f_key)

class Suffix_Feature_Template(Feature_Template):
    """
    A Suffix_Feature_Template class
    
    This class extends Feature_Template and provides the necessary functionality to compare n length suffices with the contextual tag.
    For example:
    
    Sentence = [The quick brown fox jumped over the lazy dog]
    Context = [5, (VERB)]
    
    Suffix_Feature_Template(2)'s dictionary would consist of a mapping of the key ('ed',VERB) which associates the last 2 letters of the
    word at position 5 in the sentence with the tag VERB. 
    
    """
    def __init__(self,s):
        assert(s > 0), "Suffix length must be a positive integer greater than 0"
        Feature_Template.__init__(self)
        self.s = s
        
    def __str__(self):
        return str(self.s) + "-suffix"
    
    def hash(self, sentence, context):
        i, ngram = context
        tag = ngram[-1]
        if len(tag) < self.s:
            suffix = ngram[-1]
        else:
            suffix = ngram[-1][-self.s:]
        f_key = (suffix,tag)
        return (str(self),f_key)

class Pre_Feature_Template(Feature_Template):
    """
    A Pre_Feature_Template class
    
    This class extends Feature_Template and provides the necessary functionality to compare n length prefixes with the contextual tag.
    For example:
    
    Sentence = [The quick brown fox jumped over the lazy dog]
    Context = [2, (ADV)]
    
    Prefix_Feature_Template(3)'s dictionary would consist of a mapping of the key ('qui',ADV) which associates the first 3 letters of the
    word at position 2 in the sentence with the tag ADV. 
    """
    def __init__(self,p):
        assert(p > 0), "Suffix length must be a positive integer greater than 0"
        self.p = p
        self.weights = defaultdict(lambda: 0)
        
    def __str__(self):
        return str(self.s) + "-prefix"
    
    def hash(self, sentence, context):
        i, ngram = context
        suffix = ngram[-1][:self.p]
        tag = ngram[-1]
        f_key = (suffix,tag)
        return (str(self),f_key)


if __name__ == "__main__":
    "Debugger" 
    bigram = N_Gram_Feature_Template(2)
    feature_set = Feature_Set([bigram])
  
    


