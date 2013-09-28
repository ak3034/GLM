'''
Created on Sep 22, 2013

@author: Anuj
'''

"""
As of yet FileIterator types have not been throughly tested with stdin handles.
"""
class FileIterator:
    "A wrapper for the iterator class"
    
    def __init__(self,handle):
        self.handle = handle
        
    def __iter__(self):
        return self
    
    def read(self):
        raise NotImplementedError
    
class KeyFileIterator(FileIterator):
    """
    An Iterator class designed specifically to iterator through a file numerous times. Class might not be thread safe since
    iterator/generator is never consumed.
    
    IMPORTANT: A key file must be of this type else an error will be thrown.
    """
    
    def __init__(self,handle):
        FileIterator.__init__(self, handle)
        
    def next(self):
        try:
            tagged_example = self.read().next()
        except StopIteration:
            self.handle.seek(0)
            raise StopIteration
        
        assert(tagged_example),\
        "Key File read error: None extracted"
        assert(type(tagged_example) is tuple and len(tagged_example) == 2),\
        "Key file read error: Key file child must return a 2-tuple of sentence and gold tag sequence."
        for t in tagged_example:
            assert(hasattr(t, '__iter__')),\
        "Key file read error: Return types of sentence and tag sequence must be iterable."
        
        return tagged_example
    
    def read(self):
        raise NotImplementedError
    
    
class DevFileIterator(FileIterator):
    """
    A work around the decorator pattern for reading development files. 
    
    IMPORTANT: A development/testing file must be of this type else an error will be thrown.
    
    """
    
    def __init__(self,handle):
        FileIterator.__init__(self, handle)
        
    def next(self):
        untagged_example = self.read().next()
        
        assert(hasattr(untagged_example, '__iter__')),\
        "Dev file read error: Return types of sentence must be iterable."
        
        return untagged_example
    
    def read(self):
        raise NotImplementedError
    
    
class MyKeyFileIterator(KeyFileIterator):
    """
    A concrete implementation of how the Key File should be read.
    """
    
    def __init__(self,handle):
        KeyFileIterator.__init__(self, handle)
        
    def read(self):
        sentence = []
        tag_sequence = []
        while 1:
            line = self.handle.readline()
            if not line.strip():
                if sentence != []:
                    yield sentence,tag_sequence
                    sentence = []
                    tag_sequence = []
                else:
                    return
            else:
                a,b = line.strip().split()
                sentence.append(a)
                tag_sequence.append(b)
        yield sentence, tag_sequence
        
class MyDevFileIterator(DevFileIterator):
    """
    A concrete implementation of how the Development/Testing File should be read.
    """
    def __init__(self,handle):
        DevFileIterator.__init__(self, handle)
        
    def read(self):
        sentence = []
        while 1:
            line = self.handle.readline()
            if not line.strip():
                if sentence != []:
                    yield sentence
                    sentence = []
                else:
                    return
            else:
                a = line.strip()
                sentence.append(a)
        yield sentence