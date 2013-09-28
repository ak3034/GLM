'''
Created on Sep 26, 2013

@author: Anuj
'''

"""
As of yet FileWriter types have not been throughly tested with stdout handles.
"""
class FileWriter:
    "A Wrapper for the file class"
    def __init__(self,fileName):
        self.f = open(fileName,"w+")
        
    def __enter__ (self):
        return self.f
    
    def __exit__ (self, exc_type, exc_value, traceback):
        self.f.close()
    
    def write(self,a,b):
        raise NotImplementedError

    
class DevFileWriter(FileWriter):
    """
    A contract based extension for the perceptron to format its output according to the write method. Note
    a decorator pattern could also be used and is likely more suitable.
    """
    def __init__(self,filename):
        FileWriter.__init__(self,filename)
    
    def write(self,a,b):
        for x,y in zip(a,b):
            self.f.write(str(x) + " " + str(y) +"\n")
        self.f.write("\n") 