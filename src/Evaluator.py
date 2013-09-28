#! /usr/bin/python
'''
Created on Sep 26, 2013

@author: Anuj Khanna <ak3034@columbia.edu>
@thanks: Daniel Bauer <bauer@cs.columbia.edu> 
'''

import sys
import tagger_config
from collections import defaultdict

def corpus_iterator(corpus_file, with_logprob = False):
    """
    Get an iterator object over the corpus file. The elements of the
    iterator contain (word, ne_tag) tuples. Blank lines, indicating
    sentence boundaries return (None, None).
    """
    l = corpus_file.readline()    
    tagfield = with_logprob and -2 or -1

    try:
        while l:
            line = l.strip()
            if line: # Nonempty line
                # Extract information from line.
                # Each line has the format
                # word ne_tag [log_prob]
                fields = line.split()
                ne_tag = fields[tagfield]
                word = " ".join(fields[:tagfield])
                yield word, ne_tag
            else: # Empty line
                yield (None, None)
            l = corpus_file.readline()
    except IndexError:
        sys.stderr.write("Could not read line: \n")
        sys.stderr.write("\n%s" % line)
        if with_logprob:
            sys.stderr.write("Did you forget to output log probabilities in the prediction file?\n")
        sys.exit(1)

ne_classes = tagger_config.tags

class Evaluator(object):
    """
    Stores global true/false positive/negative counts. 
    """


    ne_classes = tagger_config.tags
    
    def __init__(self):
        #Precision = class_correct_counts[X] / class_prediction_counts[X]
        #Recall = class_correct_counts[X] / class_reference_counts[X]
        #Accuracy = class_correct_counts / class_predictions_counts
        
        self.class_correct_counts = defaultdict(lambda :0)
        self.class_prediction_counts = defaultdict(lambda :0)
        self.class_gold_counts = defaultdict(lambda : 0)
        
    def compare(self, gold_standard, prediction):
        """
        Compare the prediction against a gold standard. Both objects must be
        generator or iterator objects that return a (word, ne_tag) tuple at a
        time.
        """
        for gs_word, gs_tag in gold_standard: # Move through the gold standard stream
            
            pred_word, pred_tag = prediction.next() # Get the corresponding item from the prediction stream
            
            # Make sure words in both files match up
            total = 0
            if gs_word != pred_word:
                sys.stderr.write("Could not align gold standard and predictions in line %i.\n" % (total+1))
                sys.stderr.write("Gold standard: %s  Prediction file: %s\n" % (gs_word, pred_word))
                sys.exit(1)        
    
            #correct prediction for the class
            if gs_tag == pred_tag:
                self.class_correct_counts[pred_tag] += 1
                self.class_prediction_counts[pred_tag] += 1
                self.class_gold_counts[gs_tag] += 1
            
            #incorrect prediction for the class
            if gs_tag != pred_tag:
                self.class_gold_counts[gs_tag] += 1
                self.class_prediction_counts[pred_tag] += 1
                
    def print_scores(self):
        """
        Output a table with accuracy, precision, recall and F1 score. 
        """

        prec = sum(self.class_correct_counts.values()) / float(sum(self.class_prediction_counts.values()))
        rec = sum(self.class_correct_counts.values()) / float(sum(self.class_gold_counts.values()))
        
        print("\t precision \trecall \t\tF1-Score")
        fscore = (2*prec*rec)/(prec+rec)
        print("Total:\t %f\t%f\t%f" % (prec, rec, fscore))
        for c in  ne_classes:
            c_prec = self.class_correct_counts[c] / float(self.class_prediction_counts[c]) if self.class_prediction_counts[c] != 0 else 0 
            c_rec = self.class_correct_counts[c] / float(self.class_gold_counts[c]) if self.class_gold_counts[c] != 0 else 0
            sum(self.class_correct_counts.values()) / float(sum(self.class_prediction_counts.values()))

            if c_prec + c_rec == 0:
                fscore = 0
            else:    
                fscore = (2*c_prec * c_rec)/(c_prec + c_rec)
            print("%s:\t %f\t%f\t%f" % (c, c_prec, c_rec, fscore))


def usage():
    sys.stderr.write("""
    Usage: python Evaluator.py [key_file] [prediction_file]
        Evaluate the NE-tagger output in prediction_file against
        the gold standard in key_file. Output accuracy, precision,
        recall and F1-Score for each NE tag type.\n""")

if __name__ == "__main__":

    if len(sys.argv)!=3:
        usage()
        sys.exit(1)
    gs_iterator = corpus_iterator(file(sys.argv[1]))
    pred_iterator = corpus_iterator(file(sys.argv[2]), with_logprob = False)
    evaluator = Evaluator()
    evaluator.compare(gs_iterator, pred_iterator)
    evaluator.print_scores()