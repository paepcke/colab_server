'''
Created on Jun 7, 2020

@author: paepcke
'''
import os
import unittest
import numpy as np
import pandas as pd
from preparation.text_augmentation import TextAugmenter

TEST_ALL = True
#TEST_ALL = False

class TestTextAugmentation(unittest.TestCase):

    #------------------------------------
    # setUpClass 
    #-------------------
    
    @classmethod
    def setUpClass(cls):
        df = \
        pd.DataFrame({'text' : ['hare is frog',
                                'pig is cow',
                                'this is too long a sequence!',
                                'this is too long a sequence!',
                                'this is too long a sequence!',
                                'shorty'],
                      'label' : [0,1,0,0,0,2],
                      'tokens': [['[CLS]', 'hare', 'is', 'frog', '[SEP]'],
                                 ['[CLS]', 'pig', 'is', 'cow', '[SEP]'],
                                 ['[CLS]', 'this', 'is', 'too', '[SEP]'],
                                 ['[CLS]', 'long', 'a', 'sequence', '[SEP]'],
                                 ['[CLS]', '!', '[SEP]'],
                                 ['[CLS]', 'short', '##y', '[SEP]']
                                 ]
                      },
                      index=[0,1,2,2,2,3]
                      ) 
        TestTextAugmentation.tokenized_df = df
        
    #------------------------------------
    # setUp 
    #-------------------
     
    def setUp(self):
        self.sequence_len = 5
        self.corr_tokens = [['[CLS]','hare','is','frog','[SEP]'],
                        	['[CLS]','pig','is','cow','[SEP]'],
                        	['[CLS]','this','is','too','[SEP]'],
                        	['[CLS]','long','a','sequence','[SEP]'],
                        	['[CLS]','!','[SEP]'],
                        	['[CLS]','short', '##y','[SEP]']
                            ]
        
        TestTextAugmentation.corr_ids    = \
          (np.array([[  101, 14263,  2003, 10729,   102],
    	        	 [  101, 10369,  2003, 11190,   102],
    	        	 [  101,  2023,  2003,  2205,   102],
    	        	 [  101,  2146,  1037,  5537,   102],
    	        	 [  101,   999,   102,     0,     0],
    	        	 [  101,  2460,  2100,   102,     0]]),)

        tst_msg_path = os.path.join(os.path.dirname(__file__), 'msgs_for_testing.csv')
        self.augmenter = TextAugmenter(tst_msg_path, 
                                       self.sequence_len,
                                       testing=True)

    #------------------------------------
    # tearDown 
    #-------------------

    def tearDown(self):
        pass

    #------------------------------------
    # test_tokenize 
    #-------------------

    @unittest.skipIf(not TEST_ALL, 'Temporarily skip this test.')
    def test_tokenize(self):
        tokenized_df = self.augmenter.fit_to_sequence_len(self.augmenter.train_df)
        # First .all() gives
        #     text      True
        #     label     True
        #     tokens    True
        #     dtype: bool
        # Then the second all(), summarizes those.
        
        self.assertTrue((tokenized_df == TestTextAugmentation.tokenized_df).all().all())

    #------------------------------------
    # test_padded_ids 
    #-------------------
    
    @unittest.skipIf(not TEST_ALL, 'Temporarily skip this test.')
    def test_padded_ids(self):
        padded_ids = self.augmenter.padded_ids(self.corr_tokens)
        np.allclose(padded_ids, self.corr_ids)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()