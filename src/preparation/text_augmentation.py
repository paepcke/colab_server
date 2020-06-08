'''
Created on Jun 7, 2020

@author: paepcke
'''

import os
import re
from pytorch_pretrained_bert import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd


class TextAugmenter(object):
    '''
    Minimal columns: 'text', 'label'
    '''

    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, 
                 train_files, 
                 sequence_len=128,
                 testing=False):
        '''
        Constructor
        '''
        self.sequence_len = sequence_len
        self.train_df = self.read_files(train_files)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        
        if testing:
            # Let unittests call the other methods.
            return
        token_seqs = self.fit_to_sequence_len(self.train_df)
        ids = self.padded_ids(token_seqs)

    #------------------------------------
    # fit_to_sequence_len 
    #-------------------
    
    def fit_to_sequence_len(self, train_df):
        # Add a col to the passed-in df: 'tokens'.
        # Create list the height of train_df:
        token_col = ['']*len(train_df)
        train_df['tokens'] = token_col
        
        new_rows = []
        nl_pat = re.compile(r'\n')
        for (_indx, row) in train_df.iterrows():
            # Remove \n chars;
            txt = nl_pat.sub(' ', row['text'])
            # And chop into a seq of strings:
            tokenized_txt = self.tokenizer.tokenize(txt)
            
            # Short enough to just keep?
            # The -2 allows for the [CLS] and [SEP] tokens:
            if len(tokenized_txt) <= self.sequence_len - 2:
                row['tokens'] = ['[CLS]'] + tokenized_txt + ['[SEP]']
                new_rows.append(row)
                continue

            # Go through the too-long tokenized txt, and cut into pieces
            # in which [CLS]<tokens>[SEP] are sequence_len long:
            for pos in range(0,len(tokenized_txt),self.sequence_len-2):
                sent_fragment = ['[CLS]'] + \
                                tokenized_txt[pos:pos+self.sequence_len-2]  + \
                                ['[SEP]']
                # Make a copy of the row, and fill in the token:
                new_row = row.copy()
                new_row['tokens'] = sent_fragment
                # Add to the train_df:
                new_rows.append(new_row)
        new_rows_df = pd.DataFrame(new_rows, columns=train_df.columns)
        return new_rows_df

    #------------------------------------
    # augment_text 
    #-------------------

    def augment_text(self):
        pass

    #------------------------------------
    # padded_id
    #-------------------
    
    def padded_ids(self, token_seqs):
        ids = [self.tokenizer.convert_tokens_to_ids(x) for x in token_seqs]
        padded_ids = pad_sequences(ids, 
                                   maxlen=self.sequence_len, 
                                   dtype="long", 
                                   truncating="post", # Should be truncating 
                                   padding="post")  , # If padding needed: at the end
        return padded_ids
        
    #------------------------------------
    # read_files 
    #-------------------
    
    def read_files(self, train_files):
        df = pd.DataFrame()
        if not type(train_files) == list:
            train_files = [train_files]
        for train_file in train_files:
            df_tmp = pd.read_csv(train_file, 
                                 delimiter=',', 
                                 header=0,      # Col info in row 0
                                 quotechar='"',
                                 engine='python')
            df = pd.concat([df,df_tmp])

        return df
        
if __name__ == '__main__':
    in_csv_dir = "/Users/paepcke/EclipseWorkspacesNew/colab_server/src/jupyter/"
    train_files = [os.path.join(in_csv_dir, 'left_ads_final.csv'),
                   os.path.join(in_csv_dir, 'right_ads_final.csv'),
                   os.path.join(in_csv_dir, 'neutral_ads.csv'),
                   os.path.join(in_csv_dir, 'combined-train.csv')]

    
    TextAugmenter()