'''
Created on Jun 7, 2020

@author: paepcke
'''

import os
import re
import sys

from keras.preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert import BertTokenizer

import pandas as pd


class TextAugmenter(object):
    '''
    Minimal columns: 'text', 'label'
    '''

    # Min number of tokens in a sequence to
    # consider the row for augmentation:
    MIN_AUG_LEN = 80 
    
    DEFAULT_SEQUENCE_LEN = 128

    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, 
                 train_files, 
                 sequence_len=None,
                 outfile=None,
                 text_col='text',
                 label_col='label',
                 remove_txt_col=True,
                 testing=False):
        '''
        Constructor
        '''
        if outfile is None:
            outfile = os.path.join(os.path.dirname(__file__), 'tokenized_input.csv')
        if sequence_len is None:
            sequence_len = TextAugmenter.DEFAULT_SEQUENCE_LEN
        self.text_col  = text_col
        self.label_col = label_col
        self.tokens_col = 'tokens'
        
        self.sequence_len = sequence_len
        self.train_df = self.read_files(train_files)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        
        if testing:
            # Let unittests call the other methods.
            return
        chopped_tokenized = self.fit_to_sequence_len(self.train_df)
        if remove_txt_col:
            chopped_tokenized = chopped_tokenized.drop(self.text_col, axis=1)
            
        # Add BERT ids column:
        ids = self.padded_ids(chopped_tokenized[self.tokens_col].values)
        chopped_tokenized['ids'] = ids
        chopped_tokenized.to_csv(outfile,index=False)
        
    #------------------------------------
    # fit_to_sequence_len 
    #-------------------
    
    def fit_to_sequence_len(self, train_df):
        # Add a col to the passed-in df: 'tokens'.
        # Create list the height of train_df:
        token_col = ['']*len(train_df)
        train_df[self.tokens_col] = token_col
        
        new_rows = []
        nl_pat = re.compile(r'\n')
        for (_indx, row) in train_df.iterrows():
            # Remove \n chars;
            try:
                txt = nl_pat.sub(' ', row[self.text_col])
            except TypeError:
                # Some csv entries have empty text cols.
                # Those manifest as NaN vals in the df.
                # Just skip those rows:
                continue
            # And chop into a seq of strings:
            tokenized_txt = self.tokenizer.tokenize(txt)
            
            # Short enough to just keep?
            # The -2 allows for the [CLS] and [SEP] tokens:
            if len(tokenized_txt) <= self.sequence_len - 2:
                row[self.tokens_col] = ['[CLS]'] + tokenized_txt + ['[SEP]']
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
                new_row[self.tokens_col] = sent_fragment
                # Add to the train_df:
                new_rows.append(new_row)
        new_rows_df = pd.DataFrame(new_rows, columns=train_df.columns)
        return new_rows_df

    #------------------------------------
    # augment_text 
    #-------------------

    def augment_text(self, train_df):
        '''
        Given a df with at least a column
        called 'tokens': find rows with more
        than MIN_AUG_LEN tokens. Select sequences
        that contain whole sentences, i.e. punctuation
        {.|,|!|?}. Then create new rows with all
        cols the same, except for the tokens column.
        If there is a self.text_col, its content
        will be the assembled clear text from the 
        tokens. Though token oddities will be present. 
           
        @param train_df:
        @type train_df:
        @return: new df with additional rows.
        @rtype: DataFrame
        '''
        
        end_sentence_punctuation = '.!?'
        for (_indx, row) in train_df.iterrows():
            # Long enough token seq?
            if len(row[self.tokens_col]) < TextAugmenter.MIN_AUG_LEN:
                continue
            sentence_bounds = self.get_indexes(row, end_sentence_punctuation)
            
            

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
        # return a Pandes series that can be
        # directly added as a column to a df.
        # padded_ids is a 1-tuple. Inside is 
        # a 2D numpy array. Series apparently
        # want Python arrays; therefore the
        # 'list()':
        return pd.Series(list(padded_ids[0]))

    #------------------------------------
    # get_indexes 
    #-------------------

    def get_indexes(self, arr, search_str):
        '''
        Function that returns the indexes of occurrences
        of any members in a given list within another list:
        
          Given arr = ['Earth', 'Moon', 'Earth']
          
        get_indexes(arr, 'Earth')   ==> [0,2]
        get_indexes(arr, ['Earth']) ==> [0,2]
        get_indexes(arr, ['Earth', 'Moon']) ==> [0,1,2]
        
                arr = ['[CLS]', 'The', 'Sun', '!', '[SEP]']
        
        get_indexes(arr, '.!?') ==> [3]
        
        @param arr: list within which to search
        @type arr: (<any>)
        @param search_str: what to search for
        @type search_str: str
        '''

        return [i for i in range(len(arr)) if arr[i] == search_str]


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

# ---------------------- Main ---------------

if __name__ == '__main__':
    in_csv_dir = "/Users/paepcke/EclipseWorkspacesNew/colab_server/src/jupyter/"
    
    # TRAINING:
#     train_files = [os.path.join(in_csv_dir, 'left_ads_final.csv'),
#                    os.path.join(in_csv_dir, 'right_ads_final.csv'),
#                    os.path.join(in_csv_dir, 'neutral_ads.csv'),
#                    os.path.join(in_csv_dir, 'combined-train.csv')
#                    ]
#     outfile = os.path.join(in_csv_dir, 'leanings_right_sized.csv')

    # TEST
    train_files   = os.path.join(in_csv_dir, 'final_test.csv')
    outfile = os.path.join(in_csv_dir, 'leanings_right_sized_testset.csv')
    
    if os.path.exists(outfile):
        print(f"Outfile {os.path.basename(outfile)} exists; please delete it and try again")
        sys.exit(1)
    
    TextAugmenter(train_files, outfile=outfile, text_col='message')