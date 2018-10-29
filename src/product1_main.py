#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 15:07:21 2018

@author: bursaliogluozgun
"""


import my_functions_product1 as myFC
import sys
#import pandas as pd
#import numpy as np
#==================================================================================================================


def main():
    
    try:
        script = sys.argv[0]
        filename_str = sys.argv[1]

    
    except:
        print('Input arguments are missing.')
        print('Correct format:')
        print('data/adult.data')        
        
    else:

        #file operations: checking if the input files are there etc.
        myFC.Input_file_checking(filename_str)
        
        try:
            X_df, Ybin, Zbin = myFC.read_process_data_output_bias(filename_str)
 
        finally:
            print('Data procesing job finished')
            #closing the output file       
           # output_file.close() 
            #print('----------- Output file is closed ----------------')
#=================================================================================================================
            
#==================================================================================================================
if __name__== "__main__":
    main()