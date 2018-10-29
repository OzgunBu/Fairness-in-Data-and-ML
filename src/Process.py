#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 15:29:21 2018

@author: bursaliogluozgun
"""
import src.my_functions_product1 as myFC

import numpy as np


def Main_Product(input_option,feature_path, data_filename,user_main_json_text_file,user_adv_json_text_file,h5_filename,lambda_file,p_threshold):
    train_or_untrained = 'trained'

    user_adv_json_text_file = user_main_json_text_file
    ## Input your architecture and trained model
    user_main_json_text_file,user_adv_json_text_file,h5_filename,result_fname,train_or_untrain = myFC.user_model_arch_feature_input(input_option,
                                                                                                                                feature_path,
                                                                                                                                data_filename,
                                                                                                                                user_main_json_text_file,
                                                                                                                                user_adv_json_text_file,
                                                                                                                                train_or_untrained,
                                                                                                                                h5_filename)
    
    
       ## Feature Reading
    X_train, Z_train, y_train = myFC.feature_file_reading(feature_path,'train')
    X_test, Z_test, y_test = myFC.feature_file_reading(feature_path,'test')
    
    print(user_main_json_text_file)
    ## Reading the json files for main task and adversary
    main_task_arch_json_string = myFC.read_txt_file_to_string(user_main_json_text_file)
    
   # if you want to use a different advesarial architect than the one inspired from the main. You can change this line
    adv_task_arch_json_string =  main_task_arch_json_string #myFC.read_txt_file_to_string(user_adv_json_text_file)
    adv_task_arch_json_string = adv_task_arch_json_string.replace('"batch_input_shape": [null, {}]'.format(X_train.shape[1]),'"batch_input_shape": [null, {}]'.format(1))


    #print(X_train.shape[1])
    

 

    ## Model and compile only main task: check prediction results
    save_the_weights = (train_or_untrain == 'untrained')
    main_task_accuracy, p_rule_for_Y1, y_pred = myFC.pre_train_main_task(main_task_arch_json_string,X_train, y_train,X_test,y_test,Z_test,save_the_weights=save_the_weights,h5_file_name=h5_filename)

    ## Combining the main task arch with the adversarial arch
    ### Train using pre-trained weights of main model

    tradeoff_lambda_v = np.loadtxt(feature_path+lambda_file, delimiter=',')

    pre_load_flag = True

    main_task_trained_weight_file = h5_filename#'main_task_ori_trained_model.h5'

    for tradeoff_lambda in tradeoff_lambda_v:
        print('tradeoff_lambda = ', tradeoff_lambda)

        Bacc, Bp,B_y_pred, Aacc, Ap,A_y_pred,tradeoff_lambda = myFC.run_it_for_one_lambda(tradeoff_lambda,main_task_arch_json_string,adv_task_arch_json_string,pre_load_flag,main_task_trained_weight_file,X_train, y_train,Z_train,X_test,y_test,Z_test)
        result_fname_y_pred_before_after,result_fname_acc_p_before_after = myFC.saving_performance_result(Bacc, Bp,B_y_pred, Aacc, Ap,A_y_pred,tradeoff_lambda,result_fname)


    Bacc_Bp_Aacc_Ap_results = np.zeros([len(tradeoff_lambda_v),4])

    for item in range(len(tradeoff_lambda_v)):
    
        tradeoff_lambda = tradeoff_lambda_v[item]
        result_fname_acc_p_before_after = result_fname +'Result_acc_p_BA_' + 'L'+ str(tradeoff_lambda)+'.txt'
        Bacc_Bp_Aacc_Ap = np.loadtxt(result_fname_acc_p_before_after, delimiter=',')
        Bacc_Bp_Aacc_Ap_results[item, :] = Bacc_Bp_Aacc_Ap

    #Finding if any of the lambda values satisfy the p threshold



    Aacc = Bacc_Bp_Aacc_Ap_results[:,2]
    Ap = Bacc_Bp_Aacc_Ap_results[:,3]

    lambda_to_save = -10
    Aacc_ok = Aacc[Ap>p_threshold]
    if Aacc_ok.shape[0]== 0:
        print('Increase the range of tuning parameter in the file {}'.format(lambda_file))   
    
    else:
        tradeoff_lambda_ok = tradeoff_lambda_v[Ap>p_threshold]
        tradeoff_lambda_ok = tradeoff_lambda_ok[np.argmax(Aacc_ok)]
        which_result = np.where(tradeoff_lambda_v == tradeoff_lambda_ok )
        result_row = Bacc_Bp_Aacc_Ap_results[which_result, :]
        result_row = result_row.reshape(4,)
        print('Great, we have an updated classifier satisfying the fairness. ')
        print('Accuracy is reduced from {:.2f}, to {:.2f}:'.format(result_row[0],result_row[2]))
        print('Fairness (p-score) is improved from {}, to {}'.format(result_row[1],result_row[3]))
        lambda_to_save = tradeoff_lambda_ok
    
    #TODO : Package the classifier network---------------------

    #Bacc, Bp,B_y_pred, Aacc, Ap,A_y_pred,tradeoff_lambda = myFC.run_it_for_one_lambda(lambda_to_save,main_task_arch_json_string,adv_task_arch_json_string,pre_load_flag,main_task_trained_weight_file,X_train, y_train,Z_train,X_test,y_test,Z_test)
    #result_fname_y_pred_before_after,result_fname_acc_p_before_after = myFC.saving_performance_result(Bacc, Bp,B_y_pred, Aacc, Ap,A_y_pred,tradeoff_lambda,result_fname)
#-----------------------------------------------------------
    return Bacc_Bp_Aacc_Ap_results,lambda_to_save,result_fname,Z_test, y_test


