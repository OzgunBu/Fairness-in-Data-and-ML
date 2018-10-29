#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 12:48:14 2018

@author: bursaliogluozgun
"""
import pandas as pd
import numpy as np


import keras as ke
import keras.backend as K
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.models import load_model
# model reconstruction from JSON:
from keras.models import model_from_json
import matplotlib.pyplot as plt

import seaborn as sns

import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
#==============================================================================
def FNR_FPR(y_pred,y_test):
    
    FN = ((y_test == 1)&(y_pred <=0.5)).sum()
    TP = ((y_test == 1)&(y_pred >0.5)).sum()
    TN = ((y_test == 0)&(y_pred <=0.5)).sum()
    FP = ((y_test == 0)&(y_pred >0.5)).sum()

    FNR = FN/(FN+TP)
    FPR = FP/(FP+TN)
    return FNR, FPR 
#===============================================================================
    # do this results only if a good lambda value found
def FNR_FPR_analysis(BA_y_pred,lambda_to_save, Z_test, y_test):
    Z_test = np.array(Z_test)
    y_test = np.array(y_test)

    #let's focus on sensitive class 0
    samples = Z_test==0
    y_test_samples = np.array(y_test[samples])
    y_pred = np.array(BA_y_pred[:,0])
    y_pred_samples = y_pred[samples]
    FNR, FPR = FNR_FPR(y_pred_samples,y_test_samples)
    print('Sensitive Class 0 w/o fairness FNR= {:.2f}, FPR= {:.2f}'.format(FNR,FPR))
    preFNR0 = FNR
    preFPR0 = FPR

    #let's focus on sensitive class 1
    samples = Z_test==1
    y_test_samples = np.array(y_test[samples])
    y_pred = np.array(BA_y_pred[:,0])
    y_pred_samples = y_pred[samples]
    FNR, FPR = FNR_FPR(y_pred_samples,y_test_samples)
    print('Sensitive Class 1 w/o fairness FNR= {:.2f}, FPR= {:.2f}'.format(FNR,FPR))
    preFNR1 = FNR
    preFPR1 = FPR


    #let's focus on sensitive class 0
    samples = Z_test==0
    y_test_samples = np.array(y_test[samples])
    y_pred = np.array(BA_y_pred[:,1])
    y_pred_samples = y_pred[samples]
    FNR, FPR = FNR_FPR(y_pred_samples,y_test_samples)
    print('Sensitive Class 0 w/ fairness FNR= {:.2f}, FPR= {:.2f}'.format(FNR,FPR))
    afterFNR0 = FNR
    afterFPR0 = FPR

    #let's focus on sensitive class 1
    samples = Z_test==1
    y_test_samples = np.array(y_test[samples])
    y_pred = np.array(BA_y_pred[:,1])
    y_pred_samples = y_pred[samples]
    FNR, FPR = FNR_FPR(y_pred_samples,y_test_samples)
    print('Sensitive Class 1 w/ fairness FNR= {:.2f}, FPR= {:.2f}'.format(FNR,FPR))
    afterFNR1 = FNR
    afterFPR1 = FPR

    print('-'*55+'\n')

    print('FNR for class 1 is reduced from {:.2f} to {:.2f}'.format(preFNR1,afterFNR1))

    print('FPR for class 0 is reduced from {:.2f} to {:.2f}'.format(preFPR0,afterFPR0))    
#===============================================================================
def saving_trade_off_figures(feature_path, Bacc_Bp_Aacc_Ap_results):

    #saving figures for accuracy - fairness trade off
    # figure 1
    fname = feature_path+'Figure-results/'+'Trade'
    plt.plot(Bacc_Bp_Aacc_Ap_results[:,3], 100*Bacc_Bp_Aacc_Ap_results[:,2])
    plt.xlabel('Fairness Score')
    plt.ylabel('Accuracy')
    plt.title('Accuracy & Fairness Trade-off')
    plt.xlim(0,100)
    #plt.ylim(0,100)
    plt.savefig(fname, bbox_inches='tight')

    #figure2
    fname = feature_path+'Figure-results/'+'Trade-off-zoomout'
    plt.plot(Bacc_Bp_Aacc_Ap_results[:,3], 100*Bacc_Bp_Aacc_Ap_results[:,2])
    plt.xlabel('Fairness Score')
    plt.ylabel('Accuracy')
    plt.title('Accuracy & Fairness Trade-off')
    plt.xlim(0,100)
    plt.ylim(0,100)
    plt.savefig(fname, bbox_inches='tight')    
    
    

#===============================================================================    
    
def Prediction_distribution_analysis(BA_y_pred,feature_path,Z_test):
    sensitive_class0 = 'Class 0 '
    sensitive_class1 = 'Class 1'
    good_outcome = 'y = 1'

    target_label0 = '0'
    target_label1 = '1'

    fname = feature_path+'Figure-results/'+'distribution'+'.png'



    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)


    # without adversarial training
    ax = sns.distplot(BA_y_pred[Z_test == 0,0], hist=False, 
                  kde_kws={'shade': True,},
                  label='{}'.format(sensitive_class0),ax=axes[0])

    ax = sns.distplot(BA_y_pred[Z_test == 1,0], hist=False, 
                  kde_kws={'shade': True,},
                  label='{}'.format(sensitive_class1),ax=axes[0])
    ax.set_xlim(0,1)
    ax.set_ylim(0,5)
    ax.set_yticks([])
    ax.set_title("Original Classifier")#.format(sensitive_field))
    ax.set_ylabel('Prediction Distribution')
    ax.set_xlabel('Prob( {} | Sensitive Class)'.format(good_outcome))

        # with adversarial training
    ax = sns.distplot(BA_y_pred[Z_test == 0,1], hist=False, 
                  kde_kws={'shade': True,},
                  label='{}'.format(sensitive_class0),ax=axes[1])

    ax = sns.distplot(BA_y_pred[Z_test == 1,1], hist=False, 
                  kde_kws={'shade': True,},
                  label='{}'.format(sensitive_class1),ax=axes[1])
    ax.set_xlim(0,1)
    ax.set_ylim(0,5)
    ax.set_yticks([])
    ax.set_title("Classifier w/ fairness layer")#.format(sensitive_field))
    ax.set_ylabel('Prediction Distribution')
    ax.set_xlabel('Prob( {} | Sensitive Class)'.format(good_outcome))


    plt.savefig(fname, bbox_inches='tight')
    

#==============================================================================

def comparing_decisions_of_classifiers(y_test,Z_test, BA_y_pred,feature_path):
    target_label0 = 'Ground Truth = 0'
    target_label1 = 'Ground Truth = 1'

    sensitive_class0 = 'Class 0'
    sensitive_class1 = 'Class 1'

    Z_test = np.array(Z_test)
    y_test = np.array(y_test)

    #subsample the samples
    samples = np.random.randint(Z_test.shape[0], size=np.min([1000,Z_test.shape[0]]))
    
    #predictions from the original classifier
    y_pred = np.array(BA_y_pred[:,0])
    sy_pred = y_pred[samples]
    sy_test = y_test[samples]
    sZ_test = Z_test[samples]

    #samples ordered according to their prediction values
    order_y_pred = sy_pred.argsort()

    #Let's order other vectors as well.
    Z_test_o = sZ_test[order_y_pred]
    y_test_o = sy_test[order_y_pred]
    y_pred_o = sy_pred[order_y_pred]
    vec = np.arange(sy_pred.shape[0])

    ones = y_pred_o>0.5
    border_one = np.where(y_pred_o>0.5)
    border_one = border_one[0][0]

    tar0_rows = (y_test_o == 0)
    tar1_rows = (y_test_o == 1)

    plt.figure(1+1)
    plt.scatter(vec[tar1_rows], y_pred_o[tar1_rows], s=10,c="r", alpha=0.5,marker='o',label=target_label0)
    plt.scatter(vec[tar0_rows], y_pred_o[tar0_rows], s=5,c="b", alpha=0.5,label=target_label1)
    plt.plot(vec,0.5*np.ones_like(vec),c='black')
    plt.plot(border_one*np.ones_like(np.array([0,1])),np.array([0,1]),c='black')
    plt.xlabel("ordered samples")
    plt.ylabel("Pred. Prob. w/ original classifier")
    plt.legend(loc=2)
    #plt.title('Predictions with original classifier')
    fname = feature_path+'Figure-results/'+'points_'+'.png'
    plt.savefig(fname, bbox_inches='tight')
    
    
    #---------------------
    
    #predictions from the original classifier
    y_pred = np.array(BA_y_pred[:,1])
    sy_pred = y_pred[samples]
    sy_test = y_test[samples]
    sZ_test = Z_test[samples]

    #samples ordered according to their prediction values
    #order_y_pred = sy_pred.argsort()

    #Let's order other vectors as well.
    Z_test_o = sZ_test[order_y_pred]
    y_test_o = sy_test[order_y_pred]
    y_pred_o = sy_pred[order_y_pred]
    vec = np.arange(sy_pred.shape[0])


    tar0_rows = (y_test_o == 0)
    tar1_rows = (y_test_o == 1)

    plt.figure(2+1)
    plt.scatter(vec[tar1_rows], y_pred_o[tar1_rows], s=10,c="r", alpha=0.5,marker='o',label=target_label1)
    plt.scatter(vec[tar0_rows], y_pred_o[tar0_rows], s=5,c="b", alpha=0.5,label=target_label0)
    plt.plot(vec,0.5*np.ones_like(vec),c='black')
    plt.plot(border_one*np.ones_like(np.array([0,1])),np.array([0,1]),c='black')
    plt.xlabel("ordered samples")
    plt.ylabel("Pred. Prob. w/ fair classifier")
    plt.legend(loc=2)
    #plt.title('Predictions with fair classifier')
    fname = feature_path+'Figure-results/'+'points2_'+'.png'
    plt.savefig(fname, bbox_inches='tight')
    
    
    #------------------------
    

    #predictions from the original classifier
    y_pred = np.array(BA_y_pred[:,1])
    sy_pred = y_pred[samples]
    sy_test = y_test[samples]
    sZ_test = Z_test[samples]

    #samples ordered according to their prediction values
    #order_y_pred = sy_pred.argsort()

    #Let's order other vectors as well.
    Z_test_o = sZ_test[order_y_pred]
    y_test_o = sy_test[order_y_pred]
    y_pred_o = sy_pred[order_y_pred]
    vec = np.arange(sy_pred.shape[0])


    tar0_rows = (y_test_o == 0)
    tar1_rows = (y_test_o == 1)

    #lower boundary
    tar0_y_pred1_rows_Z0 = (y_test_o[:border_one] == 0)&(y_pred_o[:border_one] > 0.5)& (Z_test_o[:border_one]==0)
    tar0_y_pred1_rows_Z1 = (y_test_o[:border_one] == 0)&(y_pred_o[:border_one] > 0.5)& (Z_test_o[:border_one]==1)

    #upper boundary
    tar1_y_pred0_rows_Z0u = (y_test_o[border_one:] == 1)&(y_pred_o[border_one:] < 0.5)& (Z_test_o[border_one:]==0)
    tar1_y_pred0_rows_Z1u = (y_test_o[border_one:] == 1)&(y_pred_o[border_one:] < 0.5)& (Z_test_o[border_one:]==1)

    plt.figure(3+1)
    plt.scatter(vec[tar1_rows], y_pred_o[tar1_rows], s=10,c="r", alpha=0.8,marker='o',label=target_label1)
    plt.scatter(vec[tar0_rows], y_pred_o[tar0_rows], s=5,c="b", alpha=0.8,label=target_label0)

    plt.scatter(vec[:border_one][tar0_y_pred1_rows_Z0], y_pred_o[:border_one][tar0_y_pred1_rows_Z0], s=50,c="g", alpha=0.5,marker='o',label='False Neg. from '+sensitive_class0)
    plt.scatter(vec[:border_one][tar0_y_pred1_rows_Z1], y_pred_o[:border_one][tar0_y_pred1_rows_Z1], s=50,c="y", alpha=0.5,marker='o',label='False Pos. from '+sensitive_class1)

    plt.scatter(vec[border_one:][tar1_y_pred0_rows_Z0u], y_pred_o[border_one:][tar1_y_pred0_rows_Z0u], s=50,c="g", alpha=0.5,marker='o')
    plt.scatter(vec[border_one:][tar1_y_pred0_rows_Z1u], y_pred_o[border_one:][tar1_y_pred0_rows_Z1u], s=50,c="y", alpha=0.5,marker='o')



    plt.plot(vec,0.5*np.ones_like(vec),c='black')
    plt.plot(border_one*np.ones_like(np.array([0,1])),np.array([0,1]),c='black')
    plt.xlabel("ordered samples")
    plt.ylabel("Pred. Prob. w/ fair classifier")
    plt.legend(loc=2)
    #plt.title('Predictions with fair classifier')
    fname = feature_path+'Figure-results/'+'points3_'+'.png'
    plt.savefig(fname, bbox_inches='tight')
    
    
    #-----------------
    #predictions from the original classifier
    y_pred = np.array(BA_y_pred[:,1])
    sy_pred = y_pred[samples]
    sy_test = y_test[samples]
    sZ_test = Z_test[samples]

    #samples ordered according to their prediction values
    #order_y_pred = sy_pred.argsort()

    #Let's order other vectors as well.
    Z_test_o = sZ_test[order_y_pred]
    y_test_o = sy_test[order_y_pred]
    y_pred_o = sy_pred[order_y_pred]
    vec = np.arange(sy_pred.shape[0])


    tar0_rows = (y_test_o == 0)
    tar1_rows = (y_test_o == 1)

    #upper boundary
    tar0_y_pred0_rows_Z0u = (y_test_o[border_one:] == 0)&(y_pred_o[border_one:] <= 0.5)& (Z_test_o[border_one:]==0)
    tar0_y_pred0_rows_Z1u = (y_test_o[border_one:] == 0)&(y_pred_o[border_one:] <= 0.5)& (Z_test_o[border_one:]==1)

    #lower boundary
    tar1_y_pred1_rows_Z0 = (y_test_o[:border_one] == 1)&(y_pred_o[:border_one] > 0.5)& (Z_test_o[:border_one]==0)
    tar1_y_pred1_rows_Z1 = (y_test_o[:border_one] == 1)&(y_pred_o[:border_one] > 0.5)& (Z_test_o[:border_one]==1)

    plt.figure(4+1)
    plt.scatter(vec[tar1_rows], y_pred_o[tar1_rows], s=10,c="r", alpha=0.8,marker='o',label=target_label1)
    plt.scatter(vec[tar0_rows], y_pred_o[tar0_rows], s=5,c="b", alpha=0.8,label=target_label0)

    plt.scatter(vec[border_one:][tar0_y_pred0_rows_Z0u], y_pred_o[border_one:][tar0_y_pred0_rows_Z0u], s=50,c="g", alpha=0.5,marker='o',label='True Neg. from '+sensitive_class0)
    plt.scatter(vec[border_one:][tar0_y_pred0_rows_Z1u], y_pred_o[border_one:][tar0_y_pred0_rows_Z1u], s=50,c="y", alpha=0.5,marker='o',label='True Pos. from '+sensitive_class1)

    plt.scatter(vec[:border_one][tar1_y_pred1_rows_Z0], y_pred_o[:border_one][tar1_y_pred1_rows_Z0], s=50,c="g", alpha=0.5,marker='o')
    plt.scatter(vec[:border_one][tar1_y_pred1_rows_Z1], y_pred_o[:border_one][tar1_y_pred1_rows_Z1], s=50,c="y", alpha=0.5,marker='o')



    plt.plot(vec,0.5*np.ones_like(vec),c='black')
    plt.plot(border_one*np.ones_like(np.array([0,1])),np.array([0,1]),c='black')
    plt.xlabel("ordered samples")
    plt.ylabel("Pred. Prob. w/ fair classifier")
    plt.legend(loc=2)
    #plt.title('Predictions with fair classifier')
    fname = feature_path+'Figure-results/'+'points4_'+'.png'
    plt.savefig(fname, bbox_inches='tight')