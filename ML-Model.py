import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score,f1_score,precision_score, roc_auc_score,roc_curve
from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import seaborn as sns
import autokeras as ak
import seaborn as sb
from keras.layers import LeakyReLU

from Common_Function import alignment_Pulses,read_file,Qtail_Qtots,save_data,sampling_reduce,equivalentEnergy
from Data_preparation import separate_signals,idx_Noise,idx_pileup,ListAppend,CreatePathAndSave

def prepare_training_validaton_data(path_neutron,path_gamma,baseline,PulseDuration,MaxStartPoint=False,TriggerPercentage=0.1)
    gamma = read_file(path_gamma)
    neutron = read_file(path_neutron)
    X = np.concatenate((neutron,gamma))
    X = alignment_Pulses(X,baseline,PulseDuration,MaxStartPoint,TriggerPercentage)
    #X = sampling_reduce(X,f_out,5)
    y = np.zeros(X.shape[0])
    y[0:neutron.shape[0]] = 1
    width = X.shape[1]
    X = X[:,0:width]
    X_train,X_val,y_train,y_val = train_test_split(X,y, test_size=0.2, shuffle=True,random_state=42,stratify=y)
    return X_train,X_val,y_train,y_val

def network(X_train,y_train,X_test,y_test,path,train=False):
    weightNeutron = sum(y_train)/(len(y_train) - sum(y_train))
    class_weight= {0:weightNeutron, 1:1-weightNeutron}
    model = Sequential()
    model.add(Dense(32, input_dim=X_train.shape[1],activation = 'relu'))
    model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
    #model.add(Dense(128, activation='relu'))
#     model.add(Dense(128, activation='relu'))
#     model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    #print(model.summary())
    opt = keras.optimizers.Adam(learning_rate=0.001)
    if(train==False):
        model.compile(optimizer=opt, loss='binary_crossentropy',metrics = ['accuracy'])


        callbacks = [EarlyStopping(monitor='val_loss', patience=20),
                 ModelCheckpoint(filepath=path, monitor='val_loss', save_best_only=True)]

        history=model.fit(X_train, y_train,epochs=300,callbacks=callbacks, batch_size=16,verbose=0,validation_data=(X_test,y_test),class_weight=class_weight)
        model.load_weights(path)
    else:
        model.load_weights(path)
    return model

def auto_network(X_train,y_train,X_val,y_val,nb_models=2): #,class_weight={0:1, 1:100}):
    weightNeutron = sum(y_train)/(len(y_train) - sum(y_train))
    class_weight= {0:weightNeutron, 1:1-weightNeutron}
    input_node = ak.StructuredDataInput()
    output_node = ak.DenseBlock()(input_node)
    output_node = ak.ClassificationHead( num_classes=2)(output_node)
    callbacks = [EarlyStopping(monitor='val_loss', patience=20),
             ModelCheckpoint(filepath='best_model_ANN.h5', monitor='val_accuracy', save_best_only=True)]
    clf = ak.AutoModel(inputs=input_node, outputs=output_node, overwrite=True, objective="val_loss",seed = 42,max_trials=nb_models)
    clf.fit(X_train, y_train,validation_data=(X_val, y_val),callbacks=callbacks,class_weight=class_weight)
    model = clf.export_model()
    return model

def loss_acc_plot(history):
    fig1, ax_acc = plt.subplots()
    temp = history.history['accuracy']
    plt.plot(temp[1:])
    temp = history.history['val_accuracy']
    plt.plot(temp[1:])
    #plt.ylim([0.9995,1])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model - Accuracy')
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.show()
    
    fig2, ax_loss = plt.subplots()
    temp = history.history['loss']
    plt.plot(temp[1:])
    temp = history.history['val_loss']
    plt.plot(temp[1:])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model- Loss')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.show()
    target_names=['neutron','gamma']
    return
	
def evaluate_model(X_test,y_test,model,threshold=0.5):
    prediction = model.predict(X_test)
    y_pred = prediction>threshold
    cnf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print('confusion matrix')
    print(cnf_matrix)
    print('report of classification')
    print(report)
    return cnf_matrix,report

def evaluation_metrics(X,y,model,threshold=0.5):
    y_pred = model.predict(X)
    y_pred = y_pred> threshold
    if (sum(y)==0):
        return {'accuracy':0, 'TPR': 0,'FPR':sum(y_pred)/len(y_pred)}
    acc = accuracy_score(y,y_pred)
    cm = confusion_matrix(y, y_pred)
    if(cm.shape[0]==2):
        TPR = cm[1,1]/(cm[1,1]+cm[1,0])
        FPR = cm[0,1]/(cm[0,1]+cm[0,0])
    return {'accuracy':acc, 'TPR': TPR,'FPR':FPR}

def visualization_results(X,y,model):
    y_test_pred =model.predict(X)
    y_test_pred = y_test_pred>0.5
    false_gamma = []
    false_neutron = []
    true_neutron = []
    true_gamma = []
    for i in range(0,len(y)):
        if(y[i]==1) & (y_test_pred[i]==0):
            false_gamma.append(X[i,:])
        if(y[i]==1) & (y_test_pred[i]==1):
            true_neutron.append(X[i,:])
        if(y[i]==0) & (y_test_pred[i]==1):
            false_neutron.append(X[i,:])
        if(y[i]==0) & (y_test_pred[i]==0):
            true_gamma.append(X[i,:])

    true_neutron = np.asarray(true_neutron)
    false_gamma = np.asarray(false_gamma)
    true_gamma = np.asarray(true_gamma)
    false_neutron = np.asarray(false_neutron)
    #
    if(len(np.argwhere(y==1))==0):
            plt.plot(np.mean(true_gamma,axis=0))
            plt.plot(np.mean(false_neutron,axis=0))
            plt.legend(['true gamma','false neutron'])
            plt.title('true gamma and false neutron')
            plt.figure()
            plt.show()
    
    if(len(np.argwhere(y==0))==0):
            plt.plot(np.mean(true_neutron,axis=0))
            plt.plot(np.mean(false_gamma,axis=0))
            plt.legend(['true neutron','false gamma'])
            plt.title('true neutron and false gamma')
            plt.figure()
            plt.show()
    if(len(np.argwhere(y==1))!=0) and (len(np.argwhere(y==0))!=0):   
        plt.plot(np.mean(true_neutron,axis=0))
        plt.plot(np.mean(true_gamma,axis=0))
        plt.legend(['true neutron','true gamma'])
        plt.title('true gamma and true neutron')
        plt.figure()
        plt.show()
        #
        plt.plot(np.mean(true_gamma,axis=0))
        plt.plot(np.mean(false_gamma,axis=0))
        plt.legend(['true gamma','false gamma'])
        plt.title('true and false gamma')
        plt.figure()
        plt.show()
        
        #
        plt.plot(np.mean(true_neutron,axis=0))
        plt.plot(np.mean(false_neutron,axis=0))
        plt.legend(['true neutron','false neutron'])
        plt.title('true and false neutron')
        plt.figure()
        plt.show()
        #
        plt.plot(np.mean(true_gamma,axis=0))
        plt.plot(np.mean(false_neutron,axis=0))
        plt.legend(['true gamma','false neutron'])
        plt.title('true gamma and false neutron')
        plt.figure()
        plt.show()
        #
        plt.plot(np.mean(true_neutron,axis=0))
        plt.plot(np.mean(false_gamma,axis=0))
        plt.legend(['true neutron','false gamma'])
        plt.title('true neutron and false gamma')
        plt.figure()
        plt.show()
    return

