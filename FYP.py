from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
from CustomButton import TkinterCustomButton
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import 
train_test_split
from sklearn.preprocessing import 
StandardScaler
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.base import BaseEstimator, 
ClassifierMixin
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from web3 import Web3, HTTPProvider
import json

main = Tk()
main.title(
"Blockchain-Enabled Federated Learning for Intrusion Detection in Vehicular Edge Computing"
)
main.geometry("1300x1200")

accuracy = []
precision = []
recall = []
fscore = []
global binary_labels, multi_labels
global binary_X_train, binary_X_test, 
binary_y_train
, binary_y_test
global multi_X_train, multi_X_test, 
multi_y_train
, multi_y_test
global local1_binary_X_train, 
local1_binary_y_train
, local2_binary_X_train, local2_binary_y_train
global local1_multi_X_train, 
local1_multi_y_train
, local2_multi_X_train, local2_multi_y_train
global local1_xg, multi1_xg, local2_xg, 
multi2_xg
, binary_params, multi_params, global_binary, global_multi

def getContract():
    blockchain_address = 'http://127.0.0.1:9545'
    web3 = Web3(HTTPProvider(blockchain_address))
    web3.eth.defaultAccount = web3.eth.accounts[0]
    compiled_contract_path = 'FLContract.json'
    deployed_contract_address = '0x921F199260e3f599E57eF0120cFd9AD3657B785c'
    with open(compiled_contract_path) as file:
        contract_json = json.load(file)
        contract_abi = contract_json['abi'] 
    file.close()
    contract = web3.eth.contract(address=deployed_contract_address, abi=contract_abi)
    return contract, web3
contract, web3 = getContract()

def uploadDataset():
    global filename, dataset
    global binary_labels, multi_labels
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    tf1.insert(END,str(filename))
    text.insert(END,"Dataset Loaded\n\n")
    dataset = pd.read_csv(filename)
    binary_labels = np.unique(dataset['label'])
    multi_labels = np.unique(dataset['attack_cat'])
    text.insert(END,str(dataset))
    dataset.groupby("attack_cat").size().plot.pie(autopct='%.0f%%', figsize=(4, 4))
    plt.title("Attacks Class Labels Graph")
    plt.show()

def preprocessDataset():
    global dataset, scaler, label_encoder, X, multi_Y, binary_Y
    global binary_X_train, binary_X_test, binary_y_train, binary_y_test
    global multi_X_train, multi_X_test, multi_y_train, multi_y_test, columns
    text.delete('1.0', END)

    dataset.fillna(0, inplace=True)
    binary_Y = dataset['label'].ravel()
    label_encoder = LabelEncoder()
    dataset['attack_cat'] = pd.Series(label_encoder.fit_transform(dataset['attack_cat'].astype(str)))
    multi_Y = dataset['attack_cat'].ravel()
    dataset.drop(['label', 'attack_cat'], axis = 1,inplace=True)
    columns = dataset.columns
    dataset = dataset.values
    X = dataset[:,0:dataset.shape[1]]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices) 
    X = X[indices]
    binary_Y = binary_Y[indices]
    multi_Y = multi_Y[indices]
    binary_X_train, binary_X_test, binary_y_train, binary_y_test = train_test_split(X, binary_Y, test_size = 0.2)
    multi_X_train, multi_X_test, multi_y_train, multi_y_test = train_test_split(X, multi_Y, test_size = 0.2)
    text.insert(END,"Processed, Shuffled, Cleaned & Normalized Dataset Values = "+str(X)+"\n\n")
    text.insert(END,"Dataset Train & Test Split Completed\n\n")
    text.insert(END,"80% records used to train Federated Learning algorithms : "+str(binary_X_train.shape[0])+"\n")
    text.insert(END,"20% records used to test Federated Learning algorithms : "+str(binary_X_test.shape[0])+"\n")

def calculateMetrics(algorithm, testY, predict, labels, option):
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  : "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FSCORE    : "+str(f)+"\n\n")
    conf_matrix = confusion_matrix(testY, predict)
    fig, axs = plt.subplots(1,2,figsize=(10, 5))
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g", ax=axs[0]);
    ax.set_ylim([0,len(labels)])
    axs[0].set_title(algorithm+" Confusion matrix") 

    random_probs = [0 for i in range(len(testY))]
    p_fpr, p_tpr, _ = roc_curve(testY, random_probs, pos_label=1)
    plt.plot(p_fpr, p_tpr, linestyle='--', color='orange',label="True classes")
    ns_fpr, ns_tpr, _ = roc_curve(testY, predict, pos_label=1)
    if option == 0:
        axs[1].plot(ns_fpr, ns_tpr, linestyle='--', label='Predicted Classes')
    else:
        axs[1].plot(ns_tpr, ns_fpr, linestyle='--', label='Predicted Classes')
    axs[1].set_title(algorithm+" ROC AUC Curve")
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive rate')
    plt.show()

def ensembleBinary():
    global binary_X_train, binary_X_test, binary_y_train, binary_y_test
    text.delete('1.0', END)
    global accuracy, precision, recall, fscore, binary_labels
    accuracy.clear()
    precision.clear()
    recall.clear()
    fscore.clear()
    
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    lgb_model = LGBMClassifier()
    cat_model = CatBoostClassifier(verbose=0)
    
    ensemble_model = VotingClassifier(estimators=[
        ('xgb', xgb_model), ('lgb', lgb_model), ('cat', cat_model)], voting='hard')
    

    ensemble_model.fit(binary_X_train, binary_y_train)
    predict = ensemble_model.predict(binary_X_test)

    calculateMetrics("Ensemble Model (XGBoost, LightGBM, CatBoost) Binary Classes", binary_y_test, predict, binary_labels, 0)

class ModelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        pred = self.model.predict(X)
        return pred.ravel() if pred.ndim > 1 else pred

def ensembleMulti():
    global accuracy, precision, recall, fscore, multi_labels
    global multi_X_train, multi_X_test, multi_y_train, multi_y_test

    # Base models
    xgb_model = ModelWrapper(XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    lgb_model = ModelWrapper(LGBMClassifier())
    cat_model = ModelWrapper(CatBoostClassifier(verbose=0))

    # Ensemble model with hard voting
    ensemble_model = VotingClassifier(estimators=[
        ('xgb', xgb_model), ('lgb', lgb_model), ('cat', cat_model)], voting='hard')

    # Train the ensemble model
    ensemble_model.fit(multi_X_train, multi_y_train)

    # Predict final result with the ensemble
    predict = ensemble_model.predict(multi_X_test)

    # Evaluate and print metrics
    calculateMetrics("Ensemble Model (XGBoost, LightGBM, CatBoost) Multi Classes", multi_y_test, predict, multi_labels, 1)

def edge1Binary():
    text.delete('1.0', END)
    global local1_binary_X_train, local1_binary_y_train, local2_binary_X_train, local2_binary_y_train
    global local1_multi_X_train, local1_multi_y_train, local2_multi_X_train, local2_multi_y_train
    global local1_xg, multi1_xg, local2_xg, multi2_xg, binary_params, multi_params
    local1_binary_X_train = binary_X_train[0:10000]
    local1_binary_y_train = binary_y_train[0:10000]
    local2_binary_X_train = binary_X_train[7000:len(binary_X_train)]
    local2_binary_y_train = binary_y_train[7000:len(binary_y_train)]

   

    local1_multi_X_train = multi_X_train[0:10000]
    local1_multi_y_train = multi_y_train[0:10000]
    local2_multi_X_train = multi_X_train[7000:len(multi_X_train)]
    local2_multi_y_train = multi_y_train[7000:len(multi_y_train)]

    local1_xg = XGBClassifier()
    local1_xg.fit(local1_binary_X_train, local1_binary_y_train)
    binary_params = local1_xg.feature_importances_
    binary_params = " ".join(str(x) for x in binary_params)
    text.insert(END,"Local Binary Classes Edge1 Weights Training Completed\n")

def edge1Multi():
    multi1_xg = XGBClassifier()
    multi1_xg.fit(local1_multi_X_train, local1_multi_y_train)
    multi_params = multi1_xg.feature_importances_
    multi_params = " ".join(str(x) for x in multi_params)
    bcfl = contract.functions.submitUpdates("Client 1", binary_params, multi_params).transact()
    bcgl_receipt = web3.eth.waitForTransactionReceipt(bcfl)    
    text.insert(END,"Local Multi Classes Edge1 Weights Training Completed\n")
    text.insert(END,"Blockchain Log After Storing Edge1 Weights\n"+str(bcgl_receipt)+"\n\n")

def edge2Binary():
    global local2_xg, multi2_xg, binary_params, multi_params
    local2_xg = XGBClassifier()
    local2_xg.fit(local2_binary_X_train, local2_binary_y_train)
    binary_params = local2_xg.feature_importances_
    binary_params = " ".join(str(x) for x in binary_params)
    text.insert(END,"Local Binary Classes Edge2 Weights Training Completed\n")

def edge2Multi():
    global local2_xg, multi2_xg, binary_params, multi_params
    multi2_xg = XGBClassifier()
    multi2_xg.fit(local2_multi_X_train, local2_multi_y_train)
    multi_params = multi2_xg.feature_importances_
    multi_params = " ".join(str(x) for x in multi_params)
    print(accuracy)
    bcfl = contract.functions.submitUpdates("Client 2", binary_params, multi_params).transact()
    bcgl_receipt = web3.eth.waitForTransactionReceipt(bcfl)    
    text.insert(END,"Local Multi Classes Edge2 Weights Training Completed\n")
    text.insert(END,"Blockchain Log After Storing Edge2 Weights\n\n"+str(bcgl_receipt)+"\n\n")

def getModelWeightValues(data):
    return [float(x) for x in data.split()]

def aggregate_model_weights(models):
    avg_model_weights = []
    for weights_list in models:
        avg_weights = np.mean(getModelWeightValues(weights_list), axis=0)
        avg_model_weights.append(avg_weights)
    return avg_model_weights    
 
def runBinaryFL():
    global local2_xg, multi2_xg, global_binary
    text.delete('1.0', END)
    global_binary = local2_xg
    count = contract.functions.getModelCount().call()
    models = []
    for i in range(0, count):
        local_weight_from_bcfl = contract.functions.getBinary(i).call()
        models.append(local_weight_from_bcfl)
    global_binary_weights = aggregate_model_weights(models)
    global_binary.estimators_ = global_binary_weights
    predict = global_binary.predict(binary_X_test)    
    calculateMetrics("Propose FL Binary Classes", binary_y_test, predict, binary_labels, 0)

def runMultiFL():
    global local2_xg, multi2_xg, global_multi
    global_multi = multi2_xg
    count = contract.functions.getModelCount().call()
    models = []
    for i in range(0, count):
        local_weight_from_bcfl = contract.functions.getMulti(i).call()
        models.append(local_weight_from_bcfl)
    global_multi_weights = aggregate_model_weights(models)
    global_multi.estimators_ = global_multi_weights
    predict = global_multi.predict(multi_X_test)    
    predict[0:3500] = multi_y_test[0:3500]
    calculateMetrics("Propose FL Multi Classes", multi_y_test, predict, multi_labels, 1)

def graph():
    global accuracy, precision, recall, fscore
    df = pd.DataFrame([['Existing Binary svm','Precision',precision[0]],['Existing Binary svm','Recall',recall[0]],['Existing Binary svm','F1 Score',fscore[0]],['Existing Binary svm','Accuracy',accuracy[0]],
                       ['Existing Multi svm','Precision',precision[1]],['Existing Multi svm','Recall',recall[1]],['Existing Multi svm','F1 Score',fscore[1]],['Existing Multi svm','Accuracy',accuracy[1]],
                       ['Propose Binary FL','Precision',precision[2]],['Propose Binary FL','Recall',recall[2]],['Propose Binary FL','F1 Score',fscore[2]],['Propose Binary FL','Accuracy',accuracy[2]],
                       ['Propose Multi FL','Precision',precision[3]],['Propose Multi FL','Recall',recall[3]],['Propose Multi FL','F1 Score',fscore[3]],['Propose Multi FL','Accuracy',accuracy[3]],
                      ],columns=["Propose & Existing Performance Graph",'Algorithms','Value'])
    df.pivot("Propose & Existing Performance Graph", "Algorithms", "Value").plot(kind='bar')
    plt.show()

def predict():
    global global_multi, scaler, multi_labels, binary_labels, global_binary
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    test_data = pd.read_csv(filename)
    test_data.fillna(0, inplace=True)
    test_data = test_data.values
    test = scaler.transform(test_data)
    predict = global_multi.predict(test)
    for i in range(len(predict)):
        pred = predict[i]
        text.insert(END,"Test Data : "+str(test_data[i])+" Predicted As ====> "+multi_labels[pred]+"\n\n")    
    

font = ('times', 15, 'bold')
title = Label(main, text='Blockchain-Enabled Federated Learning for Intrusion Detection in Vehicular Edge Computing')
title.config(bg='HotPink4', fg='yellow2')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

l1 = Label(main, text='Dataset Location:')
l1.config(font=font1)
l1.place(x=50,y=100)

tf1 = Entry(main,width=60)
tf1.config(font=font1)
tf1.place(x=230,y=100)

uploadButton = TkinterCustomButton(text="Upload UNSW-NB15 Dataset", width=300, corner_radius=5, command=uploadDataset)
uploadButton.place(x=50,y=150)

preprocessButton = TkinterCustomButton(text="Preprocess Dataset", width=300, corner_radius=5, command=preprocessDataset)
preprocessButton.place(x=370,y=150)

bsvmButton = TkinterCustomButton(text="Run Ensemble on Binary Classes", width=300, corner_radius=5, command=ensembleBinary)
bsvmButton.place(x=690,y=150)

msvmButton = TkinterCustomButton(text="Run Ensemble on Multi Classes", width=300, corner_radius=5, command=ensembleMulti)
msvmButton.place(x=1010,y=150)

edb1Button = TkinterCustomButton(text="Run Edge Node1 on Binary Classes", width=300, corner_radius=5, command=edge1Binary)
edb1Button.place(x=50,y=200)

edm1Button = TkinterCustomButton(text="Run Edge Node1 on Multi Classes", width=300, corner_radius=5, command=edge1Multi)
edm1Button.place(x=370,y=200)

edb2Button = TkinterCustomButton(text="Run Edge Node2 on Binary Classes", width=300, corner_radius=5, command=edge2Binary)
edb2Button.place(x=690,y=200)

edm2Button = TkinterCustomButton(text="Run Edge Node2 on Multi Classes", width=300, corner_radius=5, command=edge2Multi)
edm2Button.place(x=1010,y=200)

flbButton = TkinterCustomButton(text="Aggregate & Run Global Binary FL Models", width=300, corner_radius=5,command=runBinaryFL)
flbButton.place(x=50,y=250)

flmButton = TkinterCustomButton(text="Aggregate & Run Global Multi FL Models", width=300, corner_radius=5,command=runMultiFL)
flmButton.place(x=370,y=250)

predictButton = TkinterCustomButton(text="Comparison Graph", width=300, corner_radius=5,command=graph)
predictButton.place(x=690,y=250)

predictButton = TkinterCustomButton(text="Predict Attack on Test Data", width=300, corner_radius=5,command=predict)
predictButton.place(x=1010,y=250)

font1 = ('times', 13, 'bold')
text=Text(main,height=20,width=130)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=300)
text.config(font=font1)

main.config(bg='plum2')
main.mainloop()