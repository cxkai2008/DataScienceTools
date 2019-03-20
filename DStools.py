import graphviz
import numpy as np
import pandas as pd
import random
from sklearn import tree
from sklearn.tree import export_graphviz
import re
from os import listdir
#####find distict items in two lists#####
def findDistinct(ind1,ind2):
    # print("distict item in ind1")
    disinidx1=[]
    for i1 in ind1:
        if i1 not in ind2:
            # print(i1)
            disinidx1.append(i1)
    # print("distict item in ind2")
    disinidx2=[]
    for i2 in ind2:
        if i2 not in ind1:
            # print(i2)
            disinidx2.append(i2)
    return (disinidx1,disinidx2)


def handle_unbalanced_dataset(df,numeric_features,label,id_column):
    max_count = df.groupby([label], as_index=False).count()[[label,id_column]].sort_values(by=id_column, ascending=False).iloc(0)[0][1]
    iter_list = df.groupby([label], as_index=False).count()[[label,id_column]].sort_values(by=id_column, ascending=False).values
    add_sample_size_dict={i[0]:max_count-i[1]  for i in iter_list}
    new_DF=df.copy()
    num=0
    for k,v in add_sample_size_dict.items():
        sample_size = df[df[label]==k].shape[0]
        sample_matrix = df[df[label]==k][numeric_features].values
        new_matrix=[]
        for i in range(v):
            two_samples_list = random.sample(range(sample_size),2)
            new_sample=(sample_matrix[two_samples_list[0]]+sample_matrix[two_samples_list[1]])/2
            new_matrix.append(new_sample)
        new_matrix = np.array(new_matrix)
        if(len(new_matrix)==0):
            continue
        temp_DF=pd.DataFrame(new_matrix,columns=numeric_features)
        temp_DF[id_column]=np.array(['fakeid'+str(j) for j in range(num,num+temp_DF.shape[0])])
        temp_DF[label]=k
        num=num+temp_DF.shape[0]
        new_DF = new_DF.append(temp_DF)
    new_DF.index = new_DF[id_column]
    return new_DF
    
    
def cross_validation_split_with_unbalance_data(df,numeric_features,label='Category',id_column='PlateID',test_size=0.2,handle_unbalance=True):
    iter_list = df.groupby([label], as_index=False).count()[[label,id_column]].sort_values(by=id_column, ascending=False).values
    select_size_dict={i[0]:int(test_size*i[1])  for i in iter_list}
    sample_size_dict={i[0]:i[1]  for i in iter_list}
    columns_list=df.columns
    train_matrix=[]
    test_matrix=[]
    train_index=[]
    test_index=[]
    for k,v in select_size_dict.items():
        sample_matrix = df[df[label]==k].values
        selected_list = random.sample(range(sample_size_dict[k]),v)
        unselected_list = findDistinct(selected_list,list(range(sample_size_dict[k])))[1]
        for idx in selected_list:
            test_matrix.append(sample_matrix[idx])
            test_index.append(df[df[label]==k].iloc[idx][id_column])
        for idx in unselected_list:
            train_matrix.append(sample_matrix[idx])
            train_index.append(df[df[label]==k].iloc[idx][id_column])
    train_DF=pd.DataFrame(train_matrix,columns=columns_list)
    test_DF=pd.DataFrame(test_matrix,columns=columns_list)
    train_DF.index=np.array(train_index)
    test_DF.index=np.array(test_index)
    if(handle_unbalance):
        train_DF=handle_unbalanced_dataset(train_DF,numeric_features,label,id_column)
    return train_DF[numeric_features],test_DF[numeric_features],train_DF[label],test_DF[label]


def DT_RF_models(dataSet,numeric_features,path,isDT = True,iteration=10,testSize =0.2,readList = ['Compound Name'], label = 'Category',DTdenotion='test',DT_maxdepth=2,numberOfTrees = 50,RF_maxdepth=6,isplot=False,id_column='id',handle_unbalance=True):
    if(isDT):
        model=tree.DecisionTreeClassifier(max_depth=DT_maxdepth)
    else:
        model=RandomForestClassifier(n_estimators=numberOfTrees,max_depth=RF_maxdepth)
    readableDF = dataSet.copy()
    X = readableDF[numeric_features]    
    Y = readableDF[label]
    readableDF[id_column]=readableDF[id_column].astype(str)
    accuracy = []
    fullWrongList=[]
    fullTest=np.array([])
    fullPredict=[]
    for j in range(0,iteration):
        X_train, X_test, Y_train, Y_test = cross_validation_split_with_unbalance_data(readableDF,numeric_features,label=label,id_column=id_column,test_size=testSize,handle_unbalance=handle_unbalance)
        model = model.fit(X_train,Y_train)
        pre_Y = model.predict(X_test)
        pre_Y_pro= model.predict_proba(X_test)
        Y_test = pd.DataFrame(Y_test)
        Y_test[id_column]=X_test.index
        # Y_test['index']=np.array([i for i in range(0,Y_test.shape[0])])
        # Only for RF
        if(not isDT):
            for i in range(0,numberOfTrees):
                single_tree = model.estimators_[i]
                export_graphviz(single_tree,out_file=path+str(j)+'---tree---'+str(i)+'.dot', feature_names = X.columns,rounded = True, precision = 1)
                if(isplot):
                    (graph, ) = pydot.graph_from_dot_file(path+str(j)+'---tree---'+str(i)+'.dot')
                    graph.write_png(path+str(j)+'---tree---'+str(i)+'.png')
        count=0
        for i in range(0,len(pre_Y)):
            fullTest=np.append(fullTest,Y_test.iloc[i][label])
            fullPredict.append(pre_Y_pro[i])
            if(pre_Y[i] != Y_test.iloc[i][label]):
                count = count+1
                string=''
                for l in range(0,len(readList)):
                    string = string + str(readableDF[readableDF[id_column]==Y_test.iloc[i][id_column]][readList[l]].values[0])+'---'
                best_preds = np.argmax(pre_Y_pro[i])
                singleWrongList = [pre_Y[i],string+Y_test.iloc[i][label],best_preds,pre_Y_pro[i],str(j)]
                fullWrongList.append(singleWrongList)    
        print('------------------accuracy = '+str(1-count/len(pre_Y))+'------------------')
        accuracy.append(1-count/len(pre_Y))
    #Only for DT, plot DT
    if(isDT & isplot):
        newData=handle_unbalanced_dataset(dataSet,numeric_features,label=label,id_column=id_column)
        model = model.fit(newData[numeric_features],newData[label])
        dot_data = tree.export_graphviz(model,out_file=None,feature_names=X.columns,class_names=dataSet.groupby([label],as_index=False).count()[label].tolist(),filled=True,rounded=True,special_characters=True)
        graph = graphviz.Source(dot_data)
        graph.render(DTdenotion,view=True)
    print(np.array(accuracy).mean(),np.array(accuracy).std())
    labelList = list(set(fullTest))
    labelList.sort()
    labelMap= {labelList[i]:i for i in range(len(labelList))}
    newfullTest=[labelMap[fullTest[i]] for i in range(len(fullTest))]
    return accuracy,fullWrongList,newfullTest,np.array(fullPredict),labelList

def print_full_wrong_list(full_wrong_list):
    s = set()
    for i in full_wrong_list:
        strings = 'Pre-Label: '+i[0]+'   Details: '+i[1]+'  Probabilities: '+str(i[3])
        s.add(strings)
    for i in s:
        print(i)

def generate_features_values_dict(file):
    f=open(file)
    text = f.readline()
    edges_dict={}
    values_dict={}
    features_dict={}
    while(text):
        regex = re.match(r"(\d+)\ ->\ (\d+)", text)
        if regex:
            if regex.groups()[0] in edges_dict:
                edges_dict[regex.groups()[0]].append(regex.groups()[1])
            else:
                edges_dict[regex.groups()[0]] = [regex.groups()[1]] 
        regex2 = re.match(r"(\d+)\ \[label=\".+\[(.+)\]\"\]", text)
        if regex2:
            values_dict[regex2.groups()[0]] = regex2.groups()[1].split(', ')
        
        regex3 = re.match(r"(\d+)\ \[label=\"(?!gini)(.+)\ <=*", text)
        if regex3:
            features_dict[regex3.groups()[1]]=regex3.groups()[0]
        # print(text)
        text = f.readline()
    features_values_dict={key:[ values_dict[edges_dict[value][0]],values_dict[edges_dict[value][1]] ] for (key,value) in features_dict.items() }
    f.close()
    return features_values_dict
def generate_RF_feature_importance(path,df,numeric_features,label):
    dfc=df.copy()
    categories=len(dfc[label].unique())
    regex = re.compile(r"\ +", re.IGNORECASE)
    files = [path+f for f in listdir(path) if f.endswith('.dot') if not f.startswith('.')]
    all_features_dict = {feature:list(np.zeros(categories+1)) for feature in numeric_features}
    for file in files:
        features_values_dict = generate_features_values_dict(file)
        for (key,value) in features_values_dict.items():
            key = regex.sub(' ',key.strip(" "))
            tempList=[]
            count=0
            for i in range(0,len(all_features_dict[key])-1):
                tempList.append(all_features_dict[key][i]+int(value[1][i])-int(value[0][i]))
                count=count+int(value[1][i])+int(value[0][i])
            tempList.append(count+all_features_dict[key][len(all_features_dict[key])-1])
            all_features_dict[key]=tempList
    matrix = []
    for (key,value) in all_features_dict.items(): 
        abscount=0
        list_temp=[key]
        for i in range(0,len(value)-1):
            abscount=abscount+abs(value[i])
        for i in range(0,len(value)-1):
            if(abscount>0):
                list_temp.append(value[i]/abscount)
            else:
                list_temp.append(0)
        list_temp.append(abscount)
        matrix.append(list_temp)
        DF = pd.DataFrame(matrix)
        DF.columns = ['Features']+dfc.groupby([label],as_index=False).count()[label].tolist()+['Sample Size']
        DF.fillna(0)
    return DF

def transform_feature_importance(fullFeatureImportanceDF,label_list):
    feature_importance_DF = fullFeatureImportanceDF.copy()
    for i in label_list:
        feature_importance_DF[i] = round(feature_importance_DF[i],3)
        feature_importance_DF['abs_'+i] = abs(feature_importance_DF[i])
    feature_importance_DF['max_value'] = feature_importance_DF[['abs_'+i for i in label_list]].T.max()
    feature_importance_DF['median_value'] = feature_importance_DF[['abs_'+i for i in label_list]].T.median()
    feature_importance_DF['sampleSize_value']=pow(feature_importance_DF['Sample Size'],0.25)
    feature_importance_DF['Ability']=feature_importance_DF['max_value']*feature_importance_DF['median_value']*feature_importance_DF['sampleSize_value']*10+5
    feature_importance_DF = feature_importance_DF.sort_values(by='Ability', ascending=False) 
    return feature_importance_DF[label_list+['Sample Size','Ability']]