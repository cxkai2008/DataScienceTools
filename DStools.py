import xgboost as xgb
import graphviz
import numpy as np
import pandas as pd
import random
import matplotlib
import textwrap
import scipy.spatial.distance as ssd
from sklearn import tree
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, label_binarize
from sklearn.tree import export_graphviz
from sklearn.metrics import precision_score, precision_recall_curve, average_precision_score
from matplotlib import pyplot as plt
import re
import math
from os import listdir
from bokeh.layouts import gridplot
from bokeh.models import Range1d,LabelSet,Label,ColumnDataSource,HoverTool,WheelZoomTool,PanTool,BoxZoomTool,ResetTool,SaveTool,BasicTicker,ColorBar,LinearColorMapper,PrintfTickFormatter,DataSource
from bokeh.palettes import brewer,inferno,magma,viridis,grey,Category10
from bokeh.plotting import figure, show, output_file
from bokeh.transform import transform,factor_cmap
from bokeh.io import export_png
from graphviz import Source
from itertools import cycle


#####print table#####
def ppt(table,lines=30,maxWidth = 18,minWidth=10,keepDecimal=2,lineWidth=170):
    if table.shape[0]<1000:
        lines=table.shape[0]
    if(minWidth<5):
        minWidth=5
    if(maxWidth<minWidth):
        print("minWidth cannot large than maxWidth")
    if(lines == -1):
        lines = table.shape[0]
    dicType = generateTypeDic(table)
    dictWidth = generateWidthDic(table,lines,maxWidth,keepDecimal,minWidth)
    strTemp='|'
    for col in table:
        if(dictWidth[col]<minWidth):
            strTemp = strTemp + truncateString(str(col).upper(),minWidth)
        else:
            strTemp = strTemp + truncateString(str(col).upper(),dictWidth[col])
        string = textwrap.fill(text=strTemp,width=lineWidth)
    print(string)
    count = 0
    for index in range(0,table.shape[0]):
        if(count<lines):
            strTemp='|'
            for col in table:
                if(dicType[col] is 'str'):
                    strTemp = strTemp + truncateString(str(table[col].iloc[index]),dictWidth[col])
                else:
                    strTemp = strTemp + truncateNumber(decimalIdentical(table[col].iloc[index],keepDecimal),dictWidth[col])
            string = textwrap.fill(text=strTemp,width=lineWidth)
            print(string)
        else:
            break
        count +=1
        
def generateTypeDic(table):
    dicType={}
    for col in table:
        if(type(table[col].iloc[0]) is np.float64):#if type is np.float64 and the element is NaN, the type is defined as np.float64
            dicType[col]='float'
        elif(type(table[col].iloc[0]) is np.int32):
            dicType[col]='int'
        else:
            dicType[col]='str' #if type is str and the element is NaN, the type is defined as float, so that it will be asigned value as str.
    return dicType

def generateWidthDic(table,lines,maxWidth,keepDecimal=2,minWidth=5):
    dicWidth={}
    for col in table:
        width = minWidth
        colType = generateTypeDic(table)[col]
        if(colType is 'str'):
            for index in range(0,lines):
                lenth = len(str(table[col].iloc[index]))
                if(lenth > width):
                    width = lenth
            if(width > maxWidth):
                width = maxWidth
        elif(colType is 'float' or colType is 'int'):
            for index in range(0,lines):
                lenth = len(decimalIdentical(table[col].iloc[index],keepDecimal))
                if(lenth > width):
                    width = lenth
            if(width > max(9,minWidth)):
                width = max(9,minWidth)
        dicWidth[col] = width
    return dicWidth

def decimalIdentical(flt,kd):
    if(np.isnan(flt)):
        return str(flt)
    flStr = str(round(flt,kd))
    try:
        dotIndex = len(flStr)-flStr.index('.')-1
    except ValueError as ve:
        flStr = flStr+'.'
        dotIndex = 0
    for i in range(dotIndex,kd):
        flStr = flStr + '0'
    return flStr

def truncateNumber(string, length):
    strTemp=''
    if(len(string)<9):
        strTemp=string
        for _ in range(len(string),length+2):
            strTemp=" "+strTemp
        strTemp=strTemp
    else:
        scientificNotation='%.2E' % Decimal(string)
        strTemp=str(scientificNotation)
        for _ in range(len(strTemp),length+2):
            strTemp=" "+strTemp
    if(len(strTemp)!=length+2):
        print('truncateNumber is wrong!'+str(strTemp)+' '+str(length))
    else:
        return strTemp+'| '
        
def truncateString(string, length):
    strTemp=''
    if(len(string)<=length):
        strTemp=string
        for _ in range(len(string),length+2):
            strTemp=" "+strTemp
        strTemp=strTemp
    else:
        strTemp=string[0:length]
        strTemp=strTemp+".."
    if(len(strTemp)!=length+2):
        print('truncateString is wrong!'+str(strTemp))
    else:
        return strTemp+'| '


def generate_reordered_features(megadata_validation,numeric_features_validation,basic_info_features,simple_scale):
    clustering_data_validation = megadata_validation.copy()
    clustering_data_validation = scale_matrix(clustering_data_validation[numeric_features_validation],isrow=False,simple_scale=simple_scale)
    corr_validation_DF = pd.DataFrame(clustering_data_validation,columns=numeric_features_validation).corr('spearman')
    distance_validation_Matrix = corr_validation_DF.values
    for i in range(0,len(distance_validation_Matrix)):
        distance_validation_Matrix[i]=1-distance_validation_Matrix[i]
    distance_validation_Matrix = ssd.squareform(distance_validation_Matrix)
    linked = linkage(distance_validation_Matrix,'ward','euclidean',True)  
    labelList = corr_validation_DF.index
    featureDict= {i:[corr_validation_DF.index[i]] for i in range(0,len(corr_validation_DF.index))}
    for i in range(0,len(linked)):
        index = i+linked.shape[0]+1
        firstList = featureDict[linked[i][0]]
        for j in featureDict[linked[i][1]]:
            firstList.append(j)
        if(len(firstList)!=linked[i][3]):
            print("the length is not equal")
        featureDict[index]=firstList
    featureList=featureDict[linked.shape[0]*2]
    for i in range(len(basic_info_features)):
        featureList.append(basic_info_features[i])
    return featureList
    
def prepare_scaled_data(layer1_df,layer2_df,layer3_df,reordered_feature_list,simple_scale=False):
    layer1_mx = layer1_df.copy()
    layer1_mx = layer1_mx.replace(0,1E-7)
    layer1_mx = layer1_mx.fillna(1E-8)[reordered_feature_list]
    layer2_mx = layer2_df.copy()
    layer2_mx = layer2_mx.replace(0,1E-7)
    layer2_mx = layer2_mx.fillna(1E-8)[reordered_feature_list]
    layer3_mx = layer3_df.copy()
    layer3_mx = layer3_mx.replace(0,1E-7)
    layer3_mx = layer3_mx.fillna(1E-8)[reordered_feature_list]
    layer3_mx = layer3_mx.to_numpy()
    layer2_mx = layer2_mx.to_numpy()
    layer1_mx = layer1_mx.to_numpy()
    layer1_mx = scale_matrix(layer1_mx,isrow=False,simple_scale=simple_scale)
    layer2_mx = scale_matrix(layer2_mx,isrow=False,simple_scale=simple_scale)
    layer3_mx = scale_matrix(layer3_mx,isrow=False,simple_scale=simple_scale)
    return layer1_mx,layer2_mx,layer3_mx

def insert_values_between_original_data(for_image_data_matrix):
    new_matrix = []
    new_matrix.append(for_image_data_matrix[0])
    for i in range(1,len(for_image_data_matrix)):
        new_matrix.append((for_image_data_matrix[i-1]+for_image_data_matrix[i])/2)
        new_matrix.append(for_image_data_matrix[i])
    return np.array(new_matrix)

def plot_colorful_CNN_images(layer1_mx,layer2_mx,layer3_mx,cate,path,interpolation_row=0,interpolation_col=0):
    for i in list(set(cate)):
        ll=[j for j in range(len(cate)) if cate[j] == i]
        for_image_data_matrix1 = layer1_mx[ll]
        for_image_data_matrix2 = layer2_mx[ll]
        for_image_data_matrix3 = layer3_mx[ll]
        category = i
        for i in range(interpolation_row):
            for_image_data_matrix1 = insert_values_between_original_data(for_image_data_matrix1)
            for_image_data_matrix2 = insert_values_between_original_data(for_image_data_matrix2)
            for_image_data_matrix3 = insert_values_between_original_data(for_image_data_matrix3)
        for i in range(interpolation_col):
            for_image_data_matrix1 = insert_values_between_original_data(for_image_data_matrix1.T).T
            for_image_data_matrix2 = insert_values_between_original_data(for_image_data_matrix2.T).T
            for_image_data_matrix3 = insert_values_between_original_data(for_image_data_matrix3.T).T
        colorful = [[[for_image_data_matrix1[j][l],for_image_data_matrix2[j][l],for_image_data_matrix3[j][l]] for l in range(0,len(for_image_data_matrix1[j]))] for j in range(0,len(for_image_data_matrix1))]
        matplotlib.image.imsave(path+'/combined_'+category+'.png',colorful)
        matplotlib.image.imsave(path+'/layer3_'+category+'.png',for_image_data_matrix3)
        matplotlib.image.imsave(path+'/layer2_'+category+'.png',for_image_data_matrix2)
        matplotlib.image.imsave(path+'/layer1_'+category+'.png',for_image_data_matrix1)
        
def plot_colorful_images_wrapper(megadata_temp1,megadata_temp2,megadata_temp3,numeric_cols,image_col,interpolation_row,interpolation_col,path,generate_reordered_indices,simple_scale=True):
    reordered_features = generate_reordered_features(megadata_temp1,numeric_cols,[],simple_scale)
    reordered_indices = generate_reordered_indices(megadata_temp1,reordered_features)
    reordered_df1 = pd.DataFrame(megadata_temp1, index=reordered_indices)[reordered_features+[image_col]]
    reordered_df2 = pd.DataFrame(megadata_temp2, index=reordered_indices)[reordered_features+[image_col]]
    reordered_df3 = pd.DataFrame(megadata_temp3, index=reordered_indices)[reordered_features+[image_col]]
    mx1,mx2,mx3 = prepare_scaled_data(reordered_df1,reordered_df2,reordered_df3,reordered_features,simple_scale)
    plot_colorful_CNN_images(mx1,mx2,mx3,reordered_df1[image_col].values.tolist(),path,interpolation_row,interpolation_col)

#####Select certain rows from dataFrame based on the combined conditions related to index1 and index2#####
def combined_conditions_filter(condition_map,data_frame,index1,index2):
    dataFrame=data_frame.copy()
    dataFrame[index1] = dataFrame[index1].astype(str)
    dataFrame[index2] = dataFrame[index2].astype(str)
    dataFrame['filter'] = dataFrame[index1] + '***' + dataFrame[index2]
    lst = list(str(key)+'***'+str(value) for key,value in condition_map.items())
    subComLevData = dataFrame[dataFrame['filter'].isin(lst)]
    del subComLevData['filter']
    return subComLevData
#Unit vectors transformation of a Matrix. 
def generate_unit_modules(mx, isrow=True, is_scale=True, simple_scale=True):
    test = mx.copy()
    if(is_scale):
        test=scale_matrix(test,isrow=isrow,simple_scale=simple_scale)
    if(isrow):
        for i in range(0,len(test)):
            test[i] = test[i]/test[i].sum()
    else:
        test_t = np.transpose(test)
        for i in range(0,len(test_t)):
            test_t[i] = test_t[i]/test_t[i].sum()
        test = np.transpose(test_t)
    return test

#return the top or bottom n indices of a list
def topOrBottomN(lst,n,isabs=False,isbottom=False):
    if (isabs):
        sortList = []
        for i in range(0,len(lst)):
            sortList.append(abs(lst[i]))
    else:
        sortList = lst
    sortDF = pd.DataFrame({'sort':sortList})
    sortDF['index'] = sortDF.index
    sortDF = sortDF.sort_values(by='sort', ascending=isbottom)
    indexList = sortDF['index'].tolist()
    return indexList[0:n]

#scale matrix based on row or col by simple method or median_transform
def scale_matrix(tst,isrow=True,simple_scale=True):
    test=np.copy(tst)
    if(simple_scale):
        if(isrow):
            for i in range(0,len(test)):
                test[i] = (test[i]-test[i].min())/(test[i].max() - test[i].min())
        else:
            test_t = np.transpose(test)
            for i in range(0,len(test_t)):
                test_t[i] = (test_t[i]-test_t[i].min())/(test_t[i].max() - test_t[i].min())
            test = np.transpose(test_t)
    else:
        if(isrow):
            for i in range(0,len(test)):
                test[i] = median_transform(test[i],1,0)
        else:
            test_t = np.transpose(test)
            for i in range(0,len(test_t)):
                test_t[i] = median_transform(test[:,i],1,0)
            test = np.transpose(test_t)
    return test
#sort data frame
def rrd(DF,sort_column='ID',asc=True,reidx=True):
    new_DF=DF.copy()
    new_DF=DF.sort_values(by=sort_column, ascending=True)
    if(reidx):
        new_DF=new_DF.reset_index(drop=True)
    return new_DF

# dataframe to matrix
def dtm(data_frame,matrix_features,sort_column='ID',asc=True):
    data_frame_copy = data_frame.copy()
    data_frame_copy = data_frame_copy.sort_values(by=sort_column, ascending=asc)
    mtx = data_frame_copy[matrix_features].values
    return mtx
# matrix to dataframe
def mtd(mtx,numeric_features, data_frame=pd.DataFrame(),basic_info_feautes=[], sort_column='ID',asc=True):
    DF = pd.DataFrame(mtx,columns=numeric_features)
    if((data_frame.size>0)&(len(basic_info_feautes)>0)):
        DF[basic_info_feautes] = rrd(data_frame,sort_column,asc).reset_index(drop=True)[basic_info_feautes]
    return rrd(DF,sort_column)
    
def scale_transform1(lst1,lst2,scale,lowerbound):
    return ((lst1-lst2.min())/np.ptp(lst2))*scale+lowerbound
    
def scale_transform2(lst1,lst2,scale,lowerbound):
    return (lst1/lst2.max())*scale+lowerbound
#scale list
def median_transform(lst,scale,lowerbound):
    if(len(set(lst))<2):
        return np.full(len(lst), (scale+lowerbound)/2)
    if(lst.max()/lst.mean()<2):
        return 0.5*lst/lst.mean()*scale+lowerbound
    elif((lst.max()/lst.min()<10)&(lst.mean()/lst.min()>2)):
        return scale_transform1(lst,lst,scale,lowerbound)
    else: 
        scaled_list=scale_transform1(lst,lst,scale,lowerbound)
        scaled_list = scaled_list/np.median(scaled_list)
        lower_list=np.array([i for i in scaled_list if i<=1]).copy()
        upper_list=np.array([i for i in scaled_list if i>1]).copy()
        for i in range(len(scaled_list)):
            if(scaled_list[i]<=1):
                if(np.ptp(lower_list)==0):
                    scaled_list[i]=0
                else:
                    scaled_list[i]=scale_transform1(scaled_list[i],lower_list,0.5*(scale+lowerbound),lowerbound)
            else:
                scaled_list[i]=scale_transform2(scaled_list[i],upper_list,0.5*(scale+lowerbound),0.5*(scale+lowerbound))
        return scaled_list

def unify_df(rna_df,numeric_features,id_col,other_categorical_features):
    rna_scaled_df=rna_df.copy()
    rna_scaled_df['temp_indices_within_this_function']=rna_scaled_df[id_col]
    temp_mx = dtm(rna_scaled_df,numeric_features,sort_column='temp_indices_within_this_function')
    temp_mx = generate_unit_modules(temp_mx, isrow=True, is_scale=True, simple_scale=True)
    rna_scaled_df = mtd(temp_mx,numeric_features,rna_scaled_df,other_categorical_features+['temp_indices_within_this_function'], sort_column='temp_indices_within_this_function',asc=True)
    rna_scaled_df.index=rna_scaled_df['temp_indices_within_this_function']
    rna_scaled_df[id_col]=rna_scaled_df['temp_indices_within_this_function']
    del rna_scaled_df['temp_indices_within_this_function']
    temp_df=rna_df.copy()
    temp_df.index=temp_df[id_col]
    #control the scale of the mean by dividing by len(numeric_features)
    temp_df = pd.DataFrame(median_transform(temp_df[numeric_features].T.describe().T['mean'],1,0)/len(numeric_features))
    temp_df.columns=['mean_from_unify_df']
    rna_scaled_df=pd.merge(rna_scaled_df,temp_df,left_index=True,right_index=True)
    rna_scaled_df=rna_scaled_df.reset_index(drop=True)
    new_numeric_features=numeric_features+['mean_from_unify_df']
    return rna_scaled_df,new_numeric_features

#####find distict items in two lists#####
def findDistinct(ind1,ind2):
    return (list(np.setdiff1d(ind1, ind2)),list(np.setdiff1d(ind2, ind1)))


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
        new_DF = new_DF.append(temp_DF, sort=False)
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
    return feature_importance_DF[['Features']+label_list+['Sample Size','Ability']]

#####plot histogram based on a list of values#####
def plot_histogram(title, measured,outputFilePath, bins_number = 1000):
    output_file(outputFilePath)
    hist, edges = np.histogram(measured, density=True, bins=bins_number)
    p = figure(title=title, plot_width = 750, plot_height = 750,tools='', background_fill_color="#fafafa")
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
           fill_color="navy", line_color="white", alpha=0.5)
    p.y_range.start = 0
    p.legend.location = "center_right"
    p.legend.background_fill_color = "#fefefe"
    p.xaxis.axis_label = 'x'
    p.yaxis.axis_label = 'Pr(x)'
    p.grid.grid_line_color="white"
    p.x_range = Range1d(0,1.01)
    show(p)
    return p
def plotBWScatter(DataFrame ,xvalue,yvalue, sizevalue, outputFilePath, readList,plotWidth = 1200, plotHeight = 900, titleName='features importance'):
    hover = HoverTool()
    tooltipString = ""
    for ele in readList:
        readTuple = (ele.lower(),ele)
        tooltipString = tooltipString + """<br><font face="Arial" size="4">%s: @%s<font>""" % readTuple
    hover.tooltips = tooltipString
    tools= [hover,WheelZoomTool(),PanTool(),BoxZoomTool(),ResetTool(),SaveTool()]
    source= ColumnDataSource(DataFrame)
    p = figure(plot_width = plotWidth, plot_height = plotHeight, tools=tools,title=titleName,toolbar_location='right',x_axis_label=xvalue.lower(),y_axis_label=yvalue.lower(),background_fill_color='white',title_location = 'above')
    p.title.text_font_size='15pt'
    p.title.align = 'center'
    p.xaxis.axis_label_text_font_size='12pt'
    p.yaxis.axis_label_text_font_size='12pt'
    p.x_range = Range1d(DataFrame[xvalue].min()*1.1,DataFrame[xvalue].max()*1.1)
    p.y_range = Range1d(DataFrame[yvalue].min()*1.1,DataFrame[yvalue].max()*1.1)
    p.circle(x = xvalue,y = yvalue,size=sizevalue,source=source,color='grey')
    p.toolbar.active_scroll=p.select_one(WheelZoomTool)#set default active to scroll tool
    if outputFilePath.endswith('png'):
        export_png(p, filename=outputFilePath)
    else:
        output_file(outputFilePath)
        show(p)
#k-means clustering method
def k_means_DF(data_frame,numeric_features,clusters=8,is_row=True):
    clustering_data_validation = data_frame[numeric_features].copy()
    if(is_row):
        corr_validation_DF = clustering_data_validation.T.corr()
    else:
        corr_validation_DF = clustering_data_validation.corr()
    kmeans = KMeans(n_clusters=clusters,random_state=100).fit(corr_validation_DF)
    clusterDic = {corr_validation_DF.columns[i]:kmeans.labels_[i]  for i in range(0,len(kmeans.labels_))}
    npArray = np.array([[key,value]  for (key,value) in clusterDic.items() ])
    DF = pd.DataFrame(npArray)
    DF.columns = ['element','group']
    return DF

def plotHeatMap(corrDF , featureList,path_file):
    output_file(path_file)
    corrDF.columns.name = 'Features'
    df = pd.DataFrame(corrDF[featureList].stack(), columns=['Distance']).reset_index()
    df.columns=['level_0','Features','Distance']
    source = ColumnDataSource(df)
    colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
    mapper = LinearColorMapper(palette=colors, low=df.Distance.min(), high=df.Distance.max())
    p = figure(plot_width=3500, plot_height=3500, title="HeatMap",
              x_range=featureList, y_range=featureList,
              toolbar_location=None, tools="", x_axis_location="above")
    p.rect(x="Features", y="level_0", width=1, height=1, source=source,line_color=None, fill_color=transform('Distance', mapper))
    color_bar = ColorBar(color_mapper=mapper, location=(0, 0),
                         ticker=BasicTicker(desired_num_ticks=len(colors)),
                         formatter=PrintfTickFormatter(format="%d%%"))
    p.add_layout(color_bar, 'right')
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "30pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = 1.0
    show(p)
    
def heatMap(DF , path_file, x_size=3500, y_size=3500,font_size="15pt"):
    featureList=DF.columns.tolist()
    indexList=DF.index.tolist()
    DF.columns.name = 'Features'
    df = pd.DataFrame(DF[featureList].stack(), columns=['Distance']).reset_index()
    df.columns=['level_0','Features','Distance']
    source = ColumnDataSource(df)
    colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
    mapper = LinearColorMapper(palette=colors, low=df.Distance.min(), high=df.Distance.max())
    p = figure(plot_width=x_size, plot_height=y_size, title="HeatMap",
              x_range=featureList, y_range=indexList,
              toolbar_location=None, tools="", x_axis_location="above")
    p.rect(x="Features", y="level_0", width=1, height=1, source=source,line_color=None, fill_color=transform('Distance', mapper))
    color_bar = ColorBar(color_mapper=mapper, location=(0, 0),
                         ticker=BasicTicker(desired_num_ticks=len(colors)),
                         formatter=PrintfTickFormatter(format="%.2f"))
    p.add_layout(color_bar, 'right')
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = 1.0
    p.axis.major_label_text_font_size = font_size
    if path_file.endswith('png'):
        export_png(p, filename=path_file)
    else:
        output_file(path_file)
        show(p)
    
def plot_heatmap_for_kmeans_groups(data_frame,numeric_features,path,clusters=8,is_row=True):
    result_DF = k_means_DF(data_frame,numeric_features,clusters,is_row)
    for k in range(0,clusters):
        group_filter = result_DF['group'].astype(str)==str(k)
        subFeatureList = result_DF[group_filter]['element'].values
        if(is_row):
            subNormalData = data_frame[numeric_features].T[subFeatureList].copy()
        else:
            subNormalData = data_frame[subFeatureList].copy()
        if(subNormalData.shape[1]<2):
            continue
        subcorrDF = subNormalData.corr()
        subcorrDF.columns=[str(i) for i in subcorrDF.columns.tolist()]
        assert len(subFeatureList) == subcorrDF.shape[0]
        subDistMatrix = subcorrDF.values
        for i in range(0,len(subDistMatrix)):
            subDistMatrix[i]=1-subDistMatrix[i]
        subDistMatrix = ssd.squareform(subDistMatrix)
        sublinked = linkage(subDistMatrix,'ward','euclidean',True)  
        subFeatureDict= {i:[subcorrDF.columns[i]] for i in range(0,len(subcorrDF.columns))}
        for i in range(0,len(sublinked)):
            index = i+sublinked.shape[0]+1
            firstList = subFeatureDict[sublinked[i][0]]
            for j in subFeatureDict[sublinked[i][1]]:
                firstList.append(j)
            if(len(firstList)!=sublinked[i][3]):
                print("the length is not equal")
            subFeatureDict[index]=firstList
        subFeatureList=subFeatureDict[sublinked.shape[0]*2]
        strFeatureList = [str(i) for i in subFeatureList]
        subcorrDF.index=subcorrDF.columns
        subcorrDF=subcorrDF.T[subFeatureList].T
        plotHeatMap(subcorrDF[subFeatureList].reset_index(drop=True),strFeatureList,path+'/heatmap-'+str(k)+'.html')
    if clusters==1:
        return strFeatureList
def plot_precision_recall_curve(full_test,full_predict,label_list,class_num=4,title='ROC curve'):
    if(class_num==2):
        full_test=label_binarize(full_test,classes=list(range(0,3)))
        full_test=np.array([np.array([i[0],i[1]])   for i in full_test])
    else:
        full_test=label_binarize(full_test,classes=list(range(0,class_num)))
    precision = dict()
    recall = dict()
    average_precision=dict()
    for i in range(0,class_num):
        precision[i],recall[i],_ = precision_recall_curve(full_test[:,i],full_predict[:,i])
        average_precision[i] = average_precision_score(full_test[:,i],full_predict[:,i])
    precision['micro'],recall['micro'],_=precision_recall_curve(full_test.ravel(),full_predict.ravel())
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    plt.figure(figsize=(7, 8))
    labels = []
    lines = []
    for i, color in zip(range(class_num), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (AUC = {1:0.2f})'
                      ''.format(label_list[i], average_precision[i]))
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    plt.show()

def print_precision_recall_accuracy(full_test,full_predict,label_list,class_num=4):
    right=0
    wrong=0
    for i in range(len(full_test)):
        if(np.argmax(full_predict[i])  == int(full_test[i])):
            right =right+1
        else:
            wrong=wrong+1
    print("Overall Accuray: ",right/(right+wrong))
    for n in range(class_num):
        tp=0
        fp=0
        fn=0
        for i in range(len(full_test)):
            if(np.argmax(full_predict[i])==n):
                if(np.argmax(full_predict[i])  == int(full_test[i])):
                    tp=tp+1
                else:
                    fp=fp+1
            elif(int(full_test[i])==n):
                fn=fn+1
        print(label_list[n],"label size:",tp+fn)
        print(label_list[n],"Recall: ",tp/(tp+fn))
        if((tp+fp)==0):
            print(label_list[n],"Precision: ",0)
        else:
            print(label_list[n],"Precision: ",tp/(tp+fp))

##########Venn-Abers Predictor##########
### This part of codes is taken from https://github.com/ptocca/VennABERS, All credit of this part goes to the author of this repository.###

# Some elementary functions to speak the same language as the paper
# (at some point we'll just replace the occurrence of the calls with the function body itself)
def push(x,stack):
    stack.append(x)

def pop(stack):
    return stack.pop()

def top(stack):
    return stack[-1]

def nextToTop(stack):
    return stack[-2]


# perhaps inefficient but clear implementation
def nonleftTurn(a,b,c):   
    d1 = b-a
    d2 = c-b
    return np.cross(d1,d2)<=0

def nonrightTurn(a,b,c):   
    d1 = b-a
    d2 = c-b
    return np.cross(d1,d2)>=0


def slope(a,b):
    ax,ay = a
    bx,by = b
    return (by-ay)/(bx-ax)

def notBelow(t,p1,p2):
    p1x,p1y = p1
    p2x,p2y = p2
    tx,ty = t
    m = (p2y-p1y)/(p2x-p1x)
    b = (p2x*p1y - p1x*p2y)/(p2x-p1x)
    return (ty >= tx*m+b)

kPrime = None

# Because we cannot have negative indices in Python (they have another meaning), I use a dictionary

def algorithm1(P):
    global kPrime
    S = []
    P[-1] = np.array((-1,-1))
    push(P[-1],S)
    push(P[0],S)
    #put P[0] at the end of S
    for i in range(1,kPrime+1):
    #nextToTop(S):S[len(S)-2]  top(S):S[len(S)-1]  pop(S):drop the last element
    #cross product for 2 dimension vector return the value of axis z
    #cross product vector of vec1 and vec2 is the perpendicular vector with the plane consist by vec1 and vec2
        while len(S)>1 and nonleftTurn(nextToTop(S),top(S),P[i]):
            pop(S)
        push(P[i],S)
    return S

def algorithm2(P,S):
    global kPrime
    
    Sprime = S[::-1]     # reverse the stack

    F1 = np.zeros((kPrime+1,))
    for i in range(1,kPrime+1):
        F1[i] = slope(top(Sprime),nextToTop(Sprime))
        P[i-1] = P[i-2]+P[i]-P[i-1]
        if notBelow(P[i-1],top(Sprime),nextToTop(Sprime)):
            continue
        pop(Sprime)
        while len(Sprime)>1 and nonleftTurn(P[i-1],top(Sprime),nextToTop(Sprime)):
            pop(Sprime)
        push(P[i-1],Sprime)
    return F1

def algorithm3(P):
    global kPrime
    
    P[kPrime+1] = P[kPrime]+np.array((1.0,0.0))

    S = []
    push(P[kPrime+1],S)
    push(P[kPrime],S)
    for i in range(kPrime-1,0-1,-1):  # k'-1,k'-2,...,0
        while len(S)>1 and nonrightTurn(nextToTop(S),top(S),P[i]):
            pop(S)
        push(P[i],S)
    return S

def algorithm4(P,S):
    global kPrime
    
    Sprime = S[::-1]     # reverse the stack
    
    F0 = np.zeros((kPrime+1,))
    for i in range(kPrime,1-1,-1):   # k',k'-1,...,1
        F0[i] = slope(top(Sprime),nextToTop(Sprime))
        P[i] = P[i-1]+P[i+1]-P[i]
        if notBelow(P[i],top(Sprime),nextToTop(Sprime)):
            continue
        pop(Sprime)
        while len(Sprime)>1 and nonrightTurn(P[i],top(Sprime),nextToTop(Sprime)):
            pop(Sprime)
        push(P[i],Sprime)
    return F0[1:]

def prepareData(calibrPoints):
    global kPrime
    #sort score_label_list based on ascending score
    ptsSorted = sorted(calibrPoints)
    #xs score np.array, ys, label np.array, both sorted
    xs = np.fromiter((p[0] for p in ptsSorted),float)
    ys = np.fromiter((p[1] for p in ptsSorted),float)
    ptsUnique,ptsIndex,ptsInverse,ptsCounts = np.unique(xs, 
                                                        return_index=True,
                                                        return_counts=True,
                                                        return_inverse=True)
    a = np.zeros(ptsUnique.shape)
    #a: for a unique score, how many items labeled 1.
    np.add.at(a,ptsInverse,ys)
    # now a contains the sums of ys for each unique value of the objects
    w = ptsCounts
    yPrime = a/w
    #yPrime: the purity of label for each unique score
    yCsd = np.cumsum(w*yPrime)   # Might as well do just np.cumsum(a)
    #yCsd accumulation of label1 through unique score list
    xPrime = np.cumsum(w)
    #xPrime: accumulation of observations through unique score list
    kPrime = len(xPrime)
    #kPrime: the number of unique scores
    return yPrime,yCsd,xPrime,ptsUnique

def computeF(xPrime,yCsd):    
    P = {0:np.array((0,0))}
    P.update({i+1:np.array((k,v)) for i,(k,v) in enumerate(zip(xPrime,yCsd))})
    #P is (i->(xPrime[i],yCsd[i]))
    S = algorithm1(P)
    F1 = algorithm2(P,S)
    
    # P = {}
    # P.update({i+1:np.array((k,v)) for i,(k,v) in enumerate(zip(xPrime,yCsd))})    
    
    S = algorithm3(P)
    F0 = algorithm4(P,S)
    
    return F0,F1

def getFVal(F0,F1,ptsUnique,testObjects):
    pos0 = np.searchsorted(ptsUnique[1:],testObjects,side='right')
    pos1 = np.searchsorted(ptsUnique[:-1],testObjects,side='left')+1
    return F0[pos0],F1[pos1]

def ScoresToMultiProbs(calibrPoints,testObjects):
    # sort the points, transform into unique objects, with weights and updated values
    yPrime,yCsd,xPrime,ptsUnique = prepareData(calibrPoints)
    
    # compute the F0 and F1 functions from the CSD
    F0,F1 = computeF(xPrime,yCsd)
    
    # compute the values for the given test objects
    p0,p1 = getFVal(F0,F1,ptsUnique,testObjects)
                    
    return p0,p1

def computeF1(yCsd,xPrime):
    global kPrime
    
    P = {0:np.array((0,0))}
    P.update({i+1:np.array((k,v)) for i,(k,v) in enumerate(zip(xPrime,yCsd))})
    
    S = algorithm1(P)
    F1 = algorithm2(P,S)
    
    return F1

def ScoresToMultiProbsV2(calibrPoints,testObjects):
    # sort the points, transform into unique objects, with weights and updated values
    yPrime,yCsd,xPrime,ptsUnique = prepareData(calibrPoints)
   
    # compute the F0 and F1 functions from the CSD
    F1 = computeF1(yCsd,xPrime)
    pos1 = np.searchsorted(ptsUnique[:-1],testObjects,side='left')+1
    p1 = F1[pos1]
    
    yPrime,yCsd,xPrime,ptsUnique = prepareData((-x,1-y) for x,y in calibrPoints)    
    F0 = 1 - computeF1(yCsd,xPrime)
    pos0 = np.searchsorted(ptsUnique[:-1],testObjects,side='left')+1
    p0 = F0[pos0]
    return p0,p1

def generate_label_from_probability(p0,p1,testScores,isprint=True):
    p = p1/(1-p0+p1)
    full_test=np.array([np.array([1-i,i]) for i in p])
    t_p=[(int(round(testScores[i])),int(round(p[i])),testScores[i],p[i]) for i in range(0,len(p))]
    #label from score, label from probability, score, probability
    count=0
    for i in range(0,len(t_p)):
        if (t_p[i][0]!=t_p[i][1]):
            count = count+1
            if(isprint):
                print("differ",count,t_p[i])
    return t_p,full_test
##########End of Venn-Abers Predictor##########

def xgboostModel_for_venn(train,test,selectedData_Indices,n_vap=10,label = 'Control',category = 'Category',num_round = 100):
    train_temp=train.copy()
    test_temp=test.copy()
    train_temp['cv_id']=train_temp.index
    labelList = [label,'ZZZZZZZ']
    train_temp.loc[train_temp[category]!=label,category]='ZZZZZZZ'
    test_temp.loc[test_temp[category]!=label,category]='ZZZZZZZ'
    score_and_label_list=[]
    test_score=[]
    regex = re.compile(r"\[|\]|<|\ ", re.IGNORECASE)
    param = {'max_depth':2,'eta':0.3,'silent':1,'objective':'binary:logistic','learningrate':0.1}  #'binary:logistic'   'multi:softprob' 'num_class':2,
    train_temp.columns = [regex.sub('_',col) for col in train_temp.columns.values]
    test_temp.columns = [regex.sub('_',col) for col in test_temp.columns.values]
    selectedData_Indices = [regex.sub('_',col) for col in selectedData_Indices]
    labelEncoder = LabelEncoder()
    for _ in range(n_vap):
        X_train,X_test,Y_train,Y_test = cross_validation_split_with_unbalance_data(train_temp.reset_index(drop=True).copy(),selectedData_Indices,label=category,id_column='cv_id',test_size=0.2,handle_unbalance=False)
        Y_train = labelEncoder.fit_transform(Y_train.values)
        Y_test = labelEncoder.fit_transform(Y_test.values)
        dtrain = xgb.DMatrix(X_train,label=Y_train)
        dtest = xgb.DMatrix(X_test,label=Y_test)
        bst = xgb.train(param,dtrain,num_round,feval='map5eval',maximize=True)
        preds = bst.predict(dtest)
        best_preds = np.asarray([round(value) for value in preds])
        Y_test = pd.DataFrame(Y_test).reset_index()
        for i in range(0,len(best_preds)):
            score_and_label_list.append((preds[i],Y_test.iloc[i][0]))
    count=0
    X_train=train_temp[selectedData_Indices]
    X_test=test_temp[selectedData_Indices]
    Y_train=train_temp[category]
    Y_test=test_temp[category]
    Y_train = labelEncoder.fit_transform(Y_train.values)
    Y_test = labelEncoder.fit_transform(Y_test.values)
    dtrain = xgb.DMatrix(X_train,label=Y_train)
    dtest = xgb.DMatrix(X_test,label=Y_test)
    bst = xgb.train(param,dtrain,num_round,feval='map5eval',maximize=True)
    preds = bst.predict(dtest)
    best_preds = np.asarray([round(value) for value in preds])
    Y_test = pd.DataFrame(Y_test).reset_index()
    for i in range(0,len(best_preds)):
        test_score.append(preds[i])
        if(best_preds[i] != Y_test.iloc[i][0]):
            count=count+1
    fullPredict=[np.array([1-i,i]) for i in list(preds)]
    p0,p1 = ScoresToMultiProbs(score_and_label_list,test_score)
    label_from_probability,full_predic_venn = generate_label_from_probability(p0,p1,test_score,False)
    readable_pre=[i[0] for i in full_predic_venn]
    test[label]=readable_pre
    return test,Y_test[0].to_numpy(),np.array(fullPredict),labelList

def tSNEPlot(oriData,data_Indices,read_list,color_col,storing_loc,size_col = 5, iters=1000, perp=2, title='tSNE',num_components=2):
    tsne = TSNE(n_components=num_components,random_state=0,n_iter=iters,perplexity=perp)
    tSNE_DF = oriData.copy()
    tSNE_DF=tSNE_DF.reset_index(drop=True)
    tSNE_DF_2d = (tSNE_DF[data_Indices] - tSNE_DF[data_Indices].mean()) / (tSNE_DF[data_Indices].max() - tSNE_DF[data_Indices].min()).fillna(0)
    tSNE_DF_2d = tsne.fit_transform(tSNE_DF_2d.to_numpy())
    tSNE_DF_2d = pd.DataFrame(tSNE_DF_2d).reset_index(drop=True)
    tSNE_DF_2d.columns=[str(i) for i in range(1,1+num_components)]
    for i in read_list+[color_col]:
        tSNE_DF_2d[i] = tSNE_DF[i]
    tSNE_DF_2d[color_col]=tSNE_DF_2d[color_col].astype(str)
    plotColorScatter(tSNE_DF_2d ,xvalue = '1',yvalue = '2', sizevalue = size_col, outputFilePath=storing_loc,plotWidth = 750, plotHeight = 750, readList = read_list,titleName=title,colorColumn=color_col,colorPattern=viridis)
    return tSNE_DF_2d

def plotColorScatter(DataFrame ,xvalue = '0',yvalue = '1', sizevalue = 'size', outputFilePath='/abc/test.html',plotWidth = 750, plotHeight = 750, readList = ['1','2'],titleName='tSNE', colorColumn="Category", colorPattern=viridis):
    factors = DataFrame[colorColumn].unique()
    if (len(factors)>2) & (len(factors)<11):
        color_map = factor_cmap(colorColumn,factors=factors,palette=Category10[10])
    else:
        color_map = factor_cmap(colorColumn,factors=factors,palette=colorPattern(len(factors)))
    hover = HoverTool()
    tooltipString = ""
    for ele in readList:
        ele=str(ele)
        readTuple = (ele.lower(),ele)
        tooltipString = tooltipString + """<br><font face="Arial" size="4">%s: @%s<font>""" % readTuple
    hover.tooltips = tooltipString
    tools= [hover,WheelZoomTool(),PanTool(),BoxZoomTool(),ResetTool(),SaveTool()]
    source= ColumnDataSource(DataFrame)
    output_file(outputFilePath)
    p = figure(plot_width = plotWidth, plot_height = plotHeight, tools=tools,title=titleName,toolbar_location='right',x_axis_label=xvalue.lower(),y_axis_label=yvalue.lower(),background_fill_color='white',title_location = 'above')
    p.title.text_font_size='15pt'
    p.title.align = 'center'
    p.xaxis.axis_label_text_font_size='12pt'
    p.yaxis.axis_label_text_font_size='12pt'
    p.x_range = Range1d(DataFrame[xvalue].min()*1.1,DataFrame[xvalue].max()*1.1)
    p.y_range = Range1d(DataFrame[yvalue].min()*1.1,DataFrame[yvalue].max()*1.1)
    p.circle(x = xvalue,y = yvalue,size=sizevalue,source=source,color=color_map,legend=colorColumn)
    p.legend.location = "top_left"
    p.toolbar.active_scroll=p.select_one(WheelZoomTool)
    if outputFilePath.endswith('png'):
        export_png(p, filename=outputFilePath)
    else:
        output_file(outputFilePath)
        show(p)

def print_full_wrong_list(full_wrong_list):
    s = set()
    for i in full_wrong_list:
        strings = 'Pre-Label: '+i[0]+'   Original_data: '+i[1]+'  Probability: '+str(i[3])
        s.add(strings)
    for i in s:
        print(i)

def xgboost_multi_classification(input_df,numeric_features_validation,iteration=10,test_size=0.2,max_depth=2,num_trees=50,label_column='Category',id_column='PlateID',handle_unbalance=True,readList=['PlateID','Compound Name']):
    XGBData = input_df.copy()
    selectedData_Indices = numeric_features_validation.copy()  #  data_Indices
    num_class=len(XGBData[label_column].unique().tolist())
    regex = re.compile(r"\[|\]|<|\ ", re.IGNORECASE)
    param = {'max_depth':max_depth,'eta':0.3,'silent':1,'objective':'multi:softprob','num_class':num_class,'learningrate':0.1} 
    num_round = num_trees
    labelList = XGBData.groupby([label_column],as_index=False).mean()[label_column].tolist()
    label = 0
    accuracy = []
    X = XGBData.reset_index()[selectedData_Indices]
    Y = XGBData.reset_index()[label_column]
    X.columns = [regex.sub('_',col) for col in X.columns.values]
    XGBData.columns = [regex.sub('_',col) for col in XGBData.columns.values]
    selectedData_Indices = [regex.sub('_',col) for col in selectedData_Indices]
    labelEncoder = LabelEncoder()
    labelEncoded = labelEncoder.fit_transform(XGBData.reset_index()[label_column].values)
    fullWrongList=[]
    fullTest=np.array([])
    fullPredict=[]
    for j in range(0,iteration):
        X_train, X_test, Y_train, Y_test = cross_validation_split_with_unbalance_data(XGBData,selectedData_Indices,label=label_column,id_column=id_column,test_size=test_size,handle_unbalance=handle_unbalance)
        Y_train = labelEncoder.fit_transform(Y_train.values)
        Y_test = labelEncoder.fit_transform(Y_test.values)
        fullTest=np.concatenate((fullTest,Y_test),axis=0)
        dtrain = xgb.DMatrix(X_train,label=Y_train)
        dtest = xgb.DMatrix(X_test,label=Y_test)
        bst = xgb.train(param,dtrain,num_round,feval='map5eval',maximize=True)
        preds = bst.predict(dtest)
        fullPredict=fullPredict+list(preds)
        best_preds = np.asarray([np.argmax(line) for line in preds])
        precision = precision_score(Y_test,best_preds,average='macro')
        Y_test = pd.DataFrame(Y_test).reset_index()
        count=0
        for i in range(0,len(best_preds)):
            if(best_preds[i] != Y_test.iloc[i][label]):
                count=count+1
                string=''
                for l in range(0,len(readList)):
                    string = string + str(XGBData.reset_index().iloc[X_test.reset_index(drop=True).index[i]][readList[l]])+'---'
                singleWrongList = [labelList[best_preds[i]],string+labelList[Y_test.iloc[i][label]],str(j),preds[i]]
                fullWrongList.append(singleWrongList)
        print('------------------accuracy = '+str(1-count/len(best_preds))+'------------------')
        accuracy.append(1-count/len(best_preds))
        #bst.dump_model(storePath)
    pArray = np.array(accuracy)
    print(pArray.mean(),pArray.std())
    return pArray,fullWrongList,fullTest,np.array(fullPredict),labelList

def combined_eXGBT_classifier(training_set,numeric_features_validation,testing_set,n_vap=10,label_column = 'Category',max_depth=2,num_trees=50):
    num_class=len(training_set[label_column].unique().tolist())
    df_te = testing_set.copy()
    for i in training_set[label_column].unique().tolist():
        df_te,full_test,full_predict,label_list =xgboostModel_for_venn(training_set,df_te,numeric_features_validation,n_vap,label =i,category = label_column,num_round = num_trees)
    XGBData = training_set.copy()
    selectedData_Indices = numeric_features_validation  #  data_Indices
    regex = re.compile(r"\[|\]|<|\ ", re.IGNORECASE)
    param = {'max_depth':max_depth,'eta':0.3,'silent':1,'objective':'multi:softprob','num_class':num_class,'learningrate':0.1} 
    labelList = XGBData.groupby([label_column],as_index=False).mean()[label_column].tolist()
    X = XGBData.reset_index()[selectedData_Indices]
    Y = XGBData.reset_index()[label_column]
    Z = df_te[selectedData_Indices]
    X.columns = [regex.sub('_',col) for col in X.columns.values]
    Z.columns = [regex.sub('_',col) for col in Z.columns.values]
    labelEncoder = LabelEncoder()
    labelEncoded = labelEncoder.fit_transform(Y.values)
    dtrain = xgb.DMatrix(X,label=labelEncoded)
    bst = xgb.train(param,dtrain,num_trees,feval='map5eval',maximize=True)
    dtest = xgb.DMatrix(Z)
    preds = bst.predict(dtest)
    best_preds = np.asarray([np.argmax(line) for line in preds])
    readable_pre=[labelList[i] for i in best_preds]
    df_te['multi_eXGBT_pre_lable']=readable_pre
    return df_te

def transform_predict_result_DF(predict_result_DF,label_col,threshold=0.1):
    id_col='predict_result_DF_indices'
    predict_result_DF[id_col]=predict_result_DF.index
    #This is for specific purposes.
    try: 
        label_list = predict_result_DF[label_col].unique().tolist()
        predict_result_DF['max']=predict_result_DF[label_list].T.max()
    except KeyError as keyE: 
        label_list = predict_result_DF['multi_eXGBT_pre_lable'].unique().tolist()
        predict_result_DF['max']=predict_result_DF[label_list].T.max()
        print('Notice: The predicted labels were used instead of full labels')
    predict_result_DF['max']=predict_result_DF[label_list].T.max()
    min_Filter = predict_result_DF['max']<threshold
    predict_result_DF.loc[min_Filter,'F_label']=predict_result_DF.loc[min_Filter,'multi_eXGBT_pre_lable']
    max_Filter = predict_result_DF['max']>=threshold
    for i in label_list:
        analogue_filter = predict_result_DF['max']==predict_result_DF[i]
        predict_result_DF.loc[analogue_filter&max_Filter,'F_label']=i
    predict_result_DF = predict_result_DF.rename({'max': 'probability'}, axis='columns')
    temp1= predict_result_DF.groupby([id_col,'F_label'], as_index=False).mean()[[id_col,'F_label','probability']]
    temp1['ID']=temp1[id_col].astype(str)+temp1['probability'].astype(str)
    temp2= temp1.groupby([id_col], as_index=False).max()[[id_col,'probability']]
    temp2['ID']=temp2[id_col].astype(str)+temp2['probability'].astype(str)
    temp3=temp2.merge(temp1, on='ID', how='left')[[id_col+'_x','probability_x','F_label']]
    temp3.columns=[id_col,'confidence','predicted_label']
    temp3.groupby(['predicted_label'], as_index=False).count()[['predicted_label','confidence']]
    temp3=temp3.merge(predict_result_DF, on=id_col, how='left')[[id_col,'confidence','predicted_label',label_col]]
    fake_filter = temp3[id_col].astype(str).str.startswith('fake')
    return predict_result_DF,temp3[~fake_filter]
