import graphviz
import numpy as np
import pandas as pd
import random
from sklearn import tree
from sklearn.tree import export_graphviz
import re
from os import listdir
from bokeh.layouts import gridplot
from bokeh.models import Range1d,LabelSet,Label,ColumnDataSource,HoverTool,WheelZoomTool,PanTool,BoxZoomTool,ResetTool,SaveTool,BasicTicker,ColorBar,LinearColorMapper,PrintfTickFormatter,DataSource
from bokeh.palettes import brewer,inferno,magma,viridis,grey
from bokeh.plotting import figure, show, output_file
from bokeh.transform import transform,factor_cmap
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
def scale_matrix(test,isrow=True,simple_scale=True):
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
                test_t[i] = median_transform(test[i],1,0)
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
def dtm(data_frame,matrix_features,sort_column='ID'):
    data_frame_copy = data_frame.copy()
    data_frame_copy = data_frame_copy.sort_values(by=sort_column, ascending=True)
    mtx = data_frame_copy[matrix_features].as_matrix(columns=None)
    return mtx
# matrix to dataframe
def mtd(mtx,numeric_features, data_frame=pd.DataFrame(),basic_info_feautes=[], sort_column='ID',asc=True):
    DF = pd.DataFrame(mtx,columns=numeric_features)
    if((data_frame.size>0)&len(basic_info_feautes)>0):
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
        subDistMatrix = subcorrDF.as_matrix(columns=None)
        for i in range(0,len(subDistMatrix)):
            subDistMatrix[i]=1-subDistMatrix[i]
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
