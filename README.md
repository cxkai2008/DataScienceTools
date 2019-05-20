# Data Science tools
I wrote these codes during my first research project at ICL, using them makes my project much easier to be done and some of them are general methods which can be widely applied to the other data science tasks. Thus, I'd like to share them here in the case that someone may find them useful. These tools are focused on the tasks in many aspects of data science, in order to make them easy to use, they are coded as separated functions in DStools.py and can be easily used by running this script. Examples can be found in the following content as below:

Please notice: These tools are developed as references of solution for different data science use cases. Many of these tools are built based on packages like pandas, numpy, sci-learn, bokeh etc. Thus, please refer to the licences of the specific packages when using these tools.

## Data Display tools
### Find distinct elements of two list separately. 
sort and reset index based on a certain index
* ind1: the first list
* ind2: the second list
* disinidx1: distinct elements in the first list
* disinidx2: distinct elements in the second list
```
(disinidx1,disinidx2) = findDistinct(ind1,ind2)
```
### Return the top or bottom n indices of a certain list. 
The indices of top or bottom n elements in a list are calculated and outputed.
* lst: the input list
* n: the number of output indices
* isabs: elements are ordered based on their absolute values or not
* isbottom: sort by descending order or not
```
top_indices_lst = topOrBottomN(lst,n,isabs=False,isbottom=False)
```
### Select certain rows from a dataframe based on the combined conditions.
select the rows from a dataframe based on the condition that the column1 equals to the key of the condition_dict and the column2 equals to the value of the condition_dict
* condition_dict: the conditions comprised by the paired keys and values.
* data_frame: the original dataframe
* col1: the column of the key of condition_dict in data_frame
* col2: the column of the value of condition_dict in data_frame
```
sub_dataframe = combined_conditions_filter(condition_dict,data_frame,col1,col2)
```
## Data transformation tools
### Sort data frame. 
sort and reset index based on a certain index
* DF: the data frame which is to be sorted
* sort_column: the sorted index
* asc: ordered by ascending or not
* reidx: reset index or not
```
sorted_DF = rrd(DF,sort_column='ID',asc,reidx)
```
### Data frame to matrix. 
Transform data frame to matrix with a certain order
* DF: the input data frame
* matrix_features: the set of features need to be transform into matrix
* sort_column: sort before transformation
```
matrix = dtm(DF,matrix_features,sort_column='ID')
```
### Matrix to data frame. 
Transform matrix to data frame with a column name list. A sub-data-frame can be combined with the new generated data frame optionally.
* mtx: the input matrix
* numeric_features: the column names of generated matrix
* data_frame: the data frame which is required to combine with newly generated data frame
* basic_info_feautes: the columns of data_frame need to be combined with the new data frame
* sort_column: the data_frame is sorted by sort_column before combination
* asc: ordered by ascending or not
```
matrix = mtd(mtx,numeric_features, data_frame=pd.DataFrame(),basic_info_feautes=[], sort_column='ID',asc=True)
```
### List scaling. 
Scale list based on different criteria. The scaling methods vary based on the input data. For widely ranged data this algorithm use median rather than log transformation to scale it.
* lst: the input list
* scale: the range of scaled list
* lowerbound: the minima of the scaled list
```
scaled_list = median_transform(lst,scale,lowerbound)
```
### Matrix scaling. 
Scale matrix by each row or col separately. The scaling method can choose from the basic scaling method or use the method of median_transform. 
* test: the input matrix
* isrow: scale by each row or col
* simple_scale: whether use the basic method or median_transform
```
scaled_matrix = scale_matrix(test,isrow=True,simple_scale=True)
```
### Unit vectors transformation of a Matrix. 
Transform row or column vectors of a matrix into unit vectors(sum(vec)=1,min(vec)>=0).
* mx: the input matrix
* isrow: scale and transform by each row or col
* is_scale: whether scale the matrix before vector transformation
* simple_scale: whether use the basic method or median_transform
```
scaled_matrix = generate_unit_modules(mx, isrow=True, is_scale=True, simple_scale=True)
```
## Classification tools
### Handle unbalanced data. 
The sample size of minor categories are amplified to as many as the major category by randomly selecting 2 samples and use the average of the two as the newly generated sample.
* megadata_temp: input dataset, contains id column and label column 
* ['1','2','3','4']: the name list of numeric features for training models
* '5': the name of label column
* 'id': the name of id column
```
balanced_DF= handle_unbalanced_dataset(megadata_temp,['1','2','3','4'],label='5',id_column='id')
```
### MC cross validation with balancing data supported. 
* megadata_temp: input dataset, contains id column and label column 
* ['1','2','3','4']: the name list of numeric features for training models
* '5': the name of label column
* 'id': the name of id column
* test_size: the proportion of input dataset used for testing
* handle_unbalance: whether or not handle unbalanced dataset using handle_unbalanced_dataset
```
training_X,testing_X,training_Y,testing_Y = cross_validation_split_with_unbalance_data(megadata_temp,['1','2','3','4'],label='5',id_column='id',test_size=0.2,handle_unbalance=True)
```
### Decision tree and random forest models for MC cross validation. 
* megadata_temp: input dataset, contains id column and label column 
* ['1','2','3','4']: the name list of numeric features for training models
* '5': the name of label column
* 'id': the name of id column
* isDT: "True" means using decision tree model, else using random forest model
* test_size: the proportion of input dataset used for testing
* handle_unbalance: whether or not handle unbalanced dataset using handle_unbalanced_dataset
* folder_path: the path used for storing ".dot" files
* readList: the list of features required to be printed in bad predicting cases
* DTdenotion: The title of plot of decision tree model
* isplot: whether or not plot for decision tree model and random forest model
* DT_maxdepth: the maximal depth of decision tree
* RF_maxdepth: the maximal depth of trees in random forest model
* numberOfTrees: the number of trees in random forest model
* iteration: the number of iterations for MC cross validation
* accuracy: the list of accuracy for all of the iterations
* full_wrong_list: the wrong predictions in all of the iterations, which can be printed by print_full_wrong_list
```
accuracy,full_wrong_list,full_test,full_predict,label_list = DT_RF_models(megadata_temp,['1','2','3','4'],folder_path,isDT = True, iteration=5,testSize =0.1,readList = ['id'], label = '5',DTdenotion='test',DT_maxdepth=2,numberOfTrees = 100,RF_maxdepth=6,isplot=False,id_column='id',handle_unbalance=True)
```
### Feature importance, generate the affinity(relationship) between labels and features. 
* megadata_temp: input dataset, contains id column and label column 
* ['1','2','3','4']: the name list of numeric features for training models
* '5': the name of label column
* folder_path: the path used for storing ".dot" files
* label_list: the list of labels generated from DT_RF_models
```
full_feature_importance_RF_DF = generate_RF_feature_importance(folder_path,megadata_temp,['1','2','3','4'],'5')
T_full_feature_importance_RF_DF = transform_feature_importance(full_feature_importance_RF_DF,label_list) 
```
## Clustering tools
### K-means clustering. 
Use k-means to cluster the data and output the cluster id of each sample
* data_frame: the input dataframe
* numeric_features: the columns used for clustering, must be numeric
* clusters: the expected number of clusters
* is_row: clustering is based on rows or columns.
```
result_DF = k_means_DF(data_frame,numeric_features,clusters=8,is_row=True)
```
### Plot a heatmap for each K-means cluster. 
Plot a heatmap for each cluster, the features for each heatmap is reordered by hierachecal clustering, thus it is possible to have subclusters with each main cluster. It is a very useful way of clustering result visualisation.
An example:
https://github.com/cxkai2008/DataScienceTools/blob/master/materials/Screenshot_2019-05-20.png
* data_frame: the input dataframe
* numeric_features: the columns used for clustering, must be numeric
* path: the path of the saved heatmaps.
* clusters: the expected number of clusters, if equals to 1 means use the full dataset to plot the heatmap.
* is_row: clustering is based on rows or columns.
```
plot_heatmap_for_kmeans_groups(data_frame=DF,numeric_features=['1','2','3'],path='/abc/abc', clusters=1, is_row=True)
```
## Plotting tools
### Plot heatmap based on a similarity data frame. 
* corrDF: the input similarity dataframe, the values ranged from -1 to 1.
* featureList: the list of column/sample names
* path_file: the storing path and file name of the html heatmap
```
plotHeatMap(corrDF=DF.corr() , featureList=['1','2','3'],path_file='/abc/abc/heatmap.html')
```
### Plot histogram based on a list of values. 
* 'title': the title of histogram 
* listOfValues: the list of values used to plot histogram
* 'outputFilePath': the storing path of the html histogram
* bins_number: the number of bins in histogram
```
plot_histogram('title', listOfValues,'outputFilePath', bins_number = 1000)
```
### Scatter plot without colours. 
* data_frame: the dataframe which stores the plotting data
* xvalue: the column of dataframe which is used as x axis
* yvalue: the column of dataframe which is used as y axis
* sizevalue: the column of dataframe which is used as the size of dots, int value is also available.
* outputFilePath: the path and file name of output file
* readList: the columns shown in the interactive plots when the mouse hover on a dots.
* plotWidth: the width of the plot
* plotHeight: the height of the plot
* titleName: the title of the plot.
```
plotBWScatter(data_frame,xvalue = 'col1',yvalue = 'col2', sizevalue = 5, outputFilePath='/path/plot.html',readList = ['co1','col2'],plotWidth = 1200, plotHeight = 900, titleName='plot title')
```

