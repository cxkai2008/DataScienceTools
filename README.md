# Data Science tools
Useful data science tools in many aspects.

## Data transformation tools
### Find distinct elements of two list separately. 
sort and reset index based on a certain index
* ind1: the first list
* ind2: the second list
* disinidx1: distinct elements in the first list
* disinidx2: distinct elements in the second list
```
(disinidx1,disinidx2) = findDistinct(ind1,ind2)
```
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

## Plotting tools
### Plot histogram based on a list of values. 
* 'title': the title of histogram 
* listOfValues: the list of values used to plot histogram
* 'outputFilePath': the storing path of the html histogram
* bins_number: the number of bins in histogram
```
plot_histogram('title', listOfValues,'outputFilePath', bins_number = 1000)
```


