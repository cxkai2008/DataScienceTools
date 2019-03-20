# Data Science tools
Useful data science tools in many aspects.

## Classification tools
### Handle unbalanced data. 
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


