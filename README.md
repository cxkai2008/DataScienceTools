# DATA SCIENCE TOOLS

I wrote these codes during my first research project at ICL, using them makes my project much easier to be done and some of them are general methods which can be widely applied to the other data science tasks. Thus, I'd like to share them here in the case that someone may find them useful. These tools are focused on the tasks in many aspects of data science, in order to make them easy to use, they are coded as separated functions in DStools.py and can be easily used by running this script. Examples can be found in the following content as below:

Please notice: These tools are developed as references of solution for different data science use cases. Many of these tools are built based on packages like pandas, numpy, sci-learn, bokeh etc. Thus, please refer to the licences of the specific packages when using these tools.

## DATA DISPLAY TOOLS
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
## DATA TRANSFORMATION TOOLS
### Generate the reordered features of samples based on the result of hierarchical clustering. 
This function is a part of plot_colorful_images_wrapper, however it can be used individually to return the reordered list.
* megadata_validation: the data frame which is the input of hierarchical clustering
* numeric_features_validation: the numeric features used for hierarchical clustering and they are reordered based the correlation coefficents.
* basic_info_features: the categorical features which also will be combined with reordered numeric features as the output of the function.
* simple_scale: use simple scaling or median scaling
```
generate_reordered_features(megadata_validation,numeric_features_validation,basic_info_features,simple_scale)
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
* tst: the input matrix
* isrow: scale by each row or col
* simple_scale: whether use the basic method or median_transform
```
scaled_matrix = scale_matrix(tst,isrow=True,simple_scale=True)
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
## SUPERVISED LEARNING
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
* folder_path: the path used for storing ".dot" files,e.g. '/abc/abc/'
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
* folder_path: the path used for storing ".dot" files,e.g. '/abc/abc/'
* label_list: the list of labels generated from DT_RF_models
```
full_feature_importance_RF_DF = generate_RF_feature_importance(folder_path,megadata_temp,['1','2','3','4'],'5')
T_full_feature_importance_RF_DF = transform_feature_importance(full_feature_importance_RF_DF,label_list) 
```
### Plot ROC curve and calculate AUC for each label. 
The input elements of this function are mainly generated by the function of DT_RF_models.
* full_test: the true label list of testset(represent by indices rather than names), output from DT_RF_models.
* full_predict: predicted scores for each label in the order of testset, output from DT_RF_models.
* label_list: readable unique labels ranking by characters, output from DT_RF_models.
* class_num: the label size, how many classes in the classification model.
* title: the title of the ROC curve
```
plot_precision_recall_curve(full_test,full_predict,label_list,class_num=3,title='test') 
```
### Print recall, precision of each label as well as the overall accuracy. 
The input elements of this function are mainly generated by the function of DT_RF_models.
* full_test: the true label list of testset(represent by indices rather than names), output from DT_RF_models.
* full_predict: predicted scores for each label in the order of testset, output from DT_RF_models.
* label_list: readable unique labels ranking by characters, output from DT_RF_models.
* class_num: the label size, how many classes in the classification model.
```
print_precision_recall_accuracy(full_test,full_predict,label_list,class_num=3)
```
### Print the details of incorrect predictions. 
Print the details of incorrect predictions using the full_wrong_list generated by the functions of DT_RF_models or xgboost_multi_classification.
* full_wrong_list: contains details of incorrect predictions generated by DT_RF_models or xgboost_multi_classification
```
print_full_wrong_list(full_wrong_list)
```
### Binary classification model of XGBoost algorithm with Venn-Abers predictors build-in. 
The training data and the testing data is separated, the output data of full_test, full_predict, label_list are similar to the ones generated by DT_RF_models and can be used in print_precision_recall_accuracy and plot_precision_recall_curve. test_DF contains the prediction scores.
* train: the training dataset with category column.
* test: the testing dataset with category column.
* selectedData_Indices: the feature set used for training and testing.
* label: the name of a certain class in the category column, which is used for prediction.
* category: a certain class of the categories, used for binary classification.
* num_round: the number of iteration in xgboost training.
```
test_DF,full_test,full_predict,label_list = xgboostModel_for_venn(train,test,selectedData_Indices,label = 'Control',category = 'Category',num_round = 100)
```
### multi-classification model of XGBoost algorithm built and test by cross validation. 
The input data is separated by the function of cross_validation_split_with_unbalance_data, the output data of full_test, full_predict, label_list are similar to the ones generated by DT_RF_models and can be used in print_precision_recall_accuracy and plot_precision_recall_curve. full_wrong_list contains details of incorrect predictions which can be print by print_full_wrong_list.
* input_df: the input dataset
* numeric_features_validation: the columns used to train the model, must be numeric
* iteration: the number of iterations of cross validation
* test_size: the proportion of input data used for testing
* max_depth: the max depth of the trees generated by algorithm
* num_class: the number of categories for the label column
* num_trees: the number of trees generated by the algorithm
* label_column: the name of the column used for label
* id_column: the name of the column used for id
* handle_unbalance: whether or not handling the unbalanced data in the function of cross_validation_split_with_unbalance_data
* readList: the columns which is recorded in the full_wrong_list.
```
accuracy,full_wrong_list,full_test,full_predict,label_list = xgboost_multi_classification(input_df= megadata_temp, numeric_features_validation=['1','2','3','4'], iteration=10, test_size=0.2, max_depth=2, num_class=4, num_trees=50, label_column='Cate', id_column='id', handle_unbalance=True, readList=['Cate','id'])
```
### combined model of XGBoost algorithm with Venn-Abers Predictors built-in. 
This is the most powerful classification model developed in my research project. It is comprised by binary XGBoost models with Venn-Abers Predictors built-in and a multi-label XGBoost model. This model can use separated training and testing dataset. The testing dataset should contain the label column, if there isn't, please add an empty column named the same as the label column in the training dataset. df_result is the prediction result which can be transformed by transform_predict_result_DF into more readable result.
* training_set: the training dataset
* numeric_features_validation: the columns used to train the model and to predict labels, must be numeric
* testing_set: the number of iterations of cross validation
* testing_set: the testing dataset which must contain the label column.
* max_depth: the max depth of the trees generated by algorithm
* num_class: the number of categories for the label column
* num_trees: the number of trees generated by the algorithm
* label_column: the name of the column used for label
```
df_result = combined_eXGBT_classifier(training_set = df_tr,numeric_features_validation = ['1','2','3','4'],testing_set = df_te,label_column = 'Cate',max_depth=2,num_class=4,num_trees=50)
```
### Generate the predicted labels with confidence using the output of combined model of XGBoost algorithm. 
This function transorm the prediction result of combined_eXGBT_classifier into more readable result, it also decide the final predicted label of the combined classifier using a pre-set threshold.
* predict_result_DF: the prediction result generated by combined_eXGBT_classifier
* label_col: the name of column used as label in predict_result_DF
* threshold: the threshold of confidence used for deciding whether the result of binary or multi label XGBoost model is used. If the confidence of a prediction made by binary XGboost model is less than the threshold, the prediction of multi label XGBoost model will be used.
```
full_result,clean_result = transform_predict_result_DF(predict_result_DF =df_result, label_col='Cate', threshold=0.1)
```
## UNSUPERVISED LEARNING
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
### Dimensionality reduction using tSNE with plotting. 
Output a dimensionality reduction dataframe using tSNE with a color plot.
* oriData: the input dataframe
* data_Indices: the columns used for tSNE, must be numeric
* read_list: the list of columns of oriData, which will be shown interactively in the tSNE plot.
* color_col: the column used for colouring, it is usually the label column.
* storing_loc: the location of the generated plot html file, should be end as '.html'.
* size_col: the column used for determining the size of dots in the plot, an int value is used as default.
* iters: the number of iterations for tSNE training
* perp: the perplexity of tSNE.
* title: the title of tSNE plot.
* num_components: the number of dimensions of the output dataset after tSNE transformation.
* tsne_df: the output dataset after dimensionality reduction using tSNE
```
tsne_df = tSNEPlot(oriData,data_Indices,read_list,color_col,storing_loc,size_col = 5, iters=1000, perp=2, title='tSNE',num_components=2)
```
## PLOTTING TOOLS
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
### Scatter plot with colours. 
Plot a scatter colour plot using a dataframe
* data_frame: the dataframe which stores the plotting data
* xvalue: the column of dataframe which is used as x axis
* yvalue: the column of dataframe which is used as y axis
* sizevalue: the column of dataframe which is used as the size of dots, int value is also available.
* outputFilePath: the path and file name of output file
* readList: the columns shown in the interactive plots when the mouse hover on a dots.
* plotWidth: the width of the plot
* plotHeight: the height of the plot
* titleName: the title of the plot
* colorColumn: the name of column which is used for colouring
* colorPattern: the color pattern used for plotting
```
plotColorScatter(DataFrame ,xvalue = '0',yvalue = '1', sizevalue = 'size', outputFilePath='/abc/test.html',plotWidth = 750, plotHeight = 750, readList = ['1','2'],titleName='tSNE', colorColumn="Category", colorPattern=viridis)
```
### A novel visualisation method used for dataset with 100 to 1000 features. 
Plot images separated by image_col using reordered features and indices. The features are reordered by hierarchical clustering algorithm, so the adjecent columns are highly correlated. Thus, in the global view, certain patterns can appear. The indices should be reordered too(so that highly correlated rows will be close to each other), however the reordering function is different due to the original dataset and it need to be code by users.
* megadata_temp1: the first layer of image
* megadata_temp2: the second layer of image
* megadata_temp3: the third layer of image
* numeric_cols: the numeric columns used for plot images
* image_col: the column used for separate images, the unique values in this column are also used as the file name of generated images.
* interpolation_row: the times of interpolation for rows, the size of rows is doubled by each time of interpolation.
* interpolation_col: the times of interpolation for columns, the size of columns is doubled by each time of interpolation.
* path: the folder used for storing the generated images
* simple_scale: use simple scaling or median scaling
* generate_reordered_indices: the function used for reorder the indices, take the megadata_temp1 and numeric_cols as input and output a list of reordered indices
```
plot_colorful_images_wrapper(megadata_temp1, megadata_temp2, megadata_temp3, numeric_cols, image_col, interpolation_row, interpolation_col, path, simple_scale=True, generate_reordered_indices=generate_levels_reordered_megadata_DF)
```
