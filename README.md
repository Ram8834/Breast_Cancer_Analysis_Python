# Breast_Cancer_Analysis_Python
Breast cancer predictive analysis using python

Description on the Cancer_Final.py code
Location of the plot folder needs to be given, to save plot output. Code has been created using spyder

Project Folder structure:
~F- Drive
--Project 1
--Data1 (https://archive.ics.uci.edu/ml/datasets/Breast+Cancer)
--Data2 (Breast Cancer Wisconsin (Original) Data Set)
--Data3 (Breast Cancer Wisconsin (Diagnostic) Data Set)
--Plots

The complete code has been worked out using custom defined functions:
1. def OnCreateDF: Function defined for creating dataframe, function takes two arguments: 1st- location of the interested file, 2nd: list of column names

2. def OnPlotCat: Function defined for creating plot for categorical/ range attributes, function takes three arguments: 1st- dataframe, 2nd- categorical/range column name, 3rd- class (dependent) column name
Output
I have use groupby function followed by bar chart plot for categorical attributes representation and analysis.
Age independent of menstruation information (start and end of menstrual cycle) is not an important predictor of breast cancer risk. As, we can see all age range has high proportion of non-recurrence-event. Also my pathway/literature analysis document (Pathway_Analysis.doc) points on the same thing.
At the same time, patient with the age older than 45 and late onset of menopause have higher risk of breast and ovarian cancer, due to more exposure of estrogen. From our barchart output, we can see patient in the range 40+ have higher rate of recurrence-event compared to lower age group.
From research article:

3. def OnPlotBoxClass: Function defined for creating boxplot for feature ~ class continuous features, function takes two arguments: 1st- dataframe, 2nd- class (dependent) column name.
Output
Dataset2:
This so beautifully explains: high values of Clump thickness, Cell shape and size uniformity, Epithelial cell size box plot is prognostic/ deterministic factor of Malignant tumor. Also, we can see the above-mentioned features upper whisker and upper quartile touching each other, signifying point concentrating at the higher value. Also, my pathway/literature analysis document (Pathway_Analysis.doc) points on the same thing.
Contradiction: loss of tumor adhesion is a feature of malignant cancer, since loss of adhesion is required for tumor metastasis.
Low value should have been the prognostic factor for malignant tumor, while high value of adhesion should have been prognostic factor of benign tumor.
Class: 2- benign; 4- Malignant
Dataset 1:
Histology grade (deg- malig): Tumor grading deals with understanding of the tumor cell shape, spread, stage, abnormality, undifferentiated (lack normal tissue structures).
Grade 1 & 2- Normal growing cell; Grade 3 & 4- Rapidly growing and spreading cell.
High values of tumor grading (deg-malig) box plot are prognostic/ deterministic factor of recurrence event (can be linked with recurrence event based on biology). Also we can see the above mentioned features upper whisker and upper quartile touching each other, signifying point concentrating at the higher value. Also my pathway/literature analysis document (Pathway_Analysis.doc) points on the same thing.
Class 0- non recurrence event; 1 – recurrence event
Dataset 3
High values of texture, area, symmetry worse box plot are prognostic/ deterministic factor of malignant tumor. Also my pathway/literature analysis document (Pathway_Analysis.doc) points on the same thing.
Class: 4- Malignant; 2- Benign

4. def OnPlotBox: Function defined for creating plot for categorical/ range attributes, function takes three arguments: 1st- dataframe, 2nd- numeric column name
Output
Help us to understand how each feature is distributed, number of outliers etc.

5. def OnViolinPlot: OnViolinPlot is used to create a Violin Plot of the given data. Function takes two arguments, 1st- data = The DataFrame, 2nd - class_col = Either "NaN" or the dependent variable column as a List.

6. def OnSwarmpPlot: OnSwarmpPlot is used to create a Swarmp plot. Swarmp Plots are helpful to understand how finely values of independent variables are distributed among the categories of dependent variable. Function takes two arguments, 1st- data = The DataFrame, 2nd - class_col = Either "NaN" or the dependent variable column as a List.
Output
From above visualization especially Violin and Swarmp Plot, it can be seen that symmetry, fractal dimension, smoothness is not very cleanly distributed. While perimeter, area, radius, concave points are clearly distributed. Hence, we can think of dropping the feature if required.
I ran the model without dropping the variable and with dropping the variable. I was not able to see any difference in the accuracy of the models.

7. def OnPlotHeaMap: OnPlotHeaMap creates a Heat Map. The plot can be used for understanding the corelation among the variables that can help in Feature Selection. Function takes two arguments, 1st- data = The DataFrame, 2nd - class_col = Either "NaN" or the dependent variable column as a List.
Output
Dataset 2:
From above it can be seen that all the features except mitosis are strongly correlated.
Contradiction: Mitosis is a feature of malignant cancer (from molecular biology), hence class and mitosis must have high correlation.
Dataset3
From above it can be seen that radius, perimeter, area is strongly correlated with class variable, as radius, perimeter, and area are important factor in determining the morphology of cancer tissue and cell.
Contradiction: texture is a feature of malignant cancer (from molecular biology), hence class and texture must have high correlation.

8. def OnLabelEncode: OnLabelEncode is used to Label encode Categorical Features. Function takes two arguments, 1st- data = The DataFrame, 2nd - class_col = Either "NaN" or the dependent variable column as a List.

9. def OnSetDummyVars: OnSetDummyVars is used to create Dummy Variables of Label Encoded Categorical Features. Function takes two arguments, 1st- data = The DataFrame, 2nd - class_col = Either "NaN" or the dependent variable column as a List.
Points:
1. Label Encode and Dummy Encode the categorical features. The second parameter of the OnDummyEncode()
2. is a List of indices of the columns that must be Dummified starts with 1 to ensure that the dependent variable ie. "Class" is not dummified.

10. def OnReplaceValue: OnReplaceValue function is used to handel missing attributes in the dataset either Categorical or Numeric.
Input Parameters:
data = The DataFrame.
nvalue = String that represents missing values in the dataset.
myval = Custom value to replace the missing value with. Must be 'NaN' if not required.
cols = Either 'all' or the List of columns that must be processed.
dtype = Must be 'num' if given columns represent Numeric features or 'cat' if they
represent Categorical Features.
strat = Represents the replacement strategy for missing values like mean, max, min, etc.
Replacing strategy in dataset2

11. def OnMinMaxScale: OnMinMaxScale function is used to scale a DataFrame using Normalization.
Input Parameters:
data = The DataFrame.

12. def OnStdScale: OnStdScale function is used to scale a DataFrame using Normalization.
Input Parameters:
data = The DataFrame.

13. def OnSplitData: OnSplitData is used to split the dataset into Train/Test sets possessing equal propotion of all the categories of the dependent variable
Input Parameters:
data = The DataFrame.
t_size = Size of test set.


Strategy in the function:
1. Assuming the dependent variable has two categories, we first create two independent dataframes each containing data of only one of the categories of dependent variable.
2. Next, we train_test_split each of the previously created dataset individually.
3. We concat the Train Sets and Test Sets of each categories obtained from previous step to a single Train and Test set.
4. Shuffle to maintain randomness.
14. def OnRunModels: OnRunModels function is used to run KNN. Naïve Bayes, SVM and logistic regression model at one go.
Result:
1. I have used confusion matrix or classification report which calculates (accuracy, sensitivity, recall) for the given model.
2. I have identified all the four model gives allmost same accuracy across three dataset. Of which SVM and logistic giving better accuracy then naïve bayes and KNN
3. I have model twice with all variables and after droping variables showing low correlation based on plots, correlation matrix and some features which is not important for cancer. The model accuracy do not change much.
__________
SVM performed better in most of the dataset, possibly svm can be used in linear or non-linear ways with the use of a Kernel, when you have a limited set of points in many dimensions SVM tends to be very good because it should be able to find the linear separation that should exist. SVM is good with outliers as it will only use the most relevant points to find a linear separation (support vectors).
KNN is also very sensitive to bad features (attributes) so feature selection is also important. KNN is also sensitive to outliers, impacting the knn results.
__________
Some features which are having poor correlation (i.e. texture), while some having high correlation in the datasets (adhesion), as mentioned above. Are contradicting with the biology concept.
___________
Merging of datasets on Patient ID does not seem possible. This is because the first dataset lacks the information. While dataset 2 and 3 possess the Patient ID column there is not a single entry in the column of the 2 dataset that overlap/are similar, every single Patient ID in both the set is unique. Hence merging the dataset would simply mean concating the two frames and for every Patient ID of one dataset nullifying the values of all the columns of the other dataset.
____________
Converting the current model into product would require:
1. Combining all the three dataset features into one, which I was not able to perform since the patient ID was not common across.
2. Working more into feature engg part: 1.e. converting all categorical/range variable into numeric and using it in model development:
Age1 = substr(data1$age, 1,2)
Age1 = substr(data1$age, 4,5)
Age_new = Age1 + Age2 # converting range into numeric.
3. Using Advance machine learning algorithms: Adaboost, Xgboost, Ripper etc.
4. Using approach like LDA and then performing machine learning analysis on the
Outcome of the same.
5. The feature which are turning out to be poor correlated in the dataset, while the same feature is important prognostic factor in oncology, understanding the reason behind it.
_____________
