# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 20:39:48 2018

@author: Ram
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import Imputer as IMP
from sklearn.model_selection import train_test_split as tts
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder as LBLENC
from sklearn.preprocessing import StandardScaler as SSCL
from sklearn.preprocessing import MinMaxScaler as MMSCL
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.linear_model import LogisticRegression as LREG
from sklearn import svm as SVM
from sklearn.metrics import classification_report as CREP
from sklearn.metrics import confusion_matrix as CMAT

###########################################################################################################
"""
Some Constants
"""

PLOTS = 'C:/Users/10646237/Desktop/a/cancer/Plots/'

cols1 = ['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig',
         'breast', 'breast-quad', 'irradiate']
cols2 = ['ID', 'Clump_Thickness', 'Uniformity_of_Cell_Size', 'Uniformity_of_Cell_Shape',
         'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin',
         'Normal_Nucleoli', 'Mitoses', 'Class']
cols3 = ['ID', 'Class', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']

###########################################################################################################


###########################################################################################################
"""
This function creates DataFrame of Dataset
"""

def OnCreateDF( fname, sep = ',', cols = None, index = None ):
    
    data = pd.read_csv( fname, sep = sep, header = None, names = cols, index_col = index )
    return data

###########################################################################################################


###########################################################################################################
"""
The following are functions dedicated for visualization.
"""

"""
OnPlotCat function is used to visualize categorical features. The function plots Bar graph of a single 
categorical feature and shows the frequency of the feature within each class of the dependent variable.

data = The DataFrame.
col1 = Independent Categorical Feature
col2 = Dependent Variable

"""
def OnPlotCat( data, col1, col2, sav = False, name = 'out' ):
    
    datav = data.groupby( col2 )[col1].value_counts().unstack(0)
    ax = datav.plot.barh()
    if sav:
        fig = ax.get_figure()
        fig.savefig( PLOTS + name + '.png', bbox_inches = 'tight', dpi = 1000 )



"""
OnPlotBox is used to create a Box Plot. The function creates a box plot of multiple numeric variables.

data = The DataFrame.
cols = List of columns names that will be represented by the plot.

"""
def OnPlotBox( data, cols, sav = False, name = 'out2' ):
    
    vect = []
    for c in data.columns:
        if c not in ['Class']:
            vect.append( data[c] )
    
    plt.boxplot( vect, patch_artist = True )
    plt.xticks( list(range(1, len(cols)+1)), cols )
    if sav:
        plt.savefig( PLOTS + name + '.png', bbox_inches = 'tight', dpi = 1000 )



"""
OnPlotBox is used to create a Box Plot. However, unlike OnPlotBox(), this function creates a BoxPlot
with classes of the dependent variable in the X-Axis allowing us to understand the distribution of
numeric variables within each category of the dependent variable.

data = The DataFrame.
cname = Name of the column of dependent variable.

"""
def OnPlotBoxClass( data, cname = 'Class' ):
    
    boxplot = data.boxplot( by = cname )
    plt.show()



"""
OnViolinPlot is used to create a Violin Plot of the given data.

data = The DataFrame.
class_col = Either "NaN" or the dependent variable column as a List.

"""
def OnViolinPlot(data, class_col = 'NaN'):
    
    if class_col != 'NaN':
        data['Class'] = class_col
    data = pd.melt( data, id_vars = 'Class', var_name = 'features', value_name = 'value' )
    plt.figure( figsize = (10,10) )
    
    sns.violinplot( x = 'features', y = 'value', hue = 'Class', data = data, split = True, inner = 'quart' )
    plt.xticks( rotation = 90 )



"""
OnSwarmpPlot is used to create a Swarmp plot. Swarmp Plots are helpful to understand how finely values of 
independent variables are distributed among the categories of dependent variable.

data = The DataFrame.
class_col = Either "NaN" or the dependent variable column as a List.

"""
def OnSwarmpPlot(data, class_col = 'NaN'):

    if class_col != 'NaN':
        data['Class'] = class_col
    data = pd.melt( data, id_vars = 'Class', var_name = 'features', value_name = 'value' )
    plt.figure( figsize = (10,10) )

    sns.swarmplot( x = 'features', y = 'value', hue = 'Class', data = data )
    plt.xticks( rotation = 90 )


"""
OnPlotHeatMap creates a Heat Map. The plot can be used for understanding the corelation among the 
variables that can help in Feature Selection.

data = The DataFrame.
class_col = Either "NaN" or the dependent variable column as a List.

"""
def OnPlotHeaMap(data):
    
    f, ax = plt.subplots( figsize = (18,18) )
    sns.heatmap( data = data.corr(), annot = True, linewidths = 0.5, fmt = '.1f' )
    

###########################################################################################################


###########################################################################################################
"""
The following functions are dedicated to processing the datasets.
"""


"""
OnLabelEncode is used to Label encode Categorical Features.

data = The DataFrame.
class_col = Either "all" or List of columns that must be Label Encoded.

"""
def OnLabelEncode( data, cols = 'all' ):
    
    lblenc = LBLENC();

    if cols == 'all':
        for c in data.columns:
            data[c] = lblenc.fit_transform( data[c] )
        return data
    else:
        for i in cols:
            data.iloc[:, i] = lblenc.fit_transform( data.iloc[:, i] )
        return data


"""
OnSetDummyVars is used to create Dummy Variables of Label Encoded Categorical Features.

data = The DataFrame.
class_col = Either "all" or List of columns that must be converted to Dummy Variables.

"""
def OnSetDummyVars( data, cols = 'all' ):
    
    if cols == 'all':
        for c in data.columns:
            data = pd.get_dummies( data,  prefix = [ c ], columns = [ c ] )

        return data
    else:
        col_names = [ data.columns[i] for i in cols ]
        for i in col_names:
            data = pd.get_dummies( data, prefix = [ i ], columns = [ i ] )

        return data

    return data



"""
OnReplaceValue function is used to handel missing attributes in the dataset either Categorical or Numeric.

data = The DataFrame.
nvalue = String that represents missing values in the dataset.
myval = Custom value to replace the missing value with. Must be 'NaN' if not required.
cols = Either 'all' or the List of columns that must be processed.
dtype = Must be 'num' if given columns represent Numeric features or 'cat' if they represent
        Categorical Features.
strat = Represents the replacement strategy for missing values like mean, max, min, etc.

"""
def OnReplaceValue( data, nvalue = '?', myval = 'NaN', cols = 'all', dtype = 'num', strat = 'mean' ):
    
    if dtype == 'cat':
        
        if cols == 'all':
            for c in data.columns:
                if myval == 'NaN':
                    data[c] = data[c].replace( nvalue, data[c].value_counts().idxmax() )
                else:
                    data[c] = data[c].replace( nvalue, myval )

            return data
        else:
            for i in cols:
                if myval == 'NaN':
                    data.iloc[:, i] = data.iloc[:, i].replace( nvalue, data.iloc[:, i].value_counts().idxmax() )
                else:
                    data[c] = data[c].replace( nvalue, myval )

            return data

    elif dtype == 'num':
        
        if myval == 'NaN':

            if cols == 'all':
                
                impu = IMP( missing_values = nvalue, strategy = strat, axis = 0 )
                impu = impu.fit( data2.iloc[:, :] )
                data.iloc[:, :] = impu.transform( data.iloc[:, :] )
                
                return data
            else:

                for i in cols:
                    impu = IMP( missing_values = nvalue, strategy = strat, axis = 0 )
                    impu = impu.fit( data2.iloc[:, i:i+1] )
                    data.iloc[:, i:i+1] = impu.transform( data.iloc[:, i:i+1] )
                    
                return data
        else:
            if cols == 'all':
                data = data.replace( nvalue, myval )

                return data
            else:
                for i in cols:
                    data.iloc[:, i] = data.iloc[:, i].replace( nvalue, myval )

                return data
    
    return data


"""
OnMinMaxScale is used to scale a DataFrame using Normalization.

data = The DataFrame.

"""
def OnMinMaxScale( data ):
    
    mmscl = MMSCL()
    return pd.DataFrame( mmscl.fit_transform(data) )


"""
OnStdScale is used to scale a DataFrame using Standardisation.

data = The DataFrame.

"""
def OnStdScale( data ):
    
    sscl = SSCL()
    return pd.DataFrame( sscl.fit_transform(data) )
    

"""
OnSplitData is used to split the dataset into Train/Test sets possessing equal propotion of all the
categories of the dependent variable.

data = The DataFrame.
t_size = Size of test set.

"""
def OnSplitData( data, t_size = 0.2 ):
    
    """
    Assuming the dependent variable has two categories, we first create two independent dataframes each
    containing data of only one of the categories of dependent variable.
    """
    data_c1 = data.loc[ data['Class'] == data['Class'].unique()[0] ]
    data_c2 = data.loc[ data['Class'] == data['Class'].unique()[1] ]

    """
    Next, we train_test_split each of the previouly created datset individually.
    """    
    c1_xtrain, c1_xtest, c1_ytrain, c1_ytest = tts( data_c1, data_c1['Class'], test_size = t_size, random_state = 100 )
    c2_xtrain, c2_xtest, c2_ytrain, c2_ytest = tts( data_c2, data_c2['Class'], test_size = t_size, random_state = 100 )
    
    """
    We concat the Train Sets and Test Sets of each categories obtained from previous step to a single
    Train and Test set.
    """
    xtrain = pd.concat( [c1_xtrain, c2_xtrain] )
    xtest = pd.concat( [c1_xtest, c2_xtest] )
    
    """
    Shuffle to maintain randomness.
    """
    xtrain = shuffle(xtrain)
    ytrain = xtrain['Class']
    xtest = shuffle(xtest)
    ytest = xtest['Class']
    
    return xtrain, xtest, ytrain, ytest



###########################################################################################################


###########################################################################################################
"""
OnRunModels is used to run Models on the dataset.
"""


def OnRunModels(data):
    
    xtrain, xtest, ytrain, ytest = OnSplitData( data )

    knn = KNN( n_neighbors = 10, metric = 'minkowski', p = 2 )
    svm = SVM.SVC( kernel = 'linear', C = 1.0, gamma = 'auto' )
    mnb = MNB()
    lreg = LREG( random_state = 0 )
    
    knn.fit( xtrain, ytrain )
    knn_preds = knn.predict( xtest )
    print(' KNN Results : ')
    print(' Classification Report')
    print( CREP( ytest, knn_preds ) )
    print(' Confusion Matrix')
    print( CMAT( ytest, knn_preds ), '\n\n' )
    
    svm.fit( xtrain, ytrain )
    svm_preds = svm.predict( xtest )
    print(' SVM Results : ')
    print(' Classification Report')
    print( CREP( ytest, svm_preds ) )
    print(' Confusion Matrix')
    print( CMAT( ytest, svm_preds ), '\n\n' )

    mnb.fit( xtrain, ytrain )
    mnb_preds = mnb.predict( xtest )
    print(' MNB Results : ')
    print(' Classification Report')
    print( CREP( ytest, mnb_preds ) )
    print(' Confusion Matrix')
    print( CMAT( ytest, mnb_preds ), '\n\n' )

    lreg.fit( xtrain, ytrain )
    lreg_preds = lreg.predict( xtest )
    print(' LREG Results : ')
    print(' Classification Report')
    print( CREP( ytest, lreg_preds ) )
    print(' Confusion Matrix')
    print( CMAT( ytest, lreg_preds ), '\n\n' )


###########################################################################################################


###########################################################################################################
"""
Read each of the datasets into seperate dataframes.
"""


data1 = OnCreateDF( 'C:/Users/10646237/Desktop/a/cancer/breast-cancer.csv', cols = cols1 )

data2 = OnCreateDF( 'C:/Users/10646237/Desktop/a/cancer/breast-cancer-wisconsin.csv', cols = cols2, index = 'ID' )

data3 = OnCreateDF( 'C:/Users/10646237/Desktop/a/cancer/wdbc.csv', cols = cols3, index = 'ID' )

###########################################################################################################


###########################################################################################################
"""
Dataset 1
"""


data1.head(5)
data1.columns

# Replace Missing Values.
data1 = OnReplaceValue( data1, dtype = 'cat' )

# Do visulization. 

# All the columns except deg-malig are categorical in the dataset. Hence we use OnPlotCat.
OnPlotCat(data1, 'age', 'Class')

# We use to OnBoxPlotClass to view the distribution of the values in the column within each category of
# the dependent variable.
OnPlotBoxClass( data1[['Class', 'deg-malig']] )

# All features look fine and equally important, hence there is no need to drop or modify any column.

# Beacuse deg-malig is the only numeric column in the data set and rest all are categorical we save it in a
# temproary variable. This will make it easire to use OnLabelEncode() and OnSetDummyVars() as we will not
# have to call the functions multiple times specifying the columns before and after 'deg-malig' which
# would have been the case if we processed otherwise.

temp_col = list( data1['deg-malig'] )
data1.drop( 'deg-malig', inplace = True, axis = 1 )

# Label Encode and Dummy Encode the categorical features. The second parameter of the OnDummyEncode()
# is a List of indices of the columns that must be Dummified starts with 1 to ensure that the dependent
# variable ie. "Class" is not dummified.
data1 = OnLabelEncode( data1 )
data1 = OnSetDummyVars( data1, list( range( 1, len(data1.columns) ) ) )

# Recreate the deg-malig column.
data1['deg-malig'] = temp_col

# Run models.
OnRunModels(data1)

###########################################################################################################


###########################################################################################################
"""
Dataset 2
"""


data2.head(5)
data2.columns


# Replace missing values. Notice that we replace the missing values that are represented by "?" by a custom
# number NaN. This is beacuse firstly all the indenpendent variable columns in the datset are numeric.
# Second, all the columns are ranged between 1 - 10.
# Third, because the missing value is represented by "?" the columns containing missing values are
# of type String. This prevents us from using standard missing value replacement strategies like mean, max.
# Hence we first replace the missing values with NaN which is numberic.
# Next we convert the columns to type numeric. Next we replace them using standard methods.

data2 = OnReplaceValue( data2, myval = np.nan )
data2[data2.columns[5]] = data2[data2.columns[5]].apply(pd.to_numeric)
data2 = OnReplaceValue( data2, nvalue = np.nan, cols = [5] )

# Do Visulization.
OnPlotBox(data2, data2.columns[:9])
OnPlotBoxClass( data2 )
OnViolinPlot(data2)
OnSwarmpPlot(data2)
OnPlotHeaMap(data2)

# From above visulization especially Violin and Swarmp Plot, it can be seen that Mitoses is not very
# cleanly distributed. Hencce, we can drop if required.
data2.drop( 'Mitoses', inplace = True, axis = 1 )

# Run Models.
OnRunModels(data2)

###########################################################################################################


###########################################################################################################
"""
Dataset 3
"""


data3.head(5)
data3.columns

# Replace the vaues of the Class ( Diagnois ) column with numeric data. Not required though.
data3['Class'] = data3['Class'].replace( 'B', 2 )
data3['Class'] = data3['Class'].replace( 'M', 4 )

# Save the dependent variable column in a temproary variable for later use and drop it from the dataframe.
# This will help us in using the plotting functions defined above.
data3_class = list( data3['Class'] )
data3.drop( 'Class', inplace = True, axis = 1 )

# Use a feature scaling function.
data3 = OnMinMaxScale( data3 )
data3.columns = cols3[2:]

# Do Visualization.
# Because there are 30 columns in the dataset and we know that they are just variations the actual 10
# data, in groups of 10 we plot the dataset in groups of 10. This makes it easy to visualize.

OnPlotBoxClass( pd.concat( [data3.iloc[:, 0:10], pd.Series(data3_class)], axis = 1 ), cname = 0 )
OnPlotBoxClass( pd.concat( [data3.iloc[:, 10:20], pd.Series(data3_class)], axis = 1 ), cname = 0 )
OnPlotBoxClass( pd.concat( [data3.iloc[:, 20:], pd.Series(data3_class)], axis = 1 ), cname = 0 )

OnPlotHeaMap( pd.concat( [data3, pd.Series(data3_class)], axis = 1 ) )
OnPlotHeaMap(data3.iloc[:, 0:10])
OnPlotHeaMap(data3.iloc[:, 10:20])
OnPlotHeaMap(data3.iloc[:, 20:])

OnPlotBox(data3.iloc[:, 0:10], cols = data3.columns[0:10])
OnPlotBox(data3.iloc[:, 10:20], cols = data3.columns[10:20])
OnPlotBox(data3.iloc[:, 20:], cols = data3.columns[20:])

OnViolinPlot(data3.iloc[:, 0:10], data3_class)
OnViolinPlot(data3.iloc[:, 10:20], data3_class)
OnViolinPlot(data3.iloc[:, 20:], data3_class)

OnSwarmpPlot(data3.iloc[:, 0:10], data3_class)
OnSwarmpPlot(data3.iloc[:, 10:20], data3_class)
OnSwarmpPlot(data3.iloc[:, 20:], data3_class)

# drop_cols contains the names of the columns that will be droped from the dataset based on what we have
# conculded from the visualization.

# Beacuse there are 30 columns which are in 3 groups of 10 similar data just mathematical variations
# ( mean, standard error and worst ) we know that strong correlation must exist between these features.
# And the best way to understand these correlations is using the good old HeatMap. We try to eliminate
# columns with high correlations. For example, from the heat map we can see that 'area_mean',
# 'radius_mean' and 'perimeter_mean' are correlated. We drop 'radius_mean' and 'perimeter_mean' and go
# with area_mean. The reason for going with area_mean can be understood by viewing the Voilin Plot and
# Swarmp Plot of the mean ( first 10 ) features of the dataset. We see that area_mean is musch better
# distributed as compated to the other two. Hence we go with area_mean. 
# The same procedure goes for selecting a feature from other correlated ones like 'compactness_mean',
# 'concave points_mean' and 'concavity_mean' where we go with 'concavity_mean', and so on.

# Based on the visualization several combinations of features can be come up with. We decide to go with
# the below configuration.

drop_cols = ['radius_mean', 'perimeter_mean', 'compactness_mean', 'concave points_mean',
             'radius_se', 'perimeter_se', 'compactness_se', 'concave points_se',
             'radius_worst', 'area_worst', 'perimeter_worst', 'texture_worst',
             'compactness_worst', 'concave points_worst']


# Drop selected columns.

data3_new = data3.copy(deep = False)
data3_new.drop( drop_cols, inplace = True, axis = 1 )

OnPlotHeaMap( pd.concat( [data3_new, pd.Series(data3_class)], axis = 1 ) )

OnSwarmpPlot(data3_new.iloc[:, 0:6], data3_class)
OnSwarmpPlot(data3_new.iloc[:, 6:11], data3_class)
OnSwarmpPlot(data3_new.iloc[:, 12:], data3_class)

# Run Model.
data3_new['Class'] = data3_class
OnRunModels(data3_new)

###########################################################################################################

"""

Merging of datasets on Patient ID does not seem possible. This is because the first dataset lacks the
information.
While dataset 2 and 3 possess the Patient ID column there is not a single entry in the column of the 2 
dataset that overlap/are similar, every single Patient ID in both the set is unique. Hence merging
the dataset would simply mean concating the two frames and for every Patient ID of one dataset 
nullifying the values of all the columns of the other dataset.

"""
