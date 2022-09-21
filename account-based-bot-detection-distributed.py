# required libraries
import numpy as np
import math

from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp
from pyspark.sql.functions import col, udf, to_timestamp, lit
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import when, rand
from pyspark.ml.feature import Normalizer, StandardScaler, MinMaxScaler, VectorAssembler

import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, LSTM, Dense, Input, concatenate, Dropout, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD, Adam

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy

#dataset path
dataset_folder_s3 = 'data/' # 's3://bot-dataset/data/'
result_path_s3 = '' # 's3://bot-dataset/result/'

## convert into feature vector for ml model
feature_columns = ['age', 'has_location', 'is_verified', 'total_tweets', 'total_following', 
                   'total_followers', 'total_likes', 'has_avatar', 'has_background', 
                   'is_protected', 'profile_modified']

# initialize the number of epochs to train for, batch size, and
# initial learning rate
EPOCHS = 25
BS = 64
INIT_LR = 1e-3

mse = MeanSquaredError()
cce = CategoricalCrossentropy()
opt = Adam(learning_rate=INIT_LR)

# read dataset from csv
def read_dataset(spark):
    requiredColumns = ['screen_name', 'created_at', 'updated', 'location', 'verified', 'statuses_count', 'friends_count','followers_count', 'favourites_count', 'default_profile_image', 'profile_use_background_image', 'protected', 'default_profile']

    bot_accounts1 = spark.read.csv(dataset_folder_s3 + 'social_spambots_1.csv', header = True, inferSchema = True).select(requiredColumns)
    bot_accounts2 = spark.read.csv(dataset_folder_s3 + 'social_spambots_2.csv', header = True, inferSchema = True).select(requiredColumns)
    bot_accounts3 = spark.read.csv(dataset_folder_s3 + 'social_spambots_3.csv', header = True, inferSchema = True).select(requiredColumns)

    # combine multiple bot_account dataset
    bot_accounts = bot_accounts1.union(bot_accounts2.union(bot_accounts3))
    clean_accounts = spark.read.csv(dataset_folder_s3 + 'geniune_accounts.csv', header = True, inferSchema = True).select(requiredColumns)
    
    return bot_accounts, clean_accounts

# clean dataset
def cleanData(df):
    df = df.withColumn('age', lit(0)) # need to calculate from 'updated' -'created_at'
    df = df.withColumn('has_location', when((df['location'] != None), 1).otherwise(0))
    df = df.withColumn('has_avatar', when((df['default_profile_image'] != None), 1).otherwise(0))
    df = df.withColumn('has_background', when((df['profile_use_background_image'] != None), 1).otherwise(0))
    df = df.withColumn('is_verified', when((df['verified'] != None), 1).otherwise(0))
    df = df.withColumn('is_protected', when((df['protected'] != None), 1).otherwise(0))
    df = df.withColumn('profile_modified', when((df['default_profile'] != None), 1).otherwise(0))
    df = df.withColumnRenamed("screen_name", "username")
    df = df.withColumnRenamed("statuses_count", "total_tweets")
    df = df.withColumnRenamed("friends_count", "total_following")
    df = df.withColumnRenamed("followers_count", "total_followers")
    df = df.withColumnRenamed("favourites_count", "total_likes")
    
    return df.select('username', 'age', 'has_location', 'is_verified', 'total_tweets', 'total_following', 'total_followers', 'total_likes', 'has_avatar', 'has_background', 'is_protected', 'profile_modified')



def doDataScaling(df, input_column, output_column):
    ## Make data standard
    # https://spark.apache.org/docs/1.4.1/ml-features.html#standardscaler

    scaler = StandardScaler(inputCol=input_column, outputCol=output_column,
                            withStd=True, withMean=False)

    # Compute summary statistics by fitting the StandardScaler
    scalerModel = scaler.fit(df)

    # Normalize each feature to have unit standard deviation.
    scaled_df = scalerModel.transform(df)
    
    return scaled_df


def datasetSplit(X):
    # split data for training ana testing
    train_df, test_df = X.randomSplit([0.80, 0.20])

    # features --> 'BotOrNot'
    X_train = train_df.drop('BotOrNot')
    y_train = train_df.select('BotOrNot')
    X_test = test_df.drop('BotOrNot')
    y_test = test_df.select('BotOrNot')
    
    return X_train, y_train, X_test, y_test



## create model
def getDLModel():
    model = Sequential()
    model.add(Dense(500, input_dim=11))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    
    return model


# convert DataFrame column into nparray
# nparray required for model training, validation

def to_nparray_list(df, column_name):
    rows = df.select(column_name).collect()
    lists = [x[column_name] for x in rows]
    nparr = np.array(lists)
    
    return nparr


def splitDataset(n_split, X, Y):
    for train_index,test_index in KFold(n_split).split(X):

        x_train, x_test=X[train_index],X[test_index]
        y_train, y_test=Y[train_index],Y[test_index]
        print( "train: {},{} test: {},{}".format(len(x_train), len(y_train), len(x_test), len(y_test)))
        

def distributedTrainingGradients(df, feature_column, target_column, n_splits):
    print(df.count())
    each_len = df.count() // n_splits
    gradients = []
    ##split dataset into 'n_splits' part
    copy_df = df
    for i in range(n_splits):
        print(i)
        temp_df = copy_df.limit(each_len)
        copy_df = copy_df.subtract(temp_df)
        
        X = temp_df.select(feature_column)
        Y = temp_df.select(target_column)
        X_np = to_nparray_list(X, feature_column)
        Y_np = to_nparray_list(Y, target_column)
        
        grad = step(X_np, Y_np)
        gradients.append(grad)
        print(temp_df.count())
        
    return gradients

    

def step(X, y):
    print("Input count: {}, {}".format(len(X), len(y)))
    #keep track of gradients
    with tf.GradientTape() as tape:
        curr_model = getDLModel()
        #make a prediction using model
        predict = curr_model(X)
        #calculate loss
        loss = mse(y, predict)
    #calculate the gradient
    gradient = tape.gradient(loss, curr_model.trainable_variables)
    
    # return the gradient to train final model
    return gradient

def stepEPOCH(X, y):
    with tf.GradientTape() as tape:
        curr_model = getDLModel()
        for i in range(EPOCHS):
            #make a prediction using model
            predict = curr_model(X)
            #calculate loss
            loss = cce(y, predict)
            print("{}: {}".format(i, loss))
            opt.apply_gradients(zip(grad, curr_model.trainable_variables))
            
    gradient = tape.gradient(loss, curr_model.trainable_variables)
    
    return gradient
    
    
        
        

def saveModel(model):
    model.save(result_path_s3 + 'my_model.h5')

    

def app_distributed():
    # init spark
    spark = SparkSession.builder.appName('ml_account_ base_session').getOrCreate()
    
    bot_accounts, clean_accounts = read_dataset(spark)
    
    bot_accounts = cleanData(bot_accounts)
    clean_accounts = cleanData(clean_accounts)
    
    ## add BotOrNot column
    bot_accounts = bot_accounts.withColumn('BotOrNot', lit(1))
    clean_accounts = clean_accounts.withColumn('BotOrNot', lit(0))
    
    #combine clean and bot accounts data togather
    combined_df = bot_accounts.union(clean_accounts)

    # shuffle dataset
    new_df = combined_df.orderBy(rand())

    #remove 'userrname' columns from dataset
    new_df = new_df.drop('username')
    
    feature_assembler = VectorAssembler(inputCols = feature_columns, outputCol = 'independent_features')
    df_updated = feature_assembler.transform(new_df)

    # keep only required features/columns
    df_updated = df_updated.select('independent_features', 'BotOrNot')
    
    scaled_df = doDataScaling(df_updated, "independent_features", "scaled_features")
    
    
    # keep only necessary feature/column for ml model
    XY = scaled_df.select('scaled_features', 'BotOrNot')
    
    train_df, test_df = XY.randomSplit([0.80, 0.20])
    
    gradients = distributedTrainingGradients(train_df, 'scaled_features', 'BotOrNot', 5)
    
    print("[INFO] creating model...")
    model = getDLModel()
    
    for grad in gradients:
        opt.apply_gradients(zip(grad, model.trainable_variables))
        
        
    # in order to calculate accuracy using Keras' functions we first need
    # to compile the model
    model.compile(optimizer= opt, loss=cce, metrics=["acc"])
    
    
    # DataFrame(column) --> nparray
    X_test = to_nparray_list(test_df, 'scaled_features')
    y_test = to_nparray_list(test_df, 'BotOrNot')
    
    # now that the model is compiled we can compute the accuracy
    (loss, acc) = model.evaluate(X_test, y_test)
    print("[INFO] test accuracy: {:.4f}".format(acc))
    
#     saveModel(model)

    
# if __name__ == '__main__':
#     app_distributed()

    