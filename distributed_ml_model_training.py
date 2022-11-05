#libraries / dependencies
import glob
from preprocessor import api as tweet_preprocessor
from storagehelper.HDFSHelper import HDFSHelper
from storagehelper.LocalFSHelper import LocalFSHelper

import numpy as np
import math
import random

from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.context import SparkContext

from pyspark.sql.functions import col, udf, to_timestamp, lit, to_timestamp, when, rand
from pyspark.sql.types import IntegerType, LongType, DoubleType, StringType, ArrayType
from pyspark.ml.feature import Normalizer, StandardScaler, MinMaxScaler, VectorAssembler

from pyspark import StorageLevel
from pyspark.accumulators import AccumulatorParam

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Embedding, Dense, LSTM, Dense, Input, concatenate, Dropout, Activation, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD, Adam

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy

import os
import subprocess

import sys
import shutil # sutil.rmtree(dir_path)


#thread depencency
from pyspark import InheritableThread
from multiprocessing.pool import ThreadPool
from threading import Semaphore
import time

'''
import findspark

findspark.init()
findspark.find()
'''


# initialize the number of epochs to train for, batch size, and
# initial learning rate
EPOCHS = 25
BS = 64
INIT_LR = 1e-3

mse = MeanSquaredError()
cce = CategoricalCrossentropy()
opt = Adam(learning_rate=INIT_LR)


LOCAL_USERNAME = 'hadoop'
HADOOP_USERNAME = 'hadoop'

#user info
SYSTEM_NAME = {'local':'home','server':'user'}
USER_NAME = {'local':'hadoop','server':'hadoop'} # {'local':'hadoop','server':'hadoop'}
WEIGHT_FOLDER_NAME = {'local':'weights_worker_id','server':'weights'}
GRADIENT_FOLDER_NAME = {'local':'gradients_worker_id','server':'gradients'}
FILE_NAME_PREF = {'gradients': 'gradient','weights': 'weight'}

fs_semaphore = Semaphore()


#Dataset location

#Local
#bot_tweets_dataset_path = 'file:///home/ubuntu/Documents/tweet-dataset/tweet_dataset_full/bot_tweets/' #'F://TwitterBotDataset//tweet_dataset_full//bot_tweets//' 
#genuine_tweets_dataset_path = 'file:///home/ubuntu/Documents/tweet-dataset/tweet_dataset_full/genuine_tweets/' #'F://TwitterBotDataset//tweet_dataset_full//genuine_tweets//'

#S3
bot_tweets_dataset_path = 's3://tweet-dataset/bot_tweets' #'F://TwitterBotDataset//tweet_dataset_small//bot_tweets//'
genuine_tweets_dataset_path = 's3://tweet-dataset/genuine_tweets' #'F://TwitterBotDataset//tweet_dataset_small//genuine_tweets//'

#turn a line of text into d dimentional vector 
GLOVE_DIR = ""

#all columns
BOT_COLUMNS = ['id','text','source','user_id','truncated','in_reply_to_status_id', 
               'in_reply_to_user_id','in_reply_to_screen_name', 'retweeted_status_id',
               'geo','place','contributors','retweet_count', 'reply_count','favorite_count',
               'favorited', 'retweeted','possibly_sensitive','num_hashtags','num_urls',
               'num_mentions','created_at','timestamp','crawled_at', 'updated']

GENUINE_COLUMNS = ['id','text','source','user_id','truncated','in_reply_to_status_id', 
                   'in_reply_to_user_id','in_reply_to_screen_name', 'retweeted_status_id',
                   'geo','place','contributors','retweet_count', 'reply_count','favorite_count',
                   'favorited', 'retweeted','possibly_sensitive','num_hashtags','num_urls',
                   'num_mentions','REMOVE_IT', 'created_at','timestamp','crawled_at', 'updated',]

#feature used for bot detection
COLUMN_NAMES = ['text', 'retweet_count', 'reply_count', 'favorite_count',
                'num_hashtags', 'num_urls', 'num_mentions']



# #configure spark
# conf = SparkConf()
# conf.setMaster("local[8]").setAppName("ml_account_ base_session")
# conf.set("spark.executor.instances", 4)
# conf.set("spark.executor.cores", 4)
# conf.set("spark.driver.memory", 4)
# sc = SparkContext(conf=conf)

# # for spark-submit
spark = SparkSession.builder.appName('ml_account_ base_session').getOrCreate()
# spark
sc = spark.sparkContext

# for local build
# spark = SparkSession.builder.appName('ml_account_ base_session').getOrCreate()


#for local multi thread
'''
conf = SparkConf()
conf.setMaster("local[10]").setAppName("distributed_training_session")
conf.set("spark.scheduler.mode", "FAIR")
sc = SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()
'''




localFS = LocalFSHelper(LOCAL_USERNAME)
hdfs = HDFSHelper(sc, localFS, LOCAL_USERNAME,HADOOP_USERNAME)

'''
SYSTEM_NAME = {'local':'home','server':'user'}
USER_NAME = {'local':'ubuntu','server':'ubuntu'} # {'local':'hadoop','server':'hadoop'}
WEIGHT_FOLDER_NAME = {'local':'weights_worker_id','server':'weights'}
GRADIENT_FOLDER_NAME = {'local':'gradients_worker_id','server':'gradients'}
FILE_NAME_PREF = {'gradients': 'gradient','weights': 'weight'}
'''
#cash these value in all nodes after single use
broadcastUserName = sc.broadcast(USER_NAME)
broadcastWeightFolderName = sc.broadcast(WEIGHT_FOLDER_NAME)
broadcastGradientFolderName = sc.broadcast(GRADIENT_FOLDER_NAME)
broadcastFileNamePref = sc.broadcast(FILE_NAME_PREF)

# read dataset from csv
def read_dataset():
    bot_tweets = spark.read.csv(bot_tweets_dataset_path, header = True, inferSchema = True).limit(100000)
    genuine_tweets = spark.read.csv(genuine_tweets_dataset_path, header = True, inferSchema = True).limit(100000)
    
#     bot_tweets = bot_tweets.persist(StorageLevel.MEMORY_ONLY)
#     genuine_tweets = genuine_tweets.persist(StorageLevel.MEMORY_ONLY)
    return bot_tweets, genuine_tweets

def set_column_name(df, column_names):
    df = df.toDF(*column_names)
    return df

def remove_column_miss_match(df):
    ## dataset have diffrent number of columns
    ## column name of dataframe
    column_name = [cname for cname, tp in df.dtypes]
#     len(df.collect()), len(df.dtypes)
    #column_name

    #Number of column is diffrent for bot and genuine tweets data

    #genuine_tweets_df = genuine_tweets_df.toDF(*column_name)
    df = set_column_name(df, GENUINE_COLUMNS)
#     print(len(df.collect()))
    
    df = df.drop('REMOVE_IT') # remove 5th column from end
    #update column name according to 
    df = set_column_name(df, BOT_COLUMNS)
#     print(len(df.collect()))
    return df


def remove_type_miss_match(df):
    # Same column has diffrent data type. So make data type same for every column
    updated_df = df.withColumn("id",col("id").cast(IntegerType())) \
                                    .withColumn("favorite_count",col("favorite_count").cast(LongType())) \
                                    .withColumn("favorited",col("favorited").cast(IntegerType()))
    return updated_df


def resize_combine_data(bot_tweets_df, genuine_tweets_df):
    ## only keep the required column from the dataframe
    bot_tweets_df = bot_tweets_df.select(*COLUMN_NAMES)
    genuine_tweets_df = genuine_tweets_df.select(*COLUMN_NAMES)
    
#     print(len(bot_tweets_df.collect()), len(genuine_tweets_df.collect()))

    ## add BotOrNot column
    bot_tweets_df = bot_tweets_df.withColumn('BotOrNot', lit(1))
    genuine_tweets_df = genuine_tweets_df.withColumn('BotOrNot', lit(0))

    #combine clean and bot accounts data togather
    tweets_df = bot_tweets_df.union(genuine_tweets_df)

    # shuffle dataset
    tweets_df = tweets_df.orderBy(rand())

#     print(len(tweets_df.collect()))
    
    return tweets_df

text_process_udf = udf(lambda x : tweet_preprocessor.tokenize(x), StringType())
def preprocess_data(df):
    df = df.withColumn('text', text_process_udf(df.text))
    df = df.withColumn("retweet_count",col("retweet_count").cast(DoubleType()))
    df = df.withColumn("reply_count",col("reply_count").cast(DoubleType()))
    df = df.withColumn("favorite_count",col("favorite_count").cast(DoubleType()))
    df = df.withColumn("num_hashtags",col("num_hashtags").cast(DoubleType()))
    df = df.withColumn("num_urls",col("num_urls").cast(DoubleType()))
    df = df.withColumn("num_mentions",col("num_mentions").cast(DoubleType()))
    
    return df

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


def makeGloveWordEmbedder(glove_path):
    embedding_dict = {}
    with open(glove_path, 'r', encoding="utf-8") as glove_file:
        for line in glove_file:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embedding_dict[word] = vector
            
    return embedding_dict  


# # Test GLoVE result
# glove_word2vec_embedder["google"]

 ##create word embedding GLoVE model dictionary. Use pre trained model
text_feature_dimention = 25
# glove_word2vec_embedder = makeGloveWordEmbedder(GLOVE_DIR + "glove.twitter.27B.25d.txt")


#Give a word and get that word representing feature vector of dimention 25 from embedding dictionary
def word2vec(word_dict=None, word=None, dim=25):
    default_vector = np.zeros(dim)
    
    if word_dict is None or word is None:
        return default_vector
    
    word_vector = word_dict.get(word)
    
    if word_vector is None:
        return default_vector
    return word_vector

# # test a word representing feature vector
# word_vector = word2vec(glove_word2vec_embedder, "tweet", text_feature_dimention)
# print(type(word_vector), word_vector)


#----------------LSTM Model----------------------

#create a 1 layer LSTM model
#input dimention: 3d (1,7,25) input_sample = [[[1,...,25]],..,[1,...,25]]
#output dimention: 1d (1,32) output_sample [[1,2,3,,,,32]]
def lstm_model(output_dim):
    model = LSTM(output_dim, return_sequences=False, return_state=False)
    return model

def reset_lstm_model(model):
    model.resate_states() # stateful=True is required for reset states

# create LSTM model of output dimention 32. Model output feature vector will be of 32 dimention vector
lstm = lstm_model(32) 

# convver a sentence to a feature vector using LSTM(RNN) model
def sent2vec(sent):
    words = sent.split(' ')
    word_vectors = np.array([])
    count = 0;
    for word in words:
        word_vector = word2vec(broadcast_glove_dict.value, word)
#         print("word dim: {}".format(len(word_vector)))
        if word_vectors.size == 0:
            word_vectors = np.array([word_vector])
        else:
            word_vectors = np.vstack([word_vectors, word_vector])
        count = count + 1
    
#     print("Input feature vector shape before reshape(2D): {}".format(word_vectors.shape))
        
    input_feature_vectors = np.reshape(word_vectors, (1, count, text_feature_dimention))
#     print("Input feature vector shape after reshape(3d): {}".format(input_feature_vectors.shape))
#     print("LSTM requirs 3d shape inputs [batch, timesteps, feature]")
    output_vector = lstm(input_feature_vectors)
#     lstm.reset_states() # stateful = True is required for reset

#     print("result vector shape: {}".format(output_vector.shape))
#     print("Last input was: {}".format(input_feature_vectors[0][-1]))
#     print("output result: {}".format(output_vector))
    
    # (tensore --> numpy 0bject --> numpy.array --> array/list/ArrayType)
    return output_vector.numpy()[0].tolist() 
    
## For Testing sentence to vector convertion
# sent = "Twitter is a large social media network"
# res_vector = sent2vec(sent)
# type(res_vector), res_vector


# text string --> vector 32 dimention
sent_to_vector_udf = udf(lambda x : sent2vec(x), ArrayType(DoubleType()))
def processTextColumn(df, column_name, new_column_name):
    df = df.withColumn(new_column_name, sent_to_vector_udf(col(column_name)))
    return df

def sentEmbeddingGLoVE_LSTM(df):
    
    text_updated_column = 'text_features'
    updated_df = processTextColumn(df, "text", text_updated_column)

#     print(len(updated_df.collect()), type(updated_df), updated_df.printSchema()) 
    
    return updated_df



def assembleColumns(tweets_df):
    columns = ['retweet_count', 'reply_count', 'favorite_count',
               'num_hashtags' ,'num_urls', 'num_mentions', 'BotOrNot']

    tweets_df = tweets_df.select(*columns, 
                          tweets_df.text_features[0], tweets_df.text_features[1], tweets_df.text_features[2],tweets_df.text_features[3], tweets_df.text_features[4],
                          tweets_df.text_features[5], tweets_df.text_features[6], tweets_df.text_features[7],tweets_df.text_features[8], tweets_df.text_features[9], 
                          tweets_df.text_features[10], tweets_df.text_features[11], tweets_df.text_features[12],tweets_df.text_features[13], tweets_df.text_features[14],
                          tweets_df.text_features[15], tweets_df.text_features[16], tweets_df.text_features[17],tweets_df.text_features[18], tweets_df.text_features[19],
                          tweets_df.text_features[20], tweets_df.text_features[21], tweets_df.text_features[22],tweets_df.text_features[23], tweets_df.text_features[24],
                          tweets_df.text_features[25], tweets_df.text_features[26], tweets_df.text_features[27],tweets_df.text_features[28], tweets_df.text_features[29],
                          tweets_df.text_features[30], tweets_df.text_features[31])


#     print(tweets_df.columns, len(tweets_df.collect()), tweets_df.printSchema())

    #remove 

    feature_columns = ['retweet_count','reply_count','favorite_count','num_hashtags','num_urls','num_mentions',
                       'text_features[0]','text_features[1]', 'text_features[2]','text_features[3]','text_features[4]',
                       'text_features[5]','text_features[6]','text_features[7]', 'text_features[8]','text_features[9]',
                       'text_features[10]','text_features[11]','text_features[12]','text_features[13]','text_features[14]',
                       'text_features[15]','text_features[16]','text_features[17]','text_features[18]','text_features[19]',
                       'text_features[20]','text_features[21]','text_features[22]', 'text_features[23]', 'text_features[24]',
                       'text_features[25]','text_features[26]','text_features[27]', 'text_features[28]', 'text_features[29]',
                       'text_features[30]','text_features[31]']


    tweets_df = tweets_df.na.fill(value=0.0 ,subset= feature_columns)
    feature_assembler = VectorAssembler(inputCols = feature_columns, outputCol = 'independent_features')

    tweets_updated_df = feature_assembler.transform(tweets_df)

    #check
#     num = len(tweets_updated_df.collect())
#     print(num, type(tweets_updated_df), tweets_updated_df.printSchema())

    #remove unnecessary columns
    tweets_updated_df = tweets_updated_df.drop(*feature_columns)
    
    return tweets_updated_df


def to_nparray_list(df, column_name):
    rows = df.select(column_name).collect()
    lists = [x[column_name] for x in rows]
    nparr = np.array(lists)
    
    return nparr

def to_nparray_dataset(df, feature_column, target_column):
#     list(df.select('col_name').toPandas()['col_name']) 
#     feature = list(df.select(feature_column).toPandas()[feature_column])
#     target = list(df.select(target_column).toPandas()[target_column])
    feature = [row[0] for row in list(df.select(feature_column).toLocalIterator())]
    target = [row[0] for row in list(df.select(target_column).toLocalIterator())]
        
    return np.array(feature), np.array(target)    

def partition_dataset(df):
    train_df, test_df = df.randomSplit([0.80, 0.20])
#     print(len(train_df.collect()), len(test_df.collect()))

    # features --> 'BotOrNot'
#     X_train = train_df.drop('BotOrNot')
#     y_train = train_df.select('BotOrNot')
#     X_test = test_df.drop('BotOrNot')
#     y_test = test_df.select('BotOrNot')
    

    #checkpoint
#     print(len(X_train.collect()), len(y_train.collect()))
#     print(len(X_test.collect()), len(y_test.collect()))

#     X_train = to_nparray_list(X_train, 'independent_features')
#     y_train = to_nparray_list(y_train, 'BotOrNot')
#     X_test = to_nparray_list(X_test, 'independent_features')
#     y_test = to_nparray_list(y_test, 'BotOrNot')
    train_df = train_df.cache()
    test_df = test_df.cache()

    train_X,  train_Y = to_nparray_dataset(train_df, 'independent_features', 'BotOrNot')
    test_X, test_Y = to_nparray_dataset(test_df, 'independent_features', 'BotOrNot')

    
    return train_X, train_Y, test_X, test_Y # return type: numpy.array


def getTrainTestData(df, seed = 21):
    train_X, test_X = df.randomSplit([0.7, 0.3], seed)
    return train_X, test_X
    

'''
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
'''

def generateGradient(X, Y, weight, grads):
    gd, wgt = step(X, Y, weight, grads) # is working
    #gd, wgt = calculateGradientDelta(X, Y, weight, grads)
    return gd, wgt

def step(X, Y, weight, grads):
    print("Input count: {}, {}".format(len(X), len(Y)))
    #keep track of gradients
    
    curr_model = get2DenseLayeredModel(38)
    #apply previous training gradient
    if weight is not None:
        curr_model.set_weights(weight)
    
    if grads is not None:
        opt.apply_gradients(zip(grads, curr_model.trainable_variables))
    
    # gradienttape autometically watch trainable_variable
    # curr_model.trainable_variables
    # no need for tape.watch(curr_model.trainable_variables)
    
    with tf.GradientTape() as tape:    
        #make a prediction using model
        predict = curr_model(X)
        #calculate loss
        loss = mse(Y, predict)
    #calculate the gradient
    gd = tape.gradient(loss, curr_model.trainable_variables)
    
    opt.apply_gradients(zip(gd, curr_model.trainable_variables))
    wgt = curr_model.get_weights()
    # return the gradient to train final model
    return gd, wgt



'''
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

'''


def calculateGradientDelta(X, Y, weight, grad):
    grads = []
    if grad is not None:
        grads.append(grad)
    
    for e in range(EPOCHS):
        grad, weight = step(X, Y, weight, grad)
        grads.append(grad)

    
    grad_sum = grads[0]
    n = len(grads)
    for idx in range(1, n):
        grad_sum = add_grad(grad_sum, grads[idx])
    
    return grad_sum, weight
    

## create model
def get2DenseLayeredModel(input_dim):
    model = Sequential()
    model.add(Dense(500, input_dim=input_dim))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    
    return model

def model_evaluation(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train,
              batch_size=64,
              epochs=10,
              validation_data=(X_test, y_test))
    score, acc = model.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy:', acc)

    
def removeExtraColumn(df, column_names):
    if len(df.columns) == 26:
        df = remove_column_miss_match(df)
    else:
        df = set_column_name(df, column_names)
    
    return df


def add_grad(grad1, grad2):
    if grad1 is None:
        return grad2
    if grad2 is None:
        return grad1
    
    grad = []
    n = min(len(grad1), len(grad2))
    
    for idx in range(n):
        grad.append(grad1[idx]+grad2[idx])
    
    return grad


'''
username = {'local':'ubuntu','server':'hadoop'}
foldername = {'local':'gradients_worker_id','server':'gradients'}
filename = 'gradients_worker_id2'
worker_id = '2'
grad = [Tensor()...]


#first need to grad = loadFromHDFS()
saveToHDFS(USER_NAME, GRADIENT_FOLDER_NAME, 'worker_1', '1', grad)
'''
def saveToHDFS(username, foldername, filename, worker_id, grad):
    if filename is None or grad is None:
        return
    
    #First save in local filesystem
    file_path = os.path.join(os.sep, 'home', username['local'], foldername['local'] + worker_id, filename)
    
    
    ## gradients are list of tf.Tensore
    ## weights are list of numpy
    if foldername['server'] == 'gradients':
        lists = [gd.numpy() for gd in grad]
        np.savez(file_path, lists[0], lists[1], lists[2], lists[3], lists[4], lists[5])
    else:
        lists = grad
        np.savez(file_path, lists[0], lists[1], lists[2], lists[3], lists[4], lists[5])
        
    
    saved_filename = filename + '.npz'
    saved_in_local_folder = foldername['local'] + worker_id
    # save gradient to hdfs 
    #copyFileFromLocal(username, foldername, saved_filename, worker_id)
    hdfs.copyFileFromLocal(saved_in_local_folder, foldername['server'], saved_filename)
    
    print("{}: saved successful in hdfs".format(file_path))


'''
username = {'local':'ubuntu','server':'hadoop'}
foldername = {'local':'gradients_worker_id','server':'gradients'}
loadFromHDFS(USER_NAME, FOLDER_NAME, worker_id)

'''
def loadFromHDFS(foldername, worker_id):
    
    #download all gradient file from hdfs to local pc
    #copyFolderToLocal(username, foldername, worker_id) # Has can not copy folder error
    #copyFolderToLocal2(username, foldername, worker_id) # in class implementation
    hdfs.copyFolderToLocal2(foldername['server'], foldername['local']+worker_id)
    
    #now calculate gradient sum from local gradient files
    file_path_list = localFS.getLocalDirFileList(foldername['local']+worker_id)
    
    if len(file_path_list) == 0:
        return []
    
    grad_list = []
    
    for fpath in file_path_list:
        grad_layers_arr = np.load(fpath)
        keys = sorted(grad_layers_arr.files)
    
        grad_tens = []
        for key in keys:
            grad_item = np.array(grad_layers_arr[key], dtype='float32')

            ## gradients are list of tf.Tensore
            ## weights are list of numpy
            if foldername['server'] == 'gradients':
                grad_tens.append(tf.constant(grad_item))
            else:
                grad_tens.append(grad_item)
        
        grad_list.append(grad_tens)
        
    grad_sum = grad_list[0]
    for idx in range(1,len(grad_list)):
        grad_sum = add_grad(grad_sum, grad_list[idx])
        
    print("gradient: load successful")
    
    return grad_sum
        

def worker_task_eval(bot_tweets_df, genuine_tweets_df, worker_id):
   #solve column number issue
    bot_tweets_df = removeExtraColumn(bot_tweets_df, BOT_COLUMNS)
    genuine_tweets_df = removeExtraColumn(genuine_tweets_df, BOT_COLUMNS)
    
    
    bot_tweets_df = remove_type_miss_match(bot_tweets_df)
    genuine_tweets_df = remove_type_miss_match(genuine_tweets_df)
    
#     print(len(bot_tweets_df.collect()), len(genuine_tweets_df.collect()))
    
    ##preprocess data
    tweets_df = resize_combine_data(bot_tweets_df, genuine_tweets_df)
    tweets_df = preprocess_data(tweets_df)
    
#     print(len(tweets_df.collect()))
#     print(tweets_df.columns)
    
    ##text embedding using GLoVE & LSTM
    ## Word Embedding
    tweets_df = sentEmbeddingGLoVE_LSTM(tweets_df)
    
    ## Assable multiple colu,ms to create feature vector
    tweets_updated_df = assembleColumns(tweets_df)
#     print(len(tweets_updated_df.collect()), tweets_updated_df.columns)

    scaled_df = doDataScaling(tweets_updated_df, "independent_features", "scaled_independent_features")

#     tweets_updated_df = tweets_updated_df.cache()
    
    X_test, Y_test = to_nparray_dataset(scaled_df, 'scaled_independent_features', 'BotOrNot')
    
    model = get2DenseLayeredModel(38)

##    grads = loadFromHDFS(broadcastGradientFolderName.value, worker_id)
##    weight = loadFromHDFS(broadcastWeightFolderName.value, worker_id)


    fs_semaphore.acquire()
    grads = loadFromHDFS(broadcastGradientFolderName.value, worker_id)
    time.sleep(0.5)
    fs_semaphore.release()

    fs_semaphore.acquire()
    weight = loadFromHDFS(broadcastWeightFolderName.value, worker_id)
    time.sleep(0.5)
    fs_semaphore.release()
    
    print("Final w: ")
    print("Final GD: ")

    ### changed for testing  mode_weight0 --------------
    model.set_weights(weight) # actual weight
    opt.apply_gradients(zip(grads, model.trainable_variables))
        
        
    # in order to calculate accuracy using Keras' functions we first need
    # to compile the model
    model.compile(optimizer= opt, loss=cce, metrics=["acc"])
    
    
    # now that the model is compiled we can compute the accuracy
    (loss, acc) = model.evaluate(X_test, Y_test)
    print("[INFO] test accuracy: {}".format(acc))
    print("[INFO] test loss: {}".format(loss))
    


##pools = ['pool1','pool2','pool3']
##num_of_pool = len(pools)


def worker_task(bot_tweets_df, genuine_tweets_df, worker_id):

    #distributed worker in pools
##    pool_id = int(worker_id) % num_of_pool
##    sc.setLocalProperty('spark.scheduler.pool',pools[pool_id])

##    localFS.removeLocalDirectory(broadcastGradientFolderName.value['local'] + str(worker_id))
##    localFS.createLocalDirectory(broadcastGradientFolderName.value['local'] + str(worker_id))
##    localFS.removeLocalDirectory(broadcastWeightFolderName.value['local'] + str(worker_id))
##    localFS.createLocalDirectory(broadcastWeightFolderName.value['local'] + str(worker_id))
    
    #clear workar local directory
    fs_semaphore.acquire()
    localFS.removeLocalDirectory(broadcastGradientFolderName.value['local'] + str(worker_id))
    time.sleep(2)
    fs_semaphore.release()

    fs_semaphore.acquire()
    localFS.createLocalDirectory(broadcastGradientFolderName.value['local'] + str(worker_id))
    time.sleep(2)
    fs_semaphore.release()
    

    fs_semaphore.acquire()
    localFS.removeLocalDirectory(broadcastWeightFolderName.value['local'] + str(worker_id))
    time.sleep(2)
    fs_semaphore.release()

    fs_semaphore.acquire()
    localFS.createLocalDirectory(broadcastWeightFolderName.value['local'] + str(worker_id))
    time.sleep(2)
    fs_semaphore.release()
    
#     #cache df
    bot_tweets_df = bot_tweets_df.cache()
    genuine_tweets_df = genuine_tweets_df.cache()
    
    print("#bot_tweets: {} #gen_tweets: {}".format(bot_tweets_df.count(), genuine_tweets_df.count()))

    #solve column number issue
    bot_tweets_df = removeExtraColumn(bot_tweets_df, BOT_COLUMNS)
    genuine_tweets_df = removeExtraColumn(genuine_tweets_df, BOT_COLUMNS)
    
    
    bot_tweets_df = remove_type_miss_match(bot_tweets_df)
    genuine_tweets_df = remove_type_miss_match(genuine_tweets_df)
    
#     print(len(bot_tweets_df.collect()), len(genuine_tweets_df.collect()))
    
    ##preprocess data
    tweets_df = resize_combine_data(bot_tweets_df, genuine_tweets_df)
    tweets_df = preprocess_data(tweets_df)
    
#     print(len(tweets_df.collect()))
#     print(tweets_df.columns)
    
    ##text embedding using GLoVE & LSTM
    ## Word Embedding
    tweets_df = sentEmbeddingGLoVE_LSTM(tweets_df)
    
    ## Assable multiple colu,ms to create feature vector
    tweets_updated_df = assembleColumns(tweets_df)
#     print(len(tweets_updated_df.collect()), tweets_updated_df.columns)

    scaled_df = doDataScaling(tweets_updated_df, "independent_features", "scaled_independent_features")

#     tweets_updated_df = tweets_updated_df.cache()
    
    X, Y = to_nparray_dataset(scaled_df, "scaled_independent_features", 'BotOrNot')
    
    
    #load gradient and weights from hdfs parameter server
##    grad_sum = loadFromHDFS(broadcastGradientFolderName.value, worker_id)
##    weight = loadFromHDFS(broadcastWeightFolderName.value, worker_id)

    fs_semaphore.acquire() ##
    grad_sum = loadFromHDFS(broadcastGradientFolderName.value, worker_id)
    time.sleep(2) ##
    fs_semaphore.release()##

    fs_semaphore.acquire() ##
    weight = loadFromHDFS(broadcastWeightFolderName.value, worker_id)
    time.sleep(2) ##
    fs_semaphore.release()##

    #weight update and gradient calculation
    curr_gd, weight_new = generateGradient(X, Y, weight, grad_sum)
    
    grad_sum = add_grad(curr_gd, grad_sum)
    
    #save gradient sum in hdfs
    filename = broadcastFileNamePref.value['gradients'] + worker_id
    
    fs_semaphore.acquire() ##
    saveToHDFS(broadcastUserName.value, broadcastGradientFolderName.value, filename, worker_id, grad_sum)
    time.sleep(0.2) ##
    fs_semaphore.release()##
    
    #save weigh in hdfs
    filename = broadcastFileNamePref.value['weights'] + '_m'

    fs_semaphore.acquire() ##
    saveToHDFS(broadcastUserName.value, broadcastWeightFolderName.value, filename, worker_id, weight_new)
    time.sleep(0.2) ##
    fs_semaphore.release()##
    
    print(":: OK")

#
def getInitGradient(input_dim = 38):
    X = np.array([[0.0]*input_dim], dtype = 'float32')
    Y = np.array([[0.0]], dtype = 'float32')
    grad, weight = generateGradient(X, Y, None, None)
    return grad

    
# distributed training / adjustment of weights
def getAdjustedWeights(weights = None, gradient = None):
    if (weights is None):
        model = get2DenseLayeredModel(38)
        return model.get_weights()
    elif (gradient is None):
        return weights
    else:
        model = get2DenseLayeredModel(38)
        model.set_weights(weights)
        opt.apply_gradients(zip(gradient, model.trainable_variables))
        
        return model.get_weights()
    
#Ned to large data chech for performance
glove_dict = makeGloveWordEmbedder(GLOVE_DIR + "glove.twitter.27B.25d.txt")
broadcast_glove_dict = sc.broadcast(glove_dict)


localFS.removeLocalDirectory(broadcastGradientFolderName.value['server'])
localFS.createLocalDirectory(broadcastGradientFolderName.value['server'])

localFS.removeLocalDirectory(broadcastWeightFolderName.value['server'])
localFS.createLocalDirectory(broadcastWeightFolderName.value['server'])


localFS.removeLocalDirectory(broadcastGradientFolderName.value['local'] + str(0))
localFS.createLocalDirectory(broadcastGradientFolderName.value['local'] + str(0))

localFS.removeLocalDirectory(broadcastWeightFolderName.value['local'] + str(0))
localFS.createLocalDirectory(broadcastWeightFolderName.value['local'] + str(0))

#In the start remove existing parameter_server gradient data
#Create a new folder for gradient storage
hdfs.removeDirectory(broadcastGradientFolderName.value['server'])
hdfs.createNewDirectory(broadcastGradientFolderName.value['server'])


hdfs.removeDirectory(broadcastWeightFolderName.value['server'])
hdfs.createNewDirectory(broadcastWeightFolderName.value['server'])

#initial weights and gradients
mode_weight0 = getAdjustedWeights(None, None)
model_grad0  = getInitGradient(38)


#save initial gradients in parameter server
saveToHDFS(broadcastUserName.value,
           broadcastGradientFolderName.value,
           broadcastFileNamePref.value['gradients'] + '0',
           '0',
           model_grad0)

#save initial weights in parameter_server
saveToHDFS(broadcastUserName.value,
           broadcastWeightFolderName.value,
           broadcastFileNamePref.value['weights'] + '_m',
           '0',
           mode_weight0)

    
def ApplicationJob():

    print('Run Application')
    bot_tweets_df, genuine_tweets_df = read_dataset()
    print('dataset load successfully')
    
    train_bot_tweet_df, test_bot_tweet_df = bot_tweets_df.randomSplit([0.8, 0.2], seed = 21)
    train_genuine_tweets_df, test_genuine_tweets_df = genuine_tweets_df.randomSplit([0.8, 0.2], seed = 21)
    
    print('dataset split successfully')

    # split dataset for parallel data training
    num_of_model = 20
    split_weight = 1.0 / num_of_model
    split_weights = [split_weight] * num_of_model
    bot_dfs = train_bot_tweet_df.randomSplit(split_weights, seed = 71)
    gen_dfs = train_genuine_tweets_df.randomSplit(split_weights, seed = 71)
    
    print('dataset split successfully')
    wid = 1
    
    ## run a task for each small model training

    worker_threads = []##
    for idx in range(num_of_model):
        thread = InheritableThread(target = worker_task, kwargs={'bot_tweets_df': bot_dfs[idx],
                                                                 'genuine_tweets_df': gen_dfs[idx],
                                                                 'worker_id': str(wid)})
        thread.start()
        #thread.join()
        worker_threads.append(thread) ##
        print('Thread {}: created successfully'.format(wid))
        wid = wid + 1

    for thread in worker_threads:##
        thread.join()##
    
##    ## single worker or multiple worker
##    pool = ThreadPool(2)
##    func_params_list = [(bot_dfs[idx], gen_dfs[idx], str(idx+1))for idx in range(num_of_model)]
##    pool.starmap(worker_task, func_params_list)
    
    
    # testing model
#     worker_task_eval(bot_tweets_df, genuine_tweets_df, gradient)
    thread = InheritableThread(target = worker_task_eval, kwargs={'bot_tweets_df': test_bot_tweet_df, 
                                                                  'genuine_tweets_df': test_genuine_tweets_df,
                                                                  'worker_id': '100'})
    thread.start()
    thread.join()
    
    
    

if __name__ == '__main__':
    ApplicationJob()
    spark.stop()