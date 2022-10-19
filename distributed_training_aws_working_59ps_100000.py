#libraries / dependencies
import glob
from preprocessor import api as tweet_preprocessor

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


#user info
SYSTEM_NAME = {'local':'home','server':'user'}
USER_NAME = {'local':'ubuntu','server':'ubuntu'} # {'local':'hadoop','server':'hadoop'}
FOLDER_NAME = {'local':'gradients_worker_id','server':'gradients'}
WEIGHT_FOLDER_NAME = {'local':'weights_worker_id','server':'weights'}
GRADIENT_FOLDER_NAME = {'local':'gradients_worker_id','server':'gradients'}
FILE_NAME_PREF = {'gradients': 'gradient','weights': 'weight'}

#Dataset location

#Local
bot_tweets_dataset_path = 'file:///home/ubuntu/Documents/tweet-dataset/tweet_dataset_small/bot_tweets/' #'F://TwitterBotDataset//tweet_dataset_full//bot_tweets//' 
genuine_tweets_dataset_path = 'file:///home/ubuntu/Documents/tweet-dataset/tweet_dataset_small/genuine_tweets/' #'F://TwitterBotDataset//tweet_dataset_full//genuine_tweets//'

#S3
# bot_tweets_dataset_path = 's3://tweet-dataset/bot_tweets' #'F://TwitterBotDataset//tweet_dataset_small//bot_tweets//'
# genuine_tweets_dataset_path = 's3://tweet-dataset/genuine_tweets' #'F://TwitterBotDataset//tweet_dataset_small//genuine_tweets//'

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

# read dataset from csv
def read_dataset():
    bot_tweets = spark.read.csv(bot_tweets_dataset_path, header = True, inferSchema = True).limit(1000)
    genuine_tweets = spark.read.csv(genuine_tweets_dataset_path, header = True, inferSchema = True).limit(1000)
    
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
    genuine_tweets_df = df.withColumn("id",col("id").cast(IntegerType())) \
                                    .withColumn("favorite_count",col("favorite_count").cast(LongType())) \
                                    .withColumn("favorited",col("favorited").cast(IntegerType()))
    return df


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

def generateGradient(X, Y, bw0, grads):
    gd = step(X, Y, bw0, grads)
    return gd

def step(X, Y, bw0, grads):
    print("Input count: {}, {}".format(len(X), len(Y)))
    #keep track of gradients
    
    
    
    curr_model = get2DenseLayeredModel(38)
    #apply previous training gradient
    if bw0 is not None:
        curr_model.set_weights(bw0)
    
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
    
    # return the gradient to train final model
    return gd



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


def removeDirLocal(username, foldername):
    path = os.path.join(os.sep, 'home',username, foldername,'')
    if os.path.exists(path) == False:
        print("{}: dir not exist".format(path))
        return
    
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
    except (FileExistsError, FileNotFoundError, PermissionError) as err:
        print(err)

'''
username = 'hadoop'
src_dirname = 'gradients'
dst_dirname = 'gradients_worker_id5'
'''
def copyDirLocal(username, src_dirname, dst_dirname):
    src = os.path.join(os.sep, SYSTEM_NAME['local'], username, src_dirname)
    dst = os.path.join(os.sep, SYSTEM_NAME['local'], username, dst_dirname)
    if os.path.exists(src) == False:
        print("[Error] No such file or directory {}".format(src))
        return
    
    try:
        if sys.version_info[0] == 3 and sys.version_info[1] > 7:
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            if os.path.exists(dst):
                shuite.rmtree(dst)
            shutil.copytree(src, dst)
    except (shutil.Error, FileNotFoundError) as err:
        print(err)
        return
    
    print("Directory copy successfull")
    
    print("".format(src, dst))
    
def createDirLocal(username, foldername):
    path = os.path.join(os.sep, 'home',username, foldername)
    
    if os.path.exists(path) == True:
        removeDirLocal(username, foldername)
    
    try:
        os.mkdir(path)
    except (FileNotFoundError, FileExistsError, Exception) as err:
        print(error)
    

'''
params:
username = 'hadoop'
foldername = 'gradients'

return:
['/user/ubuntu/gradients/gradient0.npz', '/user/ubuntu/gradients/gradient1.npz']
'''
def getRemoteFileList(username, foldername):
    f = os.path.join(os.sep,'user', username, foldername)
    fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration())
    path = sc._jvm.org.apache.hadoop.fs.Path
    
    if fs.exists(path(f)) == False:
        printf("[Error]: not exist {}".format(f))
        return
    paths = []
    try:
        files = fs.listStatus(path(f)) #f+os.sep+file.getPath().getName() 
        paths = [f+os.sep+file.getPath().getName() for file in files if file.isFile() == True]
    except (FileNotFoundError, OSError) as err:
        print(err)
    
    return paths


#only call when you are confirm that a folder exist
def removeDirHdfs(username='hadoop', foldername='gradients'):
    fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration())
    path = sc._jvm.org.apache.hadoop.fs.Path
    dir_path = path(os.path.join(os.sep, 'user', username, foldername))
    
    try:
        if fs.exists(dir_path) == False:
            print("{}: not exist".format(dir_path))
            return
        filestatus = fs.getFileStatus(dir_path) #FileNotFoundError
        if filestatus.isDirectory() == False:
            return
        fs.delete(dir_path, True)
    except OSError as err:
        print(err)
        return
    except FileNotFoundError as err:
        print(err)
        return
    except (BaseException, Exception) as err:
        print(err)
        return
    print("{}: removed".format(dir_path))


#If folder already exist delete it and again create it
def createNewDirHdfs(username, foldername):
    fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration())
    path = sc._jvm.org.apache.hadoop.fs.Path
    dir_path = path(os.path.join(os.sep, 'user', username, foldername))
    
    if fs.exists(dir_path) == True:
        print("{}: remove existing folder".format(dir_path))
        removeDirHdfs(username, foldername);
    try:
        fs.mkdirs(dir_path)
    except OSError as err:
        print(err)
        return
    print("{}: created".format(dir_path))

# remove hdfs local file checksum file(.crc) for successful upload
def removeFileChecksumLocal(username, foldername, filename):
    file_path = os.path.join(os.sep,SYSTEM_NAME['local'],username,foldername,filename)
    print(file_path)
    
    if os.path.exists(file_path) == False:
        print("{}: File not found in local filesystem".format(file_path))
        return
    
    checksum_path = os.path.join(os.sep, SYSTEM_NAME['local'],username, foldername, '.'+filename+'.crc')
    
    if os.path.exists(checksum_path) == False:
        print("{}: checksum file not exist".format(checksum_path))
        return
    
    try:
        if os.path.isdir(checksum_path) == True:
            shutil.rmtree(checksum_path)
        else:
            os.remove(checksum_path)
            
    except OSError as err:
        print("{}: Can not remove checksum".format(file_path))
        return
    
    print("{}: checksum removed".format(checksum_path))

# .npz file extention will be added automatically

'''
username={'local': 'ubuntu','server': 'ubuntu'},
foldername={'local':'gradients_worker_id','server':'gradients'},
filename='gradients_worker_id0.npz',
worker_id = '0'
'''

def copyFileFromLocal(username={'local': 'hadoop','server': 'hadoop'},
                      foldername={'local':'gradients_worker_id','server':'gradients'},
                      filename='gradients_worker_id0.npz',
                      worker_id = '0'):
    src = os.path.join(os.sep,SYSTEM_NAME['local'],username['local'],foldername['local']+worker_id,filename)
    dest = os.path.join(os.sep,SYSTEM_NAME['server'],username['server'],foldername['server'])
    print(src)
    print(dest)
    fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration())
    path = sc._jvm.org.apache.hadoop.fs.Path
    
    if os.path.exists(src) == False:
        print("{}: File not found in local.".format(src))
        return
    #remove each hdfs files checksum file .crc from local fs
    removeFileChecksumLocal(username['local'], foldername['local']+worker_id, filename)
    
    if fs.exists(path(dest + filename)) == True:
        print("{}: File found in hdfs".format(dest))
        return
    
    fs.copyFromLocalFile(True, True,path(src),path(dest))
    print('Saved in hdfs')

def copyFolderToLocal(username={'local':'hadoop','server':'hadoop'},
                      foldername={'local':'gradients_worker_id','server':'gradients'},
                      worker_id='0'):
    src = os.path.join(os.sep, SYSTEM_NAME['server'], username['server'],foldername['server'])
    dest = os.path.join(os.sep, SYSTEM_NAME['local'],username['local'])
    
    
    fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration())
    path = sc._jvm.org.apache.hadoop.fs.Path
    
    if fs.exists(path(src)) == False:
        return
    
    print(src)
    print(dest)
    
    fs.copyToLocalFile(False, path(src), path(dest), True)
    #rename local dir 'gradients' --> 'gradients_worker_id0'
    copyDirLocal(username['local'], foldername['server'], foldername['local']+worker_id)



def copyFolderToLocal2(username={'local':'hadoop','server':'hadoop'},
                      foldername={'local':'gradients_worker_id','server':'gradients'},
                      worker_id='0'):
    src = os.path.join(os.sep, SYSTEM_NAME['server'], username['server'],foldername['server'])
    dest = os.path.join(os.sep, SYSTEM_NAME['local'],username['local'], foldername['server'])
    
    
    fs = sc._jvm.org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration())
    path = sc._jvm.org.apache.hadoop.fs.Path
    
    if fs.exists(path(src)) == False:
        return
    
    createDirLocal(username['local'], foldername['server'])
    
    print(src)
    print(dest)
    
    file_paths = getRemoteFileList(username['server'], foldername['server'])
    for fpath in file_paths:
        print(fpath)
        #copy file 'server/gradients/gradients0.npz --> local/gradients/gradients0.npz'
        fs.copyToLocalFile(False, path(fpath), path(dest), True)
    
    #copy local dir '/home/hadoop/gradients' --> /home/hadoop/gradients_worker_id0
    copyDirLocal(username['local'], foldername['server'], foldername['local']+worker_id)

def getFileList(username, foldername):
    dir_path = os.path.join(os.sep, 'home', username, foldername)
    
    file_paths = []
    
    for fname in os.listdir(dir_path):
        if fname.endswith(".npz"):
            fpath = dir_path + os.sep + fname;
            file_paths.append(fpath)
            
    return file_paths

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


#first need to grad = load_grad()
save_grad(USER_NAME, GRADIENT_FOLDER_NAME, 'worker_1', '1', grad)
'''
def save_grad(username, foldername, filename, worker_id, grad):
    if filename is None or grad is None:
        return
    #First save in local filesystem
    file_path = os.path.join(os.sep,SYSTEM_NAME['local'], username['local'], foldername['local'] + worker_id, filename)
    lists = [gd.numpy() for gd in grad]
    np.savez(file_path, lists[0], lists[1], lists[2], lists[3], lists[4], lists[5])
    
    saved_filename = filename + '.npz'
    
    # save gradient to hdfs 
    copyFileFromLocal(username, foldername, saved_filename, worker_id)
    
    print("gradient: save successful")


'''
username = {'local':'ubuntu','server':'hadoop'}
foldername = {'local':'gradients_worker_id','server':'gradients'}
load_grad(USER_NAME, FOLDER_NAME, worker_id)

'''
def load_grad(username, foldername, worker_id):
    
    #download all gradient file from hdfs to local pc
    #copyFolderToLocal(username, foldername, worker_id)
    copyFolderToLocal2(username, foldername, worker_id)
    
    #now calculate gradient sum from local gradient files
    file_path_list = getFileList(username['local'], foldername['local']+worker_id)
    
    if len(file_path_list) == 0:
        return []
    
    grad_list = []
    
    for fpath in file_path_list:
        grad_layers_arr = np.load(fpath)
        keys = sorted(grad_layers_arr.files)
    
        grad_tens = []
        for key in keys:
            grad_item = np.array(grad_layers_arr[key], dtype='float32')
            tens = tf.constant(grad_item)
            grad_tens.append(tens)
        
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
    
    bw0 = broadcast_w0.value
    
    grads = load_grad(USER_NAME, GRADIENT_FOLDER_NAME, worker_id)
    
    print("Final w: ".format(bw0))
    print("Final GD: ".format(grads))
    
    model.set_weights(bw0)
    opt.apply_gradients(zip(grads, model.trainable_variables))
        
        
    # in order to calculate accuracy using Keras' functions we first need
    # to compile the model
    model.compile(optimizer= opt, loss=cce, metrics=["acc"])
    
    
    # now that the model is compiled we can compute the accuracy
    (loss, acc) = model.evaluate(X_test, Y_test)
    print("[INFO] test accuracy: {}".format(acc))
    print("[INFO] test loss: {}".format(loss))
    
    

def worker_task(bot_tweets_df, genuine_tweets_df, username, foldername, worker_id):
    #clear workar own local directory
    removeDirLocal(username['local'], foldername['local'] + str(worker_id))
    createDirLocal(username['local'], foldername['local'] + str(worker_id))
    
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

#     tweets_updated_df = tweets_updated_df.cache()
    
    X, Y = to_nparray_dataset(tweets_updated_df, 'independent_features', 'BotOrNot')
    
    
    #load gradient sum in fs/hdfs/s3
    grad_sum = load_grad(username, foldername, worker_id)
    
    bw0 = broadcast_w0.value
    curr_gd = generateGradient(X, Y, bw0, grad_sum)
    
    grad_sum = add_grad(curr_gd, grad_sum)
    
    #save gradient sum in fs/hdfs/s3
    
    filename = FILE_NAME_PREF['gradients'] + worker_id
    save_grad(username, foldername, filename, worker_id, grad_sum)
    
    print(":: OK")

#
def getInitGradient(input_dim = 38):
    X = np.array([[0.0]*input_dim], dtype = 'float32')
    Y = np.array([[0.0]], dtype = 'float32')
    grad = generateGradient(X, Y, None, None)
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

removeDirLocal(USER_NAME['local'], GRADIENT_FOLDER_NAME['local']+'0')
createDirLocal(USER_NAME['local'], GRADIENT_FOLDER_NAME['local']+'0')

#In the start remove existing parameter_server gradient data
#Create a new folder for gradient storage
removeDirHdfs(USER_NAME['server'], GRADIENT_FOLDER_NAME['server'])
createNewDirHdfs(USER_NAME['server'], GRADIENT_FOLDER_NAME['server'])

w0 = getAdjustedWeights(None, None)
broadcast_w0 = sc.broadcast(w0)

model_grad0 = getInitGradient(38)
save_grad(USER_NAME, GRADIENT_FOLDER_NAME,FILE_NAME_PREF['gradients'] + '0', '0', model_grad0)



# broadcast glove word wmbedder to all task
def broadcastData():
    print("broadcast glove")
    glove_dict = makeGloveWordEmbedder(GLOVE_DIR + "glove.twitter.27B.25d.txt")
    broadcast_glove_dict = sc.broadcast(glove_dict)
    
    
def ApplicationJob():
    
#     broadcastData()
#     accumulateData()
    print('Run Application')
    bot_tweets_df, genuine_tweets_df = read_dataset()
    print('dataset load successfully')
    
    train_bot_tweet_df, test_bot_tweet_df = bot_tweets_df.randomSplit([0.8, 0.2], seed = 21)
    train_genuine_tweets_df, test_genuine_tweets_df = genuine_tweets_df.randomSplit([0.8, 0.2], seed = 21)
    
    print('dataset split successfully')
#     broadcastGloveDict()
    # split dataset for parallel data training
    num_of_thread = 10
    split_weight = 1.0 / num_of_thread
    split_weights = [split_weight] * num_of_thread
    bot_dfs = train_bot_tweet_df.randomSplit(split_weights, seed = 71)
    gen_dfs = train_genuine_tweets_df.randomSplit(split_weights, seed = 71)
    
    print('dataset split successfully')
    wid = 1
    ## run a task for each small model training
    for idx in range(num_of_thread):
        thread = InheritableThread(target = worker_task, kwargs={'bot_tweets_df': bot_dfs[idx],
                                                                 'genuine_tweets_df': gen_dfs[idx], 
                                                                 'username': USER_NAME, 
                                                                 'foldername': GRADIENT_FOLDER_NAME, 
                                                                 'worker_id': str(wid)})
        thread.start()
        thread.join()
        print('Thread {}: created successfully'.format(wid))
        wid = wid + 1
        
    
    ## single worker or multiple worker
    
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
