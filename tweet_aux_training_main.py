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

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Embedding, Dense, LSTM, Dense, Input, concatenate, Dropout, Activation, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD, Adam

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy


#Dataset location

#Local
# bot_tweets_dataset_path = 'F://TwitterBotDataset//tweet_dataset_full//bot_tweets//'
# genuine_tweets_dataset_path = 'F://TwitterBotDataset//tweet_dataset_full//genuine_tweets//'

#S3
bot_tweets_dataset_path = 's3://tweet-dataset/bot_tweets'
genuine_tweets_dataset_path = 's3://tweet-dataset/genuine_tweets'
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

# for spark-submit
spark = SparkSession.builder.appName('ml_account_ base_session').getOrCreate()
# spark

# for local build
# spark = SparkSession.builder.appName('ml_account_ base_session').getOrCreate()


#for local multi thread
# conf = SparkConf()
# conf.setMaster("local[20]").setAppName("ml_account_ base_session")
# sc = SparkContext(conf=conf)
# spark = SparkSession.builder.getOrCreate()

# read dataset from csv
def read_dataset():
    bot_tweets = spark.read.csv(bot_tweets_dataset_path, header = True, inferSchema = True).limit(100000)
    genuine_tweets = spark.read.csv(genuine_tweets_dataset_path, header = True, inferSchema = True).limit(100000)
    
#     print(len(bot_tweets.collect()), len(genuine_tweets.collect()))
    return bot_tweets, genuine_tweets

def set_column_name(df, column_names):
    df = df.toDF(*column_names)
#     print(len(df.collect()))
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
glove_word2vec_embedder = makeGloveWordEmbedder(GLOVE_DIR + "glove.twitter.27B.25d.txt")


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
        word_vector = word2vec(glove_word2vec_embedder, word)
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

    train_X,  train_Y = to_nparray_dataset(train_df, 'independent_features', 'BotOrNot')
    test_X, test_Y = to_nparray_dataset(test_df, 'independent_features', 'BotOrNot')

    
    return train_X, train_Y, test_X, test_Y # return type: numpy.array

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

def worker_task(bot_tweets_df, genuine_tweets_df):
    
    ##clean data / remove unwanted column
    if len(bot_tweets_df.columns) == 26:
        bot_tweets_df = remove_column_miss_match(bot_tweets_df)
    else:
        bot_tweets_df = set_column_name(bot_tweets_df, BOT_COLUMNS)
        
    if len(genuine_tweets_df.columns) == 26:
        genuine_tweets_df = remove_column_miss_match(genuine_tweets_df)
    else:
        genuine_tweets_df = set_column_name(genuine_tweets_df, BOT_COLUMNS)
    
    
    
    
    
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
    
    ##Create Dense Model
    
    ##device data into test and train parts
    X_train, y_train, X_test, y_test = partition_dataset(tweets_updated_df)
    
    ## ml model train and validation
    model = get2DenseLayeredModel(38)
    
    ##evaluate model
    model_evaluation(model, X_train, y_train, X_test, y_test)
    
    # save gradient to hdfs/s3
    


if __name__ == '__main__':
    ##load dara
    bot_tweets_df, genuine_tweets_df = read_dataset()
#     print(bot_tweets_df.count(), genuine_tweets_df.count())
    
    ## single worker or multiple worker
    worker_task(bot_tweets_df, genuine_tweets_df)
    spark.stop()