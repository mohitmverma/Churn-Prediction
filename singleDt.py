from __future__ import print_function
#from pyspark.ml.feature import VectorAssembler
from numpy import array
import pandas as pd
import sys
import csv
import random
from operator import add
from pyspark.sql import SparkSession
from pyspark.sql.functions import when
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils

sqlContext = SparkSession\
        .builder\
        .appName("400mb")\
        .getOrCreate()

train_data = sqlContext.read.load('/home/project/train3.csv', 
                          format='com.databricks.spark.csv', 
                          header='true', 
                          inferSchema='true')
test_data = sqlContext.read.load('/home/project/test.csv', 
                          format='com.databricks.spark.csv', 
                          header='true', 
                          inferSchema='true')


def labelData(data):
    return data.rdd.map(lambda row: LabeledPoint(row[-1], row[:-1]))

training_data = labelData(train_data)



model = RandomForest.trainClassifier(training_data, numClasses=2, categoricalFeaturesInfo={0:2,1:2,2:2,3:2,4:2,5:2,6:2,7:2,8:2,9:2,10:2,11:2,12:2,13:2,14:2,15:2,16:2,17:2,18:2,19:2},
  numTrees=63, featureSubsetStrategy='auto', impurity='gini', maxDepth=8, maxBins=32)


rows = test_data.rdd.map(list)
predictions = model.predict(rows).collect()
with open("output3.txt","w") as f:
    for i in predictions:
        f.write("%s\n" % i)

f.close()
