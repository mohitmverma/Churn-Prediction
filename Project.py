from __future__ import print_function
import pandas as pd
import sys
from random import random
from operator import add
from pyspark.sql import SparkSession
from pyspark.sql.functions import when
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree,RandomForest,RandomForestModel


sqlContext = SparkSession\
        .builder\
        .appName("Project")\
        .getOrCreate()
train_data = sqlContext.read.load('/home/project/churn-bigml-80.csv', 
                          format='com.databricks.spark.csv', 
                          header='true', 
                          inferSchema='true')


train_data = train_data.drop('State').drop('Area code').drop('Total day charge').drop('Total eve charge').drop('Total night charge').drop('Total intl charge')
col = 'International plan'
train_data = train_data.withColumn(col, when(train_data[col]=='Yes',1).otherwise(0))
col = 'Voice mail plan'
train_data = train_data.withColumn(col, when(train_data[col]=='Yes',1).otherwise(0))
col = 'Churn'
train_data = train_data.withColumn(col, when(train_data[col]=='False',0).otherwise(1))


def labelData(data):
    return data.rdd.map(lambda row: LabeledPoint(row[-1], row[:-1]))

training_data, testing_data = train_data.randomSplit([0.8, 0.2])
training_data = labelData(training_data)


labels = testing_data.select('Churn').collect()
test_data = testing_data.drop('Churn')
rows = test_data.rdd.map(list)
strategies = ['sqrt','log2','onethird']
#impurity = ['gini','entropy']
levels = [6,7]
for l in levels:
    for j in strategies:
        with open('gini/'+j+'_'+str(l)+'.txt',"w") as f:
            for i in range(1,101,1):
                model = RandomForest.trainClassifier(training_data, numClasses=2, categoricalFeaturesInfo={1:2,2:2},numTrees=i,featureSubsetStrategy=j, impurity='gini', maxDepth=l, maxBins=32)
                predictions = model.predict(rows).collect()
                score = 0
                p = 0
                for k in predictions:
                    if int(k) == int(labels[p].Churn):
                        score += 1
                    p += 1
                accuracy = (score*100)/len(labels)
                f.write("%d %s\n" % (i , accuracy))
        f.close()
        

