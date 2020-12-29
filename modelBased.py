from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import Row
import pyspark as ps
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.mllib.evaluation import RankingMetrics
import pandas as pd
import numpy as np

def MF(train_set, test_set, k):

    spark = ps.sql.SparkSession.builder \
            .master("local[4]") \
            .appName("building recommender") \
            .getOrCreate() # create a spark session

    train_data_spark = spark.createDataFrame(train_set)
    test_data_spark = spark.createDataFrame(test_set)

    def ALS_train(train):
        als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
        model = als.fit(train_data_spark)
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
        return model, evaluator

    #cvModel_pred = cvModel.transform(train_set)
    #cvModel_pred = cvModel_pred.filter(cvModel_pred.prediction != np.nan)
    #rmse = evaluator.evaluate(cvModel_pred)
    cvModel, evaluator = ALS_train(train_set)
    recs = cvModel.recommendForAllUsers(k)
    recommendation = recs.select('userId', 'recommendations').toPandas()

    return recommendation

# def recommend_single_user(recommend, top_n, single_userId):
#   return recommend[recommend['userId'] == single_userId]['Recommendation'].values.tolist()

def recommend_single_user(recommend, single_userId, k):
    res = recommend[recommend['userId'] == single_userId][['movieId', 'prediction']].nlargest(k, 'prediction').values.tolist()
    result = [tuple(l) for l in res] 
    return result