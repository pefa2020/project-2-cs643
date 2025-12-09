from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def load_dataset(spark, path):
    df = (
        spark.read.format("csv")
        .option("header", "true")
        .option("sep", ";")
        .option("inferSchema", "true")
        .load(path)
    )
    # Clean column names
    for c in df.columns:
        clean = c.replace('"', '').replace("'", "").strip()
        df = df.withColumnRenamed(c, clean)
    return df


def main():
    spark = SparkSession.builder \
        .appName("LocalClassificationTest") \
        .master("local[*]") \
        .getOrCreate()

    train_path = "s3://emr-logs-percy/TrainingDataset.csv" #"TrainingDataset.csv"
    val_path   = "s3://emr-logs-percy/ValidationDataset.csv" #"ValidationDataset.csv"
    model_output = "s3://emr-logs-percy/logistic_regression_model" #"local_model"

    print("Loading datasets...")
    train_df = load_dataset(spark, train_path)
    val_df   = load_dataset(spark, val_path)

    print("Training dataset schema:")
    train_df.printSchema()

    target = "quality"
    features = [c for c in train_df.columns if c != target]

    print("Indexing labels...")
    indexer = StringIndexer(inputCol=target, outputCol="label")
    
    # FIT ONCE
    label_indexer = indexer.fit(train_df)

    # TRANSFORM BOTH
    train_df = label_indexer.transform(train_df)
    val_df   = label_indexer.transform(val_df)

    assembler = VectorAssembler(inputCols=features, outputCol="features")
    train_vec = assembler.transform(train_df).select("features", "label")
    val_vec   = assembler.transform(val_df).select("features", "label")

    print("Training Logistic Regression classifier...")
    lr = LogisticRegression(featuresCol="features",
                            labelCol="label",
                            maxIter=50)

    model = lr.fit(train_vec)

    print("Evaluating...")
    preds = model.transform(val_vec)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    )

    f1 = evaluator.evaluate(preds)

    print("===== LOCAL RESULTS =====")
    print(f"F1 Score: {f1}")
    print("=========================")

    print(f"Saving LOCAL model to: {model_output}")
    model.save(model_output)

    print("Saving label indexer...")
    label_indexer.write().overwrite().save(model_output + "/label_indexer")

    spark.stop()


if __name__ == "__main__":
    main()

