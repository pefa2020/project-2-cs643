from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexerModel
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col, array_max, expr
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_dataset(spark, path):
    df = (
        spark.read.format("csv")
        .option("header", "true")
        .option("sep", ";")
        .option("inferSchema", "true")
        .load(path)
    )

    # Cleaning odd characters from header
    for c in df.columns:
        clean = c.replace('"', '').replace("'", "").strip()
        df = df.withColumnRenamed(c, clean)

    return df


def main(test_csv):
    spark = (
        SparkSession.builder
        .appName("WineModelPrediction")
        .master("local[*]")
        .config("spark.ui.showConsoleProgress", "true")
        .getOrCreate()
    )

    # Model directories (inside container)
    model_dir = os.path.join(BASE_DIR, "logistic_regression_model")
    model_path = f"file://{model_dir}"
    indexer_path = f"file://{model_dir}/label_indexer"

    # Output folder (inside container)
    output_dir = os.path.join(BASE_DIR, "prediction_output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"file://{output_dir}"

    print(f"\nUsing test CSV file: {test_csv}\n")

    # Loading CSV
    test_df = load_dataset(spark, test_csv)

    # Load model + indexer
    print("Loading logistic regression model...")
    model = LogisticRegressionModel.load(model_path)

    print("Loading label indexer...")
    label_indexer = StringIndexerModel.load(indexer_path)

    target = "quality"
    features = [c for c in test_df.columns if c != target]

    # Transform labels
    test_df = label_indexer.transform(test_df)

    # Assemble features
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    test_vec = assembler.transform(test_df).select("features", "label", target)

    print("Running predictions...")
    preds = model.transform(test_vec)

    preds = preds.withColumn("probability_array", vector_to_array(col("probability")))
    preds = preds.withColumn("max_probability", array_max(col("probability_array")))
    preds = preds.withColumn(
        "predicted_quality_index",
        expr("array_position(probability_array, max_probability) - 1")
    )

    # Map prediction index to original quality value
    labels = label_indexer.labels
    mapping = {i: float(lab) for i, lab in enumerate(labels)}

    mapping_expr = expr(
        "CASE " +
        " ".join([f"WHEN predicted_quality_index = {i} THEN {lab}" for i, lab in mapping.items()]) +
        " END"
    )

    preds = preds.withColumn("predicted_quality", mapping_expr)

    # Evaluating F1 score
    print("\nEvaluating F1 score...")
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )
    f1 = evaluator.evaluate(preds)

    print("\n===== FINAL RESULTS =====")
    print(f"F1 Score: {f1}")
    print("=========================\n")

    # Saving the results to CSV inside container
    print(f"Saving results to: {output_path}")

    preds_to_save = preds \
        .withColumn("probability_array_str", col("probability_array").cast("string")) \
        .withColumn("features_str", col("features").cast("string"))

    preds_to_save.select(
        "quality",
        "predicted_quality",
        "prediction",
        "probability_array_str",
        "max_probability",
        "features_str"
    ).coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)

    print("\nProcess completed.\n")
    spark.stop()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("ERROR: You must pass a CSV path.")
        print("Usage: docker run <image> /data/input.csv")
        sys.exit(1)

    csv_path = sys.argv[1] 
    main(csv_path)
