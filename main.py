# Databricks notebook source
# DBTITLE 1,Group Project

# COMMAND ----------

path = "/FileStore/tables/proj/"

# COMMAND ----------

# DBTITLE 1,Initialization
from pyspark.sql.functions      import *
from pyspark.ml                 import Pipeline
from pyspark.ml.feature         import OneHotEncoder, StringIndexer, RFormula
from pyspark.sql.window         import Window
from pyspark.ml.classification  import LogisticRegression
from pyspark.ml.classification  import GBTClassifier
from pyspark.ml.evaluation      import BinaryClassificationEvaluator
from pyspark.mllib.evaluation   import MulticlassMetrics
from pyspark.ml.feature         import VectorAssembler

# COMMAND ----------

# DBTITLE 1,Data Import
def getData(fileName):
    df = spark\
        .read\
        .format("csv")\
        .option("header","true")\
        .option("inferSchema","true")\
        .load(path + fileName + ".csv")
    return df

# Getting data
orders   = getData("data/orders")
items    = getData("data/order_items")
payments = getData("data/order_payments")
reviews  = getData("data/order_reviews")
prods    = getData("data/products")
cats     = getData("add-on")

# COMMAND ----------

# DBTITLE 1,Data Cleaning --- Part 1: Orders
def cleanOrders(input_orders):
    # Removing the 3 NAs rows
    output_orders = input_orders.where(col("order_id") != "NA").select("*")
    # Typecasting date columns
    output_orders = output_orders\
        .withColumn("order_purchase_timestamp",      to_timestamp(col("order_purchase_timestamp"),      "yyyy-M-d H:mm:ss"))\
        .withColumn("order_approved_at",             to_timestamp(col("order_approved_at"),             "yyyy-M-d H:mm:ss"))\
        .withColumn("order_delivered_carrier_date",  to_timestamp(col("order_delivered_carrier_date"),  "yyyy-M-d H:mm:ss"))\
        .withColumn("order_delivered_customer_date", to_timestamp(col("order_delivered_customer_date"), "yyyy-M-d H:mm:ss"))\
        .withColumn("order_estimated_delivery_date", to_timestamp(col("order_estimated_delivery_date"), "yyyy-M-d H:mm:ss"))
    return output_orders

orders_cleaned = cleanOrders(orders)

# Final check
orders_cleaned.printSchema()
print("Oldie:", orders.count(), "\nNewbie:", orders_cleaned.count())

# COMMAND ----------

# DBTITLE 1,Data Cleaning --- Part 2: Reviews
# Replacing NA strings with nulls
reviews_cleaned = reviews.replace("NA", None)
reviews_cleaned = reviews_cleaned\
    .withColumn("review_answer_timestamp", to_timestamp(col("review_answer_timestamp"), "yyyy-M-d H:mm:ss"))\

# Checking the final result
reviews_cleaned.printSchema()

# COMMAND ----------

# DBTITLE 1,Data Cleaning --- Part 3: Other tables
# Nothing to clean

# COMMAND ----------

# DBTITLE 1,Data Enrichment --- Part 1: Starting with the root - orders
def enrichOrders(input_orders):
    output_orders = input_orders\
        .select(
            "order_id",
        # Dichomotimizing status
            when(col("order_status") != "delivered", "not delivered").otherwise(col("order_status")).alias("status"),
        # Creating various time variables
            # approval_time : difference in days between order placement and order approval
            # cardlv_time   : difference in days between order apporval and arrival at logistic partner's site
            # cusdlv_time   : difference in days between arrival at logistic partner's site and successful delivery to customers
            # dlv_delay     : difference in days between estimated delivery date and actual delivery date
            datediff(col("order_approved_at"), col("order_purchase_timestamp")).alias("approval_time").alias("approval_time"),
            datediff(col("order_delivered_carrier_date"),  col("order_approved_at")).alias("cardlv_time"),
            datediff(col("order_delivered_customer_date"), col("order_delivered_carrier_date")).alias("cusdlv_time"),
            datediff(col("order_delivered_customer_date"), col("order_estimated_delivery_date")).alias("dlv_delay"),
        )
    return output_orders

orders_rich = enrichOrders(orders_cleaned)

# COMMAND ----------

# DBTITLE 1,Data Enrichment --- Part 2: Items & Products
def enrichItemProd(input_items, input_prods):
    # Merging tables. Some product_id in items don't exist in prods, an inner join is preferred.
    output_items = input_items\
        .join(input_prods, "product_id", "inner")\
        .join(cats, "product_category_name", "inner")
    # Summing Price and Shipping Cost to get Total Amount Paid
    output_items = output_items.withColumn("total_paid", col("price") + col("shipping_cost"))
    # Aggregating the table
        # num_items: total number of items in one order
        # avg_desc_len: average description length for all items in one order
        # total_paid: total amount paid for one order
        # various fields containing the count of items with the corresponding clustered categories in one order
    output_items = output_items\
        .groupBy("order_id")\
        .agg(
            max("order_item_id").alias("num_items"),
            mean("product_description_lenght").alias("avg_desc_len"),
            sum("total_paid").alias("total_paid")
        )\
        .join(
            # Rejoining with output_items to get all the clusters
            output_items\
                .groupBy("order_id")\
                .pivot("cluster")\
                .count()\
                .na.fill(0),
            "order_id", "left"
        )
    return output_items

items_rich = enrichItemProd(items, prods)

# COMMAND ----------

# DBTITLE 1,Data Enrichment --- Part 3: Getting the target variable
# This involves 2 steps:
# If an order has duplicated reviews, the latest review is prioritized
# Review score is binarized for binary classification
reviews_rich = reviews\
    .select(
        "order_id",
        (first("review_score").over(Window.partitionBy("order_id").orderBy(col("review_answer_timestamp").desc()))).alias("target")
    )\
    .groupBy("order_id").agg(max("target").alias("target"))\
    .withColumn("target", when(col("target") > 3, 1).otherwise(0))

# COMMAND ----------

# DBTITLE 1,Data Enrichment --- Part 4: Putting everything together
# Merging tables
# We forwent the payments table as it did not yield any significant feature
basetable = orders_rich\
    .join(items_rich, "order_id", "left")\
    .na.fill(0)\
    .join(reviews_rich, "order_id", "inner")

# COMMAND ----------

# DBTITLE 1,Modeling --- Part 1: Preparing the sets for fitting
def transformData(dataframe):
    # Dummy-variables the status column and applies RFormula to vectorize all features
    df = Pipeline(stages=[
        StringIndexer(inputCol="status",outputCol="statusIdx"),
        OneHotEncoder(inputCol="statusIdx",outputCol="fl_status"),
        RFormula(formula="target ~ . -order_id -status -statusIdx")
    ])\
        .fit(dataframe)\
        .transform(dataframe)\
        .select("features", "label")
    return df

vec_basetable = transformData(basetable)

# COMMAND ----------

# DBTITLE 1,Modeling --- Part 2: Splitting into training and test sets
# Splitting
blu_train, blu_test = vec_basetable.randomSplit([0.7, 0.3],seed=123)
# print(f"Training set has {blu_train.count()}")
# print(f"Test set has {blu_test.count()}")

# COMMAND ----------

# DBTITLE 1,Modeling --- Part 4a: Fit-transform for Logistic Regression
logreg_model = LogisticRegression().fit(blu_train)
logreg_pred = logreg_model.transform(blu_test)

# COMMAND ----------

# DBTITLE 1,Modeling --- Part 4b: Evaluting Logistic Regression's performance
def getPerformance(model):\
    # Gets the confusion matrix from the model
    cfsmtx = MulticlassMetrics(model.select("label", "prediction").rdd).confusionMatrix().toArray()
    
    # Prints out in a nice format. BinaryClassificationEvaluator was used to retrieve AUC score
    print(f"""
    AUC: {BinaryClassificationEvaluator().evaluate(model)}
    --{"Confusion Matrix":-^{36}}--
    TP--[ {cfsmtx[1,1]:^{10}} ]----[ {cfsmtx[0,1]:^{10}} ]--FN
    FP--[ {cfsmtx[1,0]:^{10}} ]----[ {cfsmtx[0,0]:^{10}} ]--TN
    """)
    
getPerformance(logreg_pred)

# COMMAND ----------

# DBTITLE 1,Modeling --- Part 5a: Fit-transform for Gradient Boosted Tree
gbt_model = GBTClassifier().fit(blu_train)
gbt_pred = gbt_model.transform(blu_test)

# COMMAND ----------

# DBTITLE 1,Modeling --- Part 5b: Evaluting Gradient Boosted Tree's performance
getPerformance(gbt_pred)

# COMMAND ----------

# DBTITLE 1,Prediction --- Part 1: Getting data from the hold-out set
prefix = "holdoutdata/test_"
orders_ho   = getData(prefix + "orders")
items_ho    = getData(prefix + "order_items")
payments_ho = getData(prefix + "order_payments")
prods_ho    = getData(prefix + "products")

# COMMAND ----------

# DBTITLE 1,Prediction --- Part 2: Data transformation
# Cleaning and transforming orders in one step
orders_ho_rich = enrichOrders(cleanOrders(orders_ho))
# Enriching items
items_ho_rich = enrichItemProd(items_ho, prods_ho)
# Putting together the final table
holdouttable = orders_ho_rich.join(items_ho_rich, "order_id", "left").na.fill(0)

# COMMAND ----------

# DBTITLE 1,Prediction --- Part 3: Vectorizing table
# Getting all features
inputCols = holdouttable.columns[2:]
# One-hot encoding and vector assembling
vec_ho = Pipeline(stages=[
        StringIndexer(inputCol="status",outputCol="statusIdx"),
        OneHotEncoder(inputCol="statusIdx",outputCol="fl_status"),
        VectorAssembler(inputCols = inputCols + ["fl_status"], outputCol="features")
    ])\
        .fit(holdouttable)\
        .transform(holdouttable)\
        .select("order_id", "features")

# COMMAND ----------

# DBTITLE 1,Prediction --- Part 4: Prediction
# Use the better performing model to make predictions
ho_pred = gbt_model.transform(vec_ho).select("order_id", col("prediction").alias("pred_review_score"))

# COMMAND ----------

display(ho_pred)
