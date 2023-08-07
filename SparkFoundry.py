import pyspark.sql.functions as F
from pyspark.sql.window import Window

"""
RepartitionUtil

Utility class for repartitioning data within Python data structures or frameworks such as pandas or PySpark.
The class provides mechanisms to balance data distributions, ensuring even spread across computational 
nodes or chunks, leading to efficient data operations.

Benefits:
- Balances data to avoid skewness, ensuring even computational workloads.
- Reduces overheads associated with handling imbalanced data.
- Prepares data for distributed storage and processing by creating evenly sized partitions.

Typical Use Cases:
- Dealing with skewed datasets where certain partitions or chunks are significantly larger than others.
- Pre-processing steps before distributed storage or computing to ensure optimal performance.

This block demonstrates the usage of the `RepartitionUtil` utility.

The purpose is to repartition the given DataFrame `df` based on the `PARTITION_ID` column. 
The goal is to achieve approximately 500,000 records per partition, ensuring a balanced 
distribution of records, which is beneficial for subsequent processing and efficient storage.

Furthermore, an example of writing the balanced DataFrame to an S3 location is provided, 
ensuring that data is stored in a structured and efficient manner.

dfBalanced = RepartitionUtil.repartitionWithinPartition(df, "PARTITION_ID", 500000)

# Below is an example of writing the DataFrame to S3 (assuming necessary libraries are imported):
s3Path = "s3a://[YOUR_BUCKET_NAME]/[YOUR_PATH]"

# Write the balanced DataFrame to S3, partitioned by `PARTITION_ID`
dfBalanced.write \
          .partitionBy("PARTITION_ID") \
          .format("com.databricks.spark.csv") \
          .option("delimiter", "\t") \
          .option("codec", "gzip") \
          .option("nullValue", "\\N") \
          .mode(SaveMode.Append) \
          .save(s3Path)

dfBalanced.show()
+----------+----------+--------+---------+-----------+---------------+
|   p_id   |      c_id|   count|num_files|file_offset|partition_index|
+----------+----------+--------+---------+-----------+---------------+
| ID-1631-1|     10022|15473796|       31|          0|              0|
| ID-1631-1|     10022|15473796|       31|          0|              0|
| ID-1732-1|       713| 9965716|       20|         31|             31|
| ID-1732-1|       713| 9965716|       20|         31|             31|
| ID-1732-3|       674| 6584026|       14|         51|             51|
| ID-1732-3|       674| 6584026|       14|         51|             51|
| ID-1989-1|       105| 3041839|        7|         65|             65|
| ID-1989-1|       105| 3041839|        7|         65|             65|
| ID-1989-2|       689|13942939|       28|         72|             72|
| ID-1989-2|       689|13942939|       28|         72|             72|
| ID-1989-6|        87|  554805|        2|        100|            100|
| ID-1989-6|        87|  554805|        2|        100|            100|
| ID-1989-9|       625|  439588|        1|        102|            102|
| ID-1989-9|       625|  439588|        1|        102|            102|
| ID-2003-3|       722|12070061|       25|        103|            103|
| ID-2003-3|       722|12070061|       25|        103|            103|
| ID-2208-1|       689| 7400854|       15|        128|            128|
| ID-2208-1|       689| 7400854|       15|        128|            128|
| ID-2215-1|       103|13977416|       28|        143|            143|
| ID-2215-1|       103|13977416|       28|        143|            143|
| ID-2215-4|       203|13977416|       28|        171|            171|
| ID-2215-4|       203|13977416|       28|        171|            171|
| ID-2535-1|       651| 5653858|       12|        199|            199|
| ID-2535-1|       651| 5653858|       12|        199|            199|
| ID-2590-1|       703|14446107|       29|        211|            211|
| ID-2590-1|       703|14446107|       29|        211|            211|
| ID-2721-3|       113| 1688682|        4|        240|            240|
| ID-2721-3|       113| 1688682|        4|        240|            240|
| ID-2722-1|       713|15944634|       32|        244|            244|
| ID-2722-1|       713|15944634|       32|        244|            244|
| ID-2873-1|       505|11675482|       24|        276|            276|
| ID-2873-1|       505|11675482|       24|        276|            276|
| ID-3005-3|     10012|10461967|       21|        300|            300|
| ID-3005-3|     10012|10461967|       21|        300|            300|
| ID-3035-1|       629|13623386|       28|        321|            321|
| ID-3035-1|       629|13623386|       28|        321|            321|
| ID-3052-1|       703|12833850|       26|        349|            349|
| ID-3052-1|       703|12833850|       26|        349|            349|
| ID-3086-1|       117|10111618|       21|        375|            375|
| ID-3086-1|       117|10111618|       21|        375|            375|
| ID-3086-2|       331| 5622324|       12|        396|            396|
| ID-3086-2|       331| 5622324|       12|        396|            396|
| ID-3153-1|       107| 5887068|       12|        408|            408|
| ID-3153-1|       107| 5887068|       12|        408|            408|
| ID-3155-2|       203|14033155|       29|        420|            420|
| ID-3155-2|       203|14033155|       29|        420|            420|
| ID-3200-1|       664|15336309|       31|        449|            449|
| ID-3200-1|       664|15336309|       31|        449|            449|
| ID-3222-1|       676| 8128261|       17|        480|            480|
| ID-3222-1|       676| 8128261|       17|        480|            480|
| ID-3232-1|       707| 7728343|       16|        497|            497|
| ID-3232-1|       707| 7728343|       16|        497|            497|
| ID-3274-3|       107| 3310057|        7|        513|            513|
| ID-3274-3|       107| 3310057|        7|        513|            513|
| ID-3283-1|       702| 8513616|       18|        520|            520|
| ID-3283-1|       702| 8513616|       18|        520|            520|
| ID-3302-2|       649|17460918|       35|        538|            538|
| ID-3302-2|       649|17460918|       35|        538|            538|
| ID-3302-3|       699|16969030|       34|        573|            573|
| ID-3302-3|       699|16969030|       34|        573|            573|

Reference:
- https://stackoverflow.com/questions/53037124/partitioning-a-large-skewed-dataset-in-s3-with-sparks-partitionby-method

Author: Yuan Zhao
Version: 1.0
"""
def repartition_within_partition(
    df: "pyspark.sql.dataframe.DataFrame",
    partition_col,
    records_per_partition: int = 100000,
) -> "pyspark.sql.dataframe.DataFrame":
    """Repartition data such that files are the same size, even across partitions.

    :param df: The DataFrame to repartition, partition, and then write.
    :param partition_col: The column(s) on which to partition.
    :param records_per_partition: The approximate number of records to store in each file.
    :return: A DataFrame that's ready to be written.

    Examples:
        >>> (
        ...     repartition_within_partition(df, "partition", 100_000)
        ...     .write.partitionBy("partition").parquet("/path/to/directory")
        ... )
    """
    # The record count per partition, plus the fields we need to compute the partitioning:
    partition_counts = (
        df.groupby(partition_col)
        .count()
        # The number of files to write for this partition:
        .withColumn("num_files", F.ceil(F.col("count") / records_per_partition))
        # The file offset is the cumulative sum of the number of files:
        .withColumn(
            "file_offset",
            F.sum("num_files").over(Window.rowsBetween(Window.unboundedPreceding, -1)),
        )
        .na.fill(0, "file_offset")
        .cache()
    )

    num_partitions = partition_counts.agg(F.sum("num_files")).cast("int").collect()[0][0]

    return (
        df.join(partition_counts, on=partition_col)
        .withColumn(
            "partition_index", (F.floor(F.rand() * F.col("num_files")) + F.col("file_offset")).cast("int")
        )
        # The DataFrame API doesn't let you explicitly set the partition key; only RDDs do.
        # So we convert to an RDD, repartition according to the partition index, then convert back.
        .rdd.map(lambda r: (int(r["partition_index"]), r))
        .partitionBy(num_partitions)
        .map(lambda r: r[1])
        .toDF()
        .drop("count", "num_files", "file_offset", "partition_index")
    )
