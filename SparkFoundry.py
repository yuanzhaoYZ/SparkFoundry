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

    num_partitions = partition_counts.agg(F.sum("num_files")).collect()[0][0]

    return (
        df.join(partition_counts, on=partition_col)
        .withColumn(
            "partition_index", F.floor(F.rand() * F.col("num_files")) + F.col("file_offset")
        )
        # The DataFrame API doesn't let you explicitly set the partition key; only RDDs do.
        # So we convert to an RDD, repartition according to the partition index, then convert back.
        .rdd.map(lambda r: (int(r["partition_index"]), r))
        .partitionBy(num_partitions)
        .map(lambda r: r[1])
        .toDF()
        .drop("count", "num_files", "file_offset", "partition_index")
    )
