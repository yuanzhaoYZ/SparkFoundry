import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
/**
 * RepartitionUtil
 * 
 * <p>
 * Utility class designed to handle the intricacies of repartitioning Spark DataFrames.
 * Its primary goal is to adjust the internal partitioning of a DataFrame, ensuring
 * that data chunks are balanced across the cluster. This leads to more efficient
 * computation and optimizes the writing process to distributed storage.
 * </p>
 * 
 * <p>
 * Benefits:
 * - Balances data across nodes to prevent data skewness.
 * - Optimizes the computational tasks by ensuring each task has roughly the same amount of data.
 * - Minimizes shuffling of data across the network.
 * </p>
 * 
 * <p>
 * Typical Use Case:
 * - When you have a skewed dataset where certain partitions are much larger than others, 
 *   leading to some tasks running much longer than others.
 * - Before writing a large dataset to distributed storage to ensure that the data is evenly distributed 
 *   across files and nodes.
 * </p>
 * <p>
 * This block demonstrates the usage of the `RepartitionUtil` utility.
 * 
 * The purpose is to repartition the given DataFrame `df` based on the `PARTITION_ID` column. 
 * The goal is to achieve approximately 500,000 records per partition, ensuring a balanced 
 * distribution of records, which is beneficial for subsequent processing and efficient storage.
 * 
 * Furthermore, an example of writing the balanced DataFrame to an S3 location is provided, 
 * ensuring that data is stored in a structured and efficient manner.
 * val dfBalanced = RepartitionUtil.repartitionWithinPartition(df, "PARTITION_ID", 500000)(spark)
 *
 * import org.apache.spark.sql.{Dataset, Row, SaveMode}
 * 
 * val s3Path = "s3a://[YOUR_BUCKET_NAME]/[YOUR_PATH]"
 * 
 * // Write the balanced DataFrame to S3, partitioned by `PARTITION_ID`
 * dfBalanced
 *       .write
 *       .partitionBy("PARTITION_ID")
 *       .format("com.databricks.spark.csv")
 *       .option("delimiter", "\t")
 *       .option("codec", "gzip")
 *       .option("nullValue", "\\N")
 *       .mode(SaveMode.Append)
 *       .save(s3Path)
 * </p>
 * <p>
 * Reference:
 * - https://stackoverflow.com/questions/53037124/partitioning-a-large-skewed-dataset-in-s3-with-sparks-partitionby-method
 * </p>
 *
 * @author Yuan Zhao
 * @version 1.0
 */
object RepartitionUtil {
  def repartitionWithinPartition(df: DataFrame, partitionCol: String, recordsPerPartition: Int = 100000)(implicit spark: SparkSession): DataFrame = {

    // The record count per partition, plus the fields we need to compute the partitioning:
    val partitionCounts = df.groupBy(partitionCol)
      .agg(count("*").as("count"))
      .withColumn("num_files", ceil(col("count") / recordsPerPartition))
      .withColumn("file_offset", sum("num_files").over(Window.orderBy(partitionCol).rowsBetween(Window.unboundedPreceding, -1)))
      .na.fill(0, Seq("file_offset"))
      .cache()

    val numPartitions = partitionCounts.agg(sum("num_files")).cast("int").collect()(0).getLong(0).toInt

    val dfWithPartitionIndex = df.join(partitionCounts, Seq(partitionCol))
      .withColumn("partition_index", (floor(rand() * col("num_files")) + col("file_offset")).cast("int"))
    val schema = dfWithPartitionIndex.schema
    val RddWithPartitionIndex = dfWithPartitionIndex.rdd
                          .map(r => (r.getAs[Int]("partition_index"), r))
                          .partitionBy(new org.apache.spark.HashPartitioner(numPartitions))
                          .map(_._2)
    spark.createDataFrame(RddWithPartitionIndex, schema)
                          .drop("count", "num_files", "file_offset", "partition_index")
  }
}
