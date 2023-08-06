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
 * Reference:
 * - https://stackoverflow.com/questions/53037124/partitioning-a-large-skewed-dataset-in-s3-with-sparks-partitionby-method
 * </p>
 *
 * @author Yuan Zhao
 * @version 1.0
 */
object RepartitionUtil {
  def repartitionWithinPartition(df: DataFrame, partitionCol: String, recordsPerPartition: Int)(implicit spark: SparkSession): DataFrame = {

    // The record count per partition, plus the fields we need to compute the partitioning:
    val partitionCounts = df.groupBy(partitionCol)
      .agg(count("*").as("count"))
      .withColumn("num_files", ceil(col("count") / recordsPerPartition))
      .withColumn("file_offset", sum("num_files").over(Window.orderBy(partitionCol).rowsBetween(Window.unboundedPreceding, -1)))
      .na.fill(0, Seq("file_offset"))
      .cache()

    val numPartitions = partitionCounts.agg(sum("num_files")).collect()(0).getLong(0).toInt

    df.join(partitionCounts, Seq(partitionCol))
      .withColumn("partition_index", (floor(rand() * col("num_files")) + col("file_offset")).cast("int"))
      .rdd
      .map(r => (r.getAs[Int]("partition_index"), r))
      .partitionBy(new org.apache.spark.HashPartitioner(numPartitions))
      .map(_._2)
      .toDF()
      .drop("count", "num_files", "file_offset", "partition_index")
  }
}
