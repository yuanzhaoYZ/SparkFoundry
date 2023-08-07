package us.yuanzhao.util;

import org.apache.spark.sql.*;
import static org.apache.spark.sql.functions.*;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.HashPartitioner;
import java.util.Arrays;
import scala.Tuple2;

/**
 * Utility class for common spark sql methods.
 */
public class SparkFoundry {

    /**
     * Create a new DataFrame / DataSet based on the current one. Thanks to this, we are able to break the DAG lineage and by the way we do not cause any traffic on the network 
     * @param previousDf: The dataframe to be processed
     * @param spark: spark session
     * @return a new dataframe with the lineage broken down.
     */
    public static Dataset<Row> breakLineage(Dataset<Row> previousDf, SparkSession spark) {
        return spark.createDataFrame(previousDf.toJavaRDD(), previousDf.schema());
    }

    /**
     * Repartition the DataFrame.
     * Utility class for repartitioning data within Python data structures or frameworks such as pandas or PySpark.
     * The class provides mechanisms to balance data distributions, ensuring even spread across computational 
     * nodes or chunks, leading to efficient data operations.
     * 
     * Benefits:
     * - Balances data to avoid skewness, ensuring even computational workloads.
     * - Reduces overheads associated with handling imbalanced data.
     * - Prepares data for distributed storage and processing by creating evenly sized partitions.
     * 
     * Typical Use Cases:
     * - Dealing with skewed datasets where certain partitions or chunks are significantly larger than others.
     * - Pre-processing steps before distributed storage or computing to ensure optimal performance.
     * 
     * Usage:
     * @param df                   The DataFrame to repartition.
     * @param recordsPerPartition  The approximate number of records to store in each file.
     * @param spark                An instance of SparkSession.
     * @param partitionCol         The column(s) on which to partition.
     * @return                     A repartitioned DataFrame.
     * @author Yuan Zhao
     */
    public static Dataset<Row> repartitionWithinPartition(Dataset<Row> df, int recordsPerPartition, SparkSession spark, String partitionCol) {
        
        Dataset<Row> partitionCounts = df.groupBy(partitionCol)
            .agg(count("*").as("count"))
            .withColumn("num_files", ceil(col("count").divide(recordsPerPartition)))
            .withColumn("file_offset", sum("num_files").over(Window.orderBy(partitionCol).rowsBetween(Window.unboundedPreceding(), -1)))
            .na().fill(0, new String[]{"file_offset"})
            .cache();

        int numPartitions = partitionCounts.agg(sum("num_files").cast("integer")).collectAsList().get(0).getInt(0);

        Dataset<Row> dfWithPartitionIndex = df.join(partitionCounts, partitionCol)
            .withColumn("partition_index", 
                            floor(rand().multiply(col("num_files")))
                            .plus(col("file_offset"))
                            .cast("integer")
            );

        StructType schema = dfWithPartitionIndex.schema();

        JavaRDD<Row> rddWithPartitionIndex = dfWithPartitionIndex.toJavaRDD()
            .mapToPair(new PairFunction<Row, Integer, Row>() {
                @Override
                public Tuple2<Integer, Row> call(Row row) throws Exception {
                    return new Tuple2<>(row.getAs("partition_index"), row);
                }
            })
            .partitionBy(new HashPartitioner(numPartitions))
            .values();

        return spark.createDataFrame(rddWithPartitionIndex, schema)
            .drop("count", "num_files", "file_offset", "partition_index");
    }

}
