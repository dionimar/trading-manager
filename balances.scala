// run the script inside spark-shell with :load balances.scala

// txid
// refid refers to splitted operations, for example, txid1 for spend x USDT and txid2 for receive y ZEUR
//       shares the same refid (buy y ZEUR for x USDT)

// to compute exact price, join with orders.csv in transactions based on EUR on some side to get the exact
// price. For others, join historical prices somehow

import org.apache.spark.sql.functions._

val operations = spark.read.option("header", "true").csv("ledgers.csv").
  withColumn("amount", col("amount").cast("Double")).
  withColumn("fee", col("fee").cast("Double")).
  withColumn("balance", col("balance").cast("Double")).
  withColumn("time", to_timestamp(col("time"))).
  withColumn("custom_type", when(col("amount") < 0, "sell").otherwise("buy"))

val transactions = operations.filter("txid is not null")

val changes = transactions.filter("type != 'transfer' and type != 'staking' and type != 'deposit'")

val staks = transactions.filter("type = 'transfer' or type = 'staking'")
val deposits = transactions.filter("type = 'deposit'")

val pairsLeft = changes.
  filter("custom_type = 'sell'").
  select(
    col("txid").as("txid_left"),
    col("refid"),
    col("time"),
    col("asset").as("asset_left"),
    col("amount").as("ammount_left"),
    col("fee").as("fee_left"),
    col("balance").as("balance_left")
  )

val pairsRight = changes.
  filter("custom_type = 'buy'").
  select(
    col("txid").as("txid_right"),
    col("refid"),
    col("asset").as("asset_right"),
    col("amount").as("ammount_right"),
    col("fee").as("fee_right"),
    col("balance").as("balance_right")
  )

val pairs = pairsLeft.join(
  pairsRight,
  usingColumns=Seq("refid"),
  joinType="left"
).
  withColumn("pair", concat(col("asset_left"), col("asset_right"))).
  withColumn("fee_on", when(col("fee_right") > 0, col("asset_right")).otherwise(col("asset_left"))).
  withColumn("fee", when(col("fee_left") > 0, col("fee_left")).otherwise(col("fee_right"))).
  drop("fee_left", "fee_right")



pairs.filter("asset_left = 'BNC' or asset_right = 'BNC'").sort(col("asset_left").asc, col("asset_right").asc, col("time").asc).show(truncate = false, numRows = 200)

//pairs.coalesce(1).write.json("computed_sells")

// val df = pairs.drop("refid", "txid_left", "txid_right", "pair", "fee", "fee_on", "balance_left", "balance_rigth")

// df.alias("df").
//   join(
//     df.select(
//       col("ammount_right").as("ammount_bought"),
//       col("asset_right").as("asset_bought"),
//       col("time").as("time_bought")
//     ).
//       alias("origin"), col("df.asset_left") === col("origin.asset_bought") and col("origin.time_bought") < col("df.time"), "left"
//   ).
//   sort(col("asset_left").asc, col("asset_right").asc, col("time").asc).show(truncate = false, numRows = 200)
