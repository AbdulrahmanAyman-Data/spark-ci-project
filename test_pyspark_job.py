import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (DoubleType, IntegerType, StringType, StructField, StructType)
from pyspark_job import clean_data

SCHEMA = StructType([
    StructField("id", IntegerType(), True),
    StructField("name", StringType(), True),
    StructField("amount", DoubleType(), True),
])

@pytest.fixture(scope="session")
def spark():
    
    session = (
        SparkSession.builder
        .master("local[1]")
        .appName("clean_data_tests")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )

    yield session

    session.stop()


def make_df(spark, rows):
    return spark.createDataFrame(rows, schema=SCHEMA)


def test_valid_records_are_kept(spark):
    df = make_df(spark, [(1, "Alice", 100.0), (2, "Bob", 50.0)])
 
    result = clean_data(df)
 
    assert result.count() == 2
    assert sorted(r["name"] for r in result.collect()) == ["Alice", "Bob"]


def test_amount_less_than_or_equal_to_zero_is_removed(spark):
    df = make_df(
        spark,
        [
            (1, "Alice", 100.0),
            (2, "Bob", 0.0),       
            (3, "Carol", -25.0),   
        ],
    )
 
    result = clean_data(df)
 
    assert [r["name"] for r in result.collect()] == ["Alice"]

def test_null_name_is_removed(spark):
    df = make_df(spark, [(1, "Alice", 100.0), (2, None, 80.0)])
 
    result = clean_data(df)
 
    assert result.count() == 1
    assert result.collect()[0]["name"] == "Alice"

def test_amount_with_tax_is_calculated_correctly(spark):
    df = make_df(spark, [(1, "Alice", 100.0), (2, "Bob", 50.0), (3, "Carol", 10.5)])
 
    result = clean_data(df)
 
    taxed = {r["name"]: r["amount_with_tax"] for r in result.collect()}
    assert taxed["Alice"] == pytest.approx(120.0)
    assert taxed["Bob"] == pytest.approx(60.0)
    assert taxed["Carol"] == pytest.approx(12.6)

def test_all_rules_together_and_original_columns_preserved(spark):
    df = make_df(
        spark,
        [
            (1, "Alice", 100.0),   # valid
            (2, "Bob", -5.0),      # bad amount
            (3, None, 40.0),       # bad name
            (4, "Dave", None),     # NULL amount
        ],
    )
 
    result = clean_data(df)
 
    assert result.columns == ["id", "name", "amount", "amount_with_tax"]
    rows = result.collect()
    assert len(rows) == 1
    assert rows[0]["id"] == 1

 


