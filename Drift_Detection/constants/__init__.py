#variables and constants
import os
from datetime import datetime

DATABASE_NAME = "creaditcard"
COLLECTION_NAME = "creditcard_data"
MONGODB_URL_KEY = "MONGODB_URL"

PIPELINE_NAME : str = "creditcard_fraud_detection"
ARTIFACT_DIR : str = "artifact"
FILE_NAME: str = "data.csv"
MODEL_FILE_NAME = "model.pkl"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "creditcard_data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2