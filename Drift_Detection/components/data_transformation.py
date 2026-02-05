import sys

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer

from Drift_Detection.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from Drift_Detection.entity.config_entity import DataTransformationConfig
from Drift_Detection.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from Drift_Detection.exception import USvisaException
from Drift_Detection.logger import logging
from Drift_Detection.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file, drop_columns
from Drift_Detection.entity.estimator import TargetValueMapping


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise USvisaException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise USvisaException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        try:
            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder()
            ordinal_encoder = OrdinalEncoder()

            oh_columns = self._schema_config['oh_columns']
            or_columns = self._schema_config['or_columns']
            transform_columns = self._schema_config['num_features']
            num_features = self._schema_config['num_features']

            transform_pipe = Pipeline(steps=[
                ('transformer', PowerTransformer(method='yeo-johnson'))
            ])

            preprocessor = ColumnTransformer(
                [
                    ("OneHotEncoder", oh_transformer, oh_columns),
                    ("Ordinal_Encoder", ordinal_encoder, or_columns),
                    ("Transformer", transform_pipe, transform_columns),
                    ("StandardScaler", numeric_transformer, num_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise USvisaException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            if self.data_validation_artifact.validation_status:
                logging.info("Starting data transformation")

                preprocessor = self.get_data_transformer_object()

                train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
                test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

                input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_train_df = train_df[TARGET_COLUMN]

                drop_cols = self._schema_config['drop_columns']
                input_feature_train_df = drop_columns(input_feature_train_df, drop_cols)

                target_feature_train_df = target_feature_train_df.replace(
                    TargetValueMapping()._asdict()
                )

                # -------- REMOVE NaN TARGET ROWS --------
                valid_idx = target_feature_train_df.notna()
                target_feature_train_df = target_feature_train_df[valid_idx]
                input_feature_train_df = input_feature_train_df.loc[valid_idx]
                # ------------------------------------------------

                input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_test_df = test_df[TARGET_COLUMN]
                input_feature_test_df = drop_columns(input_feature_test_df, drop_cols)

                target_feature_test_df = target_feature_test_df.replace(
                    TargetValueMapping()._asdict()
                )

                valid_idx_test = target_feature_test_df.notna()
                target_feature_test_df = target_feature_test_df[valid_idx_test]
                input_feature_test_df = input_feature_test_df.loc[valid_idx_test]

                input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
                input_feature_test_arr = preprocessor.transform(input_feature_test_df)

                # --------  REMOVE NaN ROWS FROM X BEFORE SMOTE --------
                non_nan_mask = ~np.isnan(input_feature_train_arr).any(axis=1)
                input_feature_train_arr = input_feature_train_arr[non_nan_mask]
                target_feature_train_df = target_feature_train_df.iloc[non_nan_mask]
                # -----------------------------------------------------------

                smt = SMOTEENN(sampling_strategy="minority")

                input_feature_train_final, target_feature_train_final = smt.fit_resample(
                    input_feature_train_arr, target_feature_train_df
                )

                # -------- FIX 3: DO NOT APPLY SMOTE ON TEST DATA --------
                input_feature_test_final = input_feature_test_arr
                target_feature_test_final = target_feature_test_df
                # ------------------------------------------------------

                train_arr = np.c_[
                    input_feature_train_final,
                    np.array(target_feature_train_final)
                ]

                test_arr = np.c_[
                    input_feature_test_final,
                    np.array(target_feature_test_final)
                ]

                save_object(
                    self.data_transformation_config.transformed_object_file_path,
                    preprocessor
                )
                save_numpy_array_data(
                    self.data_transformation_config.transformed_train_file_path,
                    train_arr
                )
                save_numpy_array_data(
                    self.data_transformation_config.transformed_test_file_path,
                    test_arr
                )

                return DataTransformationArtifact(
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
                )

            else:
                raise Exception(self.data_validation_artifact.message)

        except Exception as e:
            raise USvisaException(e, sys) from e