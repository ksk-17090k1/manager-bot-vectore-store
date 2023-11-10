



import json
import os
import tempfile

import pandas as pd
import boto3
import src.botdev.log_setting_utils as log_setting_utils

from logging import (
    getLogger, Formatter, StreamHandler, FileHandler,
    DEBUG, INFO,
)

# ログ設定
logger = log_setting_utils.conf_logger(DEBUG)



def save_file_to_s3(bucket_name: str, bucket_prefix: str, file_name: str, save_object) -> None:

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_path = os.path.join(tmpdirname, file_name)
        if type(save_object) == pd.DataFrame:
            save_object.to_csv(tmp_path, index=False)
        elif type(save_object) == dict:
            with open(tmp_path, mode="w", encoding="utf-8") as f:
                f.write(json.dumps(save_object))
        elif type(save_object) == str:
            with open(tmp_path, mode="w", encoding="utf-8") as f:
                f.write(save_object)
        else:
            raise "予期せぬデータ型です"

        # save to s3 bucket
        s3 = boto3.client('s3')
        file_name = bucket_prefix + file_name
        s3.upload_file(tmp_path, bucket_name, file_name)
        logger.debug(f"File '{file_name}' successfully uploaded to '{file_name}'.")



def download_data_from_s3(bucket_name, bucket_prefix, file_name, data_type):
    with tempfile.TemporaryDirectory() as tmpdirname:
        s3 = boto3.client('s3')
        local_path = os.path.join(tmpdirname, file_name)
        bucket_path = bucket_prefix + file_name
        s3.download_file(bucket_name, bucket_path, local_path)

        if data_type == "df":
            data = pd.read_csv(local_path, dtype="str")
        elif data_type == "text":
            with open(local_path, mode="r", encoding="utf-8") as f:
                data = f.read()
        else:
            raise "予期せぬデータ型です"

        logger.debug(f"File '{file_name}' successfully downloaded from '{bucket_name}/{bucket_prefix}'.")

    return data