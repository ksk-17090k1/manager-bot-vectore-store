# serverless-python-requirements を使って zip 圧縮しておいた依存ライブラリの読み込み
# それを使用しない場合はここのコードは削除しても構いません
try:
    import unzip_requirements
except ImportError:
    pass

import datetime
import logging
import os
import sys
from os.path import join, dirname
from dotenv import load_dotenv
import tempfile

from pathlib import Path
from configparser import ConfigParser

from slack_sdk import WebClient
# add the PATH for the execution by AWS Lambda
sys.path.append('/opt')
import src.botdev.langchain_utils as langchain_utils
import src.botdev.get_slack_talks as get_slack_talks
import src.botdev.mbbot_settings as mbbot_settings
import src.botdev.log_setting_utils as log_setting_utils
import src.botdev.aws_utils as aws_utils
import boto3
import pandas as pd
import json




def main():

    TIMEZONE_JST = datetime.timezone(datetime.timedelta(hours=9))
    current_time = datetime.datetime.now(TIMEZONE_JST)

    rag_chatbot = langchain_utils.BaseRAGChatBot()

    # 実行ファイルのディレクトリの絶対パスを取得
    path_base = Path(__file__).resolve().parent

    # 設定ファイルの読み込み
    SETTINGS_FILE_PATH = path_base / Path(os.environ["SETTINGS_FILE_PATH"])
    c = ConfigParser()
    c.read(SETTINGS_FILE_PATH, encoding="utf-8")
    settings = mbbot_settings.MBbotSettings(c)


    # === ユーザーマスタ読み込み ===
    client = WebClient(token=os.environ.get("TOKEN_GET_MESSAGES_APP"))
    path = ""
    dict_user_master = get_slack_talks.get_user_master_list(client, path)
    # 保存
    bucket_name = settings.get_conf_val("io.settings", "mbbot_s3_bucket_name")
    bucket_prefix = "shared/"
    file_name = Path(settings.get_conf_val("io.settings", "path_user_name_master")).name
    aws_utils.save_file_to_s3(bucket_name, bucket_prefix, file_name, dict_user_master)


    # === チャンネルマスター読み込み ===

    # Vector Store作成時で使うので、過去分をまず読み込んでおく。
    bucket_name = settings.get_conf_val("io.settings", "mbbot_s3_bucket_name")
    bucket_prefix = "shared/"    # Ex. "" / "tmp/"
    file_name = Path(settings.get_conf_val("io.settings", "path_channel_name_master")).name
    data_type = "text"   # "df", "text"
    dict_channel_master_list_old = aws_utils.download_data_from_s3(bucket_name, bucket_prefix, file_name, data_type)
    dict_channel_master_list_old = json.loads(dict_channel_master_list_old)

    # 最新版取得
    client = WebClient(token=os.environ.get("TOKEN_GET_MESSAGES_APP"))
    path = ""
    dict_channel_master_list = get_slack_talks.get_channel_master_list(client, path)
    logger.debug(f"dict_channel_master_list: {dict_channel_master_list}")
    # 保存
    bucket_prefix = "shared/"
    file_name = Path(settings.get_conf_val("io.settings", "path_channel_name_master")).name
    aws_utils.save_file_to_s3(bucket_name, bucket_prefix, file_name, dict_channel_master_list)



    # === 過去のトーク履歴読み込み ===
    bucket_name = settings.get_conf_val("io.settings", "mbbot_s3_bucket_name")
    bucket_prefix = "shared/"    # Ex. "" / "tmp/"
    file_name = Path(settings.get_conf_val("io.settings", "path_df_slack_history_4prompt")).name
    data_type = "df"   # "df", "text"
    df_talks_old = aws_utils.download_data_from_s3(bucket_name, bucket_prefix, file_name, data_type)
    df_talks_old["ts_datetime"] = pd.to_datetime(df_talks_old["ts_datetime"])
    df_talks_old["ts_reply_datetime"] = pd.to_datetime(df_talks_old["ts_reply_datetime"])


    # ====== 最新のトーク履歴をAPIから取得 ======
    slack_settings = settings.get_slack_settings()
    slack_settings.slack_channel_id_list = list(dict_channel_master_list.keys())
    # for test
    # slack_settings.slack_channel_id_list = ["C04JA828FLM", "C05C46Q738D"]
    # slack_settings.slack_channel_id_list = ["C03AKE34SD6"]

    # ts_datetimeとts_reply_datetimeのうち、最も新しい日付を取得 (チャンネルごとに)
    dict_previous_latest_datetime = {}
    for id in list(dict_channel_master_list.keys()):

        df_tmp = df_talks_old.copy().query("slack_channel_id == @id").reset_index(drop=True)

        previous_latest_datetime = df_tmp["ts_datetime"].max()
        tmp = df_tmp["ts_reply_datetime"].max()
        if previous_latest_datetime < tmp:
            previous_latest_datetime = tmp

        # NaTのとき (つまり新規チャンネルが追加された際)
        if previous_latest_datetime != previous_latest_datetime:
            TIMEZONE_JST = datetime.timezone(datetime.timedelta(hours=9))
            current_time = datetime.datetime.now(TIMEZONE_JST)
            previous_latest_datetime = current_time - datetime.timedelta(days=int(slack_settings.days_from_now))
            # datetime.datetime型をtimestamp型に直す
            previous_latest_datetime = pd.to_datetime(previous_latest_datetime)

        dict_previous_latest_datetime[id] = previous_latest_datetime

    logger.debug(f"dict_previous_latest_datetime: {dict_previous_latest_datetime}")

    # 全件更新
    # slack_settings.dict_previous_latest_datetime = None
    # 差分更新
    slack_settings.dict_previous_latest_datetime = dict_previous_latest_datetime
    df_talks = get_slack_talks.get_all_slackid_talks(
        slack_settings = slack_settings,
        dict_user_master = dict_user_master,
    )

    # --- 過去分と直近取得分のマージ ---
    # NOTE: ts_reply_datetimeがNaNかつ、ts_datetimeがprevious_latest_datetimeより古いレコードが消すべきレコード。ド・モルガンの法則。
    df_talks = df_talks.query("(ts_reply_datetime == ts_reply_datetime) | (ts_datetime >= @previous_latest_datetime)")
    logger.debug(f"len(df_talks: new): {len(df_talks)}")
    logger.debug(f"len(df_talks_old): {len(df_talks_old)}")
    df_talks = pd.concat([df_talks_old, df_talks])
    logger.debug(f"concat!!!")
    logger.debug(f"len(df_talks): {len(df_talks)}")

    # sort again
    df_talks["is_history"] = df_talks["ts_reply_datetime"].map(lambda x: 1 if x!=x else 0)
    df_talks = df_talks.sort_values(
        ["slack_channel_id", "ts_datetime", "is_history", "ts_reply_datetime"], ascending=[True, True, False, True]
    ).reset_index(drop=True)
    df_talks = df_talks[[i for i in df_talks.columns if i != "is_history"]]


    # 上司BOT関連の会話はノイズとなるので除外する
    str_bot_identifier = settings.get_conf_val("langchain", "str_bot_identifier")
    logger.debug(f"len(df_talks): {len(df_talks)}")
    df_talks = df_talks[~df_talks['user'].str.contains(str_bot_identifier)].reset_index(drop=True)
    df_talks = df_talks[~df_talks['text'].str.contains(str_bot_identifier, na=False)].reset_index(drop=True)
    logger.debug(f"上司BOT関連の会話レコードを削除！")
    logger.debug(f"len(df_talks): {len(df_talks)}")


    # 重複レコード削除
    logger.debug(f"len(df_talks): {len(df_talks)}")
    df_talks = df_talks.drop_duplicates(
        subset=["slack_channel_id", "ts_datetime", "text", "ts_reply_datetime"]
    ).reset_index(drop=True)
    logger.debug(f"重複レコード削除")
    logger.debug(f"len(df_talks): {len(df_talks)}")



    # 保存
    bucket_name = settings.get_conf_val("io.settings", "mbbot_s3_bucket_name")
    bucket_prefix = "shared/"
    file_name = Path(settings.get_conf_val("io.settings", "path_df_slack_history_4prompt")).name
    oldest_saving_datetime = current_time - datetime.timedelta(days=(int(slack_settings.days_from_now) + 30))

    aws_utils.save_file_to_s3(
        bucket_name, bucket_prefix, file_name,
        df_talks.query("ts_datetime > @oldest_saving_datetime").reset_index(drop=True),
    )



    # ====== create vector store ======

    def monthly_split_and_create_db(df_talks, month):

        df_monthly_talks = df_talks.copy()

        # df_talksのts_reply_datetimeがNaNでts_datetimeの月がmonthと一致するか、ts_reply_datetimeの月がmonthと一致するかで抽出する処理
        logger.debug(f"len(df_monthly_talks): {len(df_monthly_talks)}")
        df_monthly_talks = df_monthly_talks.query(
            "((ts_reply_datetime != ts_reply_datetime) & (ts_datetime.dt.month == @month)) \
             | ((ts_reply_datetime == ts_reply_datetime) & (ts_reply_datetime.dt.month == @month))"
        ).reset_index(drop=True)
        logger.debug(f"{month}月のdfに絞り込み！")
        logger.debug(f"len(df_monthly_talks): {len(df_monthly_talks)}")

        list_channels = df_monthly_talks["slack_channel_id"].unique().tolist()
        logger.debug(f"list_channels: {list_channels}")
        docs = []
        len_all_str_docs = 0

        # チャンネルごとにメタデータを付与したいので処理も分割して行う
        for channel_id in list_channels:
            df_talks_one_channel = df_monthly_talks.copy().query("slack_channel_id == @channel_id").reset_index(drop=True)
            str_docs = langchain_utils.create_slack_talk_prompt(df_talks_one_channel)
            docs.append(str_docs)
            # metadata = None
            # docs = langchain_utils.create_docs_from_txt(path, metadata)
            # 料金算出用
            len_all_str_docs = len_all_str_docs + len(str_docs)

        # split Document/text
        splitter_settings = settings.get_splitter_settings()
        metadatas = [{"source_channel": dict_channel_master_list[i]} for i in list_channels]
        # metadatas = [{"source": f"{i}-pl"} for i in range(len(docs))]
        splitted_docs = langchain_utils.split_docs(docs, splitter_settings, metadatas)

        # Add channel names to the beginning of a splitted document.
        for i in range(len(splitted_docs)):
            page_content = splitted_docs[i].page_content
            channel_name = splitted_docs[i].metadata["source_channel"]
            splitted_docs[i].page_content = f"以下チャンネル名「{channel_name}」での投稿" + "\n\n" + page_content

        # embeddings
        db_type = settings.get_conf_val("langchain", "db_type")
        model = settings.get_conf_val("langchain", "embedding_model")
        db = rag_chatbot.create_emb_db(db_type, splitted_docs, model)

        # --- save ---
        file_name_pkl = 'index.pkl'
        file_name_faiss = 'index.faiss'
        file_key = f"faiss/monthly/month_{month}/"
        with tempfile.TemporaryDirectory() as tmpdirname:

            # save vector store to local
            faiss_path = os.path.join(tmpdirname, Path(settings.get_conf_val("io.settings", "path_faiss_db_save")).name)
            rag_chatbot.save_emb_db(faiss_path, db)

            # save to s3 bucket
            s3 = boto3.client('s3')
            bucket_name = settings.get_conf_val("io.settings", "mbbot_s3_bucket_name")

            # upload faiss db to s3 bucket
            path_pkl = os.path.join(faiss_path, file_name_pkl)
            file_key_pkl = file_key + file_name_pkl
            s3.upload_file(path_pkl, bucket_name, file_key_pkl)

            path_faiss = os.path.join(faiss_path, file_name_faiss)
            file_key_faiss = file_key + file_name_faiss
            s3.upload_file(path_faiss, bucket_name, file_key_faiss)

            logger.debug(f"File '{file_key_pkl}' successfully uploaded to '{bucket_name}'.")
            logger.debug(f"File '{file_key_faiss}' successfully uploaded to '{bucket_name}'.")

        return db, len_all_str_docs


    # vectore storeを何ヶ月分保持しておくか
    db_storage_month_duration = 6
    month_list = [(current_time - datetime.timedelta(days=30*i)).month for i in range(db_storage_month_duration)]


    previous_channel_set = set(dict_channel_master_list_old.keys())
    current_channel_set = set(dict_channel_master_list.keys())

    # 前回と収集対象のチャンネルが同一のとき
    # if False: # 強制的に全件更新
    if previous_channel_set == current_channel_set:
        # 現在の月だけ新規作成して、残りの月は過去分を読み込んでつかう
        for i, month in enumerate(month_list):
            if i == 0:
                db, len_all_str_docs = monthly_split_and_create_db(df_talks, month)
                db_all = db
            else:
                # 過去分のdbを読み込む
                bucket_name = settings.get_conf_val("io.settings", "mbbot_s3_bucket_name")
                file_name_pkl = 'index.pkl'
                file_name_faiss = 'index.faiss'
                file_key = f"faiss/monthly/month_{month}/"
                with tempfile.TemporaryDirectory() as tmpdirname:
                    s3 = boto3.client('s3')

                    path_pkl = os.path.join(tmpdirname, file_name_pkl)
                    file_key_pkl = file_key + file_name_pkl
                    s3.download_file(bucket_name, file_key_pkl, path_pkl)
                    logger.debug(f"download {path_pkl} from {bucket_name}/{file_key_pkl}")

                    path_faiss = os.path.join(tmpdirname, file_name_faiss)
                    file_key_faiss = file_key + file_name_faiss
                    s3.download_file(bucket_name, file_key_faiss, path_faiss)
                    logger.debug(f"download {path_faiss} from {bucket_name}/{file_key_faiss}")

                    # read vector store
                    db = rag_chatbot.load_emb_db(tmpdirname)
                # merge db
                db_all.merge_from(db)

    # 前回と収集対象のチャンネルが変わっているとき
    else:
        # 各月ごとのdbを作成
        for i, month in enumerate(month_list):
            db, len_all_str_docs_monthly = monthly_split_and_create_db(df_talks, month)
            # merge db
            if i == 0:
                db_all = db
                len_all_str_docs = len_all_str_docs_monthly
            else:
                db_all.merge_from(db)
                len_all_str_docs = len_all_str_docs + len_all_str_docs_monthly


    # --- 1つに結合したdbを保存 ---
    file_name_pkl = 'index.pkl'
    file_name_faiss = 'index.faiss'
    file_key = "faiss/"
    with tempfile.TemporaryDirectory() as tmpdirname:

        # save vector store to local
        faiss_path = os.path.join(tmpdirname, Path(settings.get_conf_val("io.settings", "path_faiss_db_save")).name)
        rag_chatbot.save_emb_db(faiss_path, db_all)

        # save to s3 bucket
        s3 = boto3.client('s3')
        bucket_name = settings.get_conf_val("io.settings", "mbbot_s3_bucket_name")

        # upload faiss db to s3 bucket
        path_pkl = os.path.join(faiss_path, file_name_pkl)
        file_key_pkl = file_key + file_name_pkl
        s3.upload_file(path_pkl, bucket_name, file_key_pkl)

        path_faiss = os.path.join(faiss_path, file_name_faiss)
        file_key_faiss = file_key + file_name_faiss
        s3.upload_file(path_faiss, bucket_name, file_key_faiss)

        logger.debug(f"File '{file_key_pkl}' successfully uploaded to '{bucket_name}'.")
        logger.debug(f"File '{file_key_faiss}' successfully uploaded to '{bucket_name}'.")


    logger.debug(f"len_all_str_docs: {len_all_str_docs}")
    logger.debug(f"embeddingの超概算コスト: {(1*10**(-7) * len_all_str_docs)}ドル")




# 開発用
if __name__ == "__main__":
    load_dotenv(join(dirname(__file__), '.env'))
    # logger = logging.getLogger(__name__)
        # ログ設定
    logger = log_setting_utils.conf_logger(logging.DEBUG)

    main()


# これより以降は AWS Lambda 環境で実行したときのみ実行

# ロギングを AWS Lambda 向けに初期化
# logging.basicConfig(format="%(funcName)s %(asctime)s %(message)s", level=logger.debug)
# logger = logging.getLogger(__name__)
logger = log_setting_utils.conf_logger(logging.DEBUG)



# AWS Lambda 環境で実行される関数
def handler(event, context):
    # デバッグ用にリクエスト内容をログに出力
    logger.debug(f"event: \n{event}")
    main()
    return {}