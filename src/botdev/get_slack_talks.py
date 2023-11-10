import time
import requests
import json

from dotenv import load_dotenv
from os.path import dirname, join
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

from pandas.core.frame import DataFrame
from configparser import ConfigParser
from dataclasses import dataclass



from logging import (
    getLogger, Formatter, StreamHandler, FileHandler,
    DEBUG, INFO,
)
from slack_sdk import WebClient

from slack_sdk.errors import SlackApiError


import src.botdev.log_setting_utils as log_setting_utils
from src.botdev.mbbot_settings import SlackSettings
import importlib
importlib.reload(log_setting_utils)

from typing import Dict, List, Any, Union



# ログ設定
logger = log_setting_utils.conf_logger(DEBUG)

# 環境変数の読み込み
load_dotenv(join(dirname(__file__), '.env'))



class SlackTalkGetter():

    def __init__(self, SLACK_CHANNEL_ID) -> None:
        self.SLACK_CHANNEL_ID = SLACK_CHANNEL_ID



    def get_channels(self, output_path, oldest_unix_time):
        """指定されたSlackチャンネルからメッセージを取得し、指定された出力パスに保存する関数です。

        Args:
            SLACK_CHANNEL_ID (str): SlackチャンネルのID
            output_path (str): メッセージを保存する出力パス
            oldest_unix_time (str): 会話履歴を遡る最も過去のUNIX時間

        Returns:
            dict: 取得したメッセージのJSONデータ
        """

        def insert_histories(json_data_all, json_data):
            for i in range(len(json_data["messages"])):
                json_data_all["messages"].append(json_data["messages"][i])
            return json_data_all


        logger.debug("run!")

        # RateLimit Erroe対策
        time.sleep(1.2)

        TOKEN_GET_MESSAGES_APP = os.environ["TOKEN_GET_MESSAGES_APP"]
        SLACK_URL_HISTORY = os.environ["SLACK_URL_HISTORY"]

        payload = {
            "channel": self.SLACK_CHANNEL_ID,
            "oldest": oldest_unix_time,
            "limit": 200,
            "cursor": None,
        }
        headersAuth = {
            'Authorization': 'Bearer ' + str(TOKEN_GET_MESSAGES_APP),
        }
        response = requests.get(
            SLACK_URL_HISTORY, headers=headersAuth, params=payload)
        json_data_all = response.json()

        if "response_metadata" in json_data_all.keys():
            next_cursor = json_data_all["response_metadata"]["next_cursor"]
        else:
            next_cursor = None

        while(next_cursor != None):
            logger.debug(f"next_cursor: {next_cursor}")

            # RateLimit Erroe対策
            time.sleep(1.2)

            payload["cursor"] = next_cursor
            response = requests.get(
                SLACK_URL_HISTORY, headers=headersAuth, params=payload)
            json_data = response.json()
            json_data_all = insert_histories(json_data_all, json_data)

            if "response_metadata" in json_data.keys():
                next_cursor = json_data["response_metadata"]["next_cursor"]
            else:
                next_cursor = None

        # lambda実行のためコメントアウト
        # with open(output_path, 'w', encoding='utf-8') as f:
        #     json.dump(
        #         json_data, f,
        #         indent=4,  # インデントした状態で出力する
        #         ensure_ascii=False  # この設定をしないと日本語がうまく表示されない
        #     )

        if "error" in list(json_data_all.keys()):
            logger.debug(f"error!!")
            logger.debug(f"json_data_all: {json_data_all}")


        return json_data_all


    def get_replies(self, output_path, ts, oldest_unix_time):
        """指定されたSlackチャンネルから返信メッセージを取得し、指定された出力パスに保存する関数です。

        Args:
            SLACK_CHANNEL_ID (str): SlackチャンネルのID
            output_path (str): メッセージを保存する出力パス
            ts (str): 返信メッセージのタイムスタンプ
            oldest_unix_time (str): 会話履歴を遡る最も過去のUNIX時間

        Returns:
            dict: 取得した返信メッセージのJSONデータ
        """

        def insert_replies(json_data_all, json_data):
            for i in range(len(json_data["messages"])):
                json_data_all["messages"].append(json_data["messages"][i])
            return json_data_all


        # logger.debug(f"run!  ts: {ts}")

        # RateLimit Erroe対策
        time.sleep(1.2)

        json_data_all = {}

        TOKEN_GET_MESSAGES_APP = os.environ["TOKEN_GET_MESSAGES_APP"]
        SLACK_URL_REPLIES = os.environ["SLACK_URL_REPLIES"]


        payload = {
            "channel": self.SLACK_CHANNEL_ID,
            "ts": ts,
            "oldest": oldest_unix_time,
            "limit": 200,
            "cursor": None,
        }
        headersAuth = {
            'Authorization': 'Bearer ' + str(TOKEN_GET_MESSAGES_APP),
        }
        response = requests.get(
            SLACK_URL_REPLIES, headers=headersAuth, params=payload)
        json_data_all = response.json()

        if "response_metadata" in json_data_all.keys():
            next_cursor = json_data_all["response_metadata"]["next_cursor"]
        else:
            next_cursor = None

        while(next_cursor != None):
            logger.debug(f"next_cursor: {next_cursor}")

            # RateLimit Erroe対策
            time.sleep(1.2)

            payload["cursor"] = next_cursor
            response = requests.get(
                SLACK_URL_REPLIES, headers=headersAuth, params=payload)
            json_data = response.json()
            json_data_all = insert_replies(json_data_all, json_data)

            if "response_metadata" in json_data.keys():
                next_cursor = json_data["response_metadata"]["next_cursor"]
            else:
                next_cursor = None

        # Lambda実行のためコメントアウト
        # with open(output_path, 'w', encoding='utf-8') as f:
        #     json.dump(
        #         json_data, f,
        #         indent=4,  # インデントした状態で出力する
        #         ensure_ascii=False  # この設定をしないと日本語がうまく表示されない
        #     )

        if "error" in list(json_data_all.keys()):
            logger.debug(f"error!!")
            logger.debug(f"json_data_all: {json_data_all}")

        return json_data_all


    def convert_json_to_df(self, json_data):
        """指定されたJSONデータをDataFrameに変換する関数

        Args:
            json_data (dict): 変換するJSONデータ

        Returns:
            pandas.DataFrame: 変換されたDataFrame
        """

        # with open(file_path, "r", encoding='utf-8') as json_log:
        #     data = json.load(json_log)

        # RateLimit errorの検知
        if "messages" not in list(json_data.keys()):
            raise "slack APIからのデータ取得に失敗しました"

        df_json = pd.json_normalize(json_data["messages"], sep="_")

        # メッセージが入っていない場合は空のdfを返す
        if len(df_json) == 0:
            df_json = pd.DataFrame(columns=['user', 'text', 'ts', 'ts_reply', 'reply_count'])

        return df_json


    def get_replies_ts_list(self, df_history):
        """指定されたDataFrameから返信があるメッセージのタイムスタンプのリストを取得する関数です。

        Args:
            df_history (pandas.DataFrame): 履歴データのDataFrame

        Returns:
            list: 返信があるメッセージのタイムスタンプのリスト
        """

        if "reply_count" in list(df_history.columns):
            df_out = df_history[["ts", "reply_count"]]
            df_out = df_out.query("reply_count > 0").reset_index(drop=True)
            list_out = df_out["ts"].to_list()
        # NOTE: 誰かがチャンネルに参加したのみ等のチャンネルはmessageやtsはあるがreply_countが無いケースあり。
        else:
            list_out = []

        # 重複削除
        if len(set(list_out)) != len(list_out):
            logger.warning("tsに重複があります！")
        list_out = list(set(list_out))
        list_out.sort()

        return list_out


    def get_all_replies(self, list_replies_ts, file_path="./sample_replies.json", oldest_unix_time=0):
        """指定された返信メッセージのタイムスタンプのリストから全ての返信メッセージを取得し、DataFrameとして返す関数です。

        Args:
            list_replies_ts (list): 返信メッセージのタイムスタンプのリスト
            file_path (str, optional): メッセージを保存する出力パス。デフォルトは "./sample_replies.json" です。

        Returns:
            pandas.DataFrame: 取得した返信メッセージのDataFrame
        """
        list_cols_out = ["user", "text", "ts", "ts_reply"]

        df_replies = pd.DataFrame()

        for ts in list_replies_ts:

            json_replies = self.get_replies(file_path, ts, oldest_unix_time)
            df_replies_tmp = self.convert_json_to_df(json_replies)
            df_replies_tmp = df_replies_tmp.rename(columns={"ts": "ts_reply"})
            # historyの方のtsをts列に格納する
            df_replies_tmp["ts"] = ts

            df_replies = pd.concat(
                [df_replies, df_replies_tmp]).reset_index(drop=True)

        # 返信が何もない場合
        if len(list_replies_ts) == 0:
            df_replies = pd.DataFrame(columns=list_cols_out)

        return df_replies[list_cols_out]


    def merge_hist_and_reps(
        self,
        slack_settings, df_history: DataFrame, list_replies_ts: List[str], oldest_unix_time: float,
    ) -> DataFrame:
        """指定された履歴データと返信メッセージのタイムスタンプのリストをマージし、結合されたDataFrameを返す関数です。

        Args:
            df_history (pandas.DataFrame): 履歴データのDataFrame
            list_replies_ts (list): 返信メッセージのタイムスタンプのリスト

        Returns:
            pandas.DataFrame: マージされた履歴データと返信メッセージの結合されたDataFrame
        """

        df_replies = self.get_all_replies(
            list_replies_ts, slack_settings.replies_json_file_name, oldest_unix_time
        )
        df_replies["is_history"] = 0

        df_history_cp = df_history.copy()
        df_history_cp = df_history_cp[["user", "text", "ts"]]
        df_history_cp["is_history"] = 1
        df_history_cp["ts_reply"] = np.nan

        df_hist_and_reps = pd.concat(
            [df_history_cp, df_replies]
        ).reset_index(drop=True)

        # 返信はヒストリー側のメッセージも１レコード含まれているので、その部分を削除する。
        df_hist_and_reps = df_hist_and_reps.drop_duplicates(subset=["user", "text", "ts"], keep="first").reset_index(drop=True)

        # is_historyでソートするのはhistory側のtextを先頭にもってくるため
        df_hist_and_reps = df_hist_and_reps.sort_values(
            ["ts", "is_history", "ts_reply"],
            ascending=[True, False, True]
        ).reset_index(drop=True)

        return df_hist_and_reps[["user", "text", "ts", "ts_reply"]]




    def replace_user_id2name(self, dict_user_master: Dict[Any, Any], df: pd.DataFrame, unknown_user_name: str = "不明のユーザー") -> pd.DataFrame:
        """ユーザーIDを名前に置換する関数

        Args:
            dict_user_master (Dict[Any, Any]): ユーザーIDを名前に変換するための辞書
            df (pd.DataFrame): ユーザーIDを含むデータフレーム
            unknown_user_name (str, optional): 置換できないユーザーIDに対して使用する名前. デフォルトは"不明のユーザー".

        Returns:
            pd.DataFrame: ユーザーIDが名前に置換されたデータフレーム
        """
        logger.debug("run!")

        output_cols = ["user", "text"]

        df_out = df.copy()

        df_out[output_cols] = df_out[output_cols].replace(dict_user_master, regex=True)

        # 置換できないユーザーIDを「不明のユーザー」と置換
        unknown_user_mask = ~df_out["user"].isin(dict_user_master.values())
        list_unknown_userid = df_out.loc[unknown_user_mask, "user"].tolist()
        dict_unknown_user = dict(zip(list_unknown_userid, [unknown_user_name]*len(list_unknown_userid)))
        df_out[output_cols] = df_out[output_cols].replace(dict_unknown_user, regex=True)
        if len(list_unknown_userid) != 0:
            logger.warning(f"list_unknown_userid: {list_unknown_userid}")


        if len(df) == 0:
            df_out = pd.DataFrame(columns=output_cols)

        return df_out[output_cols]


    def convert_unix_to_jp_datetime(self, unix_timestamp):

        # logger.debug("run!")

        if unix_timestamp != unix_timestamp:
            return np.nan

        # UNIX時間をdatetimeオブジェクトに変換
        utc_time = datetime.utcfromtimestamp(float(unix_timestamp))

        # タイムゾーンを設定（東京時間はUTC+9）
        tokyo_timezone = timezone(timedelta(hours=9))

        # 東京時間に変換
        tokyo_time = utc_time.replace(tzinfo=timezone.utc).astimezone(tokyo_timezone)

        return tokyo_time




def get_all_slackid_talks(slack_settings: SlackSettings, dict_user_master: dict) -> pd.DataFrame:

    logger.debug("run!")

    list_df_out_cols = ["slack_channel_id", "user", "text", "ts_datetime", "ts_reply_datetime"]

    # 全件更新のための処理
    current_time = datetime.now()
    oldest_datetime = current_time - timedelta(days=int(slack_settings.days_from_now))
    oldest_unix_time = oldest_datetime.timestamp()
    # NOTE: 全件更新のときはchannelもrepliesも同じ時刻から取得する
    oldest_unix_time_for_channel = oldest_unix_time
    oldest_unix_time_for_replies = oldest_unix_time

    df_out = pd.DataFrame(columns=list_df_out_cols)
    list_slack_channel_id = slack_settings.slack_channel_id_list
    logger.debug(f"list_slack_channel_id: {list_slack_channel_id}")

    assert len(list_slack_channel_id) == len(set(list_slack_channel_id)), \
        f"チャンネルIDに重複があります！  {len(list_slack_channel_id)} == {len(set(list_slack_channel_id))}"

    for slack_channel_id in list_slack_channel_id:

        # 差分更新のための処理
        if slack_settings.dict_previous_latest_datetime is not None:
            oldest_datetime = slack_settings.dict_previous_latest_datetime[slack_channel_id]
            # NOTE: 差分更新のときはchannelはrepliesより２週間遡って取得する。
            #       これは、過去のチャンネル投稿に新しい時刻のreplyがされることがあるため。
            oldest_unix_time_for_channel = (oldest_datetime - timedelta(days=14)).timestamp()
            oldest_unix_time_for_replies = oldest_datetime.timestamp()

        st_getter = SlackTalkGetter(slack_channel_id)

        json_history = st_getter.get_channels(
            slack_settings.history_json_file_name, oldest_unix_time_for_channel
        )
        df_history = st_getter.convert_json_to_df(json_history)

        list_replies_ts = st_getter.get_replies_ts_list(df_history)
        df_hist_and_reps = st_getter.merge_hist_and_reps(
            slack_settings, df_history, list_replies_ts, oldest_unix_time_for_replies
        )
        # ユーザーIDをユーザー名にする
        df_hist_and_reps[["user", "text"]] = st_getter.replace_user_id2name(dict_user_master, df_hist_and_reps)
        # UNIX時間をdatetime(東京時間)にする
        df_hist_and_reps["ts_datetime"] = (
            df_hist_and_reps["ts"].map(lambda x: st_getter.convert_unix_to_jp_datetime(x))
        )
        df_hist_and_reps["ts_reply_datetime"] = (
            df_hist_and_reps["ts_reply"].map(lambda x: st_getter.convert_unix_to_jp_datetime(x))
        )

        df_hist_and_reps["slack_channel_id"] = slack_channel_id

        df_out = pd.concat([df_out, df_hist_and_reps]).reset_index(drop=True)

    return df_out[list_df_out_cols]



def get_user_master_list(client: WebClient, path: str) -> dict:
    """Slackユーザーのマスターリストを取得し、ファイルに保存します。

    Slack APIを使用してユーザーのリストを取得し、指定されたファイルに保存します。

    Args:
        client (WebClient): Slack WebClient インスタンス
        path (str): ユーザーリストを保存するファイルのパス

    Returns:
        dict: ユーザーマスターリスト

    Raises:
        SlackApiError: Slack API呼び出し時にエラーが発生した場合
    """

    # Put users into the dict
    def save_users(users_array: list, users_store: dict) -> None:
        """ユーザー情報を辞書に保存します。

        Args:
            users_array (list): Slackユーザーオブジェクトのリスト
            users_store (dict): ユーザーデータを保存する辞書
        """
        for user in users_array:
            # Key user info on their unique user ID
            user_id = user["id"]
            real_name = user["profile"]["real_name"]
            # Store the entire user object (you may not need all of the info)
            users_store[user_id] = real_name


    users_store = {}
    params = dict(
        limit=150,
    )

    try:
        # Call the users.list method using the WebClient
        # users.list requires the users:read scope
        result = client.users_list(cursor=None, **params)
        save_users(result["members"], users_store)

        next_cursor = result["response_metadata"]["next_cursor"]
        logger.debug(f"next_cursor: {next_cursor}")

        while(next_cursor != ""):
            result = client.users_list(cursor=next_cursor, **params)
            save_users(result["members"], users_store)

            next_cursor = result["response_metadata"]["next_cursor"]
            logger.debug(f"next_cursor: {next_cursor}")

        # lambdaでの実行のためコメントアウト
        # with open(path, mode="w", encoding="utf-8") as f:
        #     f.write(json.dumps(users_store))

        return users_store

    except SlackApiError as e:
        logger.error("Error creating conversation: {}".format(e))



def get_channel_master_list(client: WebClient, path: str) -> dict:
    """Slackのチャンネルリストを取得し、ファイルに保存します。

    Slack APIを使用して会話（チャンネル）のリストを取得し、指定されたファイルに保存します。

    Args:
        client (WebClient): Slack WebClient インスタンス
        path (str): 会話リストを保存するファイルのパス

    Returns:
        dict: チャンネルマスターリスト

    Raises:
        SlackApiError: Slack API呼び出し時にエラーが発生した場合
    """

    # Put conversations into the JavaScript object
    def save_conversations(conversations: list, conversations_store: dict) -> None:
        """会話（チャンネル）情報を辞書に保存します。

        Args:
            conversations (list): Slack会話オブジェクトのリスト
            conversations_store (dict): 会話データを保存する辞書
        """
        conversation_id = ""
        for conversation in conversations:
            # Key conversation info on its unique ID
            conversation_id = conversation["id"]
            name = conversation["name"]
            is_member = conversation["is_member"]

            # トーク履歴収集BOTが招待されているチャンネル情報のみを収集する
            if is_member == True:
                conversations_store[conversation_id] = name


    conversations_store = {}
    params = dict(
        types=["public_channel", "private_channel"],
        exclude_archived=True,
        limit=150,
    )

    try:
        # Call the conversations.list method using the WebClient
        result = client.conversations_list(cursor=None, **params)
        save_conversations(result["channels"], conversations_store)

        # with open("./sampleDataCL_res.json", mode="w", encoding="utf-8") as f:
        #     f.write(json.dumps(result.data))
        next_cursor = result["response_metadata"]["next_cursor"]
        logger.debug(f"next_cursor: {next_cursor}")

        while(next_cursor != ""):
            result = client.conversations_list(cursor=next_cursor, **params)
            save_conversations(result["channels"], conversations_store)

            next_cursor = result["response_metadata"]["next_cursor"]
            logger.debug(f"next_cursor: {next_cursor}")

        # Lambdaでの実行のためコメントアウト
        # with open(path, mode="w", encoding="utf-8") as f:
        #     f.write(json.dumps(conversations_store))

        return conversations_store


    except SlackApiError as e:
        logger.error("Error fetching conversations: {}".format(e))