
from dotenv import load_dotenv
from os.path import dirname, join
import os
import sys
import numpy as np
import pandas as pd

from pandas.core.frame import DataFrame
from configparser import ConfigParser
from dataclasses import dataclass

from logging import (
    getLogger, Formatter, StreamHandler, FileHandler,
    DEBUG, INFO,
)

import src.botdev.log_setting_utils as log_setting_utils
import importlib
importlib.reload(log_setting_utils)

from typing import Dict, List, Any, Union


# ログ設定
logger = log_setting_utils.conf_logger(DEBUG)
# 環境変数の読み込み
load_dotenv(join(dirname(__file__), '.env'))





class Settings:
    """
    設定ファイルを操作するための基本クラスです。

    Args:
        conf (ConfigParser): 設定ファイルを解析するためのConfigParserオブジェクト。

    Attributes:
        conf (ConfigParser): 設定ファイルを解析するためのConfigParserオブジェクト。
        default_param (dict): フォーマットや置換時のデフォルト値を指定するための辞書。

    """

    def __init__(self, conf: ConfigParser):
        self.conf = conf
        self.default_param = {}

    def get_header(self):
        """
        サブクラスで定義することを想定しています。設定セクションのヘッダーを返すため、デフォルトでは空の文字列を返します。

        Returns:
            str: 設定セクションのヘッダー。

        """
        return ''

    def get_conf_val(self, section, key):
        """
        指定されたキーに対応する値を設定ファイルから取得します。

        Args:
            key (str): 取得したい値に対応するキー。

        Returns:
            str: 指定されたキーに対応する値。

        """
        return self.conf.get(section, key)

    def get_conf_fval(self, section, key, **kwargs):
        """
        指定されたキーに対応する値を設定ファイルから取得し、引数kwargsによって指定された値でフォーマットして返します。

        Args:
            key (str): 取得したい値に対応するキー。
            **kwargs: フォーマット時の引数として指定するキーワード引数。

        Returns:
            str: フォーマットされた値。

        """
        param = self.default_param.copy()
        param.update(kwargs)
        return self.get_conf_val(section, key).format(**param)

    def get_conf_rval(self, section, key, **kwargs):
        """
        指定されたキーに対応する値を設定ファイルから取得し、引数kwargsによって指定された値で置換して返します。
        置換対象は、{{key}} の形式で指定します。

        Args:
            key (str): 取得したい値に対応するキー。
            **kwargs: 置換時に使用するキーワード引数。

        Returns:
            str: 置換された値。

        """
        param = self.default_param.copy()
        param.update(kwargs)

        buf = self.get_conf_val(section, key)
        if param:
            for k, v in param.items():
                buf = buf.replace('{{{}}}'.format(k), v)

        return buf

    def get_all(self, section):
        """
        設定ファイルの全ての情報を辞書として取得します。

        Returns:
            dict: 設定ファイルの全ての情報。

        """
        return self.conf[section]


class MBbotSettings(Settings):
    def __init__(self, conf: ConfigParser):
        super().__init__(conf)

    def get_header(self):
        return 'mbbot.settings'

    def get_slack_settings(self):
        conf_ = self.get_all("slack")
        return SlackSettings(**conf_)

    def get_splitter_settings(self):
        conf_ = self.get_all("splitter")
        return SplitterSettings(**conf_)


@dataclass
class SlackSettings():
    # directory
    slack_channel_id_list: list
    history_json_file_name: str
    replies_json_file_name: str
    days_from_now: int



@dataclass
class SplitterSettings():
    # directory
    model_name: str
    chunk_size: int
    chunk_overlap: int
    separators: list
