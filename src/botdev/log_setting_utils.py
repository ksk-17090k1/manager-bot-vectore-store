from logging import (
    getLogger, Formatter, StreamHandler, FileHandler,
    DEBUG, INFO,
)

def conf_logger(level, log_file_name='log.txt'):

    format = "%(asctime)s %(name)s:%(lineno)s %(funcName)s [%(levelname)s]:%(message)s"

    logger = get_logger(level)
    if not logger.hasHandlers():
        logger = set_streamhandler(logger, format, level)
        # logger = set_filehandler(logger, format, log_file_name, level)
    logger.propagate = False

    return logger


def get_logger(level=DEBUG):
    # ロガーを取得
    logger = getLogger(__name__)
    logger.setLevel(level)
    return logger


def set_streamhandler(logger, format, level=DEBUG):
    # 出力のフォーマットを定義
    formatter = Formatter(format)
    # sys.stderrへ出力するハンドラーを定義
    sh = StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(formatter)
    # ロガーにハンドラーを登録する
    logger.addHandler(sh)
    return logger


def set_filehandler(logger, format, log_file_name='log.txt', level=DEBUG):
    # 出力のフォーマットを定義
    formatter = Formatter(format)
    # ファイルへ出力するハンドラーを定義
    fh = FileHandler(filename=log_file_name, encoding='utf-8')
    fh.setLevel(level)
    fh.setFormatter(formatter)
    # ロガーにハンドラーを登録する
    logger.addHandler(fh)
    return logger
