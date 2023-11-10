import importlib
import json
import os
from os.path import dirname, join
from pathlib import Path
import numpy as np

import pandas as pd
import requests
from dotenv import load_dotenv
import re


import src.botdev.get_slack_talks as get_slack_talks
importlib.reload(get_slack_talks)

from logging import config, getLogger
from configparser import ConfigParser

from langchain.text_splitter import RecursiveCharacterTextSplitter

# from langchain.chains import VectorDBQA
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import sys
from logging import (
    getLogger, Formatter, StreamHandler, FileHandler,
    DEBUG, INFO,
)
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

import src.botdev.log_setting_utils as log_setting_utils
import importlib
importlib.reload(log_setting_utils)


# ログ設定
logger = log_setting_utils.conf_logger(DEBUG)

# 環境変数の読み込み
load_dotenv(join(dirname(__file__), '.env'))



def create_slack_talk_prompt(df):

    logger.debug("run!")

    if df["slack_channel_id"].nunique() != 1:
        raise Exception("チャンネルIDが複数含まれています！")

    df_cp = df.copy()
    df_cp["text"] = df_cp["text"].fillna("メッセージなし")

    prompt = ""
    unique_channels = df_cp['slack_channel_id'].unique()

    for channel_id in unique_channels:

        channel_df = df_cp[df_cp['slack_channel_id'] == channel_id]

        for user, text, ts, ts_reply in zip(channel_df['user'], channel_df['text'], channel_df['ts_datetime'], channel_df['ts_reply_datetime']):

            # text = re.sub(r"&gt;(.*?)\n", r">>>\1>>>\n", text)
            text = re.sub(r"&gt;(.*?)\n", r"\n", text)
            text = text.replace('"', '”')
            text = re.sub(r"<@(.*?)>", r"To \1 ", text)

            format_ = '%Y年%m月%d日%H時%M分'
            str_ts = ts.strftime(format_)
            if ts_reply == ts_reply:
                str_ts_reply = ts_reply.strftime(format_)

            if pd.isnull(ts_reply):
                prompt += f'{str_ts}での{user}の発言: 「{text}」\n'
            else:
                prompt += f'{str_ts_reply}での{user}の発言: 「{text}」\n'

        prompt += "\n"

    return prompt.strip()


def create_docs_from_txt(path, metadata=None):

    logger.debug("run!")

    # Documentオブジェクトはリストに格納する
    # 以下の処理と同等
    # loader = TextLoader("./index.md")
    # docs = loader.load()

    with open(path, "r", encoding="utf-8") as f:
        str_txt = f.read()

    # metadataに何か指定されていればDocument型で返す
    if metadata != None:
        docs = []
        doc = Document(
            page_content=str_txt,
            metadata=metadata,
        )
        docs.append(doc)
        return docs
    elif metadata == None:
        return str_txt


def split_docs(docs, splitter_settings, metadatas=None):

    logger.debug("run!")

    # tiktoken: 実際に利用するモデルと同等のエンコーダーを使用してテキストのtoken数をカウントしつつ分割を行ってくれる
    # 参考：https://zenn.dev/ml_bear/books/d1f060a3f166a5/viewer/9f3e99#text-splitter%E3%81%A8%E3%81%AF%EF%BC%9F
    # NOTE: splitterは入力がstr型であればstr型を格納したリスト、Document型であればDocumentを格納したリストを返す。
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name=splitter_settings.model_name,
        chunk_size=int(splitter_settings.chunk_size),
        chunk_overlap=int(splitter_settings.chunk_overlap),
        separators=json.loads(splitter_settings.separators),
    )

    if type(docs) == list:
        if type(docs[0]) == Document:
            splitted_docs = text_splitter.split_documents(docs)
        # テキストが格納されたリスト型の場合はこのタイミングでメタデータを登録する。
        elif type(docs[0]) == str:
            splitted_docs = text_splitter.create_documents(docs, metadatas=metadatas)
        else:
            raise Exception("良きせぬデータ型です！")
    # docsがstr型なら分割してもstr型
    elif type(docs) == str:
        splitted_docs = text_splitter.split_text(docs)
    else:
        raise Exception("予期せぬデータ型です！")

    return splitted_docs


class BaseRAGChatBot:

    def create_emb_db(self, db_type, docs, model):

        logger.debug("run!")

        embeddings = OpenAIEmbeddings(model=model)

        docs_type = type(docs)
        if db_type == "FAISS":
            db_class = FAISS
        elif db_type == "chroma":
            db_class = Chroma
        else:
            raise Exception("予期しないdb_typeです")

        # vector index作成
        if docs_type == list:
            db = db_class.from_documents(docs, embeddings)
        elif docs_type == str:
            db = db_class.from_texts(docs, embeddings)
        else:
            raise Exception("入力が予期しないデータ型です")

        return db


    def add_texts_to_db(self, db, text):
        logger.debug("run!")
        db.add_texts([text])


    def get_qa_from_db(self, db, chain_type, custom_prompt=None, model="gpt-3.5-turbo"):

        logger.debug("run!")

        # TODO: temperture可変にする
        llm = ChatOpenAI(model_name=model, temperature=0)

        # retriever作成
        retriever = db.as_retriever()
        # chain作成
        if custom_prompt == None:
            qa = RetrievalQA.from_chain_type(
                llm=llm, chain_type=chain_type, retriever=retriever,
                return_source_documents=True,
            )
        else:
            chain_type_kwargs = {"prompt": custom_prompt}
            qa = RetrievalQA.from_chain_type(
                llm=llm, chain_type=chain_type, retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs=chain_type_kwargs,
            )
        # VectorDBQAは非推奨になった
        # qa = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=db)

        # load_qa_chainを使った記法
        # https://ict-worker.com/ai/gpt35-with-smeca.html
        # query = "山口さんからの要求は？"
        # embedding_vector = embeddings.embed_query(query)
        # docs_and_scores = db.similarity_search_by_vector(embedding_vector)
        # chain = load_qa_chain(ChatOpenAI(temperature=0), chain_type="stuff")
        # result = chain({"input_documents": docs_and_scores, "question": query}, return_only_outputs=True)

        return qa


    def get_result_from_qa(self, qa, query):

        logger.debug("run!")

        # ソースのドキュメントも取得するには.runではなく__call__を使う。
        result = qa({"query": query})
        # result = qa.run(query)

        # ソースのドキュメントの参照は内部的にsimilarity_searchを使っている
        # searched_docs = db.similarity_search(query)


        return result["result"], result["source_documents"]


    def get_qa_from_db_and_convs(self, db, option, prompt_template=None, model="gpt-3.5-turbo"):

        logger.debug("run!")

        llm = ChatOpenAI(model_name=model)

        if option == None:
            qa = ConversationalRetrievalChain.from_llm(
                llm=llm, retriever=db.as_retriever(),
                return_source_documents=True,
            )
        elif (option == "custom_prompt") | (option == "map_reduce"):

            # NOTE: map reduceの指定とカスタムプロンプトの指定との併用はできない。
            if option == "custom_prompt":
                # QA の最終質問のprompt(stuff用)
                PROMPT = PromptTemplate(
                    template=prompt_template, input_variables=["context", "question"]
                )
                # promptを利用する stuffing のQAチェイン
                doc_chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)
            else:
                # map_reduceを使用する場合
                doc_chain = load_qa_chain(llm, chain_type="map_reduce")

            question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
            qa = ConversationalRetrievalChain(
                retriever=db.as_retriever(), # Source の vector store
                return_source_documents=True,
                question_generator=question_generator, # 要するに、何？　という質問を作る chain
                combine_docs_chain=doc_chain, # つまりユーザーの質問に対する答えは、という load_qa_chain
            )
        else:
            raise Exception("optionが予期せぬ値です")

        return qa


    # チャット履歴を保存するためのクラス。slackの関数を挟むとリストでは保持できないことがわかったので。
    class ChatHistoryMemory():

        def __init__(self) -> None:
            self.chat_history = []

        def update_chat_history(self, new_chat_history):
            self.chat_history = new_chat_history


    def get_result_from_qa_and_convs(self, qa, query, chat_history_memory: ChatHistoryMemory, max_window_memory_size=5):

        logger.debug("run!")

        chat_history = chat_history_memory.chat_history

        # search distance に閾値を設定してフィルタがかけられる
        vectordbkwargs = {"search_distance": 0.8}

        result = qa({"question": query, "chat_history": chat_history, "vectordbkwargs": vectordbkwargs})

        # chat_historyの更新(末尾に詰めていく)
        chat_history = chat_history + [(query, result["answer"])]
        if len(chat_history) > max_window_memory_size:
            # 古いのから消す
            del chat_history[0]

        chat_history_memory.update_chat_history(chat_history)

        return result["answer"], result["source_documents"]


    def load_emb_db(self, path):

        logger.debug("run!")

        embeddings = OpenAIEmbeddings()
        db = FAISS.load_local(path, embeddings)
        return db


    def save_emb_db(self, path, db):

        logger.debug("run!")

        # ベクターストアの保存
        db.save_local(str(path))



    def predict_chatmodel_message(self, system_template, human_template, prompt_params, model="gpt-3.5-turbo"):

        logger.debug("run!")

        # TODO: temperatureを可変にする
        # NOTE: だが応答速度はtempertureが低いほど早いので、基本0で良いかと。
        chat = ChatOpenAI(model_name=model, temperature=0)
        # create message
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

        # convert to message
        messages = chat_prompt.format_messages(**prompt_params)

        return chat.predict_messages(messages)


    def predict_llm(self, prompt_template, prompt_params):

        logger.debug("run!")

        llm = OpenAI(model_name="text-davinci-003")
        prompt = PromptTemplate.from_template(prompt_template)

        # It can be written using LLMChain
        # chain = LLMChain(llm=llm, prompt=prompt)
        # result = chain.run(**prompt_params)
        result = llm(prompt_template.format(**prompt_params))
        return result


    def create_boss_talk_for_translation(self, df_talks, boss_name):
        df_boss_talks = df_talks.query("user == @boss_name").reset_index(drop=True)
        assert len(df_boss_talks) > 0, "上司の名前が間違っている可能性が高いです！\n'user_name_master.json'に記載の名前にしてください。"

        # remove quoted passages from the conversation
        # TODO: なんかちゃんとできていないぽい
        df_boss_talks["text"] = df_boss_talks["text"].map(lambda x: re.sub(r"&gt;(.*?)\n", r"\n", x) if x==x else np.nan)

        # 行をシャッフル
        df_boss_talks = df_boss_talks.sample(frac=1).reset_index(drop=True)

        na_boss_talk = df_boss_talks["text"].values
        # delete nan
        na_boss_talk = na_boss_talk[~pd.isna(na_boss_talk)]
        str_boss_talk = "".join(na_boss_talk)

        # decrease the length to fit within the token limit
        # TODO: トークン数をちゃんと測る。
        str_boss_talk = str_boss_talk[-1500:]

        return str_boss_talk


    def prepare_mbbot_translate(
        self, boss_name, str_boss_talk, qa,
        talk_features_prompt_template, thought_style_prompt_template,
        model,
    ):

        logger.debug("run!")

        prompt_params = {
            "boss_name": boss_name,
            "boss_talk": str_boss_talk,
        }
        system_template = ""
        # res_talk_features = self.predict_llm(talk_features_prompt_template, prompt_params)
        res_talk_features = self.predict_chatmodel_message(
            system_template, talk_features_prompt_template, prompt_params, model
        )

        prompt_params = {
            "boss_name": boss_name,
        }
        query = thought_style_prompt_template.format(**prompt_params)
        res_thought_style, source_docs = self.get_result_from_qa(qa, query)

        result = (res_talk_features, res_thought_style)
        return result


    def mbbot_translate(
        self, boss_name, res_talk_features, res_talk_examples, input, translate_prompt_template,
        model,
    ):

        logger.debug("run!")

        prompt_params = {
            "boss_name": boss_name,
            "talk_features": res_talk_features,
            "talk_examples": res_talk_examples,
        }
        prompt_params["input"] = input
        system_template = ""
        # res_translate = self.predict_llm(translate_prompt_template, prompt_params)
        res_translate = self.predict_chatmodel_message(
            system_template, translate_prompt_template, prompt_params, model
        )

        return res_translate
