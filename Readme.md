# フォルダ/ファイル構成

- prompt/
  - プロンプトのテキストファイルが格納されているディレクトリです。
  - プロンプトは後述のデプロイ手順のところで S3 に格納します。
- src/
  - 上司 BOT 用に作成した自作モジュール群です
  - リポジトリ`myboss-bot-lambda-<boss name>`の`src/`と同じです
- app3.py
  - Slack のトーク履歴の収集、embedding によるベクターストアの作成を行うスクリプトです。
  - Slack のトーク履歴の収集は、Slack API を利用しています。
    - 毎回全件取得すると時間がかかる＆API の RateLimit に引っかかるため、差分更新できるようにしています。
    - トーク履歴の収集対象となるチャンネルは、下記で作成する slack アプリ「manager-bot-talk-collection-app」がメンバーとして参加しているチャンネルに限ります。
  - embedding によるベクターストアの作成は、OpenAI の API を利用しています。
    - こちらも毎回全件を embedding すると費用がかかる＆API の RateLimit に引っかかるため、差分更新できるようにしています。
  - ローカル環境 or lambda での実行を想定しています
- serverless.yml
  - Serverless Framework を使ってスクリプトを lambda にデプロイするための設定ファイルです。
  - `app3.py`の実行は日次に設定しています。
- package.json, package-lock.json
  - Node.js のパッケージのインストール用のファイルです
- requirements.txt
  - python のパッケージのインストール用のファイルです
- settings_conf.ini
  - 上司 Bot の設定ファイルです
- .env (git 対象外)
  - OpenAI の API キーや Slack の Signing Secret などの秘密情報を記載するファイルです

<br/>
<br/>
<br/>

# 開発言語、インフラ

- python3.11
  - バージョンが異なるとエラーとなる恐れがあるのでご注意ください。
- Serverless Framework
  - AWS Lambda へのデプロイを行うために利用しています
- Node.js (v12 以上)
  - Serverless Framework が Node.js で提供されているためインストールが必要です
- Docker Desktop
  - Serverless Framework が Docker を利用しているためインストールが必要です
- AWS CLI
- AWS Lambda
- AWS S3

<br/>
<br/>
<br/>

# デプロイ方法

※ python3.11, Node.js, Serverless Framework, Docker Desktop, AWS CLI がインストールされていることを前提とします。

<br/>
<br/>

## S3 バケットの準備

1. AWS CLI で`aws s3 mb s3://manager-bot-data --region ap-northeast-1`を実行し、バケットを作成します
2. `s3://manager-bot-data/prompt/`というディレクトリを作成します
3. `s3://manager-bot-data/prompt/`配下に以下４ファイルを格納します。
   1. `mbbot_talk_features_prompt.txt`
   2. `mbbot_thought_style_prompt.txt`
   3. `mbbot_translate_prompt.txt`
   4. `retrievalqa_stuff_custom_prompt.txt`

※ 上記４ファイルはこのリポジトリの`prompt/`配下に格納されています。

<br/>
<br/>

## slack アプリの作成

1. [Slack API: Applications | Slack](https://api.slack.com/apps)へアクセスし、「Create New App」をクリックします。
2. 「From an app manifest」をクリックします。
3. ワークスペースを選択し、「Next」をクリックします。
4. 以下の YAML を貼り付け、「Next」をクリックします。

   1. ```
      display_information:
      name: manager-bot-talk-collection-app
      description: 上司BOTで参照するトーク履歴データを収集するBOT
      background_color: "#1c44bd"
      features:
      app_home:
          home_tab_enabled: true
          messages_tab_enabled: false
          messages_tab_read_only_enabled: true
      bot_user:
          display_name: manager-bot-talk-collection-app
          always_online: true
      oauth_config:
      scopes:
          bot:
          - channels:history
          - groups:history
          - users:read
          - channels:read
          - groups:read
      settings:
      org_deploy_enabled: false
      socket_mode_enabled: false
      token_rotation_enabled: false
      ```

5. 「Create」をクリックします
6. 左側の「Basic Information」-> 「Install your app」に進み、「Install to Workspace」をクリックします
7. 「許可する」をクリックします
8. 左側の「OAuth & Permissions」-> 「OAuth Tokens for Your Workspace」に進み、「Bot User OAuth Token」をメモしておきます。

<br/>
<br/>
<br/>

## lambda へのデプロイ

1. このリポジトリをクローンします
2. 以下のような`.env`ファイルを作成します。
   (`TOKEN_GET_MESSAGES_APP`は上記でメモした「Bot User OAuth Token」の値としてください)

   1. ```
      OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
      SLACK_URL_HISTORY = "https://slack.com/api/conversations.history"
      SLACK_URL_REPLIES = "https://slack.com/api/conversations.replies"
      TOKEN_GET_MESSAGES_APP = "xoxb-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
      SETTINGS_FILE_PATH = "./settings_conf.ini"
      ```

3. `npm ci`を実行します
4. `pip install -r requirements.txt`を実行します
5. lambda の layer の作成準備のため、`xcopy .\src .\layers\src /s /e`を実行します。（コマンドプロンプトの場合）
6. `sls deploy`を実行します
7. AWS Lambda にデプロイされたことを確認し、ログに表示される API Gateway の URL をメモしておきます
8. [Slack API: Applications | Slack](https://api.slack.com/apps)から上記で作成したアプリの編集画面へ進み、左側の「App Manifest」をクリックし、YAML 内の「request_url」にメモした API Gateway の URL を貼り付けます。
9. slack で bot にメッセージを送信すると応答があるはずです。

<br/>
<br/>
<br/>

# ローカルでの実行方法

## app3.py の実行

1. `python app3.py`を実行してください
