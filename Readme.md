# フォルダ/ファイル構成

- prompt/
  - プロンプトのテキストファイルが格納されているディレクトリです。
  - プロンプトは後述のデプロイ手順のところでS3に格納します。
- src/
  - 上司BOT用に作成した自作モジュール群です
  - リポジトリ`myboss-bot-lambda-nakagawara`の`src/`と同じです
- app3.py
  - Slackのトーク履歴の収集、embeddingによるベクターストアの作成を行うスクリプトです。
  - Slackのトーク履歴の収集は、Slack APIを利用しています。
    - 毎回全件取得すると時間がかかる＆APIのRateLimitに引っかかるため、差分更新できるようにしています。
    - トーク履歴の収集対象となるチャンネルは、下記で作成するslackアプリ「manager-bot-talk-collection-app」がメンバーとして参加しているチャンネルに限ります。
  - embeddingによるベクターストアの作成は、OpenAIのAPIを利用しています。
    - こちらも毎回全件をembeddingすると費用がかかる＆APIのRateLimitに引っかかるため、差分更新できるようにしています。
  - ローカル環境 or lambdaでの実行を想定しています
- serverless.yml
  - Serverless Frameworkを使ってスクリプトをlambdaにデプロイするための設定ファイルです。
  - `app3.py`の実行は日次に設定しています。
- package.json, package-lock.json
  - Node.jsのパッケージのインストール用のファイルです
- requirements.txt
  - pythonのパッケージのインストール用のファイルです
- settings_conf.ini
  - 上司Botの設定ファイルです
- .env (git対象外)
  - OpenAIのAPIキーやSlackのSigning Secretなどの秘密情報を記載するファイルです

<br/>
<br/>
<br/>

# 開発言語、インフラ

- python3.11
  - バージョンが異なるとエラーとなる恐れがあるのでご注意ください。
- Serverless Framework
  - AWS Lambdaへのデプロイを行うために利用しています
- Node.js (v12以上)
  - Serverless FrameworkがNode.jsで提供されているためインストールが必要です
- Docker Desktop
  - Serverless FrameworkがDockerを利用しているためインストールが必要です
- AWS CLI
- AWS Lambda
- AWS S3

<br/>
<br/>
<br/>

# デプロイ方法

※ python3.11, Node.js, Serverless Framework, Docker Desktop, AWS CLIがインストールされていることを前提とします。

<br/>
<br/>


## S3バケットの準備

1. AWS CLIで`aws s3 mb s3://manager-bot-data --region ap-northeast-1`を実行し、バケットを作成します
2. `s3://manager-bot-data/prompt/`というディレクトリを作成します
3. `s3://manager-bot-data/prompt/`配下に以下４ファイルを格納します。
   1. `mbbot_talk_features_prompt.txt`
   2. `mbbot_thought_style_prompt.txt`
   3. `mbbot_translate_prompt.txt`
   4. `retrievalqa_stuff_custom_prompt.txt`

※ 上記４ファイルはこのリポジトリの`prompt/`配下に格納されています。

<br/>
<br/>


## slackアプリの作成

1. [Slack API: Applications | Slack](https://api.slack.com/apps)へアクセスし、「Create New App」をクリックします。
2. 「From an app manifest」をクリックします。
3. ワークスペース（Remain in）を選択し、「Next」をクリックします。
4. 以下のYAMLを貼り付け、「Next」をクリックします。
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
9. 左側の「OAuth & Permissions」-> 「OAuth Tokens for Your Workspace」に進み、「Bot User OAuth Token」をメモしておきます。

<br/>
<br/>
<br/>

## lambdaへのデプロイ

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
1. `npm ci`を実行します
2. `pip install -r requirements.txt`を実行します
3. lambdaのlayerの作成準備のため、`xcopy .\src .\layers\src /s /e`を実行します。（コマンドプロンプトの場合）
4. `sls deploy`を実行します
5. AWS Lambdaにデプロイされたことを確認し、ログに表示されるAPI GatewayのURLをメモしておきます
6. [Slack API: Applications | Slack](https://api.slack.com/apps)から上記で作成したアプリの編集画面へ進み、左側の「App Manifest」をクリックし、YAML内の「request_url」にメモしたAPI GatewayのURLを貼り付けます。
7. slackでbotにメッセージを送信すると応答があるはずです。


<br/>
<br/>
<br/>



# ローカルでの実行方法


## app3.pyの実行

1. `python app3.py`を実行してください

