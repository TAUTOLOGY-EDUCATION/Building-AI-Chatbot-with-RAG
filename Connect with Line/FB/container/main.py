import base64
import json
import hashlib
import hmac

from fastapi import FastAPI, Request, Response, HTTPException, status
import requests
from langchain_core.messages import BaseMessage


from facebook_chat import make_facebook_chat
from line_chat import make_line_chat

app = FastAPI()

PAGE_ID: str
PAGE_ACCESS_TOKEN: str

with open("/run/secrets/facebook.json") as facebook_page_secrets:
    secrets = json.load(facebook_page_secrets)
    PAGE_ID = secrets["PAGE_ID"]
    PAGE_ACCESS_TOKEN = secrets["PAGE_ACCESS_TOKEN"]


CHANNEL_SECRET: str
CHANNEL_ACCESS_TOKEN: str

with open("/run/secrets/line.json") as line_secrets:
    secrets = json.load(line_secrets)
    CHANNEL_SECRET = secrets["CHANNEL_SECRET"]
    CHANNEL_ACCESS_TOKEN = secrets["CHANNEL_ACCESS_TOKEN"]


@app.get("/")
def hello_world():
    return {"data": "HelloWorld"}

@app.get("/webhook")
async def listen(request:Request):

    return Response(verify_webhook(request))


def verify_webhook(req:Request)->str:
    
    calling_verify_token = req.query_params.get("hub.verify_token")
    fb_verify_token = "951357"
    if calling_verify_token == fb_verify_token:
        challenge = req.query_params.get("hub.challenge")
        if challenge:
            return challenge
        raise ValueError
    else:
        raise ValueError

user_chat_historys:dict[str,list[tuple[str, str]]] = {}

@app.post("/webhook")
async def call_back(data:dict):
    user_input = (data["entry"][0]["messaging"][0]["message"]["text"])
    recipient = data["entry"][0]["messaging"][0]["sender"]['id']
    if recipient not in user_chat_historys:
        user_chat_historys[recipient] = []
    result = make_facebook_chat(user_input, user_chat_historys[recipient])
    print(user_chat_historys[recipient])
    fb_api_url = "https://graph.facebook.com/v2.6/"

    payload = {
        "message": {
            "text": result
        },
        "recipient": {
            "id": recipient
        },
        "notification_type": "regular"
    }
    auth = {
        "access_token": PAGE_ACCESS_TOKEN
    }
    response = requests.post(
        fb_api_url+"me/messages",
        params=auth,
        json=payload
    )
    print(response.status_code)
    print(response.content)

line_user_chat_history:dict[str, list[BaseMessage]] = {}

@app.post("/line")
async def line_verify(req:Request):
    header = req.headers
    temp: str | None = header.get("x-line-signature")
    expected_signature: str
    if temp:
        expected_signature = temp
    raw = await req.body()
    body = raw.decode("utf-8")
    json_body = json.loads(body)
    hash_value = hmac.new(CHANNEL_SECRET.encode("utf-8"), body.encode("utf-8"), hashlib.sha256).digest()
    incoming_signature: str = base64.b64encode(hash_value).decode("utf-8")
    if not hash_verify(expected_signature, incoming_signature):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED)
    
    user = json_body["events"][0]['source']['userId']
    if user not in line_user_chat_history:
        line_user_chat_history[user] = []

    incoming_msg:str = json_body['events'][0]['message']['text']
    resp = requests.post("https://api.line.me/v2/bot/message/push"
                  ,headers={
                      "Authorization": f"Bearer {CHANNEL_ACCESS_TOKEN}"
                  }
                  ,json={
        "to": user,
        "messages":[
        {
            "type":"text",
            "text": make_line_chat(incoming_msg,line_user_chat_history[user])
        },]
    })
    print(resp.status_code)
    print(resp.content)


def hash_verify(hash1: str, hash2: str)->bool:
    returned: bool = True
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            returned = False
    return returned