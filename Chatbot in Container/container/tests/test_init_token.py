from fastapi.testclient import TestClient
from main import app
def test_access_token():
    client = TestClient(app)
    client.get("/webhook?hub.mode=subscribe&hub.verify_token=mytoken&hub.challenge=1158201444")

    assert False