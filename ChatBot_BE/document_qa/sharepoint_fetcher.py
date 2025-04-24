# ChatBot_BE/document_qa/sharepoint_fetcher.py

import requests
import msal
import os
from dotenv import load_dotenv

load_dotenv()

TENANT_ID = os.getenv("TENANT_ID")
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
SHAREPOINT_SITE_ID = os.getenv("SHAREPOINT_SITE_ID")

GRAPH_SCOPE = ["https://graph.microsoft.com/.default"]
GRAPH_BASE = "https://graph.microsoft.com/v1.0"

def get_graph_token():
    app = msal.ConfidentialClientApplication(
        client_id=CLIENT_ID,
        client_credential=CLIENT_SECRET,
        authority=f"https://login.microsoftonline.com/{TENANT_ID}"
    )
    result = app.acquire_token_silent(GRAPH_SCOPE, account=None)
    if not result:
        result = app.acquire_token_for_client(scopes=GRAPH_SCOPE)
    if "access_token" not in result:
        raise Exception(f"Token error: {result.get('error_description')}")
    return result["access_token"]

def list_files_in_root():
    token = get_graph_token()
    url = f"{GRAPH_BASE}/sites/{SHAREPOINT_SITE_ID}/drive/root/children"
    headers = {"Authorization": f"Bearer {token}"}
    res = requests.get(url, headers=headers)
    res.raise_for_status()
    return res.json().get("value", [])

def download_file(file_id):
    token = get_graph_token()
    url = f"{GRAPH_BASE}/sites/{SHAREPOINT_SITE_ID}/drive/items/{file_id}/content"
    headers = {"Authorization": f"Bearer {token}"}
    res = requests.get(url, headers=headers)
    res.raise_for_status()
    return res.content  # return raw bytes
