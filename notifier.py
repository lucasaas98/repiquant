# Standard Library
import os

# Third party dependencies
import requests


def notify_trade(message):
    gotify_url = os.getenv("GOTIFY_API_URL")
    app_token = os.getenv("GOTIFY_API_KEY")
    ticker = message.split("--")[1]
    files = {
        "title": (None, f"{ticker} bought!"),
        "message": (None, message),
        "priority": (None, "5"),
    }

    response = requests.post(f"{gotify_url}/message?token={app_token}", files=files)
    return response
