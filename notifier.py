# Standard Library
import os

# Third party dependencies
import requests


def notify_trade(message):
    gotify_url = os.getenv("GOTIFY_API_URL")
    app_token = os.getenv("GOTIFY_API_KEY")
    splits = message.split("--")[1]
    files = {
        "title": (None, f"{splits[1]} bought!"),
        "message": (None, splits[2]),
        "priority": (None, "5"),
    }

    response = requests.post(f"{gotify_url}/message?token={app_token}", files=files)
    return response
