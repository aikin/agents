"""Notification service for Pushover integration."""

from typing import Optional

import requests

from config.settings import settings


class NotificationService:

    def __init__(self):
        self.user = settings.pushover_user
        self.token = settings.pushover_token
        self.enabled = bool(self.user and self.token)

    def send(self, message: str, title: Optional[str] = None) -> bool:
        if not self.enabled:
            return False

        try:
            response = requests.post(
                "https://api.pushover.net/1/messages.json",
                data={
                    "token": self.token,
                    "user": self.user,
                    "message": message,
                    "title": title or "Career Chatbot"
                },
                timeout=5
            )
            return bool(response.status_code == 200)
        except Exception as e:
            print(f"Notification error: {e}")
            return False
