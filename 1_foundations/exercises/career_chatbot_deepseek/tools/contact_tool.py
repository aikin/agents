from typing import Any

from services.database_service import DatabaseService
from services.notification_service import NotificationService
from tools.base import BaseTool


class ContactTool(BaseTool):
    """Tool for recording user contact information."""

    def __init__(self, db_service: DatabaseService, notif_service: NotificationService):
        self.db_service = db_service
        self.notif_service = notif_service

    @property
    def name(self) -> str:
        return "record_user_details"

    @property
    def description(self) -> str:
        return "Record user contact details when thery want to get in touch"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "email": {"type": "string", "description": "User's email"},
                "name": {"type": "string", "description": "User's name"},
                "notes": {"type": "string", "description": "Additional notes"},
            },
            "required": ["email"],
        }

    def execute(self, **kwargs) -> dict[str, Any]:
        email = kwargs.get("email", "")
        name = kwargs.get("name", "")
        notes = kwargs.get("notes", "")

        try:
            contact_id = self.db_service.save_contact(email, name, notes)
            self.notif_service.send(f"New contact: {email} ({name})")
            return {"success": True, "message": f"Contact saved (ID: {contact_id})"}
        except Exception as e:
            return {"success": False, "message": str(e)}
