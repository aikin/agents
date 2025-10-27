from unittest.mock import Mock, patch

import pytest

from services.notification_service import NotificationService


@pytest.fixture
def notif_service():
    """Create notification service."""
    with patch('services.notification_service.settings') as mock_settings:
        mock_settings.pushover_user = "test_user"
        mock_settings.pushover_token = "test_token"
        return NotificationService()


def test_send_notification_success(notif_service):
    with patch('services.notification_service.requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        result = notif_service.send("Test message")

        assert result is True
        mock_post.assert_called_once()


def test_send_notification_disabled():
    with patch('services.notification_service.settings') as mock_settings:
        mock_settings.pushover_user = None
        mock_settings.pushover_token = None

        notif = NotificationService()
        result = notif.send("Test")

        assert result is False
