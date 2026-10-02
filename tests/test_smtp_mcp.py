"""Explicit outbound mail, recipient policy, MIME attachments, and meeting requests."""

import base64
from datetime import datetime, timedelta, timezone
from email.parser import BytesParser
from email import policy
from unittest.mock import Mock

from fastmcp import Client
from icalendar import Calendar
import pytest

from smtp_mcp import server


@pytest.fixture
def settings():
    """Synthetic sender with a narrow recipient policy."""
    return server.Settings("smtps://smtp.test", "test-user", "test-secret", "Agent <agent@example.test>",
                           ("one@example.test", "two@example.test"))


@pytest.fixture
def smtp(monkeypatch):
    """Record SMTP protocol calls without sending real messages."""
    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    client.send_message.return_value = {}
    monkeypatch.setattr(server, "SMTP", Mock(return_value=client))
    monkeypatch.setattr(server, "SMTP_SSL", Mock(return_value=client))
    return client


@pytest.mark.parametrize("invalid", ["", "a@example.test,b@example.test", "invalid", "a@example.test\nBcc:x@example.test"])
def test_invalid_addresses_fail(invalid):
    """Address entries cannot contain recipient lists or header injection."""
    with pytest.raises(ValueError):
        server.address(invalid)


@pytest.mark.parametrize("overrides", [
    {"server": "http://smtp.test"}, {"server": "smtps://u:p@smtp.test"},
    {"server": "smtps://smtp.test:0"}, {"username": ""}, {"password": ""},
    {"sender": "invalid"}, {"allowed_recipients": ("bad",)}, {"timeout": 0},
])
def test_invalid_smtp_configuration_fails(overrides):
    """Account settings reject insecure or incomplete transport before connecting."""
    with pytest.raises(ValueError):
        server.Settings(**{"server": "smtps://smtp.test", "username": "u", "password": "p",
                           "sender": "agent@example.test", **overrides})


def test_recipient_allowlist_checks_to_cc_and_sender_before_sending(settings, smtp):
    """CC, display names, and caller-supplied From cannot bypass outbound policy."""
    mailer = server.Mailer(settings)
    for to, cc, sender in [(["evil@example.test"], [], None),
                           (["one@example.test"], ["evil@example.test"], None),
                           (["one@example.test"], [], "evil@example.test"), ([], [], None)]:
        with pytest.raises(ValueError):
            mailer.send(to, cc, "Subject", "Body", [], sender)
    server.SMTP_SSL.assert_not_called()
    result = mailer.send(["One <one@example.test>"], ["two@example.test"], "Subject", "**Body**", [])
    assert result["status"] == "accepted" and len(result["accepted"]) == 2
    assert smtp.send_message.call_args.kwargs["from_addr"] == "agent@example.test"
    assert smtp.send_message.call_args.kwargs["to_addrs"] == ["one@example.test", "two@example.test"]
    message = smtp.send_message.call_args.args[0]
    assert message.get_body(preferencelist=("plain",)).get_content().strip() == "**Body**"
    assert "<strong>Body</strong>" in message.get_body(preferencelist=("html",)).get_content()


def test_empty_allowlist_denies_and_unrestricted_requires_explicit_setting(smtp):
    """No recipient restrictions are relaxed implicitly by an empty configuration."""
    restricted = server.Settings("smtps://smtp.test", "u", "p", "agent@example.test")
    with pytest.raises(ValueError):
        server.Mailer(restricted).send(["any@example.test"], [], "s", "b", [])
    unrestricted = server.Settings("smtps://smtp.test", "u", "p", "agent@example.test", allow_any_recipient=True)
    assert server.Mailer(unrestricted).send(["any@example.test"], [], "s", "b", [])["status"] == "accepted"


def test_tls_before_login_and_partial_refusals_are_reported(smtp):
    """STARTTLS precedes authentication; accepted recipients are not hidden by partial failure."""
    config = server.Settings("starttls://smtp.test:1587", "u", "p", "agent@example.test", allow_any_recipient=True)
    smtp.send_message.return_value = {"two@example.test": (550, b"Rejected")}
    result = server.Mailer(config).send(["one@example.test", "two@example.test"], [], "s", "b", [])
    assert [call[0] for call in smtp.method_calls][:3] == ["starttls", "login", "send_message"]
    assert server.SMTP.call_args.kwargs["timeout"] == 20
    assert server.SMTP.call_args.args == ("smtp.test", 1587)
    assert result["status"] == "partial" and result["accepted"] == ["one@example.test"]
    assert result["refused"] == {"two@example.test": 550}


def test_send_errors_are_not_automatically_retried(settings, smtp):
    """An uncertain send outcome does not cause duplicate outgoing messages."""
    smtp.send_message.side_effect = OSError("connection closed")
    with pytest.raises(OSError):
        server.Mailer(settings).send(["one@example.test"], [], "s", "b", [])
    smtp.send_message.assert_called_once()
    smtp.__exit__.assert_called_once()


@pytest.mark.parametrize("attachment", [
    server.Attachment(name="../secret", content=b"x"),
    server.Attachment(name="bad\nname", content=b"x"),
    server.Attachment(name="a.txt", content=b"x", content_type="text/plain\nInjected"),
    server.Attachment(name="a.txt", content=b"x", content_type="invalid"),
    server.Attachment(name="a.txt", content=b"x", calendar_method="REQUEST"),
])
def test_invalid_attachments_fail_before_connecting(settings, smtp, attachment):
    """Malformed attachment headers never reach the SMTP transport."""
    with pytest.raises(ValueError):
        server.Mailer(settings).send(["one@example.test"], [], "s", "b", [attachment])
    server.SMTP_SSL.assert_not_called()


def test_header_and_attachment_size_limits(settings, smtp):
    """Header injection and excessive decoded bytes fail before authentication."""
    mailer = server.Mailer(settings)
    with pytest.raises(ValueError):
        mailer.send(["one@example.test"], [], "bad\nheader", "b", [])
    small = server.Settings("smtps://smtp.test", "u", "p", "agent@example.test",
                            ("one@example.test",), max_attachment_bytes=2)
    with pytest.raises(ValueError, match="exceed"):
        server.Mailer(small).send(["one@example.test"], [], "s", "b", [server.Attachment(name="a", content=b"123")])
    server.SMTP_SSL.assert_not_called()


def test_invitation_preserves_timezones_and_rfc_fields():
    """Invites keep absolute time, standard attendee fields, identity, and alarms."""
    start = datetime(2026, 9, 14, 9, tzinfo=timezone(timedelta(hours=-4)))
    attachment = server.invitation("agent@example.test", ["one@example.test", "two@example.test"],
        start, start + timedelta(hours=1), "Office", "Review", "Details",
        [server.Alarm(trigger=timedelta(minutes=-10))])
    cal = Calendar.from_ical(attachment.content)
    event = cal.walk("VEVENT")[0]
    assert cal["METHOD"] == "REQUEST" and cal["VERSION"] == "2.0"
    assert event["UID"] and event["DTSTAMP"]
    assert event.decoded("DTSTART") == start
    assert event.decoded("DTEND") == start + timedelta(hours=1)
    assert len(event["ATTENDEE"]) == 2 and "ATTENDEES" not in event
    assert str(event["ORGANIZER"]) == "mailto:agent@example.test"
    assert event.walk("VALARM")[0].decoded("TRIGGER") == timedelta(minutes=-10)


@pytest.mark.parametrize("aware,offset", [(False, 1), (True, 0), (True, -1)])
def test_invalid_invitation_dates_fail(aware, offset):
    """Ambiguous local times and inverted intervals are rejected rather than silently rewritten."""
    start = datetime(2026, 9, 14, tzinfo=timezone.utc if aware else None)
    with pytest.raises(ValueError):
        server.invitation("agent@example.test", ["one@example.test"], start,
                          start + timedelta(hours=offset), "", "s", "b", [])


def test_invitation_interval_uses_absolute_time_across_dst_fold():
    """Repeated local clock hours cannot invert an accepted invitation's actual interval."""
    from zoneinfo import ZoneInfo

    zone = ZoneInfo("America/New_York")
    start = datetime(2026, 11, 1, 1, 30, tzinfo=zone, fold=1)
    end = datetime(2026, 11, 1, 1, 40, tzinfo=zone, fold=0)
    with pytest.raises(ValueError):
        server.invitation("agent@example.test", ["one@example.test"], start, end, "", "s", "b", [])
    attachment = server.invitation("agent@example.test", ["one@example.test"], end, start, "", "s", "b", [])
    event = Calendar.from_ical(attachment.content).walk("VEVENT")[0]
    assert event.decoded("DTEND") > event.decoded("DTSTART")


@pytest.mark.asyncio
async def test_mcp_binary_attachments_and_send_annotations(settings, smtp, monkeypatch):
    """Base64 bytes and platform URI argument rewriting work through real MCP validation."""
    from session_manager.mcp import inline_refs, rewrite_binary_params, resolve_uri_args

    monkeypatch.setattr(server, "mailer", server.Mailer(settings))
    async with Client(server.mcp) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
        tool = tools["send_mail"]
        assert not tool.annotations.readOnlyHint and not tool.annotations.idempotentHint
        schema = inline_refs(tool.inputSchema)
        binary_params = rewrite_binary_params(schema)
        assert binary_params

        async def resolve(uri):
            """Resolve a synthetic attachment without accessing any server or filesystem."""
            assert uri == "imap://attachments/message/0"
            return b"\x00\xffattachment"

        args = await resolve_uri_args({"to": ["one@example.test"], "subject": "Report", "body": "Attached",
            "attachments": [{"name": "report.bin", "content": "imap://attachments/message/0"}]}, binary_params, resolve)
        await client.call_tool("send_mail", args)
        start = datetime(2026, 9, 14, tzinfo=timezone.utc)
        await client.call_tool("send_event_invitation", {"attendees": ["one@example.test"],
            "start": start.isoformat(), "end": (start + timedelta(hours=1)).isoformat(),
            "location": "Office", "subject": "Review", "body": "Details"})
    sent = smtp.send_message.call_args_list[0].args[0]
    attached = list(sent.iter_attachments())[0]
    assert attached.get_payload(decode=True) == b"\x00\xffattachment"
    invited = smtp.send_message.call_args_list[1].args[0]
    calendar_part = list(invited.iter_attachments())[0]
    assert calendar_part.get_param("method") == "REQUEST"
    assert calendar_part.get_content_type() == "text/calendar"


def test_environment_and_uninitialized_service(monkeypatch):
    """Env parsing is explicit and SMTP tool access fails before initialization."""
    for name, value in {"SMTP_SERVER": "tls://smtp.test", "SMTP_USERNAME": "user", "SMTP_PASSWORD": "secret",
                        "SMTP_FROM": "agent@example.test", "SMTP_ALLOWED_RECIPIENTS": '["one@example.test"]'}.items():
        monkeypatch.setenv(name, value)
    assert server.Settings.from_env().allowed_recipients == ("one@example.test",)
    assert "secret" not in repr(server.Settings.from_env())
    monkeypatch.setenv("SMTP_ALLOW_ANY_RECIPIENT", "maybe")
    with pytest.raises(ValueError):
        server.Settings.from_env()
    monkeypatch.setenv("SMTP_ALLOWED_RECIPIENTS", "{}")
    with pytest.raises(ValueError):
        server.Settings.from_env()
    monkeypatch.setattr(server, "mailer", None)
    with pytest.raises(RuntimeError):
        server.account()


def test_trusted_plaintext_relay_can_skip_auth_without_relaxing_recipients(smtp):
    """The existing relay support still enforces its sender and recipient policy."""
    settings = server.Settings("smtp://relay.test:25", "", "", "agent@example.test", ("one@example.test",))
    result = server.Mailer(settings).send(["one@example.test"], [], "Subject", "Body", [])
    assert result["status"] == "accepted"
    smtp.starttls.assert_not_called()
    smtp.login.assert_not_called()
    assert server.SMTP.call_args.args == ("relay.test", 25)
    with pytest.raises(ValueError):
        server.Mailer(settings).send(["blocked@example.test"], [], "Subject", "Body", [])
