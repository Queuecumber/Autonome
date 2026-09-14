"""Opt-in mail services and secret-backed deployment configuration."""

import json

import pytest

from test_graphiti_chart import helm_binary, render, resources


def configured(kind, **overrides):
    """Return enabled synthetic mail values, never real account credentials."""
    values = {"enabled": True, "server": f"{'imaps' if kind == 'imap' else 'smtps'}://mail.test",
              "username": "test-user", "passwordSecretRef": {"name": "mail-passwords", "key": "PASSWORD"}}
    if kind == "smtp":
        values["from"] = "agent@example.test"
    return {"services": {f"{kind}Mcp": {**values, **overrides}}}


def deployment(objects, kind):
    """Return one rendered mail deployment."""
    return next(obj for obj in objects if obj["kind"] == "Deployment"
                and obj["metadata"]["name"] == f"embedding-test-{kind}-mcp")


def test_default_chart_does_not_deploy_mail_or_its_state(render):
    """Existing deployments remain unchanged unless mail is explicitly enabled."""
    objects = resources(render({}))
    names = {obj["metadata"]["name"] for obj in objects}
    assert not {"imap-mcp", "smtp-mcp", "embedding-test-imap", "embedding-test-imap-mcp",
                "embedding-test-smtp-mcp"} & names


def test_imap_push_configuration_and_persistent_state(render):
    """IDLE monitoring uses passive events, exact folder names, a secret, and one persistent worker."""
    objects = resources(render(configured("imap", folders=["INBOX", "Mixed Case"], sessionId="mail")))
    pod = deployment(objects, "imap")
    assert pod["spec"]["replicas"] == 1 and pod["spec"]["strategy"]["type"] == "Recreate"
    container = pod["spec"]["template"]["spec"]["containers"][0]
    env = {item["name"]: item for item in container["env"]}
    assert env["IMAP_PASSWORD"]["valueFrom"]["secretKeyRef"] == {
        "name": "mail-passwords", "key": "PASSWORD", "optional": False}
    assert json.loads(env["IMAP_FOLDERS"]["value"]) == ["INBOX", "Mixed Case"]
    assert env["IMAP_EVENT_ENERGY"]["value"] == "passive"
    assert env["IMAP_SESSION_ID"]["value"] == "mail"
    assert env["SESSION_MANAGER_URL"]["value"] == "http://session-manager:5000"
    assert env["IMAP_STATE_PATH"]["value"] == "/data/imap.sqlite3"
    assert any(obj["kind"] == "PersistentVolumeClaim" and obj["metadata"]["name"] == "embedding-test-imap"
               for obj in objects)


def test_smtp_is_restricted_and_has_no_automatic_event_loop_or_file_mount(render):
    """Outbound mail needs explicit tools; attachments travel as bytes, not host paths."""
    objects = resources(render(configured("smtp", allowedRecipients=["one@example.test"])))
    container = deployment(objects, "smtp")["spec"]["template"]["spec"]["containers"][0]
    env = {item["name"]: item for item in container["env"]}
    assert env["SMTP_FROM"]["value"] == "agent@example.test"
    assert env["SMTP_ALLOW_ANY_RECIPIENT"]["value"] == "false"
    assert json.loads(env["SMTP_ALLOWED_RECIPIENTS"]["value"]) == ["one@example.test"]
    assert "SESSION_MANAGER_URL" not in env
    assert "volumeMounts" not in container


@pytest.mark.parametrize("kind,overrides", [
    ("imap", {"server": ""}), ("imap", {"username": ""}),
    ("imap", {"passwordSecretRef": {"name": ""}}), ("imap", {"folders": []}),
    ("imap", {"folders": ["INBOX", "INBOX"]}), ("imap", {"pollSeconds": 0}),
    ("imap", {"eventEnergy": "invalid"}), ("imap", {"server": "imap://mail.test"}),
    ("smtp", {"from": ""}), ("smtp", {"allowAnyRecipient": "false"}),
    ("smtp", {"maxAttachmentBytes": -1}), ("smtp", {"server": "smtp://mail.test"}),
])
def test_invalid_mail_values_fail_before_deployment(render, kind, overrides):
    """Incomplete credentials, unsafe transports, and malformed policy cannot silently render."""
    assert render(configured(kind, **overrides)).returncode != 0
