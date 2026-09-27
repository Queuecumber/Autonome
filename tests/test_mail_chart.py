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
    assert env["IMAP_NOTIFY_SINCE"]["value"] == "startup"
    assert env["IMAP_ID_PROVIDER"]["value"] == "auto"
    assert env["IMAP_LOOKUP_MAX_MESSAGES"]["value"] == "200"
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
    ("imap", {"eventEnergy": "invalid"}), ("imap", {"server": "http://mail.test"}),
    ("imap", {"idProvider": "unknown"}), ("imap", {"lookupMaxMessages": 0}),
    ("imap", {"lookupMaxMessages": 5001}), ("imap", {"lookupMaxMessages": 1.5}),
    ("smtp", {"from": ""}), ("smtp", {"allowAnyRecipient": "false"}),
    ("smtp", {"maxAttachmentBytes": -1}), ("smtp", {"server": "http://mail.test"}),
])
def test_invalid_mail_values_fail_before_deployment(render, kind, overrides):
    """Incomplete credentials, unsafe transports, and malformed policy cannot silently render."""
    assert render(configured(kind, **overrides)).returncode != 0


@pytest.mark.parametrize("value", ["all", "2026-09-14", "2026-09-14T19:00:00Z"])
def test_imap_notification_cutoff_is_configurable(render, value):
    """Explicit floor settings reach the adapter without losing timestamp timezone information."""
    objects = resources(render(configured("imap", notifySince=value)))
    env = deployment(objects, "imap")["spec"]["template"]["spec"]["containers"][0]["env"]
    assert next(item["value"] for item in env if item["name"] == "IMAP_NOTIFY_SINCE") == value


@pytest.mark.parametrize("value", ["", "yesterday", "2026-02-31", "2026-09-14T09:00:00"])
def test_imap_notification_cutoff_rejects_invalid_dates(render, value):
    """Invalid or timezone-ambiguous chart cutoffs fail before deployment."""
    assert render(configured("imap", notifySince=value)).returncode != 0


def test_smtp_no_auth_relay_configuration_is_preserved(render):
    """Notification changes retain the user's existing trusted-relay deployment behavior."""
    objects = resources(render(configured("smtp", server="smtp://relay.test:25", username="",
                                         passwordSecretRef={"name": "", "key": "SMTP_PASSWORD"})))
    container = deployment(objects, "smtp")["spec"]["template"]["spec"]["containers"][0]
    env = {item["name"]: item for item in container["env"]}
    assert "SMTP_PASSWORD" not in env
    assert env["SMTP_SERVER"]["value"] == "smtp://relay.test:25"


def test_imap_native_identity_settings_reach_the_adapter(render):
    """Provider overrides and bounded recovery settings are available through the chart."""
    objects = resources(render(configured("imap", idProvider="proton", lookupMaxMessages=50)))
    env = deployment(objects, "imap")["spec"]["template"]["spec"]["containers"][0]["env"]
    values = {item["name"]: item.get("value") for item in env}
    assert values["IMAP_ID_PROVIDER"] == "proton"
    assert values["IMAP_LOOKUP_MAX_MESSAGES"] == "50"


def test_index_defaults_do_not_put_new_database_on_notification_pvc(render):
    """Default indexing uses a bounded pod-local directory even when global storage is NFS."""
    objects = resources(render({**configured("imap"), "storage": {"storageClass": "nfs"}}))
    pod = deployment(objects, "imap")["spec"]["template"]["spec"]
    volumes = {volume["name"]: volume for volume in pod["volumes"]}
    assert volumes["index"]["emptyDir"] == {"sizeLimit": "5Gi"}
    assert volumes["state"]["persistentVolumeClaim"]["claimName"] == "embedding-test-imap"
    env = {item["name"]: item.get("value") for item in pod["containers"][0]["env"]}
    assert env["IMAP_INDEX_DIR"] == "/index" and env["IMAP_INDEX_JOURNAL_MODE"] == "DELETE"
    assert env["IMAP_INDEX_ENABLED"] == "true" and env["IMAP_EMBEDDING_MODEL"] == ""
    assert "IMAP_EMBEDDING_API_KEY" not in env


@pytest.mark.parametrize("storage", [
    {"persistent": True, "storageClass": "local-block", "size": "8Gi"},
    {"persistent": True, "existingClaim": "mail-search"},
])
def test_index_persistence_is_separate_and_explicit(render, storage):
    """A dedicated block/local claim can preserve index progress without resizing event state."""
    objects = resources(render(configured("imap", index={"storage": storage})))
    pod = deployment(objects, "imap")["spec"]["template"]["spec"]
    volume = next(item for item in pod["volumes"] if item["name"] == "index")
    assert volume["persistentVolumeClaim"]["claimName"] == storage.get("existingClaim", "embedding-test-imap-index")
    claims = [item for item in objects if item["kind"] == "PersistentVolumeClaim"
              and item["metadata"]["name"] == "embedding-test-imap-index"]
    if "existingClaim" in storage:
        assert not claims
    else:
        assert claims[0]["spec"]["storageClassName"] == "local-block"
        assert claims[0]["spec"]["resources"]["requests"]["storage"] == "8Gi"


def test_index_embedding_settings_and_secret_references(render):
    """NVIDIA encoding and operator-selected endpoint/model credentials reach only the IMAP pod."""
    config = {"model": "nvidia/nvidia/nemotron-3-embed-1b", "baseUrl": "https://embedding.test/v1",
              "provider": "nvidia", "dim": 2048, "minScore": 0.6,
              "apiKeySecretRef": {"name": "embedding-secret", "key": "KEY"}}
    objects = resources(render(configured("imap", index={"embedding": config})))
    env = {item["name"]: item for item in deployment(objects, "imap")["spec"]["template"]["spec"]["containers"][0]["env"]}
    assert env["IMAP_EMBEDDING_MODEL"]["value"] == config["model"]
    assert env["IMAP_EMBEDDING_DIM"]["value"] == "2048"
    assert env["IMAP_EMBEDDING_API_KEY"]["valueFrom"]["secretKeyRef"] == {
        "name": "embedding-secret", "key": "KEY", "optional": False}


def test_disabling_index_omits_index_storage_and_embedding_credentials(render):
    """Live server search remains deployable without the optional local index worker."""
    objects = resources(render(configured("imap", index={"enabled": False})))
    pod = deployment(objects, "imap")["spec"]["template"]["spec"]
    assert [item["name"] for item in pod["volumes"]] == ["state"]
    env = {item["name"]: item for item in pod["containers"][0]["env"]}
    assert env["IMAP_INDEX_ENABLED"]["value"] == "false"
    assert "IMAP_INDEX_DIR" not in env and "IMAP_EMBEDDING_API_KEY" not in env


@pytest.mark.parametrize("change", [
    {"syncSeconds": 0}, {"headerBatch": 0}, {"bodyBatch": 2001}, {"flagBatch": 1.5},
    {"journalMode": "unsafe"}, {"embeddingSeconds": 0},
    {"storage": {"persistent": True}}, {"storage": {"existingClaim": "mail"}},
    {"embedding": {"model": "m"}}, {"embedding": {"dim": -1}},
    {"embedding": {"provider": "bad"}}, {"embedding": {"minScore": 2}},
])
def test_invalid_index_chart_configuration_is_rejected(render, change):
    """Unsafe persistence defaults, typos, and invalid limits fail before deployment."""
    assert render(configured("imap", index=change)).returncode != 0
