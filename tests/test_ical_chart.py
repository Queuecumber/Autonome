"""Opt-in private iCalendar feed deployment and durable state wiring."""

import pytest

from test_graphiti_chart import helm_binary, render, resources


def values(**overrides):
    """Enable a synthetic calendar deployment referencing a preexisting Kubernetes Secret."""
    return {"services": {"icalMcp": {"enabled": True,
            "urlsSecretRef": {"name": "private-calendar-feeds", "key": "URLS"}, **overrides}}}


def test_calendar_disabled_by_default(render):
    """Existing chart installs acquire no calendar resources without opt-in."""
    names = {obj["metadata"]["name"] for obj in resources(render({}))}
    assert not {"ical-mcp", "embedding-test-ical-mcp", "embedding-test-ical"} & names


def test_calendar_secret_events_and_storage(render):
    """URLs remain in a Secret; the pod receives passive routing and persistent state."""
    objects = resources(render(values(sessionId="calendar-session")))
    deployment = next(obj for obj in objects if obj["kind"] == "Deployment"
                      and obj["metadata"]["name"] == "embedding-test-ical-mcp")
    assert deployment["spec"]["replicas"] == 1
    assert deployment["spec"]["strategy"]["type"] == "Recreate"
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    env = {item["name"]: item for item in container["env"]}
    assert env["ICAL_URLS"]["valueFrom"]["secretKeyRef"] == {
        "name": "private-calendar-feeds", "key": "URLS", "optional": False}
    assert env["ICAL_EVENT_ENERGY"]["value"] == "passive"
    assert env["ICAL_SESSION_ID"]["value"] == "calendar-session"
    assert env["ICAL_TIMEZONE"]["value"] == "America/New_York"
    assert env["ICAL_REFRESH_SECONDS"]["value"] == "300"
    assert env["ICAL_STATE_PATH"]["value"] == "/data/ical.sqlite3"
    assert env["SESSION_MANAGER_URL"]["value"] == "http://session-manager:5000"
    assert any(obj["kind"] == "PersistentVolumeClaim" and obj["metadata"]["name"] == "embedding-test-ical"
               for obj in objects)


def test_calendar_timezone_override(render):
    """A service-specific floating-time zone can override the chart's global timezone."""
    objects = resources(render(values(timezone="Europe/London")))
    pod = next(obj for obj in objects if obj["kind"] == "Deployment"
               and obj["metadata"]["name"] == "embedding-test-ical-mcp")
    env = pod["spec"]["template"]["spec"]["containers"][0]["env"]
    assert next(item["value"] for item in env if item["name"] == "ICAL_TIMEZONE") == "Europe/London"


@pytest.mark.parametrize("overrides", [{"urlsSecretRef": {"name": ""}}, {"urlsSecretRef": {"key": ""}},
                                      {"refreshSeconds": 0}, {"timeoutSeconds": -1},
                                      {"maxFeedBytes": 0}, {"eventEnergy": "invalid"}])
def test_invalid_calendar_values_are_rejected(render, overrides):
    """Malformed secret references and invalid limits fail Helm rendering."""
    assert render(values(**overrides)).returncode != 0
