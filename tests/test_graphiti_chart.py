"""Render the embedding chart contract with Helm."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest
import yaml

CHART = Path(__file__).resolve().parents[1] / "charts" / "autonome"


@pytest.fixture(scope="session")
def helm_binary():
    """Resolve Helm, requiring it in CI and skipping optional local render tests otherwise."""
    binary = os.environ.get("HELM_BIN") or shutil.which("helm")
    if binary:
        return binary
    if os.environ.get("REQUIRE_HELM_TESTS") == "1":
        pytest.fail("Helm is required for chart tests", pytrace=False)
    pytest.skip("Set HELM_BIN or install Helm to run chart tests")


@pytest.fixture
def render(helm_binary, tmp_path):
    """Provide a chart renderer using only synthetic credentials and values."""
    def run(overrides):
        """Render overrides and return the Helm process result, including validation errors."""
        values = {
            "agent": {"config": "test config", "personality": "test personality"},
            "openai": {"apiKey": "test-chat-key"},
            "matrix": {"password": "test-matrix-password"},
            **overrides,
        }
        values_file = tmp_path / "values.yaml"
        values_file.write_text(yaml.safe_dump(values))
        return subprocess.run(
            [helm_binary, "template", "embedding-test", str(CHART), "-f", str(values_file)],
            capture_output=True, text=True, check=False,
        )
    return run


def resources(result):
    """Parse a successful render into Kubernetes resources, asserting Helm succeeded."""
    assert result.returncode == 0, result.stderr
    return list(yaml.safe_load_all(result.stdout))


def graph_container(objects):
    """Return the graph MCP container from rendered resources."""
    deployment = next(obj for obj in objects if obj["kind"] == "Deployment"
                      and obj["metadata"]["name"] == "embedding-test-graphiti-mcp")
    return deployment["spec"]["template"]["spec"]["containers"][0]


def test_default_chart_keeps_embeddings_disabled(render):
    """Chart defaults do not turn on a paid or unconfigured embedding endpoint."""
    objects = resources(render({}))
    env = {item["name"]: item for item in graph_container(objects)["env"]}
    assert env["EMBEDDING_MODEL"]["value"] == ""
    assert env["EMBEDDING_DIM"]["value"] == "0"
    assert env["EMBEDDING_PROVIDER"]["value"] == "nvidia"
    secret = next(obj for obj in objects if obj["kind"] == "Secret")
    assert "EMBEDDING_API_KEY" not in secret["stringData"]


def test_embedding_settings_and_inline_secret_reach_the_container(render):
    """All configured embedding settings render without putting API keys in the pod spec."""
    embedding = {
        "model": "nvidia/nvidia/example-embedding", "baseUrl": "https://gateway.test/v1",
        "provider": "nvidia", "dim": 2048, "timeoutSeconds": 12, "minScore": 0.3,
        "apiKey": "embedding-token",
    }
    objects = resources(render({"services": {"graphitiMcp": {"embedding": embedding}}}))
    container = graph_container(objects)
    env = {item["name"]: item for item in container["env"]}
    for name, value in {
        "EMBEDDING_MODEL": embedding["model"], "EMBEDDING_BASE_URL": embedding["baseUrl"],
        "EMBEDDING_PROVIDER": "nvidia", "EMBEDDING_DIM": "2048", "EMBEDDING_TIMEOUT_SECONDS": "12",
        "EMBEDDING_MIN_SCORE": "0.3",
    }.items():
        assert env[name]["value"] == value
    assert "embedding-token" not in json.dumps(container)
    secret = next(obj for obj in objects if obj["kind"] == "Secret")
    assert secret["stringData"]["EMBEDDING_API_KEY"] == "embedding-token"
    assert env["EMBEDDING_API_KEY"]["valueFrom"]["secretKeyRef"]["name"] == "embedding-test-secrets"


def test_external_embedding_secret_can_use_an_existing_gateway_key(render):
    """An existing gateway secret is referenced directly without copying its contents."""
    objects = resources(render({
        "secrets": {"create": False, "existingName": "agent-secrets"},
        "services": {"graphitiMcp": {"embedding": {
            "apiKeySecretRef": {"name": "gateway-secrets", "key": "OPENAI_API_KEY"},
        }}},
    }))
    env = {item["name"]: item for item in graph_container(objects)["env"]}
    assert env["EMBEDDING_API_KEY"]["valueFrom"]["secretKeyRef"] == {
        "name": "gateway-secrets", "key": "OPENAI_API_KEY", "optional": False,
    }
    assert not any(obj["kind"] == "Secret" for obj in objects)


def test_inline_embedding_secret_rotation_changes_pod_template(render):
    """A Helm-managed key rotation triggers a graph MCP rollout."""
    checksums = []
    for token in ["first-key", "second-key"]:
        objects = resources(render({"services": {"graphitiMcp": {"embedding": {"apiKey": token}}}}))
        deployment = next(obj for obj in objects if obj["kind"] == "Deployment"
                          and obj["metadata"]["name"] == "embedding-test-graphiti-mcp")
        checksums.append(deployment["spec"]["template"]["metadata"]["annotations"]["checksum/embedding-secret"])
    assert checksums[0] != checksums[1]


def test_unrelated_secret_rotation_does_not_restart_graph_memory(render):
    """Changing the Matrix password must not alter the embedding pod template."""
    templates = []
    for password in ["first-password", "second-password"]:
        objects = resources(render({"matrix": {"password": password}}))
        deployment = next(obj for obj in objects if obj["kind"] == "Deployment"
                          and obj["metadata"]["name"] == "embedding-test-graphiti-mcp")
        templates.append(deployment["spec"]["template"])
    assert templates[0] == templates[1]


def test_reused_gateway_key_rotation_changes_pod_template(render):
    """A shared gateway key triggers a rollout when graph memory references that key."""
    checksums = []
    for token in ["first-key", "second-key"]:
        objects = resources(render({
            "openai": {"apiKey": token},
            "services": {"graphitiMcp": {"embedding": {"apiKeySecretRef": {"key": "OPENAI_API_KEY"}}}},
        }))
        deployment = next(obj for obj in objects if obj["kind"] == "Deployment"
                          and obj["metadata"]["name"] == "embedding-test-graphiti-mcp")
        checksums.append(deployment["spec"]["template"]["metadata"]["annotations"]["checksum/embedding-secret"])
    assert checksums[0] != checksums[1]


@pytest.mark.parametrize("inline,expected", [("", "legacy-key"), ("new-key", "new-key")])
def test_existing_secret_env_embedding_key_is_preserved(render, inline, expected):
    """The dedicated key takes precedence while the earlier secretEnv route still works."""
    objects = resources(render({
        "mcp": {"secretEnv": {"EMBEDDING_API_KEY": "legacy-key", "OTHER_TOKEN": "other"}},
        "services": {"graphitiMcp": {"embedding": {"apiKey": inline}}},
    }))
    secret = next(obj for obj in objects if obj["kind"] == "Secret")
    assert secret["stringData"]["EMBEDDING_API_KEY"] == expected
    assert secret["stringData"]["OTHER_TOKEN"] == "other"


@pytest.mark.parametrize("embedding", [
    {"provider": "unsupported"}, {"dim": -1}, {"dim": 1.5}, {"timeoutSeconds": 0},
    {"minScore": 1.1}, {"minScore": -0.1},
    {"model": "nvidia/example", "baseUrl": ""},
    {"apiKey": "token", "apiKeySecretRef": {"name": "also-external"}},
    {"apiKey": "token", "apiKeySecretRef": {"key": "OTHER_KEY"}},
])
def test_invalid_embedding_configuration_is_rejected(render, embedding):
    """Invalid or conflicting settings fail before creating a broken deployment."""
    result = render({"services": {"graphitiMcp": {"embedding": embedding}}})
    assert result.returncode != 0


def test_inline_key_requires_a_chart_managed_secret(render):
    """An inline credential must not be silently ignored when secret creation is disabled."""
    result = render({
        "secrets": {"create": False},
        "services": {"graphitiMcp": {"embedding": {"apiKey": "token"}}},
    })
    assert result.returncode != 0
    assert "secrets.create=true" in result.stderr


def test_null_embedding_configuration_still_renders(render):
    """Explicitly disabling the optional configuration remains valid."""
    objects = resources(render({"services": {"graphitiMcp": {"embedding": None}}}))
    assert graph_container(objects)["name"] == "graphiti-mcp"
