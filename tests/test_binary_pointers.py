"""Tests for the binary-pointer pipeline.

Covers:
- rewrite_binary_params: schema traversal, in-place rewrite, path recording,
  union variants (anyOf/oneOf/allOf), arrays, nested objects, edge cases.
- resolve_pointer_args: pointer → bytes substitution at recorded paths,
  immutability of input, None/missing/non-string passthrough.
- BinaryStore: save/load round-trip, filename preservation, collision suffix,
  GC retention.
"""

import base64
import time

import pytest

from session_manager.binaries import BinaryStore
from session_manager.mcp import rewrite_binary_params, resolve_pointer_args


# ── rewrite_binary_params ────────────────────────────────────────────────

def test_empty_schema_yields_no_paths():
    assert rewrite_binary_params({}) == []


def test_schema_without_binary_yields_no_paths():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "count": {"type": "integer"},
        },
    }
    assert rewrite_binary_params(schema) == []
    # Non-binary fields should be untouched.
    assert schema["properties"]["name"] == {"type": "string"}


@pytest.mark.parametrize("fmt", ["byte", "binary", "base64"])
def test_single_binary_field_all_formats(fmt):
    schema = {
        "type": "object",
        "properties": {"data": {"type": "string", "format": fmt}},
    }
    paths = rewrite_binary_params(schema)
    assert paths == [("data",)]
    # Format stripped, description injected.
    assert "format" not in schema["properties"]["data"]
    assert "Pointer" in schema["properties"]["data"]["description"]


def test_binary_description_preserves_existing():
    schema = {
        "type": "object",
        "properties": {
            "data": {
                "type": "string",
                "format": "binary",
                "description": "The file bytes.",
            },
        },
    }
    rewrite_binary_params(schema)
    desc = schema["properties"]["data"]["description"]
    assert desc.startswith("The file bytes.")
    assert "Pointer" in desc


def test_multiple_binary_fields_top_level():
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "string", "format": "binary"},
            "b": {"type": "string"},
            "c": {"type": "string", "format": "base64"},
        },
    }
    paths = rewrite_binary_params(schema)
    assert set(paths) == {("a",), ("c",)}


def test_nested_object_with_binary():
    schema = {
        "type": "object",
        "properties": {
            "wrapper": {
                "type": "object",
                "properties": {
                    "data": {"type": "string", "format": "binary"},
                    "name": {"type": "string"},
                },
            },
        },
    }
    paths = rewrite_binary_params(schema)
    assert paths == [("wrapper", "data")]


def test_deeply_nested_inlined_binary():
    """Fully inlined deep nesting should be found."""
    schema = {
        "properties": {
            "input": {
                "properties": {
                    "a": {
                        "properties": {
                            "b": {
                                "properties": {
                                    "c": {"type": "string", "format": "binary"},
                                },
                            },
                        },
                    },
                },
            },
        },
    }
    paths = rewrite_binary_params(schema)
    assert paths == [("input", "a", "b", "c")]


def test_array_of_binary():
    schema = {
        "type": "object",
        "properties": {
            "blobs": {
                "type": "array",
                "items": {"type": "string", "format": "binary"},
            },
        },
    }
    paths = rewrite_binary_params(schema)
    assert paths == [("blobs", "[]")]


def test_array_of_objects_with_binary():
    schema = {
        "type": "object",
        "properties": {
            "files": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "data": {"type": "string", "format": "binary"},
                    },
                },
            },
        },
    }
    paths = rewrite_binary_params(schema)
    assert paths == [("files", "[]", "data")]


def test_nested_arrays_of_binary():
    schema = {
        "type": "object",
        "properties": {
            "grid": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "string", "format": "binary"},
                },
            },
        },
    }
    paths = rewrite_binary_params(schema)
    assert paths == [("grid", "[]", "[]")]


def test_optional_bytes_via_anyof():
    """Optional[bytes] in pydantic → {anyOf: [binary, null]}."""
    schema = {
        "type": "object",
        "properties": {
            "avatar": {
                "anyOf": [
                    {"type": "string", "format": "binary"},
                    {"type": "null"},
                ],
            },
        },
    }
    paths = rewrite_binary_params(schema)
    assert paths == [("avatar",)]
    # The binary variant got rewritten; null is untouched.
    variants = schema["properties"]["avatar"]["anyOf"]
    binary_variant = next(v for v in variants if v.get("type") == "string")
    assert "format" not in binary_variant
    assert "Pointer" in binary_variant["description"]
    null_variant = next(v for v in variants if v.get("type") == "null")
    assert null_variant == {"type": "null"}


def test_oneof_with_binary_variant():
    schema = {
        "type": "object",
        "properties": {
            "payload": {
                "oneOf": [
                    {"type": "string", "format": "base64"},
                    {"type": "integer"},
                ],
            },
        },
    }
    paths = rewrite_binary_params(schema)
    assert paths == [("payload",)]


def test_allof_with_binary_variant():
    schema = {
        "type": "object",
        "properties": {
            "chunk": {
                "allOf": [
                    {"type": "string", "format": "byte"},
                    {"description": "extra"},
                ],
            },
        },
    }
    paths = rewrite_binary_params(schema)
    assert paths == [("chunk",)]


def test_mixed_binary_and_non_binary():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "avatar": {
                "anyOf": [
                    {"type": "string", "format": "binary"},
                    {"type": "null"},
                ],
            },
            "attachments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "caption": {"type": "string"},
                        "bytes": {"type": "string", "format": "base64"},
                    },
                },
            },
        },
    }
    paths = rewrite_binary_params(schema)
    assert set(paths) == {("avatar",), ("attachments", "[]", "bytes")}


def test_format_on_non_string_is_ignored():
    """format: binary on a non-string type shouldn't match."""
    schema = {
        "type": "object",
        "properties": {
            "weird": {"type": "integer", "format": "binary"},
        },
    }
    assert rewrite_binary_params(schema) == []


def test_properties_none_does_not_crash():
    """Missing or None properties should be handled gracefully."""
    schema = {"type": "object"}
    assert rewrite_binary_params(schema) == []


def test_items_as_list_does_not_crash():
    """Tuple schemas (items: [schema, schema]) are a JSON Schema feature —
    our walker only handles items as a dict. It should at least not crash."""
    schema = {
        "type": "object",
        "properties": {
            "tuple": {"type": "array", "items": [{"type": "string", "format": "binary"}]},
        },
    }
    # Currently we don't recurse into tuple items — assert no crash.
    rewrite_binary_params(schema)


# ── resolve_pointer_args ─────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path):
    return BinaryStore(store_dir=tmp_path / "binaries", retention_days=30)


def _save_hello(store: BinaryStore, filename: str = "hello.txt") -> str:
    return store.save(b"hello world", "text/plain", filename=filename)


def test_resolve_empty_paths_passthrough(store):
    args = {"x": "some-pointer.txt"}
    out = resolve_pointer_args(args, [], store)
    assert out == args


def test_resolve_single_string_pointer(store):
    pointer = _save_hello(store)
    args = {"data": pointer}
    out = resolve_pointer_args(args, [("data",)], store)
    assert out["data"] == base64.b64encode(b"hello world").decode()


def test_resolve_does_not_mutate_input(store):
    pointer = _save_hello(store)
    args = {"data": pointer}
    snapshot = dict(args)
    resolve_pointer_args(args, [("data",)], store)
    assert args == snapshot  # original untouched


def test_resolve_missing_pointer_passes_through(store):
    args = {"data": "does-not-exist.bin"}
    out = resolve_pointer_args(args, [("data",)], store)
    assert out["data"] == "does-not-exist.bin"


def test_resolve_non_string_value_passes_through(store):
    args = {"data": 42}
    out = resolve_pointer_args(args, [("data",)], store)
    assert out["data"] == 42


def test_resolve_none_at_path_passes_through(store):
    """Optional arg that wasn't provided — value is None, path still recorded."""
    args = {"avatar": None}
    out = resolve_pointer_args(args, [("avatar",)], store)
    assert out["avatar"] is None


def test_resolve_list_of_pointers(store):
    p1 = _save_hello(store, "a.txt")
    p2 = _save_hello(store, "b.txt")
    args = {"blobs": [p1, p2]}
    out = resolve_pointer_args(args, [("blobs", "[]")], store)
    encoded = base64.b64encode(b"hello world").decode()
    assert out["blobs"] == [encoded, encoded]


def test_resolve_deeply_nested_pointer(store):
    pointer = _save_hello(store)
    args = {"input": {"a": {"b": {"c": pointer}}}}
    out = resolve_pointer_args(args, [("input", "a", "b", "c")], store)
    assert out["input"]["a"]["b"]["c"] == base64.b64encode(b"hello world").decode()


def test_resolve_array_of_objects_with_pointer(store):
    pointer = _save_hello(store)
    args = {"files": [{"name": "x", "data": pointer}, {"name": "y", "data": pointer}]}
    out = resolve_pointer_args(args, [("files", "[]", "data")], store)
    encoded = base64.b64encode(b"hello world").decode()
    assert out["files"][0]["data"] == encoded
    assert out["files"][1]["data"] == encoded
    # Non-binary sibling untouched.
    assert out["files"][0]["name"] == "x"


def test_resolve_multiple_independent_paths(store):
    p1 = _save_hello(store, "a.txt")
    p2 = _save_hello(store, "b.txt")
    args = {"first": p1, "second": p2, "skip": "leave-me"}
    out = resolve_pointer_args(args, [("first",), ("second",)], store)
    encoded = base64.b64encode(b"hello world").decode()
    assert out["first"] == encoded
    assert out["second"] == encoded
    assert out["skip"] == "leave-me"


def test_resolve_missing_intermediate_key_does_not_crash(store):
    """Path refers to a nested key that isn't present in args."""
    args = {"input": {"a": {}}}  # no "b" key
    out = resolve_pointer_args(args, [("input", "a", "b")], store)
    assert out == {"input": {"a": {}}}


# ── BinaryStore ──────────────────────────────────────────────────────────

def test_save_and_load_roundtrip(store):
    pointer = store.save(b"payload", "text/plain", filename="note.txt")
    raw, mime = store.load(pointer)
    assert raw == b"payload"


def test_save_preserves_filename(store):
    pointer = store.save(b"x", "text/plain", filename="report.md")
    assert pointer == "report.md"


def test_save_without_filename_uses_extension(store):
    pointer = store.save(b"\x89PNG\r\n\x1a\n", "image/png")
    assert pointer == "blob.png"


def test_save_collision_appends_suffix(store):
    p1 = store.save(b"first", "text/plain", filename="note.txt")
    p2 = store.save(b"second", "text/plain", filename="note.txt")
    p3 = store.save(b"third", "text/plain", filename="note.txt")
    assert p1 == "note.txt"
    assert p2 == "note_0.txt"
    assert p3 == "note_1.txt"
    # All three files exist with their distinct contents.
    assert store.load(p1)[0] == b"first"
    assert store.load(p2)[0] == b"second"
    assert store.load(p3)[0] == b"third"


def test_load_missing_pointer_raises(store):
    with pytest.raises(FileNotFoundError):
        store.load("nope.bin")


def test_load_rejects_path_traversal(store):
    """_sanitize should strip path components before lookup."""
    with pytest.raises(FileNotFoundError):
        store.load("../etc/passwd")


def test_gc_removes_old_files(tmp_path):
    store = BinaryStore(store_dir=tmp_path / "bins", retention_days=1)
    pointer = store.save(b"old", "text/plain", filename="old.txt")
    # Backdate the file to just past the retention cutoff.
    path = store.store_dir / pointer
    old_time = time.time() - (2 * 86400)
    import os
    os.utime(path, (old_time, old_time))
    removed = store.gc()
    assert removed == 1
    assert not path.exists()


def test_gc_preserves_recent_files(store):
    pointer = store.save(b"fresh", "text/plain", filename="fresh.txt")
    removed = store.gc()
    assert removed == 0
    assert (store.store_dir / pointer).exists()
