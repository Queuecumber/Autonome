"""Tests for the binary-pointer pipeline.

Covers:
- inline_refs: $ref expansion + $defs removal.
- rewrite_binary_params: jsonpath-based discovery, in-place rewrite, and
  collection of argument property names that may carry pointers.
- resolve_pointer_args: pointer:// → base64 substitution scoped to known-
  binary keys; passthrough for unprefixed values and unrelated keys.
- BinaryStore: save/load round-trip, filename preservation, collision
  suffix, GC retention.
"""

import base64
import os
import time

import pytest

from session_manager.binaries import BinaryStore
from session_manager.mcp import (
    POINTER_PREFIX,
    inline_refs,
    resolve_pointer_args,
    rewrite_binary_params,
)


def _ptr(pointer: str) -> str:
    return f"{POINTER_PREFIX}{pointer}"


# ── rewrite_binary_params: flat / simple ─────────────────────────────────

def test_empty_schema_yields_no_names():
    assert rewrite_binary_params({}) == set()


def test_schema_without_binary_yields_no_names():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "count": {"type": "integer"},
        },
    }
    assert rewrite_binary_params(schema) == set()
    assert schema["properties"]["name"] == {"type": "string"}


@pytest.mark.parametrize("fmt", ["byte", "binary", "base64"])
def test_single_binary_field_all_formats(fmt):
    schema = {
        "type": "object",
        "properties": {"data": {"type": "string", "format": fmt}},
    }
    names = rewrite_binary_params(schema)
    assert names == {"data"}
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
    assert rewrite_binary_params(schema) == {"a", "c"}


def test_format_on_non_string_is_ignored():
    schema = {
        "type": "object",
        "properties": {"weird": {"type": "integer", "format": "binary"}},
    }
    assert rewrite_binary_params(schema) == set()


def test_properties_none_does_not_crash():
    schema = {"type": "object"}
    assert rewrite_binary_params(schema) == set()


# ── rewrite_binary_params: nested / arrays / unions ──────────────────────

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
    assert rewrite_binary_params(schema) == {"data"}


def test_deeply_nested_inlined_binary():
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
    assert rewrite_binary_params(schema) == {"c"}


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
    assert rewrite_binary_params(schema) == {"blobs"}


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
    assert rewrite_binary_params(schema) == {"data"}


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
    assert rewrite_binary_params(schema) == {"grid"}


def test_optional_bytes_via_anyof():
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
    assert rewrite_binary_params(schema) == {"avatar"}
    variants = schema["properties"]["avatar"]["anyOf"]
    binary_variant = next(v for v in variants if v.get("type") == "string")
    assert "format" not in binary_variant


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
    assert rewrite_binary_params(schema) == {"payload"}


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
    assert rewrite_binary_params(schema) == {"chunk"}


def test_anyof_object_variant_with_binary():
    """Previously failed — now succeeds: jsonpath recursive descent finds the
    binary leaf no matter how many union wrappers enclose it."""
    schema = {
        "type": "object",
        "properties": {
            "payload": {
                "oneOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "properties": {"data": {"type": "string", "format": "binary"}},
                    },
                ],
            },
        },
    }
    assert rewrite_binary_params(schema) == {"data"}


def test_optional_object_with_binary_inside():
    """Optional[Model] with a binary field. Previously failed."""
    schema = {
        "type": "object",
        "properties": {
            "profile": {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {"avatar": {"type": "string", "format": "binary"}},
                    },
                    {"type": "null"},
                ],
            },
        },
    }
    assert rewrite_binary_params(schema) == {"avatar"}


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
    assert rewrite_binary_params(schema) == {"avatar", "bytes"}


def test_multiple_binaries_at_varying_depths():
    schema = {
        "type": "object",
        "properties": {
            "top": {"type": "string", "format": "binary"},
            "container": {
                "type": "object",
                "properties": {
                    "mid": {"type": "string", "format": "base64"},
                    "deep": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "leaf": {"type": "string", "format": "byte"},
                            },
                        },
                    },
                },
            },
        },
    }
    assert rewrite_binary_params(schema) == {"top", "mid", "leaf"}


def test_dict_of_binary_via_additional_properties():
    """dict[str, bytes] → additionalProperties of a binary string."""
    schema = {
        "type": "object",
        "properties": {
            "files": {
                "type": "object",
                "additionalProperties": {"type": "string", "format": "binary"},
            },
        },
    }
    assert rewrite_binary_params(schema) == {"files"}


def test_tuple_items_with_binary():
    """JSON Schema tuple arrays — items is a list of per-position schemas."""
    schema = {
        "type": "object",
        "properties": {
            "tuple": {
                "type": "array",
                "items": [
                    {"type": "string"},
                    {"type": "string", "format": "binary"},
                ],
            },
        },
    }
    assert rewrite_binary_params(schema) == {"tuple"}


# ── inline_refs + rewrite_binary_params: $ref cases ─────────────────────

def test_inline_refs_drops_defs():
    schema = {
        "$defs": {"Blob": {"type": "object", "properties": {"c": {"type": "string"}}}},
        "type": "object",
        "properties": {"input": {"$ref": "#/$defs/Blob"}},
    }
    resolved = inline_refs(schema)
    assert "$defs" not in resolved
    assert resolved["properties"]["input"]["properties"]["c"]["type"] == "string"


def test_ref_at_root_property():
    schema = {
        "$defs": {
            "Blob": {
                "type": "object",
                "properties": {"c": {"type": "string", "format": "binary"}},
            },
        },
        "type": "object",
        "properties": {"input": {"$ref": "#/$defs/Blob"}},
    }
    inlined = inline_refs(schema)
    assert rewrite_binary_params(inlined) == {"c"}


def test_chain_of_refs():
    schema = {
        "$defs": {
            "ActualDataBlob": {
                "type": "object",
                "properties": {"c": {"type": "string", "format": "binary"}},
            },
            "DataHolder": {
                "type": "object",
                "properties": {"b": {"$ref": "#/$defs/ActualDataBlob"}},
            },
            "Wrapper": {
                "type": "object",
                "properties": {"a": {"$ref": "#/$defs/DataHolder"}},
            },
        },
        "type": "object",
        "properties": {"input": {"$ref": "#/$defs/Wrapper"}},
    }
    assert rewrite_binary_params(inline_refs(schema)) == {"c"}


def test_optional_ref_with_binary_inside():
    schema = {
        "$defs": {
            "Avatar": {
                "type": "object",
                "properties": {"data": {"type": "string", "format": "binary"}},
            },
        },
        "type": "object",
        "properties": {
            "profile": {
                "anyOf": [
                    {"$ref": "#/$defs/Avatar"},
                    {"type": "null"},
                ],
            },
        },
    }
    assert rewrite_binary_params(inline_refs(schema)) == {"data"}


def test_list_of_refs_with_binary_inside():
    schema = {
        "$defs": {
            "File": {
                "type": "object",
                "properties": {"bytes": {"type": "string", "format": "base64"}},
            },
        },
        "type": "object",
        "properties": {
            "files": {"type": "array", "items": {"$ref": "#/$defs/File"}},
        },
    }
    assert rewrite_binary_params(inline_refs(schema)) == {"bytes"}


def test_allof_merged_with_ref():
    schema = {
        "$defs": {
            "Blob": {
                "type": "object",
                "properties": {"data": {"type": "string", "format": "binary"}},
            },
        },
        "type": "object",
        "properties": {
            "x": {
                "allOf": [
                    {"$ref": "#/$defs/Blob"},
                    {"description": "extra"},
                ],
            },
        },
    }
    assert rewrite_binary_params(inline_refs(schema)) == {"data"}


def test_discriminated_union_with_binary():
    schema = {
        "$defs": {
            "TextMessage": {
                "type": "object",
                "properties": {
                    "kind": {"const": "text"},
                    "text": {"type": "string"},
                },
            },
            "FileMessage": {
                "type": "object",
                "properties": {
                    "kind": {"const": "file"},
                    "data": {"type": "string", "format": "binary"},
                },
            },
        },
        "type": "object",
        "properties": {
            "message": {
                "oneOf": [
                    {"$ref": "#/$defs/TextMessage"},
                    {"$ref": "#/$defs/FileMessage"},
                ],
                "discriminator": {"propertyName": "kind"},
            },
        },
    }
    assert rewrite_binary_params(inline_refs(schema)) == {"data"}


# Self-referential schemas can't be fully inlined without proxies=True.
# jsonref.replace_refs(proxies=False) explodes infinitely. Documented gap.
@pytest.mark.xfail(reason="Self-referential $ref expands infinitely with proxies=False")
def test_self_referential_schema():
    schema = {
        "$defs": {
            "TreeNode": {
                "type": "object",
                "properties": {
                    "blob": {"type": "string", "format": "binary"},
                    "children": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/TreeNode"},
                    },
                },
            },
        },
        "type": "object",
        "properties": {"root": {"$ref": "#/$defs/TreeNode"}},
    }
    assert rewrite_binary_params(inline_refs(schema)) == {"blob"}


# ── resolve_pointer_args ─────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path):
    return BinaryStore(store_dir=tmp_path / "binaries", retention_days=30)


def _save_hello(store: BinaryStore, filename: str = "hello.txt") -> str:
    """Save b'hello world' and return the full pointer:// URI."""
    pointer_id = store.save(b"hello world", "text/plain", filename=filename)
    return _ptr(pointer_id)


HELLO_B64 = base64.b64encode(b"hello world").decode()


def test_resolve_empty_names_passthrough(store):
    args = {"data": _ptr("hello.txt")}
    assert resolve_pointer_args(args, set(), store) == args


def test_resolve_simple_string_pointer(store):
    pointer = _save_hello(store)
    out = resolve_pointer_args({"data": pointer}, {"data"}, store)
    assert out == {"data": HELLO_B64}


def test_resolve_does_not_resolve_unprefixed_values(store):
    """A plain string at a binary-typed key is NOT treated as a pointer."""
    # Pre-save a file whose name could collide with a literal.
    pointer_id = _save_hello(store, filename="photo.jpg").removeprefix(POINTER_PREFIX)
    out = resolve_pointer_args({"data": pointer_id}, {"data"}, store)
    # Without the pointer:// prefix, the literal passes through untouched.
    assert out == {"data": pointer_id}


def test_resolve_does_not_mutate_input(store):
    pointer = _save_hello(store)
    args = {"data": pointer}
    snapshot = {"data": pointer}
    resolve_pointer_args(args, {"data"}, store)
    assert args == snapshot


def test_resolve_expired_pointer_raises(store):
    """Pointer prefix present but binary not found → loud error."""
    args = {"data": _ptr("gone.txt")}
    with pytest.raises(ValueError, match="not found"):
        resolve_pointer_args(args, {"data"}, store)


def test_resolve_none_at_known_key_passes_through(store):
    out = resolve_pointer_args({"avatar": None}, {"avatar"}, store)
    assert out == {"avatar": None}


def test_resolve_non_string_passes_through(store):
    out = resolve_pointer_args({"data": 42}, {"data"}, store)
    assert out == {"data": 42}


def test_resolve_ignores_keys_not_in_names(store):
    """Other keys — even `pointer://` ones — are left alone."""
    out = resolve_pointer_args(
        {"description": _ptr("hello.txt"), "data": _ptr("hello.txt")},
        {"data"},
        store,
    )
    assert out["description"] == _ptr("hello.txt")  # unchanged


def test_resolve_list_under_known_key(store):
    p1 = _save_hello(store, "a.txt")
    p2 = _save_hello(store, "b.txt")
    out = resolve_pointer_args({"blobs": [p1, p2]}, {"blobs"}, store)
    assert out == {"blobs": [HELLO_B64, HELLO_B64]}


def test_resolve_nested_dict_searches_for_known_key(store):
    pointer = _save_hello(store)
    args = {"input": {"a": {"b": {"c": pointer}}}}
    out = resolve_pointer_args(args, {"c"}, store)
    assert out["input"]["a"]["b"]["c"] == HELLO_B64


def test_resolve_array_of_objects_with_known_key(store):
    pointer = _save_hello(store)
    args = {"files": [{"name": "x", "data": pointer}, {"name": "y", "data": pointer}]}
    out = resolve_pointer_args(args, {"data"}, store)
    assert out["files"][0]["data"] == HELLO_B64
    assert out["files"][1]["data"] == HELLO_B64
    assert out["files"][0]["name"] == "x"


def test_resolve_multiple_known_keys(store):
    p1 = _save_hello(store, "a.txt")
    p2 = _save_hello(store, "b.txt")
    out = resolve_pointer_args(
        {"first": p1, "second": p2, "skip": "leave-me"},
        {"first", "second"},
        store,
    )
    assert out["first"] == HELLO_B64
    assert out["second"] == HELLO_B64
    assert out["skip"] == "leave-me"


def test_resolve_dict_values_under_known_key(store):
    """dict[str, bytes] shape — key is known-binary, value is a dict of pointers."""
    p1 = _save_hello(store, "a.txt")
    p2 = _save_hello(store, "b.txt")
    out = resolve_pointer_args({"files": {"alpha": p1, "beta": p2}}, {"files"}, store)
    assert out == {"files": {"alpha": HELLO_B64, "beta": HELLO_B64}}


def test_resolve_missing_key_does_not_crash(store):
    args = {"input": {"a": {}}}
    out = resolve_pointer_args(args, {"c"}, store)
    assert out == {"input": {"a": {}}}


# ── BinaryStore ──────────────────────────────────────────────────────────

def test_save_and_load_roundtrip(store):
    pointer = store.save(b"payload", "text/plain", filename="note.txt")
    raw, _ = store.load(pointer)
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
    assert store.load(p1)[0] == b"first"
    assert store.load(p2)[0] == b"second"
    assert store.load(p3)[0] == b"third"


def test_load_missing_pointer_raises(store):
    with pytest.raises(FileNotFoundError):
        store.load("nope.bin")


def test_load_rejects_path_traversal(store):
    with pytest.raises(FileNotFoundError):
        store.load("../etc/passwd")


def test_gc_removes_old_files(tmp_path):
    store = BinaryStore(store_dir=tmp_path / "bins", retention_days=1)
    pointer = store.save(b"old", "text/plain", filename="old.txt")
    path = store.store_dir / pointer
    old_time = time.time() - (2 * 86400)
    os.utime(path, (old_time, old_time))
    assert store.gc() == 1
    assert not path.exists()


def test_gc_preserves_recent_files(store):
    pointer = store.save(b"fresh", "text/plain", filename="fresh.txt")
    assert store.gc() == 0
    assert (store.store_dir / pointer).exists()
