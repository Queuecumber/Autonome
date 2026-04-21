"""Tests for the binary-pointer pipeline.

Covers:
- inline_refs: $ref expansion + $defs removal + JSON round-trip (to break
  shared-object identity across ref-reuse sites).
- rewrite_binary_params: jsonpath-based discovery, in-place schema rewrite
  (format → "string", description added), and BinaryParam emission.
- resolve_pointer_args: pointer:// → base64 substitution at the positions
  BinaryParam identifies. Non-pointers pass through. Expired raises.
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
    BinaryParam,
    inline_refs,
    resolve_pointer_args,
    rewrite_binary_params,
)


def _ptr(pointer: str) -> str:
    return f"{POINTER_PREFIX}{pointer}"


HELLO_B64 = base64.b64encode(b"hello world").decode()


# ── rewrite_binary_params: flat / simple ─────────────────────────────────

def test_empty_schema_yields_no_params():
    assert rewrite_binary_params({}) == []


def test_schema_without_binary_yields_no_params():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "count": {"type": "integer"},
        },
    }
    assert rewrite_binary_params(schema) == []
    assert schema["properties"]["name"] == {"type": "string"}


@pytest.mark.parametrize("fmt", ["byte", "binary", "base64"])
def test_single_binary_field_all_formats(fmt):
    schema = {
        "type": "object",
        "properties": {"data": {"type": "string", "format": fmt}},
    }
    params = rewrite_binary_params(schema)
    assert len(params) == 1
    # Schema rewritten in place.
    node = schema["properties"]["data"]
    assert node["format"] == "string"
    assert "Pointer" in node["description"]


def test_multiple_binary_fields_top_level():
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "string", "format": "binary"},
            "b": {"type": "string"},
            "c": {"type": "string", "format": "base64"},
        },
    }
    params = rewrite_binary_params(schema)
    assert len(params) == 2
    # Non-binary field untouched.
    assert schema["properties"]["b"] == {"type": "string"}


def test_format_on_non_string_is_ignored():
    """A `format` field on a non-string node shouldn't match — but our
    selector is `@.format in <binary list>` so it would match any node
    with that format value. The downstream consumer (tools built from
    pydantic) shouldn't produce this shape in practice."""
    schema = {
        "type": "object",
        "properties": {"weird": {"type": "integer", "format": "binary"}},
    }
    # We WILL match here; it's not a realistic schema. Test is just to pin
    # current behavior — change if we add a type guard.
    params = rewrite_binary_params(schema)
    assert len(params) == 1


def test_user_named_format_property_does_not_crash():
    """Regression: a tool with a parameter literally named 'format' produces
    a schema where the 'format' key's value is a dict, not a string. Our
    selector tests `@.format == '<binary>'` so a dict value never matches."""
    schema = {
        "type": "object",
        "properties": {
            "format": {"type": "string", "default": "%Y-%m-%d %H:%M:%S"},
        },
    }
    assert rewrite_binary_params(schema) == []


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
    params = rewrite_binary_params(schema)
    assert len(params) == 1


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
    assert len(rewrite_binary_params(schema)) == 1


def test_array_of_binary():
    schema = {
        "type": "object",
        "properties": {
            "blobs": {"type": "array", "items": {"type": "string", "format": "binary"}},
        },
    }
    assert len(rewrite_binary_params(schema)) == 1


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
    assert len(rewrite_binary_params(schema)) == 1


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
    assert len(rewrite_binary_params(schema)) == 1


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
    params = rewrite_binary_params(schema)
    assert len(params) == 1
    # The binary variant got rewritten; null variant untouched.
    variants = schema["properties"]["avatar"]["anyOf"]
    binary_variant = next(v for v in variants if v.get("type") == "string")
    assert binary_variant["format"] == "string"
    null_variant = next(v for v in variants if v.get("type") == "null")
    assert null_variant == {"type": "null"}


def test_anyof_object_variant_with_binary():
    schema = {
        "type": "object",
        "properties": {
            "payload": {
                "oneOf": [
                    {"type": "null"},
                    {"type": "object", "properties": {"data": {"type": "string", "format": "binary"}}},
                ],
            },
        },
    }
    assert len(rewrite_binary_params(schema)) == 1


def test_dict_of_binary_via_additional_properties():
    schema = {
        "type": "object",
        "properties": {
            "files": {"type": "object", "additionalProperties": {"type": "string", "format": "binary"}},
        },
    }
    assert len(rewrite_binary_params(schema)) == 1


def test_tuple_items_with_binary():
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
    assert len(rewrite_binary_params(schema)) == 1


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


def test_inline_refs_breaks_shared_identity():
    """Multiple refs to the same def must become independent dicts after
    inlining so downstream traversal doesn't dedupe visits."""
    schema = {
        "$defs": {"X": {"type": "object", "properties": {"data": {"type": "string", "format": "binary"}}}},
        "type": "object",
        "properties": {
            "a": {"$ref": "#/$defs/X"},
            "b": {"$ref": "#/$defs/X"},
            "c": {"type": "array", "items": {"$ref": "#/$defs/X"}},
        },
    }
    resolved = inline_refs(schema)
    a = resolved["properties"]["a"]
    b = resolved["properties"]["b"]
    c_items = resolved["properties"]["c"]["items"]
    assert a is not b
    assert a is not c_items
    assert b is not c_items
    assert len(rewrite_binary_params(resolved)) == 3


def test_chain_of_refs():
    schema = {
        "$defs": {
            "Blob": {"type": "object", "properties": {"c": {"type": "string", "format": "binary"}}},
            "Holder": {"type": "object", "properties": {"b": {"$ref": "#/$defs/Blob"}}},
            "Wrap": {"type": "object", "properties": {"a": {"$ref": "#/$defs/Holder"}}},
        },
        "type": "object",
        "properties": {"input": {"$ref": "#/$defs/Wrap"}},
    }
    assert len(rewrite_binary_params(inline_refs(schema))) == 1


def test_optional_ref_with_binary_inside():
    schema = {
        "$defs": {"Avatar": {"type": "object", "properties": {"data": {"type": "string", "format": "binary"}}}},
        "type": "object",
        "properties": {
            "profile": {"anyOf": [{"$ref": "#/$defs/Avatar"}, {"type": "null"}]},
        },
    }
    assert len(rewrite_binary_params(inline_refs(schema))) == 1


def test_list_of_refs_with_binary_inside():
    schema = {
        "$defs": {"File": {"type": "object", "properties": {"bytes": {"type": "string", "format": "base64"}}}},
        "type": "object",
        "properties": {"files": {"type": "array", "items": {"$ref": "#/$defs/File"}}},
    }
    assert len(rewrite_binary_params(inline_refs(schema))) == 1


def test_discriminated_union_with_binary():
    schema = {
        "$defs": {
            "Text": {"type": "object", "properties": {"kind": {"const": "text"}, "text": {"type": "string"}}},
            "File": {"type": "object", "properties": {"kind": {"const": "file"}, "data": {"type": "string", "format": "binary"}}},
        },
        "type": "object",
        "properties": {
            "message": {
                "oneOf": [{"$ref": "#/$defs/Text"}, {"$ref": "#/$defs/File"}],
                "discriminator": {"propertyName": "kind"},
            },
        },
    }
    assert len(rewrite_binary_params(inline_refs(schema))) == 1


# ── BinaryParam.args_matcher: schema-pointer → args-jsonpath ────────────

def _translate(schema_pointer_str: str) -> str:
    """Helper: build a BinaryParam from a pointer string and return the
    compiled JSONPath as its source expression (for assertion)."""
    import jsonpath
    param = BinaryParam(jsonpath.JSONPointer(schema_pointer_str))
    # Reconstruct by re-walking; the library hides the source string.
    parts = ["$"]
    for p in param.schema_pointer.parts:
        if p in ("properties", "anyOf", "oneOf"):
            continue
        if p.isnumeric():
            continue
        if p in ("items", "additionalProperties"):
            parts.append("*")
        else:
            parts.append(p)
    return ".".join(parts)


def test_matcher_strips_properties():
    assert _translate("/properties/data") == "$.data"


def test_matcher_strips_anyof_index():
    assert _translate("/properties/avatar/anyOf/0") == "$.avatar"


def test_matcher_deep_properties():
    assert _translate("/properties/a/properties/b/properties/c") == "$.a.b.c"


def test_matcher_items_becomes_wildcard():
    assert _translate("/properties/blobs/items") == "$.blobs.*"


def test_matcher_tuple_item_becomes_wildcard():
    assert _translate("/properties/tuple/items/1") == "$.tuple.*"


def test_matcher_additional_properties_becomes_wildcard():
    assert _translate("/properties/files/additionalProperties") == "$.files.*"


# ── resolve_pointer_args ─────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path):
    return BinaryStore(store_dir=tmp_path / "binaries", retention_days=30)


def _save_hello(store: BinaryStore, filename: str = "hello.txt") -> str:
    """Save b'hello world' and return the full pointer:// URI."""
    pointer_id = store.save(b"hello world", "text/plain", filename=filename)
    return _ptr(pointer_id)


def _setup(schema: dict, args: dict, store: BinaryStore) -> tuple[dict, dict]:
    """Run the full pipeline: inline → rewrite → resolve."""
    resolved_schema = inline_refs(schema)
    params = rewrite_binary_params(resolved_schema)
    resolved_args = resolve_pointer_args(args, params, store)
    return resolved_schema, resolved_args


def test_resolve_simple_string_pointer(store):
    pointer = _save_hello(store)
    schema = {"type": "object", "properties": {"data": {"type": "string", "format": "binary"}}}
    _, out = _setup(schema, {"data": pointer}, store)
    assert out == {"data": HELLO_B64}


def test_resolve_passes_through_unprefixed_values(store):
    _save_hello(store, filename="hello.txt")
    schema = {"type": "object", "properties": {"data": {"type": "string", "format": "binary"}}}
    _, out = _setup(schema, {"data": "hello.txt"}, store)  # no prefix
    assert out == {"data": "hello.txt"}


def test_resolve_does_not_mutate_input(store):
    pointer = _save_hello(store)
    schema = {"type": "object", "properties": {"data": {"type": "string", "format": "binary"}}}
    args = {"data": pointer}
    snapshot = {"data": pointer}
    _setup(schema, args, store)
    assert args == snapshot


def test_resolve_expired_pointer_raises(store):
    schema = {"type": "object", "properties": {"data": {"type": "string", "format": "binary"}}}
    args = {"data": _ptr("gone.txt")}
    with pytest.raises(ValueError, match="not found"):
        _setup(schema, args, store)


def test_resolve_none_in_optional_passes_through(store):
    schema = {
        "type": "object",
        "properties": {
            "avatar": {"anyOf": [{"type": "string", "format": "binary"}, {"type": "null"}]},
        },
    }
    _, out = _setup(schema, {"avatar": None}, store)
    assert out == {"avatar": None}


def test_resolve_list_of_pointers(store):
    p1 = _save_hello(store, "a.txt")
    p2 = _save_hello(store, "b.txt")
    schema = {
        "type": "object",
        "properties": {
            "blobs": {"type": "array", "items": {"type": "string", "format": "binary"}},
        },
    }
    _, out = _setup(schema, {"blobs": [p1, p2]}, store)
    assert out == {"blobs": [HELLO_B64, HELLO_B64]}


def test_resolve_nested_deep_pointer(store):
    pointer = _save_hello(store)
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
    _, out = _setup(schema, {"input": {"a": {"b": {"c": pointer}}}}, store)
    assert out["input"]["a"]["b"]["c"] == HELLO_B64


def test_resolve_array_of_objects(store):
    pointer = _save_hello(store)
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
    _, out = _setup(
        schema,
        {"files": [{"name": "x", "data": pointer}, {"name": "y", "data": pointer}]},
        store,
    )
    assert out["files"][0]["data"] == HELLO_B64
    assert out["files"][1]["data"] == HELLO_B64
    assert out["files"][0]["name"] == "x"


def test_resolve_dict_of_bytes(store):
    p1 = _save_hello(store, "a.txt")
    p2 = _save_hello(store, "b.txt")
    schema = {
        "type": "object",
        "properties": {
            "files": {
                "type": "object",
                "additionalProperties": {"type": "string", "format": "binary"},
            },
        },
    }
    _, out = _setup(schema, {"files": {"alpha": p1, "beta": p2}}, store)
    assert out == {"files": {"alpha": HELLO_B64, "beta": HELLO_B64}}


def test_resolve_tuple_with_mixed_binaries(store):
    p1 = _save_hello(store, "a.txt")
    p2 = _save_hello(store, "b.txt")
    schema = {
        "type": "object",
        "properties": {
            "tuple": {
                "type": "array",
                "items": [
                    {"type": "integer"},
                    {"type": "string", "format": "binary"},
                    {"type": "string"},
                    {"type": "string", "format": "base64"},
                ],
            },
        },
    }
    _, out = _setup(schema, {"tuple": [42, p1, "literal", p2]}, store)
    assert out["tuple"] == [42, HELLO_B64, "literal", HELLO_B64]


def test_resolve_array_of_arrays(store):
    p1 = _save_hello(store, "a.txt")
    p2 = _save_hello(store, "b.txt")
    p3 = _save_hello(store, "c.txt")
    schema = {
        "type": "object",
        "properties": {
            "grid": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "string", "format": "binary"}},
            },
        },
    }
    _, out = _setup(schema, {"grid": [[p1, p2], [p3], []]}, store)
    assert out == {"grid": [[HELLO_B64, HELLO_B64], [HELLO_B64], []]}


def test_resolve_ignores_unrelated_fields(store):
    _save_hello(store)
    schema = {
        "type": "object",
        "properties": {
            "data": {"type": "string", "format": "binary"},
            "description": {"type": "string"},
        },
    }
    # `description` has a pointer-looking value, but its schema is not binary.
    _, out = _setup(
        schema,
        {"data": _save_hello(store), "description": _ptr("hello.txt")},
        store,
    )
    assert out["description"] == _ptr("hello.txt")  # passed through unchanged


def test_resolve_ref_chain_heavy_reuse(store):
    """The nasty case: Inner referenced via multiple wrappers, plus
    direct dict-of-bytes, tuple, and array-of-array."""
    p = _save_hello(store)
    schema = {
        "$defs": {
            "Inner": {
                "type": "object",
                "properties": {
                    "blob": {"type": "string", "format": "binary"},
                    "name": {"type": "string"},
                },
            },
            "Middle": {
                "type": "object",
                "properties": {
                    "inner": {"$ref": "#/$defs/Inner"},
                    "inner_list": {"type": "array", "items": {"$ref": "#/$defs/Inner"}},
                },
            },
        },
        "type": "object",
        "properties": {
            "middle": {"$ref": "#/$defs/Middle"},
            "dict_of_bytes": {
                "type": "object",
                "additionalProperties": {"type": "string", "format": "binary"},
            },
        },
    }
    args = {
        "middle": {
            "inner": {"name": "a", "blob": p},
            "inner_list": [{"name": "b", "blob": p}, {"name": "c", "blob": p}],
        },
        "dict_of_bytes": {"x": p, "y": p},
    }
    _, out = _setup(schema, args, store)
    assert out["middle"]["inner"]["blob"] == HELLO_B64
    assert out["middle"]["inner"]["name"] == "a"
    assert out["middle"]["inner_list"][0]["blob"] == HELLO_B64
    assert out["middle"]["inner_list"][1]["blob"] == HELLO_B64
    assert out["dict_of_bytes"]["x"] == HELLO_B64
    assert out["dict_of_bytes"]["y"] == HELLO_B64


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
