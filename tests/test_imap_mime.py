"""Inline MIME discovery, CID resource links, and image delivery to agent context."""

import base64
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from unittest.mock import Mock

import pytest

from imap_mcp import events, model, server
from session_manager.mcp import mcp_content_to_openai
from test_imap_identity import fresh_mcp, mailbox, protocol

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a9ZkAAAAASUVORK5CYII=")


@pytest.fixture
def key(mailbox):
    """Provide a mailbox-scoped identifier for the synthetic raw messages."""
    return model.MessageKey(account=mailbox.settings.account, folder="INBOX", validity=21, uid=7)


def related_message(html=None, cid="<scan@example.test>", filename="scan.png"):
    """Build HTML and an inline image inside multipart/related, matching the digest's nesting."""
    related = EmailMessage()
    related.set_content(html or '<table><tr><td><img alt="Mail scan" src="cid:scan%40example.test"></td></tr></table>',
                        subtype="html")
    related.add_related(PNG, maintype="image", subtype="png", cid=cid,
                        disposition="inline", filename=filename)
    return related


def digest_message(with_file=False):
    """Wrap a related HTML body in multipart/mixed, with an optional existing numbered file."""
    message = EmailMessage()
    message["Subject"] = "Synthetic daily digest"
    message["From"] = "sender@example.test"
    message[model.PROTON_HEADER] = "inline-native-id"
    message.make_mixed()
    message.attach(related_message())
    if with_file:
        message.add_attachment(b"existing attachment", maintype="application", subtype="octet-stream",
                               filename="report.bin")
    return message


def test_nested_inline_images_are_discoverable_and_body_links_are_fetchable(key):
    """The mixed/related structure from the report has a fetchable image despite no top-level files."""
    mail = digest_message()
    assert list(mail.iter_attachments()) == []
    result = model.parse_message(mail.as_bytes(), key)
    assert result.has_attachments and len(result.attachment_metadata) == 1
    image = result.attachment_metadata[0]
    assert image.id == "part-0.0.1"
    assert image.inline and image.content_id == "scan@example.test"
    assert image.content_type == "image/png" and image.size == len(PNG)
    assert image.name == "scan.png"
    assert image.uri in result.body
    assert "cid:" not in result.body
    assert base64.b64encode(PNG).decode() not in result.body


def test_new_inline_images_do_not_renumber_existing_attachments(key):
    """Previously issued numeric attachment IDs continue to identify the original file."""
    message = digest_message(with_file=True)
    original = list(message.iter_attachments())
    assert len(original) == 1 and original[0].get_filename() == "report.bin"
    result = model.parse_message(message.as_bytes(), key)
    file, image = result.attachment_metadata
    assert file.id == "0" and file.name == "report.bin" and not file.inline
    assert image.id == "part-0.0.1" and image.inline
    parts = dict(model.attachment_parts(BytesParser(policy=policy.default).parsebytes(message.as_bytes())))
    assert model.attachment_bytes(parts["0"]) == b"existing attachment"
    assert model.attachment_bytes(parts[image.id]) == PNG


def test_top_level_related_images_keep_their_old_numeric_ids(key):
    """Images already exposed by iter_attachments are neither duplicated nor renamed."""
    message = related_message()
    result = model.parse_message(message.as_bytes(), key)
    assert [item.id for item in result.attachment_metadata] == ["0"]
    assert result.attachment_metadata[0].uri in result.body


def test_inline_images_in_alternatives_are_listed_even_when_plain_text_is_preferred(key):
    """Choosing the plain body does not hide the HTML alternative's embedded images."""
    message = EmailMessage()
    message.set_content("Plain text with <literal> formatting")
    message.make_alternative()
    message.attach(related_message())
    result = model.parse_message(message.as_bytes(), key)
    assert result.body.strip() == "Plain text with <literal> formatting"
    assert [item.id for item in result.attachment_metadata] == ["part-0.1.1"]
    assert result.attachment_metadata[0].inline


def test_related_root_start_parameter_and_existing_attachment_order(key):
    """A non-first related root still uses the standard library's body and attachment selection."""
    image = EmailMessage()
    image.set_content(PNG, maintype="image", subtype="png", cid="<scan@example.test>")
    body = EmailMessage()
    body.set_content('<p><img src="cid:scan@example.test"></p>', subtype="html")
    body["Content-ID"] = "<html-root>"
    related = EmailMessage()
    related.make_related()
    related.set_param("start", "<html-root>")
    related.attach(image)
    related.attach(body)
    result = model.parse_message(related.as_bytes(), key)
    assert len(result.attachment_metadata) == 1
    assert result.attachment_metadata[0].id == "0"
    assert result.attachment_metadata[0].uri in result.body


def test_duplicate_content_ids_are_not_guessed(key):
    """Every part remains fetchable when repeated CIDs prevent an unambiguous HTML binding."""
    message = related_message('<p><img src="cid:scan@example.test"></p>')
    message.add_related(b"different bytes", maintype="image", subtype="png",
                        cid="<scan@example.test>", disposition="inline")
    result = model.parse_message(message.as_bytes(), key)
    assert len(result.attachment_metadata) == 2
    assert {item.content_id for item in result.attachment_metadata} == {"scan@example.test"}
    assert all(item.uri not in result.body for item in result.attachment_metadata)


def test_missing_content_ids_and_remote_images_are_not_fabricated_or_fetched(key, monkeypatch):
    """Only bytes actually present in MIME become attachments; ordinary URLs remain links."""
    import httpx

    monkeypatch.setattr(httpx, "get", Mock(side_effect=AssertionError("No remote fetch")))
    message = EmailMessage()
    message.set_content(
        '<p><img src="cid:missing" alt="Missing"><img src="https://example.test/image.png" alt="Remote">'
        '<img src="blob:https://example.test/local" alt="Browser-local"></p>', subtype="html")
    result = model.parse_message(message.as_bytes(), key)
    assert not result.has_attachments
    assert "imap://attachments/" not in result.body
    assert "https://example.test/image.png" in result.body


@pytest.mark.parametrize("values,expected", [([], None), (["<>"], None),
                                          (["bare-id"], "bare-id"),
                                          (["< id >"], "id"), (["<a>", "<b>"], None)])
def test_content_id_normalization(values, expected):
    """Content IDs are normalized without choosing arbitrarily among duplicate headers."""
    part = EmailMessage()
    for value in values:
        part["Content-ID"] = value
    assert model._content_id(part) == expected


def test_cid_links_to_non_image_inline_files_use_the_same_catalog(key):
    """CID-backed non-image MIME resources remain usable by other tools without fetching a URL."""
    message = EmailMessage()
    message.make_mixed()
    related = EmailMessage()
    related.set_content('<a href="CID:document%40example.test">Document</a>', subtype="html")
    related.add_related(b"document bytes", maintype="application", subtype="pdf",
                        cid="<document@example.test>", disposition="inline")
    message.attach(related)
    result = model.parse_message(message.as_bytes(), key)
    attachment = result.attachment_metadata[0]
    assert attachment.inline and attachment.content_type == "application/pdf"
    assert attachment.uri in result.body


def test_nested_explicit_files_and_attached_emails_remain_single_downloads(key):
    """Attached RFC822 messages are not expanded into the parent email's image catalog."""
    inner = digest_message()
    message = EmailMessage()
    message.set_content("See attached mail")
    message.add_attachment(inner, filename="forwarded.eml")
    result = model.parse_message(message.as_bytes(), key)
    assert len(result.attachment_metadata) == 1
    assert result.attachment_metadata[0].content_type == "message/rfc822"
    assert result.attachment_metadata[0].id == "0"
    assert not result.attachment_metadata[0].inline

    container = EmailMessage()
    container.make_alternative()
    container.attach(inner)
    attachment = EmailMessage()
    attachment.set_content(b"nested file", maintype="application", subtype="pdf", disposition="attachment")
    container.attach(attachment)
    nested = model.parse_message(container.as_bytes(), key)
    assert {item.id for item in nested.attachment_metadata} == {"part-0.0.0.1", "part-0.1"}


def test_an_image_only_message_is_fetchable(key):
    """A single-part image without a filename or CID still has a deterministic attachment ID."""
    message = EmailMessage()
    message.set_content(PNG, maintype="image", subtype="png")
    result = model.parse_message(message.as_bytes(), key)
    assert result.body == ""
    image = result.attachment_metadata[0]
    assert image.id == "part-0" and image.name == "attachment-part-0"
    assert image.inline and image.content_id is None


def test_header_summaries_do_not_claim_or_decode_inline_images(key, monkeypatch):
    """Cheap search summaries retain the not-fetched contract for attachment fields."""
    monkeypatch.setattr(model, "attachment_parts", Mock(side_effect=AssertionError("Unexpected MIME walk")))
    summary = model.parse_message(digest_message().as_bytes(), key, summary=True)
    assert summary.body is summary.has_attachments is summary.attachment_metadata is None


def test_inline_image_lookup_returns_exact_bytes_after_folder_move(mailbox, protocol):
    """Inline part IDs and native message IDs preserve the image URI across archiving."""
    protocol.messages["INBOX"][7] = (digest_message().as_bytes(), "SyntheticEmailId")
    identifier = mailbox.search("ALL", "INBOX")[0].id
    image = mailbox.get(identifier).attachment_metadata[0]
    protocol.messages["Archive"][9] = protocol.messages["INBOX"].pop(7)
    data, metadata = mailbox.attachment(identifier, image.id)
    assert data == PNG and metadata == image
    assert mailbox.get(identifier).attachment_metadata[0].uri == image.uri
    with pytest.raises(KeyError):
        mailbox.attachment(identifier, "part-0.99")


@pytest.mark.asyncio
async def test_inline_image_tools_and_resources_reach_the_agent_as_images(mailbox, protocol, monkeypatch, tmp_path):
    """Both attachment routes preserve image MIME types through MCP and the platform renderer."""
    from fastmcp import Client
    from mcp.types import EmbeddedResource

    protocol.messages["INBOX"][7] = (digest_message(with_file=True).as_bytes(), "SyntheticEmailId")
    monkeypatch.setattr(server.Settings, "from_env", lambda: mailbox.settings)
    monkeypatch.setattr(server, "Mailbox", lambda settings, state_path=None: mailbox)
    monkeypatch.setattr(events.Monitor, "watch", lambda self, folder: self.stop.wait(5))
    monkeypatch.setenv("IMAP_STATE_PATH", str(tmp_path / "events.sqlite3"))
    async with Client(fresh_mcp()) as client:
        result = await client.call_tool("search_mail", {"search": "ALL", "folder": "INBOX"})
        identifier = result.structured_content["result"][0]["id"]
        result = await client.call_tool("get_mail", {"message_id": identifier})
        mail_detail = result.structured_content
        metadata = next(item for item in mail_detail["attachment_metadata"] if item["inline"])
        assert metadata["uri"] in mail_detail["body"]
        fetched = await client.call_tool("get_attachment", {
            "message_id": identifier, "attachment_id": metadata["id"]})
        images = [part for part in mcp_content_to_openai(fetched.content) if part["type"] == "input_image"]
        assert len(images) == 1 and images[0]["image_url"] == f"data:image/png;base64,{base64.b64encode(PNG).decode()}"
        contents = await client.read_resource(metadata["uri"])
        assert contents[0].mimeType == "image/png"
        assert base64.b64decode(contents[0].blob) == PNG
        parts = mcp_content_to_openai([EmbeddedResource(type="resource", resource=contents[0])])
        assert [part for part in parts if part["type"] == "input_image"] == images
        file_metadata = next(item for item in mail_detail["attachment_metadata"] if not item["inline"])
        assert file_metadata["id"] == "0"
        file_contents = await client.read_resource(file_metadata["uri"])
        assert file_contents[0].mimeType == "application/octet-stream"
        assert base64.b64decode(file_contents[0].blob) == b"existing attachment"
