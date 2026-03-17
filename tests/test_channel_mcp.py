import pytest
from adapters.signal.channel_mcp import create_channel_mcp, StagedAttachments


def test_staged_attachments_stage_and_drain():
    """Staging attachments queues them; draining returns and clears the queue."""
    staged = StagedAttachments()
    staged.stage("conv-1", "/tmp/image.png", "image/png")
    staged.stage("conv-1", "/tmp/doc.pdf", "application/pdf")

    items = staged.drain("conv-1")
    assert len(items) == 2
    assert items[0]["file_path"] == "/tmp/image.png"
    assert items[1]["mime_type"] == "application/pdf"

    # Queue should be empty after drain
    assert staged.drain("conv-1") == []


def test_staged_attachments_separate_conversations():
    """Attachments are isolated per conversation_id."""
    staged = StagedAttachments()
    staged.stage("conv-1", "/tmp/a.png", "image/png")
    staged.stage("conv-2", "/tmp/b.png", "image/png")

    assert len(staged.drain("conv-1")) == 1
    assert len(staged.drain("conv-2")) == 1


@pytest.mark.asyncio
async def test_mcp_has_expected_tools():
    """The MCP server exposes the expected Signal tools."""
    mcp = create_channel_mcp()
    tools = await mcp.list_tools()
    tool_names = {t.name for t in tools}
    assert "stage_attachment" in tool_names
    assert "send_attachment" in tool_names
    assert "react" in tool_names
    assert "set_typing" in tool_names
