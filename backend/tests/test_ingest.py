import io

from copilot.ingest import extract_text


def test_plain_text_file_is_decoded():
    content = "Hello, world! Café.".encode()
    assert extract_text("notes.txt", content) == "Hello, world! Café."


def test_pdf_extension_routes_through_pypdf():
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)

    # A blank page has no text, but this exercises the PDF code path
    # end-to-end (parsing, not crashing) rather than the plain-text fallback.
    result = extract_text("document.pdf", buf.getvalue())
    assert result == ""


def test_unknown_extension_falls_back_to_text_decode():
    content = b"raw content without a recognized extension"
    assert extract_text("data.unknown", content) == content.decode("utf-8")
