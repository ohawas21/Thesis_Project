"""
    Auxiliary functions to render HTMl into different document format types
"""
from os import path
def clean_html(html):
    html = html.strip('```html').strip('```')
    html = html.replace("\\n", "")
    interpreted_html = html.encode('utf-8').decode('unicode_escape').strip()
    return interpreted_html

def save_html(documents_html:list, output_path:str):
    """
    Save HTML content as an HTML file
    Args:
        documents_html: A list with HTML content to save.
        output_path: Full path (including filename) for the output of the HTML files.

    Returns:
        ValueError: If the output path's directory does not exist.
    """
    for i, document in enumerate(documents_html):
        doc_path = path.join(output_path, document['document_type'] + f'_{i}.pdf')
        with open(doc_path, 'wb') as f:
            f.write(document.encode('utf-8'))

def render_html_to_pdf(documents_html: list, output_path: str):
    """
    Render HTML content (as a string) into a PDF file.

    Args:
        html_string (list): A list with HTML content to render.
        pdf_output_path (str): Full path (including filename) for the output PDF.

    Raises:
        ValueError: If the output path's directory does not exist.
    """
    from weasyprint import HTML

    for i, document in enumerate(documents_html):
        doc_path = path.join(output_path, document['document_type']+f'_{i}.pdf')
        HTML(string=document['html'],
             encoding='UTF-8').write_pdf(doc_path)

def render_html_to_docx(documents_html: list, output_path: str):
    """
    Render HTML content (as a string) into individual DOCX files.

    Args:
        documents_html (list): A list of dictionaries, each with:
            - 'document_type': A label for the document name
            - 'html': HTML string content to render
        output_path (str): Directory path where the DOCX files will be saved.

    Raises:
        ValueError: If the output path directory does not exist.
    """
    if not path.isdir(output_path):
        raise ValueError(f"The directory '{output_path}' does not exist.")

    try:
        from docx import Document
        from html4docx import HtmlToDocx
    except ImportError:
        raise Exception("Missing dependencies: Please install 'python-docx' and 'html2docx'.")

    for i, document in enumerate(documents_html):
        docx_path = path.join(output_path, f"{document['document_type']}_{i}.docx")

        doc = Document()
        new_parser = HtmlToDocx()
        new_parser.add_html_to_document(document['html'], doc)

        doc.save(docx_path)

