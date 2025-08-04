def clean_html(html): ...
def save_html(documents_html: list, output_path: str):
    """
    Save HTML content as an HTML file
    Args:
        documents_html: A list with HTML content to save.
        output_path: Full path (including filename) for the output of the HTML files.

    Returns:
        ValueError: If the output path's directory does not exist.
    """
def render_html_to_pdf(documents_html: list, output_path: str):
    """
    Render HTML content (as a string) into a PDF file.

    Args:
        html_string (list): A list with HTML content to render.
        pdf_output_path (str): Full path (including filename) for the output PDF.

    Raises:
        ValueError: If the output path's directory does not exist.
    """
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
