"""
    Prompt per document type
"""
from enum import Enum
from typing import Optional

class DocumentType(Enum):
    INVOICE = "invoice"
    BALANCE_SHEET = "balance sheet"
    CASH_FLOW = "cash flow"
    EARNINGS_SUMMARY = "earnings summary"
    CREDIT_CARD_STATEMENT = "credit card statement"

    @classmethod
    def normalize(cls, doc_type: str) -> str:
        return doc_type.strip().replace("_", " ").replace("-", " ").lower()

    @classmethod
    def get_html_prompt(cls, doc_type: str) -> Optional[str]:
        normalized = cls.normalize(doc_type)
        for member in cls:
            if member.value == normalized:
                return _DOCUMENT_TYPE_HTML_PROMPTS.get(member)
        return None

    @classmethod
    def get_content_prompt(cls, doc_type: str) -> Optional[str]:
        normalized = cls.normalize(doc_type)
        for member in cls:
            if member.value == normalized:
                return _DOCUMENT_TYPE_CONTENT_PROMPTS.get(member)
        return None

_DOCUMENT_TYPE_CONTENT_PROMPTS = {
    DocumentType.INVOICE: """
You are a professional document generation assistant with expertise across financial and operational documentation.

Your task is to generate a polished and realistic invoice based on the inputs provided.

### Required Inputs:
- **Document Type:** Invoice
- **Audience:** {audience}
- **Tone:** {tone}
- **Purpose:** {purpose}

### Optional Inputs:
- **Region/Locale:** {region}
- **Language:** {language}
- **Length Preference:** {length}
- **Key Topics or Sections:** {topics}
- **Reference Style or Brand Voice:** {style_guide}

### Output Guidelines:
- Generate the full **invoice content**, including invoice number, dates, company and client names, line items, tax, and total due.
- Use realistic line items (e.g., software licenses, consulting hours, delivery fees) and monetary amounts.
- Avoid placeholders or generic labels — use plausible fictional entities.
- The content should be structured for integration into an invoice layout, with itemized billing and payment terms.
{prompt_ending}
Return only the complete document content, without any commentary or explanation.
""",

    DocumentType.BALANCE_SHEET: """
You are a professional document generation assistant specialized in financial statements.

Your task is to generate a realistic balance sheet based on the inputs provided.

### Required Inputs:
- **Document Type:** Balance Sheet
- **Audience:** {audience}
- **Tone:** {tone}
- **Purpose:** {purpose}

### Optional Inputs:
- **Region/Locale:** {region}
- **Language:** {language}
- **Length Preference:** {length}
- **Key Topics or Sections:** {topics}
- **Reference Style or Brand Voice:** {style_guide}

### Output Guidelines:
- Present a full balance sheet with **Assets**, **Liabilities**, and **Equity**.
- Include subcategories (e.g., Current Assets, Long-Term Liabilities).
- Use consistent currency and financial terminology.
- The content should reflect real-world accounting structure and values.
{prompt_ending}
Return only the complete document content, without any commentary or explanation.
""",

    DocumentType.CASH_FLOW: """
You are a professional assistant skilled in corporate financial reporting.

Your task is to generate a detailed and realistic **cash flow statement**.

### Required Inputs:
- **Document Type:** Cash Flow Statement
- **Audience:** {audience}
- **Tone:** {tone}
- **Purpose:** {purpose}

### Optional Inputs:
- **Region/Locale:** {region}
- **Language:** {language}
- **Length Preference:** {length}
- **Key Topics or Sections:** {topics}
- **Reference Style or Brand Voice:** {style_guide}

### Output Guidelines:
- Organize into **Operating**, **Investing**, and **Financing** activities.
- Include cash inflows/outflows for each, with proper labels and values.
- Use a realistic narrative around business transactions if needed.
{prompt_ending}
Return only the complete document content, without any commentary or explanation.
""",

    DocumentType.EARNINGS_SUMMARY: """
You are a document generation assistant with a focus on investor communications and executive summaries.

Your task is to generate a complete and professional **earnings summary** document.

### Required Inputs:
- **Document Type:** Earnings Summary
- **Audience:** {audience}
- **Tone:** {tone}
- **Purpose:** {purpose}

### Optional Inputs:
- **Region/Locale:** {region}
- **Language:** {language}
- **Length Preference:** {length}
- **Key Topics or Sections:** {topics}
- **Reference Style or Brand Voice:** {style_guide}

### Output Guidelines:
- Include meaningful and plausible financial metrics such as **revenue**, **net income**, **earnings per share (EPS)**, and **profit margins**.
- Use values, phrasing, and context that reflect real-world financial communication standards — the document should feel like it was written for an actual business scenario.
- Include comparison to previous periods or forecast where relevant.
- Use realistic company names, product lines, industries, and timeframes.
- Do **not** include any placeholder values such as [Company Name], [Date], [Your Name], or [Metric].

{prompt_ending}

Return only the complete document content, without any commentary or explanation.
""",

    DocumentType.CREDIT_CARD_STATEMENT: """
You are an assistant skilled in generating realistic financial and billing documents.

Your task is to produce a **credit card statement** that could plausibly be issued by a financial institution.

### Required Inputs:
- **Document Type:** Credit Card Statement
- **Audience:** {audience}
- **Tone:** {tone}
- **Purpose:** {purpose}

### Optional Inputs:
- **Region/Locale:** {region}
- **Language:** {language}
- **Length Preference:** {length}
- **Key Topics or Sections:** {topics}
- **Reference Style or Brand Voice:** {style_guide}

### Output Guidelines:
- Include all standard summary fields: masked account number, billing period, statement date, total due, minimum payment, and payment due date.
- Provide a realistic transaction history table with **multiple line items** (5 to 15), including: Date, Description, Category, and Amount.
- Use plausible merchant names and categories (e.g., grocery, travel, utilities, entertainment).
- Ensure amounts and dates are consistent with the billing period.
- Do not use placeholders — use realistic, fictionalized data that could appear in a real statement.

{prompt_ending}
Return only the complete document content, without any commentary or explanation.
"""
}

#Document type HTML generation
_DOCUMENT_TYPE_HTML_PROMPTS = {
    DocumentType.INVOICE: """
You are a creative and professional HTML document designer with expertise in financial document presentation.

Your task is to convert the provided plain text content of an **invoice** into a complete, visually appealing HTML document.

### Context:
- Document Type: Invoice
- Audience: {audience}
- Tone: {tone}

### Creative Guidelines:
- Use semantic HTML tags (<header>, <section>, <table>, etc.) to structure the document meaningfully.
- Apply a modern and aesthetically pleasing design using embedded <style> tags.
- Exercise creativity in layout, typography, and spacing — feel free to use visual hierarchy, whitespace, and subtle visual accents.
- Ensure the document maintains a professional appearance, appropriate for a business or financial institution.
- The invoice should include clearly defined sections:
  - Header with invoice number, date, and parties involved
  - Table of line items (item, description, unit price, quantity, total)
  - Summary section with tax, total due, and payment terms

### Output Instructions:
Put ```html at the beginning and ``` at the end of the script to separate the code from the text.

Here is the content:
---
{document_content}
---

Do not change the wording of the content. Return only valid HTML code that can be rendered as a standalone document.
Do not include comments, explanations, or placeholders.
""",

    DocumentType.BALANCE_SHEET: """
    You are a professional HTML document designer with a strong background in financial reporting design.

    Your task is to transform the provided plain text content of a **balance sheet** into a fully structured, visually refined HTML document.

    ### Objective:
    Convert the existing balance sheet content into modern, readable, and semantically organized HTML — without altering any of the wording or values.

    ### Design Context:
    - Document Type: Balance Sheet
    - Audience: {audience}
    - Tone: {tone}

    ### Design and Layout Requirements:
    - Use semantic HTML tags (<header>, <section>, <table>, etc.) to reflect the logical structure of the balance sheet.
    - Apply embedded <style> tags to:
      - Improve readability and alignment
      - Ensure clear hierarchy and spacing
    - Right-align numeric values where appropriate and emphasize totals using styling (e.g., bold or highlighted rows).

    ### Output Instructions:
    - Do not change the wording or structure of the content itself — your task is visual formatting only.
    - Wrap the entire response in `<html>...</html>` tags.

    Here is the content:
    ---
    {document_content}
    ---

    Return only the valid HTML code. Do not include any comments, explanations, or placeholder values.
    """,

    DocumentType.CASH_FLOW: """
    You are a professional HTML document designer with expertise in formatting financial statements for the web.

    Your task is to convert the provided plain text content of a **cash flow statement** into a clean, well-structured,
    and visually organized HTML document.

    ### Objective:
    Focus entirely on the visual layout and semantic HTML structure. The content has already been generated and must not be altered.

    ### Context:
    - Document Type: Cash Flow Statement
    - Audience: {audience}
    - Tone: {tone}

    ### HTML and Styling Guidelines:
    - Use semantic HTML elements such as <header>, <section>, <table>, <tbody>, <tr>, and <td> to structure the document logically.
    - Use embedded <style> tags to apply clean and readable formatting, including spacing, alignment, and font styling.
    - Ensure consistent alignment, especially for any numerical values, but do not infer or generate totals.
    - Apply general design principles to enhance readability (e.g., whitespace, section separation), while keeping the layout simple and professional.

    ### Output Instructions:
    - Do not change or interpret the content — only apply HTML layout and design.
    - Wrap the entire response in `<html>...</html>` tags.

    Here is the content:
    ---
    {document_content}
    ---

    Return **only valid HTML code**, without comments or extra explanation.
    """,

    DocumentType.EARNINGS_SUMMARY: """
    You are a professional HTML layout and formatting assistant with expertise in corporate reporting and investor communications.

    Your task is to transform the provided plain text content of an **earnings summary** into a clean, structured, and visually refined HTML document.

    ### Objective:
    Focus entirely on layout, structure, and styling. The content has already been generated and must not be modified.

    ### Context:
    - Document Type: Earnings Summary
    - Audience: {audience}
    - Tone: {tone}

    ### Layout and Styling Guidelines:
    - Use semantic HTML elements such as <header>, <section>, <article>, <table>, and <ul>/<li> to organize the document clearly.
    - Structure the content using headings, paragraphs, and bullet points to highlight financial performance and key metrics.
    - Apply embedded <style> tags to ensure a clean, professional appearance — including font consistency, spacing, alignment, and visual hierarchy.
    - Maintain readability and balance without introducing decorative elements or unnecessary complexity.
    - If applicable, use tables for comparisons (e.g., quarterly or YoY results), but do not infer structure that is not present in the content.

    ### Output Instructions:
    - Do not change the wording or meaning of the content.
    - Wrap the entire response in `<html>...</html>` tags.

    Here is the content:
    ---
    {document_content}
    ---

    Return **only valid HTML code**, without comments or extra explanation.
""",

    DocumentType.CREDIT_CARD_STATEMENT: """
    You are a professional HTML presentation assistant with expertise in formatting consumer financial documents for digital delivery.

    Your task is to transform the provided credit card statement content into a visually polished and professionally styled HTML document.

    ### Objective:
    Focus exclusively on visual presentation and formatting. The content has already been generated and should remain unchanged.

    ### Context:
    - Document Type: Credit Card Statement
    - Audience: {audience}
    - Tone: {tone}

    ### Design Guidelines:
    - Use semantic HTML and embedded <style> tags for styling, layout, spacing, and font.
    - Apply visual hierarchy using font size, weight, and spacing.
    - When the content includes a list of transactions (typically repeated lines with date, description, and amount), format them as a structured, styled table.
    - Ensure that all repeated entries — such as transaction lines — are clearly and consistently formatted as table rows.
    - Use alternating row background colors to improve readability.
    - Right-align numeric values (e.g., amounts) and left-align descriptions.
    - The final layout should feel like a professional document from a financial institution — not plain or default HTML.


    ### Output Instructions:
    - Do not change or interpret the content — format only.
    - Wrap the full response in `<html>...</html>` tags.
    - Return only valid, styled HTML — no extra text, comments, or explanations.


    Here is the content:
    ---
    {document_content}
    ---

    Return **only valid HTML code**, without comments or extra explanation.
"""
}
