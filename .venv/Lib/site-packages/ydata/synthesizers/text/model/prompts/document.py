"""
    Document generation prompts
"""
# I can inject this later in the code
VARIANTS = ['Generate a version with more detail', 'Swap synonyms in the prompt or vary tone descriptors',
            'Use natural phrasing and vary sentence structure.']

GENERAL_PURPOSE_DOCUMENTS_TOPICS = """
    You are a professional document generation assistant with expertise across industries, formats, and communication styles.

    Your task is to generate a polished and realistic {document_type} based on the inputs provided.

    ### Required Inputs:
    - **Document Type:** {document_type} *(e.g., report, letter, proposal, memo, legal contract, marketing brochure)*
    - **Audience:** {audience} *(e.g., potential client, internal team, legal department, students)*
    - **Tone:** {tone} *(e.g., formal, conversational, persuasive, neutral)*
    - **Purpose:** {purpose} *(e.g., pitch a product, summarize research, send an official notice, propose a partnership)*

    ### Optional Inputs:
    - **Region/Locale:** {region} *(e.g., US, UK, EU, India — for date formats, phrasing, etc.)*
    - **Language:** {language} *(e.g., en, fr, de, es)*
    - **Length Preference:** {length} *(e.g., short, medium, detailed)*
    - **Key Topics or Sections:** {topics} *(structured list or freeform notes)*
    - **Reference Style or Brand Voice:** {style_guide} *(e.g., APA, internal guidelines, or tone of a known brand)*

    ### Output Guidelines:
    - Generate the full content of the {document_type} as if it were being used in a real business or professional context.
    - Use fully written-out names, descriptions, and details that feel **plausible and industry-appropriate** — avoid anything that signals placeholder content.
    - **Do not use fake names like "John Doe", "Jane Smith", "Company XYZ", "ACME Corp", or any other obvious stand-ins.**
    - Instead, choose **realistic but fictional names** that sound like they could belong to actual people or companies (e.g., "Nora Patel", "Blueleaf Analytics", "MediCore Health").
    - Do not include square brackets or comments like `[insert name]`, `[insert product]`, or anything similar.
    - Use structure and formatting appropriate to the {document_type} (headings, bullet points, paragraphs).
    - Maintain clarity, professionalism, and realism in all content.
    - Content should feel publish-ready or client-ready, not like a draft or template.

    {prompt_ending}
    Return only the complete document content, without any commentary or explanation.
"""

GENERAL_PURPOSE_HTML_SYSTEM_PROMPT = """
    You are a meticulous HTML expert. When asked to generate HTML for a specific type of document with defined content, you must:
    - Produce valid, well-indented, and semantic HTML5.
    - Include appropriate tags for structure (like <header>, <main>, <section>, <footer>, etc.).
    - Use placeholder text or links only if content is not specified.
    - Include comments to indicate the start and end of major sections.
    - Ensure the document is accessible and standards-compliant (e.g., use alt attributes on images). Respond only with the HTML code, unless otherwise asked.
"""

DOCUMENT_GENERATION_PROMPT_ENDINGS = [
    "Ensure the content flows naturally and maintains reader engagement.",
    "Use rich examples or analogies where relevant.",
    "Start with a compelling hook to draw the reader in.",
    "Maintain a balanced tone and be mindful of the audience’s background.",
    "Avoid unnecessary jargon unless clearly defined.",
    "Use storytelling techniques to convey the message more effectively.",
    "Incorporate bullet points or structure where appropriate for clarity.",
    "Focus on clarity, coherence, and reader-friendly formatting.",
    "Inspire trust and convey authority without sounding rigid.",
    "Add subtle rhetorical devices to emphasize key ideas.",
    "Keep the writing fluid, using transitions for smoother reading.",
    "Encourage action or reflection by the end of the document.",
    "Incorporate a sense of empathy into the delivery.",
    "Use varied sentence structures to improve rhythm and tone.",
    "Make it sound as if it was written by a seasoned professional.",
    "Include a brief summary or takeaway at the end.",
    "Prioritize directness, but allow for expressive phrasing.",
    "Let the writing breathe — avoid dense paragraphs.",
    "Reflect a sense of calm confidence in the writing.",
    "Write as though it were reviewed by an editorial team.",
    "Avoid filler content and focus on what truly matters.",
    "Use visual metaphors or descriptive imagery when fitting.",
    "Highlight critical insights or unique perspectives subtly.",
    "End on a thought-provoking note or a call to consideration.",
    "Make the tone subtly adaptive to the context of the reader.",
    "Keep pacing steady and adapt to the expected reading scenario.",
    "Write like you are having a one-on-one conversation with the reader.",
    "Make it sound intelligent, but accessible to a wide audience.",
    "Avoid clichés and lean toward original, vivid phrasing.",
    "Write with clarity, precision, and just a touch of elegance."
]

DOCUMENTS_TONE_POOL = {
    "formal": [
        "formal", "professional", "objective", "serious", "factual"
    ],
    "casual": [
        "casual", "friendly", "relaxed", "conversational", "lighthearted"
    ],
    "persuasive": [
        "persuasive", "convincing", "assertive", "motivational", "directive"
    ],
    "empathetic": [
        "empathetic", "compassionate", "reassuring", "respectful", "supportive"
    ],
    "inspirational": [
        "inspirational", "uplifting", "visionary", "encouraging", "artistic"
    ],
    "enthusiastic": [
        "enthusiastic", "passionate", "vibrant", "dynamic", "engaging"
    ],
    "humorous": [
        "humorous", "witty", "quirky", "playful", "tongue-in-cheek"
    ],
    "neutral": [
        "neutral", "balanced", "even-toned", "unbiased", "matter-of-fact"
    ],
    "professional": [
        "professional", "businesslike", "polished", "corporate", "authoritative", "objective", "refined", "clear and concise", "sophisticated", "neutral"
    ]

}


GENERAL_PURPOSE_DOCUMENTS_HTML = """
    You are a creative document designer assistant.

    Given the plain text content of a {document_type}, your task is to generate a clean, modern HTML version of the document.
    Apply semantic HTML tags and layout consistent with the tone: {tone} and target audience: {audience}.
    Apply modern design HTML style to make it more appeal - colour accents and other elements.

    ### Instructions:
    - Structure the document using headings, paragraphs, and optional bullet points
    - Maintain readability, professionalism, and clarity
    - Use CSS inline or as embedded <style> tags for layout if needed
    - Do not change the wording of the content

    ### Output Requirements:
    Put ```html at the beginning and ``` at the end of the script to separate the code from the text.

    Here is the content:
    ---
    {document_content}
    ---

    Please don't answer with any additional text in the script, your whole response should be the HTML code which can be directly executed.
"""

GENERAL_PURPOSE_DOCUMENTS_JSON = """
    You are a creative document designer assistant.

    Given the plain text content of a {document_type}, your task is to convert it into a structured JSON format that represents the layout of a modern, clean Microsoft Word document.
    The output should maintain the document’s logical structure while applying formatting appropriate for the tone: {tone} and audience: {audience}.

    ### Instructions:
    - Structure the content into a sequence of elements: titles, paragraphs, headings, lists, tables, or page breaks
    - Use appropriate JSON keys such as "heading", "paragraph", "list", "table", etc.
    - Preserve the exact text; do not paraphrase or rewrite the content
    - You may split large sections into multiple structured elements if helpful
    - Do not include any stylesheets or HTML—this is not an HTML document

    ### Output Format:
    Wrap your output using triple backticks and `json` to clearly separate the JSON code block.

    Example format:
    ```json
    [
      {{"type": "heading", "level": 1, "text": "Document Title"}},
      {{"type": "paragraph", "text": "Introduction paragraph goes here."}},
      {{"type": "heading", "level": 2, "text": "Section Title"}},
      {{"type": "list", "items": ["Point one", "Point two", "Point three"]}},
      {{"type": "paragraph", "text": "Concluding remarks."}}
    ]
    ```
    Here is the content:
    ---
    {document_content}
    ---
    Please do not include anything outside the JSON block. Your entire response must be a JSON array representing the Word document's structure.
"""
