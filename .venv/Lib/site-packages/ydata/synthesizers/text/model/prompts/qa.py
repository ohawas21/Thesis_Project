"""
    Prompts for QA expert
"""
SYSTEM_INSTRUCTION = """
    You are a question-answer generation assistant.

    Your task is to read chunks of text from documents and generate high-quality, relevant question-answer pairs based solely on the content provided.

    Guidelines:
    - Each question should be clear, concise, and answerable based on the text.
    - Each answer should be self-contained and directly address the question.
    - Avoid hallucinating information or making assumptions beyond the text.
    - Prefer fact-based, specific questions rather than abstract or general ones.
    - Output format (JSON) with the fo
    - Return only the final JSON. Do not include code, explanation, or markdown fences.
"""

OUTPUT_REQUEST = """
    Please follow the format strictly and do not include any additional text keys or nesting at the beginning or end of your response.
    Do not include code, explanation, or markdown fences. Return only the final JSON with the exact provided structure.
"""

QA_GENERATION_GENERAL = """
    You are an expert at analyzing and understanding structured and unstructured documents across various formats (PDFs, Word documents, and scanned images with text).

    Your task is to carefully review the content of the provided document and generate a set of **relevant and grounded question-and-answer pairs** that reflect the information found **only** in the document.

    ---

    ### Requirements:

    1. **Content Boundaries:**
       - Do not assume or fabricate any information.
       - Only generate questions and answers based on what is explicitly or clearly implied in the document content.

    2. **Question Types:**
       - Include a variety of:
         - Factual questions (e.g., “What is the document title?”)
         - Detail-based questions (e.g., “Who signed the agreement?”)
         - Summary-level or reasoning-based questions (e.g., “What is the document's main purpose?”)
         - Contextual or cross-referenced questions (e.g., “What product was quoted in the pricing section?”)

    3. **Answer Expectations:**
       - Answers must be accurate, complete, and specific.
       - Avoid ambiguous, inferred, or overly general responses.

    4. **Output Format:**
       - Return a JSON with the following structure:
       [
            {{
                "question", ..
                "answer": ..,
                "difficulty": ..
            }}
       ]

       - Difficulty should be one of: `"easy"`, `"medium"`, or `"hard"`
         - Use `"easy"` for surface-level or fact recall
         - Use `"medium"` for multi-step references or structural reasoning
         - Use `"hard"` for contextual, interpretive, or highly specific questions

    ---

    ### JSON Input:
    ```json
    {info_json}

    ---
    {output_request}
"""

QA_GENERATION_INVOICE = """
    You are an expert at analyzing structured data and generating high-quality, context-aware question-and-answer (Q&A) pairs for training and testing purposes.

    Given a JSON object representing an invoice, your task is to create a dataset of realistic, relevant, and useful questions and answers that reflect the content of the invoice. These Q&A pairs should be useful for real-world applications such as customer support, invoice automation, comprehension testing, or chatbot integration.

    ---

    ### Guidelines:

    1. **Question Relevance:**
       - Focus only on information that is actually present in the provided JSON.
       - Avoid speculative or generic questions (e.g., "What is the refund policy?" if not in the JSON).

    2. **Question Types:**
       - Include a mix of:
         - Factual questions (e.g., “What is the invoice total?”)
         - Reference-based questions (e.g., “Who is the recipient of this invoice?”)
         - Reasoning questions (e.g., “Why is the tax amount $0?”)
         - Contextual business questions (e.g., “What services were provided?”)

    3. **Answer Quality:**
       - Answers must be complete, specific, and drawn only from the invoice content.
       - Use proper formatting for monetary values, dates, etc.

    4. **Output Format:**
       - Return an array of objects with the following structure: question, answer, category, difficulty.

       - Use the following **example categories** (when applicable):
         `invoiceMetadata`, `recipientInfo`, `companyInfo`, `services`, `totals`, `taxDetails`, `paymentInfo`, `notes`
       - Difficulty levels should be one of: `"easy"`, `"medium"`, or `"hard"`

    ---

    ### JSON Input:
    ```json
    {info_json}

    ---

    {output_request}
"""
