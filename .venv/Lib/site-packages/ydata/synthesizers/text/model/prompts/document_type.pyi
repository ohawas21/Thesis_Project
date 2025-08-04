from enum import Enum

class DocumentType(Enum):
    INVOICE = 'invoice'
    BALANCE_SHEET = 'balance sheet'
    CASH_FLOW = 'cash flow'
    EARNINGS_SUMMARY = 'earnings summary'
    CREDIT_CARD_STATEMENT = 'credit card statement'
    @classmethod
    def normalize(cls, doc_type: str) -> str: ...
    @classmethod
    def get_html_prompt(cls, doc_type: str) -> str | None: ...
    @classmethod
    def get_content_prompt(cls, doc_type: str) -> str | None: ...
