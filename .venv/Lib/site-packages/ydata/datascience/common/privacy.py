from ydata.core.enum import StringEnum


class PrivacyLevel(StringEnum):
    """Privacy level exposed to the end-user."""
    HIGH_FIDELITY = "HIGH_FIDELITY"
    """High fidelity"""
    HIGH_PRIVACY = "HIGH_PRIVACY"
    """High privacy"""
    BALANCED_PRIVACY_FIDELITY = "BALANCED_PRIVACY_FIDELITY"
    """Balanced privacy/fidelity"""

    def __str__(self):
        if self.value == self.HIGH_FIDELITY.value:
            return "High Fidelity"
        if self.value == self.HIGH_PRIVACY.value:
            return "High Privacy"
        if self.value == self.BALANCED_PRIVACY_FIDELITY.value:
            return "Balanced Privacy/Fidelity"
        return "N/D"
