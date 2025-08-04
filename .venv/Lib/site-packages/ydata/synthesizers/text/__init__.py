"""
    Init file for Text generators
"""
missing_dependencies = []
try:
    import openai
except ImportError:
    missing_dependencies.append("openai")

try:
    import anthropic
except ImportError:
    missing_dependencies.append("anthropic")

if missing_dependencies:
    raise ImportError(
        f"The 'ydata_synthesizers.text' module requires the following packages: "
        + ", ".join(missing_dependencies) +
        ". Please install with: pip install \"ydata-sdk[text]\""
    )
