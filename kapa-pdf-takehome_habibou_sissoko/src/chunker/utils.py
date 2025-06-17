import html


def sanitize_title(heading: str) -> str:
    """
    Converts encoded HTML entities like &lt; (<), &#x27; (') from markdown content.

    Ex: "## FAQ&#x27s" -> "FAQ's"

    NOTE: not designed to be used with links. It will remove the anchor tag if present.
    """
    return html.unescape(heading).replace("#", "").strip()
