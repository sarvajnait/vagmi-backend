"""
Kannada Text Converter Utility
Converts legacy ASCII-encoded Kannada text to proper Unicode
"""
import os
from pathlib import Path
from py_mini_racer import py_mini_racer
from loguru import logger


class KannadaConverter:
    """Singleton converter for Kannada text using kn.js"""

    _instance = None
    _ctx = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the JavaScript context with kn.js"""
        try:
            # Find kn.js in app/scripts directory
            scripts_dir = Path(__file__).parent.parent / "scripts"
            kn_js_path = scripts_dir / "kn.js"

            if not kn_js_path.exists():
                logger.warning(f"kn.js not found at {kn_js_path}")
                self._ctx = None
                return

            # Load and evaluate kn.js
            self._ctx = py_mini_racer.MiniRacer()
            with open(kn_js_path, "r", encoding="utf-8") as f:
                js_code = f.read()
                self._ctx.eval(js_code)

            # Initialize kn instance and create bound wrapper functions
            self._ctx.eval("""
                var knInstance = new Kn();

                // Create wrapper functions that maintain correct 'this' context
                function asciiToUnicode(text, englishNumbers, removeExtraSpace) {
                    return knInstance.ascii_to_unicode(text, englishNumbers, removeExtraSpace);
                }

                function unicodeToAscii(text, englishNumbers, removeExtraSpace) {
                    return knInstance.unicode_to_ascii(text, englishNumbers, removeExtraSpace);
                }
            """)

            logger.info("Kannada converter initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Kannada converter: {e}")
            self._ctx = None

    def is_available(self) -> bool:
        """Check if converter is available"""
        return self._ctx is not None

    def ascii_to_unicode(
        self,
        text: str,
        english_numbers: bool = False,
        remove_extra_space: bool = True
    ) -> str:
        """
        Convert ASCII-encoded Kannada text to Unicode

        Args:
            text: Input text in ASCII encoding
            english_numbers: Keep numbers in English (default: False)
            remove_extra_space: Remove extra spaces (default: True)

        Returns:
            Unicode Kannada text
        """
        if not self.is_available():
            logger.warning("Kannada converter not available, returning original text")
            return text

        try:
            # Use ctx.call() with the bound wrapper function
            result = self._ctx.call("asciiToUnicode", text, english_numbers, remove_extra_space)
            return result
        except Exception as e:
            logger.error(f"Error converting ASCII to Unicode: {e}")
            return text

    def unicode_to_ascii(
        self,
        text: str,
        english_numbers: bool = False,
        remove_extra_space: bool = True
    ) -> str:
        """
        Convert Unicode Kannada text to ASCII encoding

        Args:
            text: Input text in Unicode
            english_numbers: Keep numbers in English (default: False)
            remove_extra_space: Remove extra spaces (default: True)

        Returns:
            ASCII-encoded Kannada text
        """
        if not self.is_available():
            logger.warning("Kannada converter not available, returning original text")
            return text

        try:
            # Use ctx.call() with the bound wrapper function
            result = self._ctx.call("unicodeToAscii", text, english_numbers, remove_extra_space)
            return result
        except Exception as e:
            logger.error(f"Error converting Unicode to ASCII: {e}")
            return text


def detect_kannada_encoding(text: str) -> str:
    """
    Detect if text is in Unicode Kannada or legacy ASCII encoding

    Returns:
        "unicode" if text contains Unicode Kannada characters
        "ascii" if text appears to be legacy ASCII encoding
        "unknown" if cannot determine
    """
    if not text:
        return "unknown"

    # Check for Unicode Kannada characters (U+0C80 to U+0CFF)
    unicode_kannada_count = sum(1 for c in text if '\u0C80' <= c <= '\u0CFF')

    # If we find Unicode Kannada characters, it's Unicode
    if unicode_kannada_count > 10:  # Threshold to avoid false positives
        return "unicode"

    # Check for common legacy ASCII patterns
    # Legacy Kannada often uses Latin chars with diacritics
    suspicious_chars = sum(1 for c in text if c in 'ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ')

    # If more than 5% of text is suspicious Latin chars, likely legacy ASCII
    if len(text) > 0 and (suspicious_chars / len(text)) > 0.05:
        return "ascii"

    return "unknown"


def convert_kannada_text(text: str) -> str:
    """
    Smart conversion: Detect encoding and convert if needed

    Args:
        text: Input text (may be Unicode or legacy ASCII)

    Returns:
        Text in proper Unicode Kannada
    """
    encoding = detect_kannada_encoding(text)

    if encoding == "unicode":
        # Already Unicode, return as-is
        logger.debug("Text is already in Unicode Kannada")
        return text
    elif encoding == "ascii":
        # Convert from legacy ASCII to Unicode
        logger.info("Detected legacy ASCII Kannada, converting to Unicode")
        converter = KannadaConverter()
        return converter.ascii_to_unicode(text)
    else:
        # Unknown encoding, try conversion anyway (it won't hurt if it's already Unicode)
        logger.debug("Unknown encoding, attempting conversion")
        converter = KannadaConverter()
        converted = converter.ascii_to_unicode(text)

        # If conversion produced Unicode Kannada, use it; otherwise use original
        if detect_kannada_encoding(converted) == "unicode":
            return converted
        return text


# Singleton instance
_converter = KannadaConverter()


def ascii_to_unicode(text: str, english_numbers: bool = False, remove_extra_space: bool = True) -> str:
    """Convenience function for ASCII to Unicode conversion"""
    return _converter.ascii_to_unicode(text, english_numbers, remove_extra_space)


def unicode_to_ascii(text: str, english_numbers: bool = False, remove_extra_space: bool = True) -> str:
    """Convenience function for Unicode to ASCII conversion"""
    return _converter.unicode_to_ascii(text, english_numbers, remove_extra_space)
