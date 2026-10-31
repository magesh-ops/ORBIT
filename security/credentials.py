"""Credential management using system keyring for secure storage.

All API keys are stored in the system's secure credential store:
- macOS: Keychain
- Windows: Credential Manager
- Linux: Secret Service (libsecret)

NO plaintext storage, NO .env fallback, NO exceptions.
"""

from __future__ import annotations

import re
from typing import Optional

try:
    import keyring
    import keyring.errors
except ImportError as exc:
    raise RuntimeError(
        "keyring library is not installed. Install it with: pip install keyring"
    ) from exc


SERVICE_NAME = "ORBIT"
SUPPORTED_PROVIDERS = ["anthropic", "openai", "gemini", "perplexity"]


# ── Key format validation patterns ──────────────────────────────────────

_KEY_PATTERNS: dict[str, re.Pattern[str]] = {
    "anthropic": re.compile(r"^sk-ant-[a-zA-Z0-9_-]{20,}$"),
    "openai": re.compile(r"^sk-[a-zA-Z0-9_-]{20,}$"),
    "gemini": re.compile(r"^AI[a-zA-Z0-9_-]{20,}$"),
    "perplexity": re.compile(r"^pplx-[a-zA-Z0-9]{20,}$"),
}


def validate_api_key(provider: str, key: str) -> bool:
    """Validate that an API key matches the expected format for its provider.

    Args:
        provider: Provider name (anthropic, openai, gemini, perplexity)
        key: The API key to validate

    Returns:
        True if the key format is valid, False otherwise
    """
    if not key or not isinstance(key, str):
        return False

    pattern = _KEY_PATTERNS.get(provider)
    if pattern is None:
        # Unknown provider — accept any non-empty string
        return len(key.strip()) > 0

    return bool(pattern.match(key))


def save_api_key(provider: str, key: str) -> None:
    """Store an API key securely in the system keyring.

    Args:
        provider: Provider name (anthropic, openai, gemini, perplexity)
        key: The API key to store

    Raises:
        RuntimeError: If keyring storage fails or is unavailable
        ValueError: If the key format is invalid
    """
    if not validate_api_key(provider, key):
        raise ValueError(f"Invalid API key format for provider '{provider}'")

    try:
        keyring.set_password(SERVICE_NAME, provider, key)
    except Exception as exc:
        # Catch any keyring error or general exception
        error_type = type(exc).__name__
        if "Keyring" in error_type or "keyring" in str(exc).lower():
            raise RuntimeError(
                f"Failed to store API key for '{provider}'. "
                f"System keyring is unavailable or access was denied."
            ) from exc
        raise RuntimeError(
            f"Unexpected error storing API key for '{provider}'. "
            f"Check that your system keyring is properly configured."
        ) from exc


def get_api_key(provider: str) -> str | None:
    """Retrieve an API key from the system keyring.

    Args:
        provider: Provider name (anthropic, openai, gemini, perplexity)

    Returns:
        The API key if found, None if not configured

    Raises:
        RuntimeError: If keyring access fails
    """
    try:
        key = keyring.get_password(SERVICE_NAME, provider)
        return key
    except Exception as exc:
        # Catch any keyring error or general exception
        error_type = type(exc).__name__
        if "Keyring" in error_type or "keyring" in str(exc).lower():
            raise RuntimeError(
                f"Failed to retrieve API key for '{provider}'. "
                f"System keyring is unavailable or access was denied."
            ) from exc
        raise RuntimeError(
            f"Unexpected error retrieving API key for '{provider}'. "
            f"Check that your system keyring is properly configured."
        ) from exc


def delete_api_key(provider: str) -> None:
    """Remove an API key from the system keyring.

    Args:
        provider: Provider name (anthropic, openai, gemini, perplexity)

    Raises:
        RuntimeError: If keyring access fails

    Note:
        Silently succeeds if the key doesn't exist.
    """
    try:
        keyring.delete_password(SERVICE_NAME, provider)
    except Exception as exc:
        # Check if this is a "password not found" error
        error_type = type(exc).__name__
        error_msg = str(exc).lower()

        if "PasswordDelete" in error_type or "not found" in error_msg or "no password" in error_msg:
            # Key doesn't exist — this is not an error, just return
            return

        # Any other keyring error
        if "Keyring" in error_type or "keyring" in error_msg:
            raise RuntimeError(
                f"Failed to delete API key for '{provider}'. "
                f"System keyring is unavailable or access was denied."
            ) from exc

        raise RuntimeError(
            f"Unexpected error deleting API key for '{provider}'. "
            f"Check that your system keyring is properly configured."
        ) from exc


def list_configured_providers() -> list[str]:
    """Return a list of providers that have API keys stored.

    Returns:
        List of provider names with stored keys

    Raises:
        RuntimeError: If keyring access fails
    """
    configured: list[str] = []

    for provider in SUPPORTED_PROVIDERS:
        try:
            key = keyring.get_password(SERVICE_NAME, provider)
            if key is not None:
                configured.append(provider)
        except keyring.errors.KeyringError as exc:
            raise RuntimeError(
                f"Failed to check configured providers. "
                f"System keyring is unavailable or access was denied."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Unexpected error checking configured providers. "
                f"Check that your system keyring is properly configured."
            ) from exc

    return configured


# ── Legacy compatibility layer (DEPRECATED) ─────────────────────────────
# These functions remain for backward compatibility with existing code
# but will be removed in a future version.

def mask_key(key: str) -> str:
    """Return a masked version of an API key for safe display.

    Example: sk-ant-abc...wxyz → sk-an•••wxyz
    """
    _MASK_SUFFIX_LEN = 4
    if len(key) <= _MASK_SUFFIX_LEN + 4:
        return "•" * len(key)
    return key[:4] + "•" * (len(key) - 4 - _MASK_SUFFIX_LEN) + key[-_MASK_SUFFIX_LEN:]


def require_credential(provider: str) -> str:
    """Retrieve an API key or raise if not configured.

    DEPRECATED: Use get_api_key() and handle None explicitly.

    Raises:
        CredentialMissingError: If no key is configured for the provider
        RuntimeError: If keyring access fails
    """
    key = get_api_key(provider)
    if not key:
        raise CredentialMissingError(provider)
    return key


class CredentialMissingError(Exception):
    """Raised when a required API key is not configured."""

    def __init__(self, provider: str) -> None:
        self.provider = provider
        super().__init__(
            f"No API key configured for '{provider}'. "
            f"Add one with: /add-key {provider}"
        )
