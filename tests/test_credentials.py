"""Tests for credential management with keyring integration."""

import pytest
from unittest.mock import patch, MagicMock

from security.credentials import (
    CredentialMissingError,
    delete_api_key,
    get_api_key,
    list_configured_providers,
    mask_key,
    require_credential,
    save_api_key,
    validate_api_key,
)


class TestMaskKey:
    def test_long_key(self):
        key = "sk-ant-abcdefghijklmnop"
        masked = mask_key(key)
        assert masked.startswith("sk-a")
        assert masked.endswith("mnop")
        assert "•" in masked
        assert key != masked

    def test_short_key(self):
        key = "abcd1234"
        masked = mask_key(key)
        assert masked == "••••••••"

    def test_very_short_key(self):
        key = "abc"
        masked = mask_key(key)
        assert masked == "•••"


class TestValidateApiKey:
    def test_anthropic_valid(self):
        assert validate_api_key("anthropic", "sk-ant-abcdefghijklmnopqrstuv")

    def test_anthropic_invalid(self):
        assert not validate_api_key("anthropic", "bad-key")

    def test_openai_valid(self):
        assert validate_api_key("openai", "sk-abcdefghijklmnopqrstuvwxyz")

    def test_gemini_valid(self):
        assert validate_api_key("gemini", "AIzaSyABC123def456GHI789jklMNO")

    def test_perplexity_valid(self):
        assert validate_api_key("perplexity", "pplx-abc123def456ghi789jklmno")

    def test_empty_key_invalid(self):
        assert not validate_api_key("anthropic", "")

    def test_none_key_invalid(self):
        assert not validate_api_key("anthropic", None)

    def test_unknown_provider_accepts_nonempty(self):
        assert validate_api_key("unknown_provider", "anything-goes-here")


class TestSaveApiKey:
    @patch("security.credentials.keyring")
    def test_save_valid_key(self, mock_keyring):
        mock_keyring.set_password = MagicMock()
        
        result = save_api_key("anthropic", "sk-ant-test1234567890abcdef")
        
        assert result is None
        mock_keyring.set_password.assert_called_once_with(
            "ORBIT", "anthropic", "sk-ant-test1234567890abcdef"
        )

    @patch("security.credentials.keyring")
    def test_save_invalid_key_raises(self, mock_keyring):
        with pytest.raises(ValueError, match="Invalid API key format"):
            save_api_key("anthropic", "bad-key")
        
        mock_keyring.set_password.assert_not_called()

    @patch("security.credentials.keyring")
    def test_save_keyring_error_raises_runtime_error(self, mock_keyring):
        mock_keyring.set_password.side_effect = RuntimeError("keyring: Keyring unavailable")
        
        with pytest.raises(RuntimeError, match="Failed to store API key"):
            save_api_key("anthropic", "sk-ant-test1234567890abcdef")

    @patch("security.credentials.keyring")
    def test_replace_valid_with_invalid_preserves_original(self, mock_keyring):
        mock_keyring.set_password = MagicMock()
        save_api_key("anthropic", "sk-ant-valid1234567890abcdef")
        
        with pytest.raises(ValueError, match="Invalid API key format"):
            save_api_key("anthropic", "bad-key")
        
        assert mock_keyring.set_password.call_count == 1


class TestGetApiKey:
    @patch("security.credentials.keyring")
    def test_get_existing_key(self, mock_keyring):
        mock_keyring.get_password.return_value = "sk-ant-test1234567890abcdef"
        
        key = get_api_key("anthropic")
        
        assert key == "sk-ant-test1234567890abcdef"
        mock_keyring.get_password.assert_called_once_with("ORBIT", "anthropic")

    @patch("security.credentials.keyring")
    def test_get_missing_key_returns_none(self, mock_keyring):
        mock_keyring.get_password.return_value = None
        
        key = get_api_key("anthropic")
        
        assert key is None

    @patch("security.credentials.keyring")
    def test_keyring_unavailable_raises(self, mock_keyring):
        mock_keyring.get_password.side_effect = RuntimeError("keyring: backend unavailable")
        
        with pytest.raises(RuntimeError, match="Failed to retrieve API key"):
            get_api_key("anthropic")


class TestDeleteApiKey:
    @patch("security.credentials.keyring")
    def test_delete_existing_key(self, mock_keyring):
        mock_keyring.delete_password = MagicMock()
        
        result = delete_api_key("anthropic")
        
        assert result is None
        mock_keyring.delete_password.assert_called_once_with("ORBIT", "anthropic")

    @patch("security.credentials.keyring")
    def test_delete_missing_key_silently_succeeds(self, mock_keyring):
        mock_keyring.delete_password.side_effect = Exception("not found in keyring")
        
        result = delete_api_key("anthropic")
        
        assert result is None

    @patch("security.credentials.keyring")
    def test_keyring_unavailable_raises(self, mock_keyring):
        mock_keyring.delete_password.side_effect = RuntimeError("keyring: backend unavailable")
        
        with pytest.raises(RuntimeError, match="Failed to delete API key"):
            delete_api_key("anthropic")


class TestListConfiguredProviders:
    @patch("security.credentials.keyring")
    def test_list_with_keys(self, mock_keyring):
        def mock_get(service, provider):
            return "some-key" if provider in ["anthropic", "openai"] else None
        
        mock_keyring.get_password = MagicMock(side_effect=mock_get)
        
        providers = list_configured_providers()
        
        assert "anthropic" in providers
        assert "openai" in providers
        assert "gemini" not in providers
        assert "perplexity" not in providers

    @patch("security.credentials.keyring")
    def test_list_empty(self, mock_keyring):
        mock_keyring.get_password.return_value = None
        
        providers = list_configured_providers()
        
        assert providers == []

    @patch("security.credentials.keyring")
    def test_list_never_returns_secrets(self, mock_keyring):
        mock_keyring.get_password.return_value = "sk-ant-secret1234567890"
        
        providers = list_configured_providers()
        
        for provider in providers:
            assert "sk-" not in provider
            assert provider in ["anthropic", "openai", "gemini", "perplexity"]


class TestRequireCredential:
    @patch("security.credentials.keyring")
    def test_require_missing_key_raises(self, mock_keyring):
        mock_keyring.get_password.return_value = None
        
        with pytest.raises(CredentialMissingError, match="anthropic"):
            require_credential("anthropic")

    @patch("security.credentials.keyring")
    def test_require_present_key_returns_it(self, mock_keyring):
        mock_keyring.get_password.return_value = "sk-ant-test1234567890abcdef"
        
        key = require_credential("anthropic")
        
        assert key == "sk-ant-test1234567890abcdef"
