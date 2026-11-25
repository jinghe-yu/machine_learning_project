import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from llm_caller import AINativeError, DEFAULT_MODEL, call_llm
from openai import omit


class AINativeTests(unittest.TestCase):
    def test_call_uses_configured_model_and_returns_content(self):
        client = _mock_client("  x AND y  ")
        with patch.dict(
            os.environ,
            {
                "AINATIVE_MODEL": "test/model",
            },
            clear=True,
        ), patch("llm_caller.OpenAI", return_value=client) as openai:
            result = call_llm("prompt", 0.3)

        self.assertEqual(result, "x AND y")
        openai.assert_called_once_with(timeout=90, max_retries=3)
        client.chat.completions.create.assert_called_once_with(
            model="test/model",
            messages=[
                {
                    "role": "system",
                    "content": "You solve Boolean-genetics hypothesis discovery tasks.",
                },
                {"role": "user", "content": "prompt"},
            ],
            temperature=0.3,
        )

    def test_default_model_is_used_when_no_override_is_set(self):
        client = _mock_client("x AND y")
        with patch.dict(os.environ, {}, clear=True), patch(
            "llm_caller.OpenAI", return_value=client
        ):
            call_llm("prompt", 0.3)

        self.assertEqual(
            client.chat.completions.create.call_args.kwargs["model"], DEFAULT_MODEL
        )

    def test_custom_endpoint_uses_placeholder_key_without_leaking_ainative_key(self):
        client = _mock_client("x AND y")
        with patch.dict(
            os.environ,
            {
                "LLM_BASE_URL": "https://example.test/v1",
                "LLM_MODEL": "test/provider-model",
                "AINATIVE_API_KEY": "ainative-secret",
                "AINATIVE_MODEL": "ignored/model",
            },
            clear=True,
        ), patch("llm_caller.OpenAI", return_value=client) as openai:
            call_llm("prompt", 0.3)

        openai.assert_called_once_with(
            timeout=90,
            max_retries=3,
            api_key="not-needed",
            base_url="https://example.test/v1",
        )
        self.assertEqual(
            client.chat.completions.create.call_args.kwargs["model"],
            "test/provider-model",
        )

    def test_hugging_face_endpoint_uses_llm_key_and_model(self):
        client = _mock_client("x AND y")
        with patch.dict(
            os.environ,
            {
                "LLM_BASE_URL": "https://router.huggingface.co/v1",
                "LLM_API_KEY": "hf_test_token",
                "LLM_MODEL": "meta-llama/Llama-3.3-70B-Instruct",
            },
            clear=True,
        ), patch("llm_caller.OpenAI", return_value=client) as openai:
            call_llm("prompt", 0.3)

        openai.assert_called_once_with(
            timeout=90,
            max_retries=3,
            api_key="hf_test_token",
            base_url="https://router.huggingface.co/v1",
        )
        self.assertEqual(
            client.chat.completions.create.call_args.kwargs["model"],
            "meta-llama/Llama-3.3-70B-Instruct",
        )

    def test_openrouter_key_selects_free_model_and_endpoint(self):
        client = _mock_client("x AND y")
        with patch.dict(
            os.environ,
            {"OPENROUTER_API_KEY": "or_test_key"},
            clear=True,
        ), patch("llm_caller.OpenAI", return_value=client) as openai:
            call_llm("prompt", 0.3)

        openai.assert_called_once_with(
            timeout=90,
            max_retries=3,
            api_key="or_test_key",
            base_url="https://openrouter.ai/api/v1",
        )
        self.assertEqual(
            client.chat.completions.create.call_args.kwargs["model"],
            "openrouter/free",
        )

    def test_non_ascii_openrouter_key_fails_before_client_initialization(self):
        with patch.dict(
            os.environ,
            {"OPENROUTER_API_KEY": "你的 OpenRouter API Key"},
            clear=True,
        ), patch("llm_caller.OpenAI") as openai:
            with self.assertRaisesRegex(AINativeError, "non-ASCII"):
                call_llm("prompt", 0.3)

        openai.assert_not_called()

    def test_ovh_endpoint_explicitly_omits_authorization_header(self):
        client = _mock_client("x AND y")
        with patch.dict(
            os.environ,
            {
                "LLM_BASE_URL": "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
                "LLM_API_KEY": "should-not-be-used",
                "AINATIVE_API_KEY": "ainative-secret",
            },
            clear=True,
        ), patch("llm_caller.OpenAI", return_value=client) as openai:
            call_llm("prompt", 0.3)

        openai.assert_called_once_with(
            timeout=90,
            max_retries=3,
            api_key="not-needed",
            base_url="https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
        )
        self.assertEqual(
            client.chat.completions.create.call_args.kwargs["extra_headers"],
            {"Authorization": omit},
        )

    def test_invalid_temperature_fails_before_client_initialization(self):
        with patch("llm_caller.OpenAI") as openai:
            with self.assertRaises(ValueError):
                call_llm("prompt", 2.1)

        openai.assert_not_called()

    def test_provider_errors_are_reported_as_ainative_errors(self):
        client = _mock_client("unused")
        client.chat.completions.create.side_effect = RuntimeError("provision failed")
        with patch("llm_caller.OpenAI", return_value=client):
            with self.assertRaisesRegex(AINativeError, "provision failed"):
                call_llm("prompt", 0.3)

    def test_empty_completion_is_rejected(self):
        client = _mock_client("   ")
        with patch("llm_caller.OpenAI", return_value=client):
            with self.assertRaisesRegex(AINativeError, "empty or non-text"):
                call_llm("prompt", 0.3)


def _mock_client(content):
    completion = SimpleNamespace(message=SimpleNamespace(content=content))
    response = SimpleNamespace(choices=[completion])
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=unittest.mock.Mock(return_value=response))
        )
    )


if __name__ == "__main__":
    unittest.main()
