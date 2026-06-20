"""Unit tests for burnt_toast.tools nested-brace JSON parsing (no Ollama required)."""

from __future__ import annotations

import unittest

from burnt_toast.tools import parse_final_answer, parse_tool_call


class TestParseToolCall(unittest.TestCase):
    def test_tool_prefix_format_still_works(self) -> None:
        call = parse_tool_call('TOOL: search_context {"query": "secret agent code"}')
        self.assertIsNotNone(call)
        self.assertEqual(call.name, "search_context")
        self.assertEqual(call.arguments, {"query": "secret agent code"})

    def test_fenced_json_format_still_works(self) -> None:
        text = '```json\n{"tool": "search_context", "arguments": {"query": "x"}}\n```'
        call = parse_tool_call(text)
        self.assertIsNotNone(call)
        self.assertEqual(call.name, "search_context")
        self.assertEqual(call.arguments, {"query": "x"})

    def test_bare_json_without_nesting_still_works(self) -> None:
        call = parse_tool_call('{"tool": "search_context", "arguments": {}}')
        self.assertIsNotNone(call)
        self.assertEqual(call.name, "search_context")

    def test_bare_json_with_nested_arguments_now_parses(self) -> None:
        # This is exactly the case the old [^{}]* regex could not match --
        # the nested "arguments" object broke the character class.
        text = '{"tool": "search_context", "arguments": {"query": "secret agent code"}}'
        call = parse_tool_call(text)
        self.assertIsNotNone(call)
        self.assertEqual(call.name, "search_context")
        self.assertEqual(call.arguments, {"query": "secret agent code"})

    def test_no_tool_call_returns_none(self) -> None:
        self.assertIsNone(parse_tool_call("I don't know the answer."))


class TestParseFinalAnswer(unittest.TestCase):
    def test_unnested_bare_json_still_works(self) -> None:
        result = parse_final_answer('{"secret_code": 4821}')
        self.assertEqual(result, {"secret_code": 4821})

    def test_single_quoted_bare_json_still_works(self) -> None:
        result = parse_final_answer("{'secret_code': 4821}")
        self.assertEqual(result, {"secret_code": 4821})

    def test_fenced_json_still_works(self) -> None:
        result = parse_final_answer('```json\n{"secret_code": 4821}\n```')
        self.assertEqual(result, {"secret_code": 4821})

    def test_whole_text_json_still_works(self) -> None:
        result = parse_final_answer('{"secret_code": 4821}')
        self.assertEqual(result, {"secret_code": 4821})

    def test_secret_code_nested_in_wrapper_object_still_resolves(self) -> None:
        # A weaker model wrapping the answer, e.g. {"result": {"secret_code": 4821}}.
        # The innermost-match preference must still pick out {"secret_code": 4821},
        # not the outer wrapper (which has no top-level secret_code key).
        result = parse_final_answer('{"result": {"secret_code": 4821}}')
        self.assertEqual(result, {"secret_code": 4821})

    def test_no_secret_code_returns_none(self) -> None:
        self.assertIsNone(parse_final_answer("I don't know the answer."))


if __name__ == "__main__":
    unittest.main()
