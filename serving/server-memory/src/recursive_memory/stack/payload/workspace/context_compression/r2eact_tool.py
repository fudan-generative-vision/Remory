"""Canonical r^2eAct facade for the safe ReAct tool runtime."""

from __future__ import annotations

from context_compression import grepseek_tool as _legacy


ALLOWED_COMMANDS = _legacy.ALLOWED_COMMANDS
CORPUS_FILENAME_ACTUAL = _legacy.CORPUS_FILENAME_ACTUAL
CORPUS_FILENAME_LOGICAL = _legacy.CORPUS_FILENAME_LOGICAL
CORPUS_PREFLIGHT_COMMAND = _legacy.CORPUS_PREFLIGHT_COMMAND
DEFAULT_BROWSER_MAX_CALLS = _legacy.DEFAULT_BROWSER_MAX_CALLS
DEFAULT_BROWSER_MAX_LENGTH = _legacy.DEFAULT_BROWSER_MAX_LENGTH
DEFAULT_BROWSER_TIMEOUT_SECONDS = _legacy.DEFAULT_BROWSER_TIMEOUT_SECONDS
DEFAULT_MAX_OUTPUT_TOKENS = _legacy.DEFAULT_MAX_OUTPUT_TOKENS
DEFAULT_TIMEOUT_SECONDS = _legacy.DEFAULT_TIMEOUT_SECONDS
DEFAULT_WEB_SEARCH_MAX_CALLS = _legacy.DEFAULT_WEB_SEARCH_MAX_CALLS
DEFAULT_WEB_SEARCH_TIMEOUT_SECONDS = _legacy.DEFAULT_WEB_SEARCH_TIMEOUT_SECONDS
DEFAULT_WEB_SEARCH_TOP_K = _legacy.DEFAULT_WEB_SEARCH_TOP_K
REACT_MEMORY_CANONICAL_SHELL_PATH = _legacy.REACT_MEMORY_CANONICAL_SHELL_PATH
REACT_MEMORY_QUERY_RESULT_ARCHIVE_POLICY = (
    _legacy.REACT_MEMORY_QUERY_RESULT_ARCHIVE_POLICY
)
REACT_MEMORY_QUERY_RESULT_ARCHIVE_STUB = _legacy.REACT_MEMORY_QUERY_RESULT_ARCHIVE_STUB
SHELL_CORPUS_MODES = _legacy.SHELL_CORPUS_MODES
SHELL_CORPUS_REACT_MEMORY = _legacy.SHELL_CORPUS_REACT_MEMORY
SHELL_CORPUS_WIKI18 = _legacy.SHELL_CORPUS_WIKI18
TOOL_RESPONSE_KEYS = _legacy.TOOL_RESPONSE_KEYS
ToolError = _legacy.ToolError
ToolResult = _legacy.ToolResult
append_browser_system_prompt = _legacy.append_browser_system_prompt
append_web_search_system_prompt = _legacy.append_web_search_system_prompt
canonical_browser_command = _legacy.canonical_browser_command
canonical_web_search_command = _legacy.canonical_web_search_command
parse_browser_command = _legacy.parse_browser_command
parse_web_search_command = _legacy.parse_web_search_command
preflight_corpus_tool = _legacy.preflight_corpus_tool
redact_react_memory_query_results_for_archive = (
    _legacy.redact_react_memory_query_results_for_archive
)
react_memory_shell_system_prompt = _legacy.react_memory_shell_system_prompt
run_tool = _legacy.run_tool
truncate_tokens = _legacy.truncate_tokens
validate_pipeline = _legacy.validate_pipeline

R2EActCorpusToolRunner = _legacy.GrepSeekCorpusToolRunner
R2EActToolModeRunner = _legacy.GrepSeekToolModeRunner


__all__ = [
    "ALLOWED_COMMANDS",
    "CORPUS_FILENAME_ACTUAL",
    "CORPUS_FILENAME_LOGICAL",
    "CORPUS_PREFLIGHT_COMMAND",
    "DEFAULT_BROWSER_MAX_CALLS",
    "DEFAULT_BROWSER_MAX_LENGTH",
    "DEFAULT_BROWSER_TIMEOUT_SECONDS",
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_WEB_SEARCH_MAX_CALLS",
    "DEFAULT_WEB_SEARCH_TIMEOUT_SECONDS",
    "DEFAULT_WEB_SEARCH_TOP_K",
    "R2EActCorpusToolRunner",
    "R2EActToolModeRunner",
    "REACT_MEMORY_CANONICAL_SHELL_PATH",
    "REACT_MEMORY_QUERY_RESULT_ARCHIVE_POLICY",
    "REACT_MEMORY_QUERY_RESULT_ARCHIVE_STUB",
    "SHELL_CORPUS_MODES",
    "SHELL_CORPUS_REACT_MEMORY",
    "SHELL_CORPUS_WIKI18",
    "TOOL_RESPONSE_KEYS",
    "ToolError",
    "ToolResult",
    "append_browser_system_prompt",
    "append_web_search_system_prompt",
    "canonical_browser_command",
    "canonical_web_search_command",
    "parse_browser_command",
    "parse_web_search_command",
    "preflight_corpus_tool",
    "redact_react_memory_query_results_for_archive",
    "react_memory_shell_system_prompt",
    "run_tool",
    "truncate_tokens",
    "validate_pipeline",
]
