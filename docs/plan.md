# ADA Self-Editing Capability Implementation Plan

**NOTE** Consider this plan file to be a living document. Update any todos as they are completed. Do not remove completed todos ever. The Working Notes should reflect the current project state and goals.

## Summary

This plan outlines the implementation of self-editing capabilities for ADA (Agentic Digital Assistant), enabling it to modify its own codebase, execute code, and handle multi-step tasks within a REPL-style interactive loop. The implementation focuses on project-scoped file operations, Python-based code execution, autonomous operation with safety checks, and task breakdown capabilities.

**Key Principles:**
- Project-scoped operations only (no system-wide file access)
- Python-based capabilities (no arbitrary shell commands)
- Autonomous with safety checks (validate paths, backup critical files)
- Multi-step task planning and execution

## Research

### Areas to Investigate
- Best practices for sandboxed Python execution (subprocess vs other isolation methods)
- Context window management strategies for iterative tool use
- File diff generation libraries (difflib, unified diff format)
- AST-based code editing vs text-based search/replace

### References
- [x] Current ADA architecture review (agent.py, tool_box.py, response.py)
- [ ] Python subprocess security best practices
- [ ] Token budget management in multi-turn conversations
- [ ] AST manipulation libraries (ast, astroid, libcst)

## Working Notes

### Current Architecture Strengths
- Async TaskGroup architecture supports concurrent operations
- Tool calling mechanism already functional via `__handle_tool_calls`
- Hot-reload system for memories can be extended to tools
- Backend abstraction allows testing with different models

### Current Limitations
- Tool execution is single-turn (no iterative loop)
- No file system access capabilities
- No code execution abilities
- No task planning or multi-step coordination

### Design Decisions Made
- **Scope**: Project directory only (safety-first approach)
- **Execution**: Python subprocess isolation (timeout + error handling)
- **Autonomy**: Autonomous with safety checks (backup + validation)
- **Planning**: Task breakdown with sequential execution

## Open Questions

- [ ] Should we implement automatic context summarization when approaching token limits?
- [ ] Do we need a session-level undo/rollback system for all file changes?
- [ ] Should file edits show diffs even in autonomous mode?
- [ ] How should we handle large tool results (file contents, search results) that could overflow context?
- [ ] Should task execution be strictly sequential or allow parallelism for independent tasks?
- [ ] What's the right maximum iteration limit for the tool loop? (currently thinking 10)
- [ ] Should we persist tasks across conversation sessions?
- [ ] How do we handle mid-task interruption (ctrl+c, /stop command)?

---

## Implementation Checklist

### Phase 1: Core File Operations (Week 1)

#### 1.1 File Reading Tools
- [ ] 1.1.1 Create `ada/tools/file_read.py` - ReadFile tool
  - [ ] 1.1.1.1 Implement basic file reading with path validation
  - [ ] 1.1.1.2 Add line number prefixing for reference
  - [ ] 1.1.1.3 Add optional line range parameters (start_line, end_line)
  - [ ] 1.1.1.4 Handle file not found errors gracefully
  - [ ] 1.1.1.5 Add file size limit (prevent reading huge files)
- [ ] 1.1.2 Create `ada/tools/file_list.py` - ListFiles tool
  - [ ] 1.1.2.1 Implement directory listing
  - [ ] 1.1.2.2 Add glob pattern support (*.py, **/*.json)
  - [ ] 1.1.2.3 Add recursive vs non-recursive option
  - [ ] 1.1.2.4 Return file metadata (size, modified time)
- [ ] 1.1.3 Create `ada/tools/file_search.py` - SearchFiles tool
  - [ ] 1.1.3.1 Implement regex-based content search
  - [ ] 1.1.3.2 Add file pattern filtering (search only *.py files)
  - [ ] 1.1.3.3 Return file paths with line numbers and context
  - [ ] 1.1.3.4 Add case-sensitive/insensitive option

#### 1.2 File Writing Tools
- [ ] 1.2.1 Create `ada/tools/file_write.py` - WriteFile tool
  - [ ] 1.2.1.1 Implement basic file writing with path validation
  - [ ] 1.2.1.2 Add create parent directories if needed
  - [ ] 1.2.1.3 Add overwrite protection (check if file exists)
  - [ ] 1.2.1.4 Return success/failure status with message
- [ ] 1.2.2 Create `ada/tools/file_edit.py` - EditFile tool
  - [ ] 1.2.2.1 Implement line-based replacement (replace lines X-Y)
  - [ ] 1.2.2.2 Implement search/replace functionality
  - [ ] 1.2.2.3 Add multiple edit operations in single call
  - [ ] 1.2.2.4 Generate and return unified diff of changes

#### 1.3 File Management Tools
- [ ] 1.3.1 Create `ada/tools/file_ops.py` - File operations
  - [ ] 1.3.1.1 Implement CreateDirectory tool
  - [ ] 1.3.1.2 Implement DeleteFile tool with safety checks
  - [ ] 1.3.1.3 Implement MoveFile/RenameFile tool
  - [ ] 1.3.1.4 Add CopyFile tool

#### 1.4 Safety Infrastructure
- [ ] 1.4.1 Create `ada/safety.py` - Path validation and backups
  - [ ] 1.4.1.1 Implement project root detection
  - [ ] 1.4.1.2 Implement path validation (prevent directory traversal)
  - [ ] 1.4.1.3 Define critical files list (config.json, pyproject.toml, .git/)
  - [ ] 1.4.1.4 Implement file backup mechanism (copy to .ada/backups/)
  - [ ] 1.4.1.5 Implement backup cleanup (limit backup retention)
  - [ ] 1.4.1.6 Add restore_backup utility function

#### 1.5 Tool Registration and Testing
- [ ] 1.5.1 Update `ada/tool_box.py` to register file tools
- [ ] 1.5.2 Create unit tests for file tools in `tests/ada/tools/`
  - [ ] 1.5.2.1 Test ReadFile with various scenarios
  - [ ] 1.5.2.2 Test WriteFile with safety checks
  - [ ] 1.5.2.3 Test EditFile with line and search/replace modes
  - [ ] 1.5.2.4 Test ListFiles with glob patterns
  - [ ] 1.5.2.5 Test SearchFiles with regex patterns
- [ ] 1.5.3 Create integration tests for path validation
- [ ] 1.5.4 Manual testing with ADA REPL

### Phase 2: Code Execution (Week 1-2)

#### 2.1 Python Execution Tool
- [ ] 2.1.1 Create `ada/tools/execute_python.py` - ExecutePython tool
  - [ ] 2.1.1.1 Implement subprocess-based Python execution
  - [ ] 2.1.1.2 Add timeout mechanism (default 30s, configurable)
  - [ ] 2.1.1.3 Capture stdout and stderr separately
  - [ ] 2.1.1.4 Return exit code, stdout, stderr in structured format
  - [ ] 2.1.1.5 Handle subprocess errors and timeouts gracefully
  - [ ] 2.1.1.6 Add working directory parameter (default to project root)
  - [ ] 2.1.1.7 Set environment variables (PYTHONPATH, etc.)

#### 2.2 Testing Tools
- [ ] 2.2.1 Create `ada/tools/run_tests.py` - RunTests tool
  - [ ] 2.2.1.1 Execute pytest with file/pattern/marker filters
  - [ ] 2.2.1.2 Add verbose/quiet output options
  - [ ] 2.2.1.3 Parse pytest output for pass/fail counts
  - [ ] 2.2.1.4 Return structured results (passed, failed, errors)
  - [ ] 2.2.1.5 Add timeout support
- [ ] 2.2.2 Create `ada/tools/run_lint.py` - RunLint tool
  - [ ] 2.2.2.1 Execute ruff check with path parameter
  - [ ] 2.2.2.2 Parse lint output for errors and warnings
  - [ ] 2.2.2.3 Return structured results with file/line numbers
  - [ ] 2.2.2.4 Add auto-fix option (run ruff check --fix)

#### 2.3 Make Command Tool
- [ ] 2.3.1 Create `ada/tools/run_command.py` - RunCommand tool
  - [ ] 2.3.1.1 Execute allowlisted make targets (test, lint, check, clean)
  - [ ] 2.3.1.2 Add command validation (prevent arbitrary commands)
  - [ ] 2.3.1.3 Capture output and return structured results
  - [ ] 2.3.1.4 Add timeout support

#### 2.4 Testing and Integration
- [ ] 2.4.1 Create unit tests for execution tools
  - [ ] 2.4.1.1 Test ExecutePython with simple scripts
  - [ ] 2.4.1.2 Test timeout behavior
  - [ ] 2.4.1.3 Test error handling (syntax errors, runtime errors)
  - [ ] 2.4.1.4 Test RunTests with sample test files
  - [ ] 2.4.1.5 Test RunLint with sample code
- [ ] 2.4.2 Register tools in ToolBox
- [ ] 2.4.3 Manual testing with ADA REPL

### Phase 3: Iterative Tool Loop (Week 2)

#### 3.1 Response Handler Enhancement
- [ ] 3.1.1 Modify `ada/response.py` for multi-turn tool execution
  - [ ] 3.1.1.1 Track tool execution depth/iteration count
  - [ ] 3.1.1.2 Prevent infinite loops (max 10 iterations)
  - [ ] 3.1.1.3 Add tool result formatting for LLM context

#### 3.2 Agent Loop Modification
- [ ] 3.2.1 Modify `ada/agent.py` `__process_message` method
  - [ ] 3.2.1.1 Implement iterative tool calling loop
  - [ ] 3.2.1.2 Feed tool results back to LLM as messages
  - [ ] 3.2.1.3 Add token budget tracking across iterations
  - [ ] 3.2.1.4 Add stop conditions (max iterations, token limit, no more tools)
  - [ ] 3.2.1.5 Handle errors during tool iteration gracefully

#### 3.3 Conversation History Updates
- [ ] 3.3.1 Modify `ada/conversation.py` for tool messages
  - [ ] 3.3.1.1 Add support for tool call messages
  - [ ] 3.3.1.2 Add support for tool result messages
  - [ ] 3.3.1.3 Update message formatting for backend compatibility

#### 3.4 Control Commands
- [ ] 3.4.1 Add `/stop` command to interrupt tool execution
- [ ] 3.4.2 Add `/continue` command to resume after manual stop
- [ ] 3.4.3 Update help text with new commands

#### 3.5 Testing
- [ ] 3.5.1 Create integration tests for iterative tool loop
  - [ ] 3.5.1.1 Test with mocked backend and tool calls
  - [ ] 3.5.1.2 Test max iteration limit
  - [ ] 3.5.1.3 Test token budget limits
  - [ ] 3.5.1.4 Test stop conditions
- [ ] 3.5.2 Manual end-to-end testing with real LLM
  - [ ] 3.5.2.1 Test multi-step file editing scenario
  - [ ] 3.5.2.2 Test code search → read → edit workflow
  - [ ] 3.5.2.3 Test error recovery

### Phase 4: Task Planning System (Week 3)

#### 4.1 Task Models
- [ ] 4.1.1 Create `ada/task_planner.py` with Pydantic models
  - [ ] 4.1.1.1 Define Task model (id, description, status, result)
  - [ ] 4.1.1.2 Define TaskStatus enum (pending, in_progress, completed, failed, skipped)
  - [ ] 4.1.1.3 Define TaskPlan model (list of tasks, current task index)
  - [ ] 4.1.1.4 Add task serialization for conversation context

#### 4.2 Task Planning Logic
- [ ] 4.2.1 Implement TaskPlanner class
  - [ ] 4.2.1.1 Parse task list from LLM response
  - [ ] 4.2.1.2 Track current task execution
  - [ ] 4.2.1.3 Handle task state transitions
  - [ ] 4.2.1.4 Generate task summary for context

#### 4.3 Agent Integration
- [ ] 4.3.1 Add task_plan attribute to Agent class
- [ ] 4.3.2 Modify `__process_message` to support task execution mode
- [ ] 4.3.3 Add task context to system prompt when plan exists
- [ ] 4.3.4 Add task management commands
  - [ ] 4.3.4.1 `/tasks` - Show current task plan
  - [ ] 4.3.4.2 `/skip` - Skip current task
  - [ ] 4.3.4.3 `/retry` - Retry failed task
  - [ ] 4.3.4.4 `/cancel_plan` - Cancel current task plan
- [ ] 4.3.5 Update help text

#### 4.4 Prompt Engineering
- [ ] 4.4.1 Update default persona to encourage task planning
- [ ] 4.4.2 Add task planning examples to system prompt
- [ ] 4.4.3 Add task status reporting guidance

#### 4.5 Testing
- [ ] 4.5.1 Create unit tests for task models
- [ ] 4.5.2 Create integration tests for task planning
  - [ ] 4.5.2.1 Test task creation and state transitions
  - [ ] 4.5.2.2 Test skip and retry commands
  - [ ] 4.5.2.3 Test task plan cancellation
- [ ] 4.5.3 Manual testing with complex multi-step requests

### Phase 5: Advanced Tools & Refinement (Week 3-4)

#### 5.1 Introspection Tools
- [ ] 5.1.1 Create `ada/tools/introspection.py`
  - [ ] 5.1.1.1 GetToolDefinitions - List all registered tools
  - [ ] 5.1.1.2 GetPersonas - List available personas programmatically
  - [ ] 5.1.1.3 GetConfig - Read current configuration
  - [ ] 5.1.1.4 InspectModule - Get info about Python modules/classes
- [ ] 5.1.2 Register introspection tools
- [ ] 5.1.3 Test introspection tools

#### 5.2 Git Integration Tools (Optional)
- [ ] 5.2.1 Create `ada/tools/git_ops.py`
  - [ ] 5.2.1.1 GitStatus - Get current git status
  - [ ] 5.2.1.2 GitDiff - Show uncommitted changes
  - [ ] 5.2.1.3 GitLog - View recent commits
  - [ ] 5.2.1.4 GitCommit - Commit changes (with confirmation)
- [ ] 5.2.2 Register git tools
- [ ] 5.2.3 Test git tools

#### 5.3 Enhanced Safety Features
- [ ] 5.3.1 Implement session-level change tracking
- [ ] 5.3.2 Add `/undo` command to revert last file change
- [ ] 5.3.3 Add `/rollback` command to revert all session changes
- [ ] 5.3.4 Improve backup management
  - [ ] 5.3.4.1 Add backup listing command `/backups`
  - [ ] 5.3.4.2 Add restore from specific backup
  - [ ] 5.3.4.3 Add backup cleanup on session end

#### 5.4 Context Management
- [ ] 5.4.1 Implement token counting for tool results
- [ ] 5.4.2 Add automatic truncation for large results
- [ ] 5.4.3 Add pagination support for large file reads
- [ ] 5.4.4 Consider context summarization strategy

#### 5.5 Documentation
- [ ] 5.5.1 Update README.md with self-editing capabilities
- [ ] 5.5.2 Update AGENTS.md with new tools and architecture
- [ ] 5.5.3 Add tool development guide with file operation examples
- [ ] 5.5.4 Add safety guidelines documentation

#### 5.6 End-to-End Testing
- [ ] 5.6.1 Test scenario: "Add a new tool"
  - [ ] 5.6.1.1 ADA searches for existing tool examples
  - [ ] 5.6.1.2 ADA reads relevant files
  - [ ] 5.6.1.3 ADA creates new tool file
  - [ ] 5.6.1.4 ADA registers tool in ToolBox
  - [ ] 5.6.1.5 ADA runs tests
- [ ] 5.6.2 Test scenario: "Fix a bug"
  - [ ] 5.6.2.1 ADA searches for bug location
  - [ ] 5.6.2.2 ADA reads context around bug
  - [ ] 5.6.2.3 ADA edits file to fix bug
  - [ ] 5.6.2.4 ADA runs tests to verify fix
- [ ] 5.6.3 Test scenario: "Refactor code"
  - [ ] 5.6.3.1 ADA reads multiple related files
  - [ ] 5.6.3.2 ADA plans refactoring steps
  - [ ] 5.6.3.3 ADA edits multiple files
  - [ ] 5.6.3.4 ADA runs lint and tests
- [ ] 5.6.4 Test safety mechanisms
  - [ ] 5.6.4.1 Attempt to access parent directory (should fail)
  - [ ] 5.6.4.2 Attempt to delete critical file (should warn/backup)
  - [ ] 5.6.4.3 Test backup and restore functionality
- [ ] 5.6.5 Test error recovery
  - [ ] 5.6.5.1 Handle file not found errors
  - [ ] 5.6.5.2 Handle syntax errors in code execution
  - [ ] 5.6.5.3 Handle test failures

#### 5.7 Performance Optimization
- [ ] 5.7.1 Profile tool execution times
- [ ] 5.7.2 Optimize file search for large codebases
- [ ] 5.7.3 Cache frequently accessed files (if needed)
- [ ] 5.7.4 Optimize conversation history size

#### 5.8 Final Polish
- [ ] 5.8.1 Review all error messages for clarity
- [ ] 5.8.2 Add helpful hints in tool descriptions
- [ ] 5.8.3 Improve logging for debugging
- [ ] 5.8.4 Add configuration options for new features
  - [ ] 5.8.4.1 max_tool_iterations
  - [ ] 5.8.4.2 enable_backups
  - [ ] 5.8.4.3 backup_retention_days
  - [ ] 5.8.4.4 max_file_size
- [ ] 5.8.5 Update config.json.example with new options

---

## Success Criteria

- [ ] ADA can read any file in the project directory
- [ ] ADA can create, edit, and delete files with proper safety checks
- [ ] ADA can execute Python code and capture results
- [ ] ADA can run tests and interpret results
- [ ] ADA can break down complex requests into tasks
- [ ] ADA can iterate through multiple tool calls to complete tasks
- [ ] All safety mechanisms work (path validation, backups, critical file protection)
- [ ] Token budget management prevents context overflow
- [ ] Error handling is robust and provides helpful messages
- [ ] End-to-end scenarios work: add feature, fix bug, refactor code

## Future Enhancements (Post-Plan)

- [ ] Web search capability for documentation lookup
- [ ] Package installation tool (uv add, pip install)
- [ ] Database tools for data persistence
- [ ] API tools for external service integration
- [ ] Code quality metrics (complexity, coverage)
- [ ] Automated PR creation and review
- [ ] Multi-file refactoring with AST manipulation
- [ ] Visual diff display in terminal
- [ ] Collaborative editing with human-in-the-loop approval
