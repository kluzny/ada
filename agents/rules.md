# Agent Rules

This document contains additional rules and guidelines that agents must always follow when working with this codebase.

## General Rules

1. **Always read before writing**: Never modify a file without reading it first to understand its current state
2. **Follow the existing patterns**: Maintain consistency with existing code structure and conventions
3. **Run tests after changes**: Always run `make test` or `pytest` after making code changes to verify nothing breaks
4. **Lint before committing**: Run `make lint` to ensure code follows formatting and linting standards

## Code Modification Rules

1. **Preserve existing functionality**: When modifying code, ensure existing behavior is not broken
2. **Add tests for new features**: Any new functionality should include corresponding test coverage
3. **Update documentation**: If adding new features or changing behavior, update relevant documentation (AGENTS.md, README.md, docstrings)
4. **Follow type hints**: All new code must include proper type hints following the project's conventions

## File Operation Rules

1. **Use relative paths**: When working with files, use relative paths from the application root.
2. **Respect the directory structure**: Keep files organized according to the existing structure
3. **Don't create unnecessary files**: Only create new files when explicitly needed, prefer editing existing ones

## Testing Rules

1. **Mirror test structure**: Tests should mirror the source code structure under `tests/ada/`
2. **Use fixtures**: Leverage existing test fixtures from `tests/helpers/fixtures.py`
3. **Test edge cases**: Consider error conditions, edge cases, and invalid inputs
4. **Keep tests focused**: Each test should verify a specific behavior or scenario

## Configuration Rules

1. **Never commit secrets**: Do not commit API keys, tokens, or sensitive data to config files
2. **Use config.json.example**: When adding new config options, update the example file
3. **Document new settings**: Add documentation for any new configuration options in AGENTS.md

## Persona and Memory Rules

1. **Alphabetical loading**: Memory files are loaded alphabetically, name them with numeric prefixes (001_, 002_, etc.)
2. **Keep memories focused**: Each memory file should contain a single, focused piece of context
3. **Test persona changes**: After modifying persona definitions, test them in the REPL

## Backend Development Rules

1. **Follow the Base interface**: All backends must implement the `Base` abstract class methods
2. **Return OpenAI format**: Backends must return responses in OpenAI-compatible format
3. **Handle errors gracefully**: Include proper error handling and descriptive error messages
4. **Document configuration**: Clearly document all backend-specific configuration options

## Tool Development Rules

1. **Inherit from Base**: All tools must inherit from `ada.tools.Base`
2. **Provide clear descriptions**: Tool descriptions should clearly explain what the tool does
3. **Define JSON schema**: Use proper JSON schema for tool parameters
4. **Register tools**: Add new tools to `ToolBox.AVAILABLE_TOOLS` in `ada/tool_box.py`

## Git and Version Control Rules

1. **Write clear commit messages**: Follow the existing commit message style in the repository
2. **Keep commits focused**: Each commit should address a single concern or feature
3. **Don't commit generated files**: Avoid committing logs, conversation history, or downloaded models

## Documentation Rules

1. **Use Mermaid for diagrams**: When adding diagrams to markdown documentation files, use Mermaid syntax by default
2. **Validate Mermaid diagrams**: All Mermaid diagrams must be validated with a test compile command:
   ```bash
   mmdc --width 1400 --scale=2 -i $file -o tmp/$file
   ```
3. **Test diagrams before committing**: Ensure all diagrams render correctly before committing documentation changes

## Todo Format Rules

1. **Use markdown format**: All todos must be created as markdown files with proper formatting
2. **Organize into numbered phases**: Structure todos into numbered phases (Phase 1, Phase 2, etc.)
3. **Use hierarchical numbering**: Format each todo item as `- [ ] 1.2.3 item` following a hierarchical numbering scheme
4. **Include standard sections**: Every todo file should include:
   - **Summary**: Brief overview of the task at the top
   - **Research**: Section for research findings and references
   - **Working Notes**: Section for ongoing notes and observations
   - **Open Questions**: Section at the very end for unanswered questions and uncertainties
5. **Maintain hierarchy**: Use proper indentation to show parent-child relationships in the numbered structure

## Communication Rules

1. **Be concise**: Provide clear, concise explanations and updates
2. **Show progress**: Use appropriate tools to show what you're doing
3. **Ask when unclear**: If requirements are ambiguous, ask for clarification before proceeding
4. **Report errors clearly**: When errors occur, provide clear information about what went wrong and how to fix it
