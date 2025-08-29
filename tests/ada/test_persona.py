import os

from unittest.mock import patch
from pathlib import Path
from textwrap import dedent

from ada.persona import Persona

TEST_MEMORY_PATH = Path("tests/fixtures/memories")


def test_persona():
    persona = Persona(
        name="test", description="A test persona.", prompt="This is a test."
    )
    assert persona.name == "test"
    assert persona.prompt == "This is a test."
    assert persona.description == "A test persona."
    assert str(persona) == "test: A test persona."
    assert (
        repr(persona)
        == "Persona(name='test', description='A test persona.', prompt='This is a test.')"
    )


def test_persona_get_prompt():
    persona = Persona(
        name="test",
        description="A test persona.",
        prompt="This is a test.",
    )
    assert persona.get_prompt() == "This is a test."


def test_persona_get_memory_files():
    persona = Persona(
        name="test",
        description="A test persona.",
        prompt="This is a test.",
    )

    with patch(
        "ada.persona.Persona._memory_path", return_value=TEST_MEMORY_PATH
    ) as _memory_path:
        assert persona._get_memory_files() == [
            os.path.join(TEST_MEMORY_PATH, "001.txt"),
            os.path.join(TEST_MEMORY_PATH, "002.txt"),
        ]


def test_persona_commands():
    persona = Persona(
        name="test",
        description="A test persona.",
        prompt="This is a test.",
    )

    with patch(
        "ada.persona.Persona._memory_path", return_value=TEST_MEMORY_PATH
    ) as _memory_path:
        assert persona._commands() == [
            dedent("""
            <memory>
            1
            </memory>
            """).lstrip(),
            dedent("""
            <memory>
            2
            </memory>
            """).lstrip(),
        ]


def test_persona_get_prompt_appends_memories():
    persona = Persona(
        name="test",
        description="A test persona.",
        prompt="This is a test.",
    )

    with patch(
        "ada.persona.Persona._memory_path", return_value=TEST_MEMORY_PATH
    ) as _memory_path:
        assert (
            persona.get_prompt()
            == dedent("""
                    This is a test.

                    IMPORTANT: Additional instructions are wrapped with <memory></memory>
                    <memory>
                    1
                    </memory>

                    <memory>
                    2
                    </memory>
                """).lstrip()
        )


def test_persona_clear_cached_memories():
    persona = Persona(
        name="test",
        description="A test persona.",
        prompt="This is a test.",
    )

    persona._cached_memories = dedent("""
            <memory>
            foo
            </memory>
            """)
    assert (
        persona.get_prompt()
        == dedent("""
                This is a test.

                IMPORTANT: Additional instructions are wrapped with <memory></memory>

                <memory>
                foo
                </memory>
        """).lstrip()
    )

    persona.clear_cached_memories()
    assert persona.get_prompt() == "This is a test."
