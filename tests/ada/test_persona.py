from ada.persona import Persona


def test_persona():
    p = Persona(name="test", description="A test persona.", prompt="This is a test.")
    assert p.name == "test"
    assert p.prompt == "This is a test."
    assert p.description == "A test persona."
    assert str(p) == "test: A test persona."
    assert (
        repr(p)
        == "Persona(name='test', description='A test persona.', prompt='This is a test.')"
    )
