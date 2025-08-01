from ada.persona import Persona
from ada.personas import Personas


def test_persona_creation():
    p = Persona(name="test", prompt="This is a test.")
    assert p.name == "test"
    assert p.prompt == "This is a test."
    assert str(p) == "Persona(test)"
    assert repr(p) == "Persona(name='test', prompt='This is a test.')"


def test_personas_all_contains_default():
    personas = Personas.all()
    assert any(p.name == "default" for p in personas)
    assert all(isinstance(p, Persona) for p in personas)


def test_personas_get_found():
    default = Personas.get("default")
    assert isinstance(default, Persona)
    assert default.name == "default"


def test_personas_get_not_found():
    assert Personas.get("notarealpersona") is None
