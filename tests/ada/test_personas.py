from ada.persona import Persona
from ada.personas import Personas


def test_personas_all_returns_personas():
    assert all(isinstance(p, Persona) for p in Personas.all())


def test_personas_all_contains_default():
    assert Personas.get("default") is not None


def test_personas_get_found():
    default = Personas.get("default")
    assert isinstance(default, Persona)
    assert default.name == "default"


def test_personas_get_not_found():
    assert Personas.get("notarealpersona") is None
