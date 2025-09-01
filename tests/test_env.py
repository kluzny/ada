import os


def test_env():
    assert os.environ.get("APP_ENV") == "test"
