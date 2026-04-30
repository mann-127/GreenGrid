"""Smoke test — verifies the package imports without error."""


def test_import_greengrid():
    import greengrid

    assert greengrid


def test_import_greengrid_cli():
    from greengrid import cli

    assert cli


def test_import_greengrid_settings():
    from greengrid import settings

    assert settings
