from interfaze_langchain import __all__

PUBLIC_EXPORTS = ["ChatInterfaze", "__version__"]


def test_public_exports_are_pinned() -> None:
    assert sorted(__all__) == sorted(PUBLIC_EXPORTS)
