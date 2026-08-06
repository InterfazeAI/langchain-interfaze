from langchain_interfaze import __all__

EXPECTED = ["ChatInterfaze", "__version__"]


def test_all_imports() -> None:
    assert sorted(__all__) == sorted(EXPECTED)
