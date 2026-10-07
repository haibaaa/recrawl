"""Tests for URL normalisation."""

import pytest

from recrawl.normalize import host_of, normalize_url


def test_lowercases_scheme_and_host() -> None:
    assert normalize_url("HTTP://WWW.Example.COM/Path") == "http://www.example.com/Path"


def test_strips_fragment() -> None:
    assert normalize_url("https://example.com/a#section") == "https://example.com/a"


def test_strips_tracking_params_and_sorts_query() -> None:
    result = normalize_url("https://example.com/s?b=2&a=1&utm_source=rss&fbclid=xyz")
    assert result == "https://example.com/s?a=1&b=2"


def test_strips_session_params() -> None:
    assert (
        normalize_url("https://example.com/p?jsessionid=ABC123&q=x") == "https://example.com/p?q=x"
    )


def test_removes_default_ports_only() -> None:
    assert normalize_url("https://example.com:443/x") == "https://example.com/x"
    assert normalize_url("http://example.com:80/x") == "http://example.com/x"
    assert normalize_url("http://example.com:8080/x") == "http://example.com:8080/x"


def test_collapses_duplicate_slashes() -> None:
    assert normalize_url("https://example.com//a///b") == "https://example.com/a/b"


def test_root_path_preserved() -> None:
    assert normalize_url("https://example.com") == "https://example.com/"


def test_rejects_missing_host() -> None:
    with pytest.raises(ValueError):
        normalize_url("/relative/path")


def test_rejects_unsupported_scheme() -> None:
    with pytest.raises(ValueError):
        normalize_url("ftp://example.com/file")


def test_rejects_overlong_url() -> None:
    with pytest.raises(ValueError):
        normalize_url("https://example.com/" + "a" * 3000)


def test_host_of() -> None:
    assert host_of("https://WWW.BBC.com/news") == "www.bbc.com"
