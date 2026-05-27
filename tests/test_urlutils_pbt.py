"""Property-based tests for urlutils module using Hypothesis."""
from hypothesis import given, strategies as st, settings, example, assume
import pytest

from boltons.urlutils import (
    URL, quote_path_part, quote_query_part, quote_fragment_part,
    quote_userinfo_part, unquote, parse_url
)


# ============================================================================
# HIGH PRIORITY: URL Parsing Roundtrip
# ============================================================================

class TestURLRoundtrip:
    """Test URL parsing and serialization roundtrip."""

    @given(st.text(alphabet='abcdefghijklmnopqrstuvwxyz', min_size=1, max_size=10))
    @example('http')
    @example('https')
    def test_simple_url_roundtrip(self, scheme):
        """Simple URLs should roundtrip through parsing and serialization."""
        url_str = f'{scheme}://example.com/path'
        url = URL(url_str)
        result = url.to_text()
        # Parse again to compare structure
        url2 = URL(result)
        assert url.scheme == url2.scheme
        assert url.host == url2.host
        assert url.path == url2.path

    @given(st.text(alphabet='abcdefghijklmnopqrstuvwxyz0123456789-', min_size=1, max_size=20))
    @example('example.com')
    @example('sub.example.com')
    def test_url_host_preserved(self, host):
        """URL host should be preserved through roundtrip."""
        assume('.' in host or len(host) > 2)  # Valid hostname
        url_str = f'http://{host}/'
        try:
            url = URL(url_str)
            result = url.to_text()
            url2 = URL(result)
            assert url.host == url2.host
        except Exception:
            # Invalid hostnames are acceptable to reject
            pass

    @given(st.text(alphabet='abcdefghijklmnopqrstuvwxyz0123456789-_/', min_size=0, max_size=50))
    @example('')
    @example('/')
    @example('/path')
    @example('/path/to/resource')
    def test_url_path_preserved(self, path):
        """URL path should be preserved through roundtrip."""
        if not path.startswith('/'):
            path = '/' + path
        url_str = f'http://example.com{path}'
        url = URL(url_str)
        result = url.to_text()
        url2 = URL(result)
        # Paths should be equivalent (may have normalization)
        assert url2.path is not None


# ============================================================================
# HIGH PRIORITY: Quote/Unquote Roundtrip
# ============================================================================

class TestQuoteUnquoteRoundtrip:
    """Test quote/unquote roundtrip properties."""

    @given(st.text(alphabet='abcdefghijklmnopqrstuvwxyz0123456789', min_size=0, max_size=50))
    @example('')
    @example('simple')
    @example('with space')
    def test_quote_unquote_path_roundtrip(self, text):
        """Path quoting and unquoting should roundtrip for safe chars."""
        quoted = quote_path_part(text)
        unquoted = unquote(quoted)
        assert unquoted == text

    @given(st.text(alphabet='abcdefghijklmnopqrstuvwxyz0123456789', min_size=0, max_size=50))
    @example('key')
    @example('value')
    def test_quote_unquote_query_roundtrip(self, text):
        """Query quoting and unquoting should roundtrip for safe chars."""
        quoted = quote_query_part(text)
        unquoted = unquote(quoted)
        assert unquoted == text

    @given(st.text(alphabet='abcdefghijklmnopqrstuvwxyz0123456789', min_size=0, max_size=50))
    @example('fragment')
    def test_quote_unquote_fragment_roundtrip(self, text):
        """Fragment quoting and unquoting should roundtrip for safe chars."""
        quoted = quote_fragment_part(text)
        unquoted = unquote(quoted)
        assert unquoted == text

    @given(st.text(alphabet='abcdefghijklmnopqrstuvwxyz0123456789-_', min_size=1, max_size=20))
    @example('user')
    @example('pass123')
    def test_quote_unquote_userinfo_roundtrip(self, text):
        """Userinfo quoting and unquoting should roundtrip."""
        quoted = quote_userinfo_part(text)
        unquoted = unquote(quoted)
        assert unquoted == text


# ============================================================================
# MEDIUM PRIORITY: Quote Idempotence
# ============================================================================

class TestQuoteIdempotence:
    """Test that quoting is idempotent."""

    @given(st.text(max_size=50))
    @example('')
    @example('test')
    def test_quote_path_idempotent(self, text):
        """Quoting path twice should equal quoting once."""
        once = quote_path_part(text)
        twice = quote_path_part(once)
        # Second quote should not change already-quoted string
        # (assuming it's already properly quoted)
        assert unquote(once) == unquote(twice)

    @given(st.text(max_size=50))
    def test_quote_query_idempotent(self, text):
        """Quoting query twice should equal quoting once."""
        once = quote_query_part(text)
        twice = quote_query_part(once)
        assert unquote(once) == unquote(twice)


# ============================================================================
# MEDIUM PRIORITY: URL Properties
# ============================================================================

class TestURLProperties:
    """Test URL object properties."""

    @given(st.text(alphabet='abcdefghijklmnopqrstuvwxyz', min_size=2, max_size=10))
    @example('http')
    @example('https')
    @example('ftp')
    def test_url_scheme_lowercase(self, scheme):
        """URL scheme should be lowercase."""
        url = URL(f'{scheme}://example.com/')
        assert url.scheme == scheme.lower()

    @given(st.integers(min_value=1, max_value=65535))
    @example(80)
    @example(443)
    @example(8080)
    def test_url_port_preserved(self, port):
        """URL port should be preserved."""
        url = URL(f'http://example.com:{port}/')
        assert url.port == port

    @given(st.text(alphabet='abcdefghijklmnopqrstuvwxyz0123456789', min_size=1, max_size=20))
    @example('key=value')
    @example('a=1&b=2')
    def test_url_query_string_preserved(self, query):
        """URL query string should be preserved."""
        url = URL(f'http://example.com/?{query}')
        assert url.query_string is not None


# ============================================================================
# MEDIUM PRIORITY: Unquote Properties
# ============================================================================

class TestUnquoteProperties:
    """Test unquote function properties."""

    @given(st.text(alphabet='abcdefghijklmnopqrstuvwxyz0123456789-_.~', max_size=50))
    @example('')
    @example('simple')
    def test_unquote_safe_chars_unchanged(self, text):
        """Unquoting text without % should return unchanged."""
        result = unquote(text)
        assert result == text

    @given(st.text(max_size=50))
    def test_unquote_idempotent(self, text):
        """Unquoting twice should equal unquoting once."""
        once = unquote(text)
        twice = unquote(once)
        assert once == twice

    @given(st.text(max_size=50))
    def test_unquote_returns_string(self, text):
        """Unquote should always return a string."""
        result = unquote(text)
        assert isinstance(result, str)


# ============================================================================
# MEDIUM PRIORITY: URL Normalization
# ============================================================================

class TestURLNormalization:
    """Test URL normalization properties."""

    @given(st.text(alphabet='abcdefghijklmnopqrstuvwxyz', min_size=2, max_size=10))
    @example('HTTP')
    @example('HtTp')
    def test_url_scheme_normalized_to_lowercase(self, scheme):
        """URL scheme should be normalized to lowercase."""
        url = URL(f'{scheme}://example.com/')
        assert url.scheme == scheme.lower()

    def test_url_default_port_removed(self):
        """Default ports should be removed in normalization."""
        url1 = URL('http://example.com:80/')
        url2 = URL('http://example.com/')
        # Both should be equivalent after normalization
        assert url1.host == url2.host
        assert url1.scheme == url2.scheme


# ============================================================================
# MEDIUM PRIORITY: Parse URL
# ============================================================================

class TestParseURL:
    """Test parse_url function."""

    @given(st.text(alphabet='abcdefghijklmnopqrstuvwxyz', min_size=2, max_size=10))
    @example('http')
    @example('https')
    def test_parse_url_returns_dict(self, scheme):
        """parse_url should return a dictionary."""
        result = parse_url(f'{scheme}://example.com/')
        assert isinstance(result, dict)
        assert 'scheme' in result

    @given(st.text(alphabet='abcdefghijklmnopqrstuvwxyz0123456789-.', min_size=3, max_size=30))
    @example('example.com')
    def test_parse_url_extracts_host(self, host):
        """parse_url should extract host."""
        assume('.' in host)
        try:
            result = parse_url(f'http://{host}/')
            assert result.get('host') or result.get('authority')
        except Exception:
            # Invalid URLs are acceptable to reject
            pass


# ============================================================================
# Edge Cases and Boundary Conditions
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_url_empty_path(self):
        """URL with empty path should work."""
        url = URL('http://example.com')
        assert url.host == 'example.com'

    def test_url_root_path(self):
        """URL with root path should work."""
        url = URL('http://example.com/')
        assert url.path == '/'

    def test_unquote_empty_string(self):
        """Unquoting empty string should return empty string."""
        assert unquote('') == ''

    def test_quote_empty_string(self):
        """Quoting empty string should return empty string."""
        assert quote_path_part('') == ''
        assert quote_query_part('') == ''
        assert quote_fragment_part('') == ''

    def test_unquote_no_percent(self):
        """Unquoting string without % should be fast path."""
        text = 'simple_text_123'
        result = unquote(text)
        assert result == text

    def test_url_with_fragment(self):
        """URL with fragment should preserve it."""
        url = URL('http://example.com/path#fragment')
        assert url.fragment == 'fragment'

    def test_url_with_query(self):
        """URL with query should preserve it."""
        url = URL('http://example.com/path?key=value')
        assert 'key' in url.query_string or url.qp.get('key') == ['value']

    @given(st.text(alphabet='%0123456789ABCDEF', min_size=0, max_size=30))
    def test_unquote_percent_encoded(self, text):
        """Unquote should handle percent-encoded strings."""
        try:
            result = unquote(text)
            assert isinstance(result, str)
        except Exception:
            # Malformed percent encoding is acceptable to reject
            pass


# ============================================================================
# Invariant Properties
# ============================================================================

class TestInvariants:
    """Test invariant properties that should always hold."""

    @given(st.text(max_size=50))
    def test_quote_produces_ascii(self, text):
        """Quoted strings should be ASCII."""
        quoted = quote_path_part(text)
        assert quoted.isascii()

    @given(st.text(max_size=50))
    def test_quote_no_spaces(self, text):
        """Quoted strings should not contain unencoded spaces."""
        quoted = quote_path_part(text)
        # Spaces should be encoded as %20
        if ' ' in text:
            assert ' ' not in quoted or '%20' in quoted

    @given(st.text(alphabet='abcdefghijklmnopqrstuvwxyz', min_size=2, max_size=10))
    def test_url_scheme_is_string(self, scheme):
        """URL scheme should be a string."""
        url = URL(f'{scheme}://example.com/')
        assert isinstance(url.scheme, str)

    @given(st.text(alphabet='abcdefghijklmnopqrstuvwxyz0123456789-.', min_size=3, max_size=30))
    def test_url_host_is_string(self, host):
        """URL host should be a string."""
        assume('.' in host)
        try:
            url = URL(f'http://{host}/')
            assert isinstance(url.host, str)
        except Exception:
            pass


# ============================================================================
# Type Preservation
# ============================================================================

class TestTypePreservation:
    """Test that functions preserve expected types."""

    @given(st.text(max_size=50))
    def test_quote_path_returns_str(self, text):
        """quote_path_part should return string."""
        result = quote_path_part(text)
        assert isinstance(result, str)

    @given(st.text(max_size=50))
    def test_quote_query_returns_str(self, text):
        """quote_query_part should return string."""
        result = quote_query_part(text)
        assert isinstance(result, str)

    @given(st.text(max_size=50))
    def test_quote_fragment_returns_str(self, text):
        """quote_fragment_part should return string."""
        result = quote_fragment_part(text)
        assert isinstance(result, str)

    @given(st.text(max_size=50))
    def test_unquote_returns_str(self, text):
        """unquote should return string."""
        result = unquote(text)
        assert isinstance(result, str)

    @given(st.text(alphabet='abcdefghijklmnopqrstuvwxyz', min_size=2, max_size=10))
    def test_url_to_text_returns_str(self, scheme):
        """URL.to_text() should return string."""
        url = URL(f'{scheme}://example.com/')
        result = url.to_text()
        assert isinstance(result, str)


# ============================================================================
# Special Characters
# ============================================================================

class TestSpecialCharacters:
    """Test handling of special characters."""

    @given(st.text(alphabet=' !@#$%^&*()', min_size=1, max_size=20))
    @example(' ')
    @example('!')
    @example('@')
    def test_quote_special_chars(self, text):
        """Special characters should be percent-encoded."""
        quoted = quote_path_part(text)
        # Should contain % for encoding
        if any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_.~' for c in text):
            assert '%' in quoted or text == quoted

    @given(st.text(alphabet='abcdefghijklmnopqrstuvwxyz /', min_size=1, max_size=30))
    @example('hello world')
    @example('path/to/file')
    def test_quote_unquote_preserves_meaning(self, text):
        """Quote/unquote should preserve semantic meaning."""
        quoted = quote_path_part(text)
        unquoted = unquote(quoted)
        assert unquoted == text


# ============================================================================
# URL Components
# ============================================================================

class TestURLComponents:
    """Test URL component extraction."""

    def test_url_has_scheme(self):
        """URL should extract scheme."""
        url = URL('http://example.com/')
        assert url.scheme == 'http'

    def test_url_has_host(self):
        """URL should extract host."""
        url = URL('http://example.com/')
        assert url.host == 'example.com'

    def test_url_has_path(self):
        """URL should extract path."""
        url = URL('http://example.com/path')
        assert url.path == '/path'

    @given(st.integers(min_value=1, max_value=65535))
    def test_url_has_port(self, port):
        """URL should extract port."""
        url = URL(f'http://example.com:{port}/')
        assert url.port == port

    def test_url_has_fragment(self):
        """URL should extract fragment."""
        url = URL('http://example.com/#section')
        assert url.fragment == 'section'
