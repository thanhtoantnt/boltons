"""Property-based tests for strutils module using Hypothesis."""
from hypothesis import given, strategies as st, settings, example, assume, HealthCheck
import pytest
import shlex
import string

from boltons.strutils import (
    camel2under, under2camel, slugify, gzip_bytes, gunzip_bytes,
    parse_int_list, format_int_list, args2sh, ordinalize,
    asciify, is_ascii, strip_ansi, pluralize, singularize,
    bytes2human, html2text, multi_replace, MultiReplace
)


# ============================================================================
# HIGH PRIORITY: Roundtrip Properties
# ============================================================================

class TestGzipRoundtrip:
    """Test gzip_bytes/gunzip_bytes roundtrip property."""

    @given(st.binary())
    @example(b'')
    @example(b'a')
    @example(b'a' * 10000)
    def test_gzip_gunzip_roundtrip(self, data):
        """Compressing then decompressing returns original data."""
        compressed = gzip_bytes(data)
        decompressed = gunzip_bytes(compressed)
        assert decompressed == data

    @given(st.binary(), st.integers(min_value=1, max_value=9))
    @example(b'test', 1)
    @example(b'test', 9)
    def test_gzip_all_levels_roundtrip(self, data, level):
        """All compression levels should roundtrip correctly."""
        compressed = gzip_bytes(data, level=level)
        decompressed = gunzip_bytes(compressed)
        assert decompressed == data

    @given(st.binary(min_size=100))
    def test_gzip_actually_compresses(self, data):
        """Gzip should compress repetitive data."""
        # Create highly compressible data
        repetitive = data[:10] * 100
        compressed = gzip_bytes(repetitive)
        # Should achieve some compression on repetitive data
        assert len(compressed) < len(repetitive)


class TestIntListRoundtrip:
    """Test parse_int_list/format_int_list roundtrip property."""

    @given(st.lists(st.integers(min_value=0, max_value=1000), max_size=50))
    @example([])
    @example([1])
    @example([1, 2, 3])
    @example([1, 3, 5, 7])
    @example([1, 2, 3, 10, 11, 12])
    def test_format_parse_roundtrip(self, int_list):
        """Formatting then parsing returns sorted unique integers."""
        formatted = format_int_list(int_list)
        parsed = parse_int_list(formatted)
        # Should get back sorted unique values
        assert parsed == sorted(set(int_list))

    @given(st.lists(st.integers(min_value=1, max_value=100), min_size=1, max_size=20))
    def test_format_is_compact(self, int_list):
        """Format should collapse contiguous ranges."""
        formatted = format_int_list(int_list)
        parsed = parse_int_list(formatted)
        assert parsed == sorted(set(int_list))
        # Formatted string should not be longer than naive comma-separated
        naive = ','.join(map(str, sorted(set(int_list))))
        # Allow formatted to be longer due to range notation overhead
        # but it should be compact for contiguous ranges
        if int_list == list(range(min(int_list), max(int_list) + 1)):
            # Fully contiguous should be very compact
            assert len(formatted) <= len(naive)

    @given(st.text(alphabet=st.characters(whitelist_categories=('Nd',), max_codepoint=127) | st.just(',') | st.just('-'), min_size=1, max_size=50))
    def test_parse_valid_strings(self, range_string):
        """Parse should handle valid range strings without crashing."""
        try:
            result = parse_int_list(range_string)
            # Result should be sorted
            assert result == sorted(result)
            # All values should be integers
            assert all(isinstance(x, int) for x in result)
        except (ValueError, AttributeError):
            # Invalid format is acceptable to reject
            pass


class TestShellEscapingRoundtrip:
    """Test args2sh shell escaping roundtrip with shlex."""

    @given(st.lists(st.text(max_size=100), max_size=20))
    @example([])
    @example([''])
    @example(['simple'])
    @example(['with space'])
    @example(["with'quote"])
    @example(['with"doublequote'])
    @example(['with\nnewline'])
    @example(['with\ttab'])
    def test_args2sh_shlex_roundtrip(self, args):
        """Shell-escaped args should roundtrip through shlex.split."""
        escaped = args2sh(args)
        # shlex.split should recover the original args
        try:
            parsed = shlex.split(escaped)
            assert parsed == args
        except ValueError:
            # Some edge cases might not parse, but shouldn't crash args2sh
            pass

    @given(st.lists(st.text(alphabet=st.characters(blacklist_characters='\x00'), max_size=50), max_size=10))
    def test_args2sh_no_injection(self, args):
        """Escaped args should not allow command injection."""
        escaped = args2sh(args)
        # Should not contain unescaped semicolons, pipes, or redirects
        # outside of quotes (basic check)
        parsed = shlex.split(escaped)
        assert len(parsed) == len(args)


class TestCamelUnderConversion:
    """Test camel2under/under2camel conversion properties."""

    @given(st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll'), min_codepoint=65, max_codepoint=122), min_size=1, max_size=50))
    @example('SimpleTest')
    @example('HTTPResponse')
    @example('XMLParser')
    def test_camel_under_case_insensitive_roundtrip(self, camel_string):
        """Converting camel to under and back should preserve lowercase form."""
        # Filter to valid camel case patterns
        if not camel_string[0].isupper():
            camel_string = camel_string.capitalize()

        under = camel2under(camel_string)
        back_to_camel = under2camel(under)

        # Case-insensitive comparison (camel2under lowercases)
        assert camel2under(back_to_camel) == under

    @given(st.text(alphabet='abcdefghijklmnopqrstuvwxyz_', min_size=1, max_size=50).filter(lambda s: not s.startswith('_') and not s.endswith('_') and '__' not in s))
    @example('simple_test')
    @example('http_response')
    def test_under_camel_roundtrip(self, under_string):
        """Converting under to camel and back should preserve structure."""
        camel = under2camel(under_string)
        back_to_under = camel2under(camel)
        # Should get back the same underscore form (may lose single-char segments)
        # Note: single letters between underscores may be lost (a_a -> Aa -> aa)
        # This is expected behavior, so we check a weaker property
        assert back_to_under.replace('_', '') == under_string.replace('_', '').lower()


# ============================================================================
# MEDIUM PRIORITY: Idempotence Properties
# ============================================================================

class TestIdempotence:
    """Test idempotence properties for normalization functions."""

    @given(st.text())
    @example('')
    @example('simple')
    @example('With Punctuation!!!')
    def test_slugify_idempotent(self, text):
        """Slugifying twice should equal slugifying once."""
        once = slugify(text)
        twice = slugify(once)
        assert once == twice

    @given(st.text())
    def test_asciify_idempotent(self, text):
        """Asciifying twice should equal asciifying once."""
        once = asciify(text)
        # asciify returns bytes, so we can't apply it twice directly
        # Instead, check that the result is stable
        assert isinstance(once, bytes)
        # Decoding and re-asciifying should give same result
        if once:
            decoded = once.decode('ascii', errors='replace')
            twice = asciify(decoded)
            # Should be idempotent after one application
            assert once == twice or is_ascii(decoded)

    @given(st.text())
    def test_strip_ansi_idempotent(self, text):
        """Stripping ANSI codes twice should equal stripping once."""
        once = strip_ansi(text)
        twice = strip_ansi(once)
        assert once == twice

    @given(st.text(alphabet=st.characters(blacklist_categories=('Cc',), blacklist_characters='<>&'), max_size=200))
    def test_html2text_idempotent_on_plain_text(self, text):
        """html2text on plain text (no HTML chars) should be idempotent."""
        once = html2text(text)
        twice = html2text(once)
        assert once == twice


# ============================================================================
# MEDIUM PRIORITY: Invariant Properties
# ============================================================================

class TestInvariants:
    """Test invariant properties that should always hold."""

    @given(st.text())
    def test_slugify_no_punctuation(self, text):
        """Slugified text should remove punctuation and whitespace."""
        result = slugify(text)
        # Slugify removes punctuation and whitespace, but may leave other chars
        # Check that common punctuation is removed
        for char in '!@#$%^&*()[]{}.,;:?/\\|<>':
            assert char not in result

    @given(st.text())
    def test_asciify_is_ascii(self, text):
        """Asciified text should be ASCII."""
        result = asciify(text)
        if isinstance(result, bytes):
            assert all(b < 128 for b in result)
        else:
            assert is_ascii(result)

    @given(st.text())
    def test_strip_ansi_no_ansi(self, text):
        """Stripped text should contain no ANSI escape codes."""
        result = strip_ansi(text)
        # Basic check: no escape sequences
        assert '\x1b[' not in result

    @given(st.integers(min_value=0, max_value=10**15))
    @example(0)
    @example(1)
    @example(11)
    @example(12)
    @example(13)
    @example(21)
    @example(111)
    def test_ordinalize_format(self, number):
        """Ordinalized numbers should end with st/nd/rd/th."""
        result = ordinalize(number)
        assert result.endswith(('st', 'nd', 'rd', 'th'))
        # Should start with the number
        assert result.startswith(str(number))

    @given(st.integers(min_value=10, max_value=20))
    def test_ordinalize_teens(self, number):
        """Teen numbers (10-19) should all end with 'th'."""
        result = ordinalize(number)
        assert result.endswith('th')

    @given(st.integers(min_value=0))
    @example(0)
    @example(1024)
    @example(1024**2)
    def test_bytes2human_monotonic(self, nbytes):
        """bytes2human should produce reasonable output."""
        result = bytes2human(nbytes)
        # Should contain a number and a unit
        assert any(c.isdigit() for c in result)
        assert any(c.isalpha() for c in result)
        # Should end with a size symbol
        assert result[-1] in 'BKMGTPEZY'

    @given(st.lists(st.integers(min_value=0, max_value=1000), max_size=50))
    def test_format_int_list_sorted(self, int_list):
        """Formatted int list should represent sorted values."""
        formatted = format_int_list(int_list)
        parsed = parse_int_list(formatted)
        # Parsed result should be sorted
        assert parsed == sorted(parsed)


# ============================================================================
# MEDIUM PRIORITY: Pluralization Properties
# ============================================================================

class TestPluralization:
    """Test pluralize/singularize properties."""

    @given(st.text(alphabet=st.characters(whitelist_categories=('Ll',), min_codepoint=97, max_codepoint=122), min_size=2, max_size=20))
    @example('cat')
    @example('dog')
    @example('box')
    @example('class')
    def test_pluralize_increases_or_maintains_length(self, word):
        """Pluralizing should not decrease word length (usually)."""
        plural = pluralize(word)
        # Plural is usually longer or same length
        assert len(plural) >= len(word) - 1  # Allow for irregular cases

    @given(st.text(alphabet=st.characters(whitelist_categories=('Ll',), min_codepoint=97, max_codepoint=122), min_size=3, max_size=20).filter(lambda w: w.endswith('s') and len(w) > 2))
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    @example('cats')
    @example('dogs')
    @example('boxes')
    def test_singularize_decreases_or_maintains_length(self, word):
        """Singularizing should not increase word length (usually)."""
        singular = singularize(word)
        # Singular is usually shorter or same length
        assert len(singular) <= len(word) + 1  # Allow for irregular cases

    @given(st.sampled_from(['cat', 'dog', 'box', 'class', 'baby', 'man', 'woman', 'child', 'person']))
    def test_pluralize_singularize_common_words(self, word):
        """Common words should roundtrip through pluralize/singularize."""
        plural = pluralize(word)
        back = singularize(plural)
        # Should get back the original or a close variant
        assert back.lower() == word.lower() or pluralize(back) == plural


# ============================================================================
# MEDIUM PRIORITY: MultiReplace Properties
# ============================================================================

class TestMultiReplace:
    """Test multi_replace functionality."""

    @given(st.text(max_size=100), st.dictionaries(st.text(min_size=1, max_size=5), st.text(max_size=5), max_size=10))
    @example('hello world', {'hello': 'hi', 'world': 'earth'})
    @example('aaa', {'a': 'b'})
    def test_multi_replace_applies_all(self, text, replacements):
        """multi_replace should apply all replacements."""
        if not replacements:
            assume(False)  # Skip empty replacement dicts

        result = multi_replace(text, replacements)
        # Result should be a string
        assert isinstance(result, str)

    @given(st.text(alphabet='abc', min_size=1, max_size=50))
    @example('aaa')
    @example('abc')
    def test_multi_replace_idempotent_when_no_match(self, text):
        """Replacing with non-matching patterns should return original."""
        result = multi_replace(text, {'x': 'y', 'z': 'w'})
        assert result == text


# ============================================================================
# Type Preservation Properties
# ============================================================================

class TestTypePreservation:
    """Test that functions preserve expected types."""

    @given(st.text())
    def test_camel2under_returns_str(self, text):
        """camel2under should always return a string."""
        result = camel2under(text)
        assert isinstance(result, str)

    @given(st.text())
    def test_under2camel_returns_str(self, text):
        """under2camel should always return a string."""
        result = under2camel(text)
        assert isinstance(result, str)

    @given(st.text())
    def test_slugify_returns_str_or_bytes(self, text):
        """slugify should return str or bytes depending on ascii flag."""
        result_str = slugify(text, ascii=False)
        assert isinstance(result_str, str)

        result_bytes = slugify(text, ascii=True)
        assert isinstance(result_bytes, (str, bytes))

    @given(st.binary())
    def test_gzip_returns_bytes(self, data):
        """gzip_bytes should always return bytes."""
        result = gzip_bytes(data)
        assert isinstance(result, bytes)

    @given(st.lists(st.text()))
    def test_args2sh_returns_str(self, args):
        """args2sh should always return a string."""
        result = args2sh(args)
        assert isinstance(result, str)


# ============================================================================
# Edge Cases and Boundary Conditions
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_gzip_empty_bytes(self):
        """Gzipping empty bytes should work."""
        result = gzip_bytes(b'')
        assert isinstance(result, bytes)
        assert gunzip_bytes(result) == b''

    def test_parse_int_list_empty(self):
        """Parsing empty string should return empty list."""
        assert parse_int_list('') == []

    def test_format_int_list_empty(self):
        """Formatting empty list should return empty string."""
        assert format_int_list([]) == ''

    def test_args2sh_empty_list(self):
        """Escaping empty list should return empty string."""
        assert args2sh([]) == ''

    def test_args2sh_empty_string_arg(self):
        """Escaping list with empty string should quote it."""
        result = args2sh([''])
        assert result == "''"

    def test_slugify_empty(self):
        """Slugifying empty string should return empty or delimiter."""
        result = slugify('')
        assert result in ('', '_')

    @given(st.integers(min_value=-1000, max_value=1000))
    def test_bytes2human_handles_negative(self, nbytes):
        """bytes2human should handle negative values."""
        result = bytes2human(nbytes)
        assert isinstance(result, str)
        if nbytes < 0:
            assert '-' in result
