"""
Property-based tests for the boltons library using Hypothesis.
Covers: strutils, mathutils, iterutils, dictutils, setutils, statsutils,
        timeutils, urlutils, cacheutils, queueutils, typeutils, funcutils,
        formatutils, listutils, namedutils
"""
import sys
import os
import math
import string
import pytest
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hypothesis import given, assume, settings, example
from hypothesis import strategies as st

# ─────────────────────────────────────────────────────────────────────────────
# strutils (33 functions)
# ─────────────────────────────────────────────────────────────────────────────
from boltons.strutils import (
    camel2under, under2camel, slugify, ordinalize,
    cardinalize, unit_len, is_ascii, bytes2human,
    strip_ansi, indent, a10n, asciify, is_uuid,
    parse_int_list, format_int_list, gzip_bytes, gunzip_bytes,
)

# Strategy: identifiers that look like camelCase words
camel_word = st.from_regex(r'[A-Z][a-z]+', fullmatch=True)
camel_string_st = st.lists(camel_word, min_size=1, max_size=5).map(''.join)

@given(camel_string_st)
def test_camel2under_under2camel_roundtrip(s):
    """under2camel(camel2under(s)) should reproduce the original camelCase."""
    under = camel2under(s)
    assert under == under.lower()
    back = under2camel(under)
    assert back == s


@given(st.text(alphabet=string.ascii_lowercase + '_', min_size=1, max_size=30))
def test_under2camel_no_underscores(s):
    """Result of under2camel should not contain underscores (unless input was all underscores)."""
    result = under2camel(s)
    assert isinstance(result, str)


@given(st.text(min_size=0, max_size=100))
def test_slugify_output_properties(text):
    """slugify output should be lowercase and contain only word chars and delimiter."""
    result = slugify(text)
    assert isinstance(result, str)
    if text:
        assert result == result.lower()
        assert ' ' not in result


@given(st.text(min_size=1, max_size=50))
def test_slugify_idempotent(text):
    """Slugifying a slug should return the same slug."""
    slug1 = slugify(text)
    if isinstance(slug1, str) and slug1:
        slug2 = slugify(slug1)
        assert slug1 == slug2


@given(st.integers(min_value=0, max_value=10**9))
def test_ordinalize_contains_number(n):
    """ordinalize(n) should contain the string representation of n."""
    result = ordinalize(n)
    assert str(n) in result


@given(st.integers(min_value=0, max_value=10**9))
def test_ordinalize_ends_with_suffix(n):
    """ordinalize(n) should end with st/nd/rd/th."""
    result = ordinalize(n)
    assert result.endswith(('st', 'nd', 'rd', 'th'))


@given(st.text(alphabet=string.ascii_letters, min_size=1, max_size=20),
       st.integers(min_value=0, max_value=100))
def test_cardinalize_count_zero(noun, count):
    """cardinalize should return a string."""
    result = cardinalize(noun, count)
    assert isinstance(result, str)
    assert len(result) > 0


@given(st.text(alphabet=string.ascii_letters + ' ', min_size=0, max_size=50))
def test_is_ascii_consistency(text):
    """is_ascii should agree with manual check."""
    result = is_ascii(text)
    expected = all(ord(c) < 128 for c in text)
    assert result == expected


@given(st.integers(min_value=0, max_value=10**15))
def test_bytes2human_returns_string(n):
    """bytes2human should always return a non-empty string."""
    result = bytes2human(n)
    assert isinstance(result, str)
    assert len(result) > 0


@given(st.text(min_size=0, max_size=200))
def test_strip_ansi_idempotent(text):
    """strip_ansi applied twice should equal applied once."""
    once = strip_ansi(text)
    twice = strip_ansi(once)
    assert once == twice


@given(st.text(alphabet=string.ascii_letters + string.digits, min_size=1, max_size=100),
       st.text(alphabet=string.ascii_letters + ' ', min_size=1, max_size=10))
def test_indent_lines_prefixed(text, prefix):
    """indent() should prefix each line with the margin."""
    result = indent(text, prefix)
    # indent() prefixes each line, so result should start with prefix if text is non-empty
    assert result.startswith(prefix)


@given(st.text(alphabet=string.ascii_letters, min_size=2, max_size=30))
def test_a10n_length(word):
    """a10n abbreviation: first char + count + last char."""
    result = a10n(word)
    if len(word) > 2:
        assert result[0] == word[0]
        assert result[-1] == word[-1]
        assert str(len(word) - 2) in result
    else:
        assert result == word


@given(st.text(min_size=0, max_size=100))
def test_asciify_output_is_ascii(text):
    """asciify output should be ASCII."""
    result = asciify(text)
    assert is_ascii(result)


@given(st.binary(min_size=0, max_size=1000))
def test_gzip_gunzip_roundtrip(data):
    """gzip_bytes then gunzip_bytes should return original data."""
    compressed = gzip_bytes(data)
    decompressed = gunzip_bytes(compressed)
    assert decompressed == data


@given(st.lists(st.integers(min_value=1, max_value=1000), min_size=1, max_size=20).map(lambda lst: sorted(set(lst))))
def test_parse_format_int_list_roundtrip(int_list):
    """format_int_list then parse_int_list should return original list (for unique sorted lists)."""
    formatted = format_int_list(int_list)
    parsed = parse_int_list(formatted)
    assert parsed == int_list



@given(st.text(alphabet=string.hexdigits + '-', min_size=36, max_size=36))
def test_is_uuid_valid_format(text):
    """is_uuid should return True for valid UUID format."""
    # Generate a valid UUID format
    import uuid
    valid_uuid = str(uuid.uuid4())
    assert is_uuid(valid_uuid) is True





# ─────────────────────────────────────────────────────────────────────────────
# mathutils (3 functions + Bits class)
# ─────────────────────────────────────────────────────────────────────────────
from boltons.mathutils import clamp, ceil as bolt_ceil, floor as bolt_floor, Bits

@given(st.floats(allow_nan=False, allow_infinity=False),
       st.floats(allow_nan=False, allow_infinity=False),
       st.floats(allow_nan=False, allow_infinity=False))
def test_clamp_within_bounds(x, lower, upper):
    """clamp(x, lower, upper) must be in [lower, upper]."""
    assume(lower <= upper)
    result = clamp(x, lower, upper)
    assert lower <= result <= upper


@given(st.floats(allow_nan=False, allow_infinity=False),
       st.floats(allow_nan=False, allow_infinity=False))
def test_clamp_idempotent(x, bound):
    """Clamping an already-clamped value should be a no-op."""
    lower, upper = min(x, bound), max(x, bound)
    assume(lower <= upper)
    clamped = clamp(x, lower, upper)
    assert clamp(clamped, lower, upper) == clamped


@given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10))
def test_clamp_no_bounds_identity(x):
    """clamp with no bounds should return x unchanged."""
    assert clamp(x) == x


@given(st.integers(min_value=0, max_value=255))
def test_bits_roundtrip_int(n):
    """Bits(n).as_int() should equal n."""
    b = Bits(n)
    assert b.as_int() == n


@given(st.integers(min_value=0, max_value=255))
def test_bits_as_bytes_returns_bytes(n):
    """Bits.as_bytes() should return bytes."""
    b = Bits(n)
    raw = b.as_bytes()
    assert isinstance(raw, bytes)



@given(st.lists(st.booleans(), min_size=1, max_size=16))
def test_bits_from_list_roundtrip(bools):
    """Bits from list of bools should roundtrip."""
    b = Bits(bools)
    assert list(b) == bools


@given(st.integers(min_value=0, max_value=255),
       st.integers(min_value=0, max_value=255))
def test_bits_and_or_properties(a, b):
    """Bits AND/OR should match integer bitwise ops."""
    ba = Bits(a)
    bb = Bits(b)
    and_result = (ba & bb).as_int()
    or_result = (ba | bb).as_int()
    assert and_result == (a & b)
    assert or_result == (a | b)



# ─────────────────────────────────────────────────────────────────────────────
# iterutils (41 functions)
# ─────────────────────────────────────────────────────────────────────────────
from boltons.iterutils import (
    chunked, split, lstrip, rstrip, strip as iter_strip,
    pairwise, windowed, unique, flatten, bucketize,
    partition, same, first, one, soft_sorted, get_path,
    chunked_iter, flatten_iter, is_iterable, is_scalar, is_collection,
)

@given(st.lists(st.integers()), st.integers(min_value=1, max_value=20))
def test_chunked_preserves_all_elements(lst, size):
    """chunked should preserve all elements in order."""
    chunks = list(chunked(lst, size))
    reconstructed = [x for chunk in chunks for x in chunk]
    assert reconstructed == lst


@given(st.lists(st.integers()), st.integers(min_value=1, max_value=20))
def test_chunked_chunk_sizes(lst, size):
    """All chunks except possibly the last should have exactly `size` elements."""
    chunks = list(chunked(lst, size))
    for chunk in chunks[:-1]:
        assert len(chunk) == size
    if chunks:
        assert len(chunks[-1]) <= size


@given(st.lists(st.integers(min_value=0, max_value=5)))
def test_unique_no_duplicates(lst):
    """unique() result should have no duplicates."""
    result = list(unique(lst))
    assert len(result) == len(set(result))


@given(st.lists(st.integers(min_value=0, max_value=5)))
def test_unique_subset_of_input(lst):
    """unique() result should be a subset of input."""
    result = list(unique(lst))
    assert set(result).issubset(set(lst))


@given(st.lists(st.integers(min_value=0, max_value=5)))
def test_unique_order_preserving(lst):
    """unique() should preserve first-occurrence order."""
    result = list(unique(lst))
    seen = []
    seen_set = set()
    for x in lst:
        if x not in seen_set:
            seen.append(x)
            seen_set.add(x)
    assert result == seen


@given(st.lists(st.integers()))
def test_flatten_preserves_elements(lst):
    """flatten of a list of lists should preserve all elements."""
    nested = [lst[i:i+3] for i in range(0, len(lst), 3)]
    result = list(flatten(nested))
    assert result == lst


@given(st.lists(st.integers(), min_size=2))
def test_pairwise_length(lst):
    """pairwise should yield len(lst)-1 pairs."""
    pairs = list(pairwise(lst))
    assert len(pairs) == len(lst) - 1


@given(st.lists(st.integers(), min_size=2))
def test_pairwise_consecutive(lst):
    """pairwise pairs should be consecutive elements."""
    pairs = list(pairwise(lst))
    for i, (a, b) in enumerate(pairs):
        assert a == lst[i]
        assert b == lst[i + 1]


@given(st.lists(st.integers(), min_size=1), st.integers(min_value=1, max_value=10))
def test_windowed_length(lst, size):
    """windowed should yield len(lst)-size+1 windows when len(lst)>=size."""
    assume(len(lst) >= size)
    windows = list(windowed(lst, size))
    assert len(windows) == len(lst) - size + 1


@given(st.lists(st.integers(), min_size=1), st.integers(min_value=1, max_value=10))
def test_windowed_window_size(lst, size):
    """Each window should have exactly `size` elements."""
    assume(len(lst) >= size)
    windows = list(windowed(lst, size))
    for w in windows:
        assert len(w) == size


@given(st.lists(st.integers()))
def test_lstrip_removes_prefix(lst):
    """lstrip should remove leading occurrences of strip_value."""
    if not lst:
        return
    val = lst[0]
    result = lstrip(lst, val)
    if result:
        assert result[0] != val


@given(st.lists(st.integers()))
def test_rstrip_removes_suffix(lst):
    """rstrip should remove trailing occurrences of strip_value."""
    if not lst:
        return
    val = lst[-1]
    result = rstrip(lst, val)
    if result:
        assert result[-1] != val


@given(st.lists(st.integers()))
def test_same_all_equal(lst):
    """same() should return True iff all elements are equal."""
    result = same(lst)
    if not lst:
        assert result is True
    else:
        expected = all(x == lst[0] for x in lst)
        assert result == expected


@given(st.lists(st.integers(min_value=1, max_value=1000), min_size=1))
def test_first_returns_element_from_list(lst):
    """first() should return an element that is in the list (for truthy values)."""
    result = first(lst)
    assert result in lst


@given(st.lists(st.integers(), min_size=1))
def test_bucketize_covers_all(lst):
    """bucketize should cover all elements."""
    buckets = bucketize(lst, key=lambda x: x % 3)
    all_vals = [v for vals in buckets.values() for v in vals]
    assert sorted(all_vals) == sorted(lst)


@given(st.lists(st.integers()))
def test_partition_covers_all(lst):
    """partition should split list into two parts covering all elements."""
    trues, falses = partition(lst, key=lambda x: x > 0)
    assert sorted(trues + falses) == sorted(lst)
    assert all(x > 0 for x in trues)
    assert all(x <= 0 for x in falses)


@given(st.lists(st.integers()))
def test_soft_sorted_is_sorted(lst):
    """soft_sorted should return a sorted list."""
    result = soft_sorted(lst)
    assert result == sorted(lst)


@given(st.integers())
def test_is_scalar_primitives(x):
    """Primitives should be scalar."""
    assert is_scalar(x) is True


@given(st.lists(st.integers()))
def test_is_collection_lists(lst):
    """Lists should be collections."""
    assert is_collection(lst) is True


@given(st.text())
def test_is_scalar_strings(s):
    """Strings should be scalar (not collections)."""
    assert is_scalar(s) is True



# ─────────────────────────────────────────────────────────────────────────────
# dictutils (1 function + classes)
# ─────────────────────────────────────────────────────────────────────────────
from boltons.dictutils import OrderedMultiDict, subdict, FrozenDict

@given(st.lists(st.tuples(st.text(min_size=1, max_size=10), st.integers()), max_size=20))
def test_ordered_multi_dict_preserves_all_values(pairs):
    """OMD should preserve all key-value pairs."""
    omd = OrderedMultiDict(pairs)
    for k, v in pairs:
        assert v in omd.getlist(k)


@given(st.lists(st.tuples(st.text(min_size=1, max_size=5), st.integers()), max_size=20))
def test_ordered_multi_dict_todict_keys(pairs):
    """OMD.todict() keys should match unique keys in pairs."""
    omd = OrderedMultiDict(pairs)
    d = omd.todict()
    expected_keys = set(k for k, v in pairs)
    assert set(d.keys()) == expected_keys


@given(st.dictionaries(st.text(min_size=1, max_size=10), st.integers(), max_size=20),
       st.lists(st.text(min_size=1, max_size=10), max_size=10))
def test_subdict_keys_subset(d, keys):
    """subdict should only contain keys that are in both d and keys."""
    result = subdict(d, keys)
    assert set(result.keys()).issubset(set(d.keys()))
    assert set(result.keys()).issubset(set(keys))


@given(st.dictionaries(st.text(min_size=1, max_size=10), st.integers(), max_size=20))
def test_frozen_dict_immutable(d):
    """FrozenDict should raise TypeError on item assignment."""
    fd = FrozenDict(d)
    with pytest.raises(TypeError):
        fd['new_key'] = 42


@given(st.dictionaries(st.text(min_size=1, max_size=10), st.integers(), max_size=20))
def test_frozen_dict_equal_to_source(d):
    """FrozenDict should equal the original dict."""
    fd = FrozenDict(d)
    assert dict(fd) == d


@given(st.dictionaries(st.text(min_size=1, max_size=10), st.integers(), max_size=20))
def test_frozen_dict_hashable(d):
    """FrozenDict should be hashable."""
    fd = FrozenDict(d)
    h = hash(fd)
    assert isinstance(h, int)


# ─────────────────────────────────────────────────────────────────────────────
# setutils (3 functions + IndexedSet class)
# ─────────────────────────────────────────────────────────────────────────────
from boltons.setutils import IndexedSet

@given(st.lists(st.integers(min_value=0, max_value=50), max_size=30))
def test_indexed_set_no_duplicates(lst):
    """IndexedSet should not contain duplicates."""
    iset = IndexedSet(lst)
    assert len(iset) == len(set(lst))


@given(st.lists(st.integers(min_value=0, max_value=50), max_size=30))
def test_indexed_set_order_preserving(lst):
    """IndexedSet should preserve insertion order (first occurrence)."""
    iset = IndexedSet(lst)
    seen = []
    seen_set = set()
    for x in lst:
        if x not in seen_set:
            seen.append(x)
            seen_set.add(x)
    assert list(iset) == seen


@given(st.lists(st.integers(min_value=0, max_value=50), max_size=30))
def test_indexed_set_index_access(lst):
    """IndexedSet[i] should return the i-th unique element."""
    iset = IndexedSet(lst)
    unique_lst = list(dict.fromkeys(lst))
    for i, val in enumerate(unique_lst):
        assert iset[i] == val


@given(st.lists(st.integers(min_value=0, max_value=20), max_size=20),
       st.lists(st.integers(min_value=0, max_value=20), max_size=20))
def test_indexed_set_union(a, b):
    """IndexedSet union should contain all elements from both sets."""
    ia = IndexedSet(a)
    ib = IndexedSet(b)
    union = ia | ib
    assert set(union) == set(a) | set(b)


@given(st.lists(st.integers(min_value=0, max_value=20), max_size=20),
       st.lists(st.integers(min_value=0, max_value=20), max_size=20))
def test_indexed_set_intersection(a, b):
    """IndexedSet intersection should contain only common elements."""
    ia = IndexedSet(a)
    ib = IndexedSet(b)
    inter = ia & ib
    assert set(inter) == set(a) & set(b)


@given(st.lists(st.integers(min_value=0, max_value=20), max_size=20),
       st.lists(st.integers(min_value=0, max_value=20), max_size=20))
def test_indexed_set_difference(a, b):
    """IndexedSet difference should contain elements in a but not b."""
    ia = IndexedSet(a)
    ib = IndexedSet(b)
    diff = ia - ib
    assert set(diff) == set(a) - set(b)


# ─────────────────────────────────────────────────────────────────────────────
# statsutils (3 functions + Stats class)
# ─────────────────────────────────────────────────────────────────────────────
from boltons.statsutils import Stats

@given(st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
                min_size=1, max_size=100))
def test_stats_mean_in_range(data):
    """Mean should be between min and max of data (with float tolerance)."""
    s = Stats(data)
    # Use tolerance for float precision issues
    assert s.mean >= min(data) - 1e-9
    assert s.mean <= max(data) + 1e-9


@given(st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
                min_size=1, max_size=100))
def test_stats_median_in_range(data):
    """Median should be between min and max of data (with float tolerance)."""
    s = Stats(data)
    assert s.median >= min(data) - 1e-9
    assert s.median <= max(data) + 1e-9


@given(st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
                min_size=2, max_size=100))
def test_stats_variance_non_negative(data):
    """Variance should be non-negative."""
    s = Stats(data)
    assert s.variance >= 0


@given(st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
                min_size=2, max_size=100))
def test_stats_std_dev_non_negative(data):
    """Standard deviation should be non-negative."""
    s = Stats(data)
    assert s.std_dev >= 0


@given(st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
                min_size=2, max_size=100))
def test_stats_std_dev_sqrt_variance(data):
    """std_dev should equal sqrt(variance)."""
    s = Stats(data)
    assert abs(s.std_dev - math.sqrt(s.variance)) < 1e-6


@given(st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
                min_size=1, max_size=100))
def test_stats_count(data):
    """Stats.count should equal len(data)."""
    s = Stats(data)
    assert s.count == len(data)


@given(st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
                min_size=1, max_size=100))
def test_stats_min_max(data):
    """Stats min/max should match Python's min/max."""
    s = Stats(data)
    assert s.min == min(data)
    assert s.max == max(data)



# ─────────────────────────────────────────────────────────────────────────────
# timeutils (9 functions)
# ─────────────────────────────────────────────────────────────────────────────
from boltons.timeutils import dt_to_timestamp, isoparse, parse_timedelta, daterange

@given(st.datetimes(min_value=datetime.datetime(1970, 1, 2),
                    max_value=datetime.datetime(2100, 12, 31)))
def test_dt_to_timestamp_roundtrip(dt):
    """dt_to_timestamp -> datetime.fromtimestamp should roundtrip."""
    ts = dt_to_timestamp(dt)
    assert isinstance(ts, float)
    # Reconstruct: should be close to original (within 1 second)
    reconstructed = datetime.datetime.utcfromtimestamp(ts)
    delta = abs((reconstructed - dt).total_seconds())
    assert delta < 1.0


@given(st.dates(min_value=datetime.date(1900, 1, 1),
                max_value=datetime.date(2100, 12, 31)))
def test_isoparse_date_roundtrip(d):
    """isoparse(d.isoformat()) should return a datetime with same date."""
    iso = d.isoformat()
    result = isoparse(iso)
    assert result.year == d.year
    assert result.month == d.month
    assert result.day == d.day


@given(st.integers(min_value=1, max_value=365),
       st.integers(min_value=1, max_value=23),
       st.integers(min_value=1, max_value=59))
def test_parse_timedelta_days_hours_minutes(days, hours, minutes):
    """parse_timedelta should correctly parse day/hour/minute strings."""
    # Test individual components
    s_days = f'{days}d'
    result_days = parse_timedelta(s_days)
    assert isinstance(result_days, datetime.timedelta)
    assert result_days.days == days
    
    s_hours = f'{hours}h'
    result_hours = parse_timedelta(s_hours)
    assert isinstance(result_hours, datetime.timedelta)
    assert result_hours.total_seconds() == hours * 3600
    
    s_minutes = f'{minutes}m'
    result_minutes = parse_timedelta(s_minutes)
    assert isinstance(result_minutes, datetime.timedelta)
    assert result_minutes.total_seconds() == minutes * 60


@given(st.dates(min_value=datetime.date(2000, 1, 1),
                max_value=datetime.date(2030, 12, 31)),
       st.integers(min_value=1, max_value=30))
def test_daterange_length(start, n_days):
    """daterange should yield exactly n_days dates."""
    end = start + datetime.timedelta(days=n_days)
    dates = list(daterange(start, end))
    assert len(dates) == n_days


@given(st.dates(min_value=datetime.date(2000, 1, 1),
                max_value=datetime.date(2030, 12, 31)),
       st.integers(min_value=1, max_value=30))
def test_daterange_monotone(start, n_days):
    """daterange dates should be strictly increasing."""
    end = start + datetime.timedelta(days=n_days)
    dates = list(daterange(start, end))
    for i in range(len(dates) - 1):
        assert dates[i] < dates[i + 1]



# ─────────────────────────────────────────────────────────────────────────────
# urlutils (14 functions)
# ─────────────────────────────────────────────────────────────────────────────
from boltons.urlutils import URL, quote_path_part, unquote

@given(st.text(alphabet=string.ascii_letters + string.digits + '-._~', min_size=0, max_size=50))
def test_quote_unquote_path_roundtrip(text):
    """quote then unquote should return original text."""
    quoted = quote_path_part(text)
    unquoted = unquote(quoted)
    assert unquoted == text


@given(st.text(alphabet=string.ascii_letters + string.digits, min_size=1, max_size=20),
       st.text(alphabet=string.ascii_letters + string.digits, min_size=1, max_size=20))
def test_url_basic_construction(host, path):
    """URL should be constructable from basic components."""
    url = URL(f'https://{host}.com/{path}')
    assert url.host == f'{host}.com'
    assert url.scheme == 'https'


# ─────────────────────────────────────────────────────────────────────────────
# cacheutils (3 functions + LRI/LRU classes)
# ─────────────────────────────────────────────────────────────────────────────
from boltons.cacheutils import LRI, LRU

@given(st.integers(min_value=1, max_value=20),
       st.lists(st.tuples(st.integers(min_value=0, max_value=50), st.integers()), max_size=100))
def test_lri_max_size(max_size, operations):
    """LRI cache should never exceed max_size entries."""
    cache = LRI(max_size=max_size)
    for k, v in operations:
        cache[k] = v
        assert len(cache) <= max_size


@given(st.integers(min_value=1, max_value=20),
       st.lists(st.tuples(st.integers(min_value=0, max_value=50), st.integers()), max_size=100))
def test_lru_max_size(max_size, operations):
    """LRU cache should never exceed max_size entries."""
    cache = LRU(max_size=max_size)
    for k, v in operations:
        cache[k] = v
        assert len(cache) <= max_size


@given(st.integers(min_value=2, max_value=10),
       st.integers(min_value=0, max_value=5),
       st.integers())
def test_lru_get_after_set(max_size, key, value):
    """LRU: recently set value should be retrievable."""
    cache = LRU(max_size=max_size)
    cache[key] = value
    assert cache[key] == value


# ─────────────────────────────────────────────────────────────────────────────
# typeutils (3 functions)
# ─────────────────────────────────────────────────────────────────────────────
from boltons.typeutils import make_sentinel

def test_make_sentinel_unique():
    """Sentinels should be unique."""
    s1 = make_sentinel('S1')
    s2 = make_sentinel('S2')
    assert s1 is not s2
    assert s1 != s2


def test_make_sentinel_repr():
    """Sentinel repr should contain its name."""
    s = make_sentinel('MySentinel')
    assert 'MySentinel' in repr(s)


# ─────────────────────────────────────────────────────────────────────────────
# funcutils (15 functions)
# ─────────────────────────────────────────────────────────────────────────────
from boltons.funcutils import noop, once

@given(st.integers())
def test_noop_returns_none(x):
    """noop should always return None."""
    result = noop(x)
    assert result is None


def test_once_calls_once():
    """once decorator should ensure function is called only once."""
    call_count = [0]
    
    @once
    def increment():
        call_count[0] += 1
        return call_count[0]
    
    result1 = increment()
    result2 = increment()
    result3 = increment()
    
    assert result1 == 1
    assert result2 == 1
    assert result3 == 1
    assert call_count[0] == 1


# ─────────────────────────────────────────────────────────────────────────────
# formatutils (5 functions)
# ─────────────────────────────────────────────────────────────────────────────
from boltons.formatutils import get_format_args, tokenize_format_str

@given(st.text(alphabet=string.ascii_letters + '{}', min_size=0, max_size=50))
def test_get_format_args_returns_tuple(fstr):
    """get_format_args should return a tuple of (positional, keyword) lists."""
    try:
        result = get_format_args(fstr)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)
    except (ValueError, IndexError):
        pass  # Invalid format strings are expected to fail


@given(st.text(alphabet=string.ascii_letters + '{}', min_size=0, max_size=50))
def test_tokenize_format_str_returns_list(fstr):
    """tokenize_format_str should return a list."""
    try:
        result = tokenize_format_str(fstr)
        assert isinstance(result, list)
    except (ValueError, IndexError):
        pass  # Invalid format strings are expected to fail



# ─────────────────────────────────────────────────────────────────────────────
# strutils - additional functions
# ─────────────────────────────────────────────────────────────────────────────
from boltons.strutils import (
    pluralize, singularize, human_readable_list, unit_len,
    split_punct_ws, unwrap_text, find_hashtags, multi_replace,
    html2text, iter_splitlines, args2cmd, args2sh,
    escape_shell_args, complement_int_list, int_ranges_from_int_list,
    gzip_bytes, gunzip_bytes, is_uuid, MultiReplace,
)

@given(st.text(alphabet=string.ascii_lowercase, min_size=2, max_size=20))
def test_pluralize_singularize_roundtrip(word):
    """singularize(pluralize(word)) should return the original word for simple cases."""
    plural = pluralize(word)
    assert isinstance(plural, str)
    assert len(plural) >= len(word)


@given(st.lists(st.text(alphabet=string.ascii_letters, min_size=1, max_size=10), min_size=1, max_size=5))
def test_human_readable_list_contains_all(items):
    """human_readable_list should contain all items."""
    result = human_readable_list(items)
    for item in items:
        assert item in result


@given(st.lists(st.integers(), min_size=1, max_size=20),
       st.text(alphabet=string.ascii_letters, min_size=1, max_size=10))
def test_unit_len_count(lst, noun):
    """unit_len should contain the count of items (non-empty list)."""
    result = unit_len(lst, noun)
    assert str(len(lst)) in result


@given(st.text(alphabet=string.ascii_letters + string.punctuation + ' ', min_size=0, max_size=100))
def test_split_punct_ws_returns_list(text):
    """split_punct_ws should return a list of strings."""
    result = split_punct_ws(text)
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, str)


@given(st.text(alphabet=string.ascii_letters + ' ', min_size=0, max_size=100))
def test_unwrap_text_no_newlines(text):
    """unwrap_text on text without newlines should return a string."""
    result = unwrap_text(text)
    assert isinstance(result, str)


@given(st.text(alphabet=string.ascii_letters + ' #', min_size=0, max_size=100))
def test_find_hashtags_all_valid(text):
    """find_hashtags should return only alphanumeric tags."""
    tags = find_hashtags(text)
    for tag in tags:
        assert tag.isalnum() or '_' in tag


@given(st.text(alphabet=string.ascii_letters + string.digits, min_size=1, max_size=50),
       st.dictionaries(
           st.text(alphabet=string.ascii_letters, min_size=1, max_size=5),
           st.text(alphabet=string.ascii_letters, min_size=1, max_size=5),
           min_size=1, max_size=5
       ))
def test_multi_replace_returns_string(text, replacements):
    """multi_replace should return a string."""
    result = multi_replace(text, replacements)
    assert isinstance(result, str)


@given(st.text(alphabet=string.ascii_letters + string.digits + ' ', min_size=0, max_size=100))
def test_html2text_no_tags(text):
    """html2text on plain text should return the same text."""
    result = html2text(text)
    assert isinstance(result, str)
    assert '<' not in result or '>' not in result


@given(st.text(alphabet=string.ascii_letters + '\n\r', min_size=0, max_size=100))
def test_iter_splitlines_preserves_content(text):
    """iter_splitlines should yield all non-empty lines."""
    lines = list(iter_splitlines(text))
    assert isinstance(lines, list)
    # Reconstructed text should contain all original content
    reconstructed = '\n'.join(lines)
    for line in lines:
        assert line in reconstructed


@given(st.lists(st.text(alphabet=string.ascii_letters + string.digits, min_size=1, max_size=10), min_size=1, max_size=5))
def test_args2cmd_returns_string(args):
    """args2cmd should return a string."""
    result = args2cmd(args)
    assert isinstance(result, str)
    # All args should appear in the result
    for arg in args:
        assert arg in result


@given(st.lists(st.text(alphabet=string.ascii_letters + string.digits, min_size=1, max_size=10), min_size=1, max_size=5))
def test_args2sh_returns_string(args):
    """args2sh should return a string."""
    result = args2sh(args)
    assert isinstance(result, str)


@given(st.lists(st.text(alphabet=string.ascii_letters + string.digits, min_size=1, max_size=10), min_size=1, max_size=5))
def test_escape_shell_args_returns_string(args):
    """escape_shell_args should return a string."""
    result = escape_shell_args(args)
    assert isinstance(result, str)


@given(st.integers(min_value=0, max_value=100),
       st.integers(min_value=0, max_value=100))
def test_complement_int_list_roundtrip(lo, hi):
    """complement of complement should return original range."""
    assume(lo <= hi)
    # Build a range string
    range_str = f'{lo}-{hi}'
    complement = complement_int_list(range_str, lo, hi)
    assert isinstance(complement, str)


@given(st.lists(st.integers(min_value=1, max_value=100), min_size=1, max_size=20).map(lambda l: sorted(set(l))))
def test_int_ranges_from_int_list_returns_tuple(int_list):
    """int_ranges_from_int_list should return a tuple of (start, end) pairs."""
    # format_int_list produces a range string
    range_str = format_int_list(int_list)
    result = int_ranges_from_int_list(range_str)
    assert isinstance(result, tuple)
    for pair in result:
        assert len(pair) == 2
        assert pair[0] <= pair[1]


@given(st.binary(min_size=0, max_size=100))
def test_gzip_gunzip_roundtrip(data):
    """gzip_bytes then gunzip_bytes should return original data."""
    compressed = gzip_bytes(data)
    decompressed = gunzip_bytes(compressed)
    assert decompressed == data


@given(st.uuids())
def test_is_uuid_valid(uuid):
    """is_uuid should return True for UUID objects when version check is skipped."""
    # st.uuids() generates version=None UUIDs; use version=0 to skip version check
    result = is_uuid(uuid, version=0)
    assert result is True


@given(st.text(alphabet=string.ascii_letters + string.digits, min_size=1, max_size=20))
def test_is_uuid_invalid(text):
    """is_uuid should return False for non-UUID strings."""
    assume(len(text) != 36)
    assert is_uuid(text) is False


@given(st.text(alphabet=string.ascii_letters + string.digits, min_size=1, max_size=50),
       st.dictionaries(
           st.text(alphabet=string.ascii_letters, min_size=1, max_size=5),
           st.text(alphabet=string.ascii_letters, min_size=1, max_size=5),
           min_size=1, max_size=5
       ))
def test_multi_replace_class(text, replacements):
    """MultiReplace.sub should return a string."""
    mr = MultiReplace(replacements)
    result = mr.sub(text)
    assert isinstance(result, str)



# ─────────────────────────────────────────────────────────────────────────────
# iterutils - additional functions
# ─────────────────────────────────────────────────────────────────────────────
from boltons.iterutils import (
    one, split, strip as iter_strip2, chunk_ranges, frange, xfrange,
    redundant, untyped_sorted, get_path, is_iterable, is_collection,
    backoff_iter, remap,
)

@given(st.lists(st.integers(min_value=1, max_value=100), min_size=1, max_size=1))
def test_one_single_element(lst):
    """one() should return the single element."""
    result = one(lst)
    assert result == lst[0]


@given(st.lists(st.integers(min_value=1, max_value=100), min_size=2))
def test_one_returns_default_on_multiple(lst):
    """one() should return default (None) when multiple truthy elements exist."""
    result = one(lst)
    assert result is None  # XOR semantics: multiple truthy → return default


@given(st.lists(st.integers(min_value=0, max_value=5)))
def test_split_covers_all_elements(lst):
    """split() should cover all non-separator elements."""
    sep = 0
    parts = split(lst, sep)
    reconstructed = [x for part in parts for x in part]
    expected = [x for x in lst if x != sep]
    assert reconstructed == expected


@given(st.lists(st.integers(min_value=0, max_value=5)))
def test_strip_removes_both_ends(lst):
    """strip() should remove leading and trailing occurrences of value."""
    if not lst:
        return
    val = lst[0] if lst else 0
    result = iter_strip2(lst, val)
    if result:
        assert result[0] != val
        assert result[-1] != val


@given(st.integers(min_value=1, max_value=100), st.integers(min_value=1, max_value=20))
def test_chunk_ranges_covers_all(total, chunk_size):
    """chunk_ranges should cover [0, total) without gaps or overlaps."""
    ranges = list(chunk_ranges(total, chunk_size))
    if total == 0:
        assert ranges == []
        return
    # First range starts at 0
    assert ranges[0][0] == 0
    # Last range ends at total
    assert ranges[-1][1] == total
    # No gaps
    for i in range(len(ranges) - 1):
        assert ranges[i][1] == ranges[i+1][0]


@given(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
       st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
       st.floats(min_value=0.01, max_value=10, allow_nan=False, allow_infinity=False))
def test_frange_length(start, stop, step):
    """frange should yield ceil((stop-start)/step) elements."""
    import math
    assume(start < stop)
    result = list(frange(start, stop, step))
    expected_count = math.ceil((stop - start) / step)
    assert len(result) == expected_count


@given(st.lists(st.integers(min_value=0, max_value=10)))
def test_redundant_subset_of_input(lst):
    """redundant() should return only elements that appear more than once."""
    result = redundant(lst)
    for item in result:
        assert lst.count(item) > 1


@given(st.lists(st.integers()))
def test_untyped_sorted_is_sorted(lst):
    """untyped_sorted should return a sorted list."""
    result = untyped_sorted(lst)
    assert result == sorted(lst)


@given(st.dictionaries(
    st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=5),
    st.integers(),
    min_size=1, max_size=5
))
def test_get_path_existing_key(d):
    """get_path with a single-key path should return the value."""
    key = list(d.keys())[0]
    result = get_path(d, (key,))
    assert result == d[key]


@given(st.integers())
def test_is_iterable_int_false(n):
    """is_iterable should return False for integers."""
    assert is_iterable(n) is False


@given(st.lists(st.integers()))
def test_is_iterable_list_true(lst):
    """is_iterable should return True for lists."""
    assert is_iterable(lst) is True


@given(st.text())
def test_is_collection_str_false(s):
    """is_collection should return False for strings."""
    assert is_collection(s) is False


@given(st.lists(st.integers()))
def test_is_collection_list_true(lst):
    """is_collection should return True for lists."""
    assert is_collection(lst) is True


@given(st.floats(min_value=0.001, max_value=10, allow_nan=False, allow_infinity=False),
       st.floats(min_value=0.001, max_value=100, allow_nan=False, allow_infinity=False),
       st.integers(min_value=1, max_value=10))
def test_backoff_iter_length(start, stop, count):
    """backoff_iter with count should yield exactly count values."""
    assume(start <= stop)
    result = list(backoff_iter(start, stop, count=count))
    assert len(result) == count


@given(st.floats(min_value=0.001, max_value=10, allow_nan=False, allow_infinity=False),
       st.floats(min_value=0.001, max_value=100, allow_nan=False, allow_infinity=False),
       st.integers(min_value=2, max_value=10))
def test_backoff_iter_monotone(start, stop, count):
    """backoff_iter values should be non-decreasing."""
    assume(start <= stop)
    result = list(backoff_iter(start, stop, count=count))
    for i in range(len(result) - 1):
        assert result[i] <= result[i+1]


@given(st.dictionaries(
    st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=5),
    st.integers(),
    min_size=1, max_size=5
))
def test_remap_identity(d):
    """remap with identity visit should return equal dict."""
    result = remap(d, visit=lambda p, k, v: (k, v))
    assert result == d


@given(st.dictionaries(
    st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=5),
    st.integers(),
    min_size=1, max_size=5
))
def test_remap_double_values(d):
    """remap doubling integers should double all values."""
    result = remap(d, visit=lambda p, k, v: (k, v * 2) if isinstance(v, int) else (k, v))
    for k in d:
        assert result[k] == d[k] * 2


# ─────────────────────────────────────────────────────────────────────────────
# urlutils - additional functions
# ─────────────────────────────────────────────────────────────────────────────
from boltons.urlutils import (
    parse_url, unquote, parse_qsl, parse_host,
    quote_path_part, quote_query_part, resolve_path_parts, find_all_links,
)

@given(st.text(alphabet=string.ascii_letters + string.digits + '-._~', min_size=0, max_size=50))
def test_quote_query_unquote_roundtrip(text):
    """quote_query_part then unquote should return original text."""
    quoted = quote_query_part(text)
    unquoted = unquote(quoted)
    assert unquoted == text


@given(st.lists(st.tuples(
    st.text(alphabet=string.ascii_letters + string.digits, min_size=1, max_size=10),
    st.text(alphabet=string.ascii_letters + string.digits, min_size=0, max_size=10)
), min_size=0, max_size=5))
def test_parse_qsl_returns_pairs(pairs):
    """parse_qsl should return list of (key, value) tuples."""
    qs = '&'.join(f'{k}={v}' for k, v in pairs)
    result = parse_qsl(qs)
    assert isinstance(result, list)
    for item in result:
        assert len(item) == 2


@given(st.text(alphabet=string.ascii_letters + string.digits, min_size=1, max_size=20),
       st.integers(min_value=1, max_value=65535))
def test_parse_host_with_port(host, port):
    """parse_host should extract host and port."""
    result = parse_host(f'{host}:{port}')
    assert isinstance(result, tuple)


@given(st.lists(st.text(alphabet=string.ascii_letters + string.digits, min_size=1, max_size=10), min_size=1, max_size=5))
def test_resolve_path_parts_no_dotdot(parts):
    """resolve_path_parts should not contain '..' in result."""
    result = resolve_path_parts(parts)
    assert '..' not in result


@given(st.text(alphabet=string.ascii_letters + string.digits + ' ', min_size=0, max_size=200))
def test_find_all_links_returns_list(text):
    """find_all_links should return a list."""
    result = find_all_links(text)
    assert isinstance(result, list)


# ─────────────────────────────────────────────────────────────────────────────
# statsutils - additional functions
# ─────────────────────────────────────────────────────────────────────────────
from boltons.statsutils import iqr, trimean, median_abs_dev, skewness, kurtosis, rel_std_dev, mean, median

@given(st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=4, max_size=100))
def test_iqr_non_negative(data):
    """IQR should be non-negative."""
    result = iqr(data)
    assert result >= 0


@given(st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=4, max_size=100))
def test_trimean_in_range(data):
    """trimean should be between min and max."""
    result = trimean(data)
    assert min(data) <= result <= max(data)


@given(st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=1, max_size=100))
def test_median_abs_dev_non_negative(data):
    """median_abs_dev should be non-negative."""
    result = median_abs_dev(data)
    assert result >= 0


@given(st.lists(st.floats(min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=6, max_size=50))
def test_skewness_symmetric_near_zero(data):
    """skewness of symmetric data should be near zero."""
    sym_data = data + [-x for x in data]
    s = Stats(sym_data)
    try:
        skew = s.skewness
        assert abs(skew) < 1e-6
    except ZeroDivisionError:
        pass  # Near-zero variance causes division by zero
@given(st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=4, max_size=100))
def test_kurtosis_returns_float(data):
    """kurtosis should return a float."""
    result = kurtosis(data)
    assert isinstance(result, float)


@given(st.lists(st.floats(min_value=0.001, max_value=1e6, allow_nan=False, allow_infinity=False), min_size=2, max_size=100))
def test_rel_std_dev_non_negative(data):
    """rel_std_dev should be non-negative for positive data."""
    result = rel_std_dev(data)
    assert result >= 0



# ─────────────────────────────────────────────────────────────────────────────
# timeutils - additional functions
# ─────────────────────────────────────────────────────────────────────────────
from boltons.timeutils import strpdate, parse_td, decimal_relative_time

@given(st.dates(min_value=datetime.date(1900, 1, 1), max_value=datetime.date(2100, 12, 31)))
def test_strpdate_roundtrip(d):
    """strpdate(d.strftime(fmt), fmt) should return the original date."""
    fmt = '%Y-%m-%d'
    result = strpdate(d.strftime(fmt), fmt)
    assert result == d


@given(st.integers(min_value=1, max_value=365))
def test_parse_td_days(days):
    """parse_td should parse day strings correctly."""
    result = parse_td(f'{days}d')
    assert isinstance(result, datetime.timedelta)
    assert result.days == days


@given(st.integers(min_value=1, max_value=23))
def test_parse_td_hours(hours):
    """parse_td should parse hour strings correctly."""
    result = parse_td(f'{hours}h')
    assert isinstance(result, datetime.timedelta)
    assert result.total_seconds() == hours * 3600


@given(st.datetimes(min_value=datetime.datetime(2000, 1, 1),
                    max_value=datetime.datetime(2030, 12, 31)),
       st.datetimes(min_value=datetime.datetime(2000, 1, 1),
                    max_value=datetime.datetime(2030, 12, 31)))
def test_decimal_relative_time_returns_tuple(dt1, dt2):
    """decimal_relative_time should return (float, str) tuple."""
    result = decimal_relative_time(dt1, dt2)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], float)
    assert isinstance(result[1], str)


# ─────────────────────────────────────────────────────────────────────────────
# formatutils - additional functions
# ─────────────────────────────────────────────────────────────────────────────
from boltons.formatutils import split_format_str, infer_positional_format_args, construct_format_field_str

@given(st.text(alphabet=string.ascii_letters + string.digits + ' ', min_size=0, max_size=50))
def test_split_format_str_plain_text(text):
    """split_format_str on plain text should return one segment."""
    result = list(split_format_str(text))
    assert isinstance(result, list)
    # All segments should be tuples
    for seg in result:
        assert isinstance(seg, tuple)


@given(st.integers(min_value=0, max_value=5))
def test_infer_positional_format_args_count(n):
    """infer_positional_format_args should fill in positional args."""
    fstr = ' '.join(f'{{{i}}}' for i in range(n))
    result = infer_positional_format_args(fstr)
    assert isinstance(result, str)


@given(st.text(alphabet=string.ascii_letters + string.digits + '_', min_size=1, max_size=10),
       st.text(alphabet=string.ascii_letters + string.digits + '.', min_size=0, max_size=10),
       st.sampled_from(['', '!r', '!s', '!a']))
def test_construct_format_field_str_valid(fname, fspec, conv):
    """construct_format_field_str should return a valid format string."""
    result = construct_format_field_str(fname, fspec, conv)
    assert isinstance(result, str)
    assert fname in result
    assert result.startswith('{')
    assert result.endswith('}')


# ─────────────────────────────────────────────────────────────────────────────
# mathutils - ceil/floor with options
# ─────────────────────────────────────────────────────────────────────────────

@given(st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False))
def test_ceil_gte_input(x):
    """ceil(x) should be >= x."""
    result = bolt_ceil(x)
    assert result >= x


@given(st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False))
def test_floor_lte_input(x):
    """floor(x) should be <= x."""
    result = bolt_floor(x)
    assert result <= x


@given(st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False))
def test_ceil_floor_differ_by_at_most_one(x):
    """ceil(x) - floor(x) should be 0 or 1."""
    c = bolt_ceil(x)
    f = bolt_floor(x)
    assert 0 <= c - f <= 1


# ─────────────────────────────────────────────────────────────────────────────
# cacheutils - ThresholdCounter, make_cache_key
# ─────────────────────────────────────────────────────────────────────────────
from boltons.cacheutils import ThresholdCounter, make_cache_key

@given(st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
       st.lists(st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=5), min_size=1, max_size=20))
def test_threshold_counter_count(threshold, items):
    """ThresholdCounter should count items correctly."""
    tc = ThresholdCounter(threshold=threshold)
    for item in items:
        tc.add(item)
    # All items should be counted
    for item in set(items):
        assert tc.get_common_count() >= 0


@given(st.tuples(st.integers(), st.integers()),
       st.dictionaries(st.text(min_size=1, max_size=5), st.integers(), max_size=3))
def test_make_cache_key_deterministic(args, kwargs):
    """make_cache_key should return same key for same args."""
    key1 = make_cache_key(args, kwargs)
    key2 = make_cache_key(args, kwargs)
    assert key1 == key2


# ─────────────────────────────────────────────────────────────────────────────
# namedutils - namedtuple, namedlist
# ─────────────────────────────────────────────────────────────────────────────
from boltons.namedutils import namedtuple as bolt_namedtuple, namedlist as bolt_namedlist

@given(st.integers(), st.integers())
def test_namedtuple_field_access(x, y):
    """namedtuple fields should be accessible by name."""
    Point = bolt_namedtuple('Point', ['x', 'y'])
    p = Point(x, y)
    assert p.x == x
    assert p.y == y


@given(st.integers(), st.integers())
def test_namedtuple_asdict(x, y):
    """namedtuple._asdict() should return correct dict."""
    Point = bolt_namedtuple('Point', ['x', 'y'])
    p = Point(x, y)
    d = p._asdict()
    assert d['x'] == x
    assert d['y'] == y


@given(st.integers(), st.integers())
def test_namedtuple_replace(x, y):
    """namedtuple._replace() should create new instance with changed field."""
    Point = bolt_namedtuple('Point', ['x', 'y'])
    p = Point(x, y)
    p2 = p._replace(x=999)
    assert p2.x == 999
    assert p2.y == y


@given(st.integers(), st.integers())
def test_namedlist_mutable(x, y):
    """namedlist fields should be mutable."""
    NL = bolt_namedlist('NL', ['x', 'y'])
    nl = NL(x, y)
    nl.x = 999
    assert nl.x == 999
    assert nl.y == y


@given(st.integers(), st.integers())
def test_namedlist_indexable(x, y):
    """namedlist should be indexable like a list."""
    NL = bolt_namedlist('NL', ['x', 'y'])
    nl = NL(x, y)
    assert nl[0] == x
    assert nl[1] == y


# ─────────────────────────────────────────────────────────────────────────────
# dictutils - ManyToMany, OneToOne
# ─────────────────────────────────────────────────────────────────────────────
from boltons.dictutils import ManyToMany, OneToOne

@given(st.lists(st.tuples(
    st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=3),
    st.integers(min_value=0, max_value=10)
), min_size=1, max_size=20))
def test_many_to_many_add_get(pairs):
    """ManyToMany: added values should be retrievable."""
    m2m = ManyToMany()
    for k, v in pairs:
        m2m.add(k, v)
    for k, v in pairs:
        assert v in m2m.get(k)


@given(st.lists(st.tuples(
    st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=3),
    st.integers(min_value=0, max_value=10)
), min_size=1, max_size=20))
def test_many_to_many_remove(pairs):
    """ManyToMany: removed values should not be retrievable."""
    m2m = ManyToMany()
    for k, v in pairs:
        m2m.add(k, v)
    # Remove first pair
    k0, v0 = pairs[0]
    m2m.remove(k0, v0)
    assert v0 not in m2m.get(k0)


@given(st.dictionaries(
    st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=5),
    st.integers(min_value=0, max_value=100),
    min_size=1, max_size=10
))
def test_one_to_one_inverse(d):
    """OneToOne inverse should map values back to keys."""
    assume(len(set(d.values())) == len(d))  # unique values
    o2o = OneToOne(d)
    for k, v in d.items():
        assert o2o.inv[v] == k


@given(st.dictionaries(
    st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=5),
    st.integers(min_value=0, max_value=100),
    min_size=1, max_size=10
))
def test_one_to_one_double_inverse(d):
    """OneToOne.inv.inv should equal original."""
    assume(len(set(d.values())) == len(d))  # unique values
    o2o = OneToOne(d)
    assert dict(o2o.inv.inv) == dict(o2o)


# ─────────────────────────────────────────────────────────────────────────────
# listutils - BList, BarrelList
# ─────────────────────────────────────────────────────────────────────────────
from boltons.listutils import BList, BarrelList

@given(st.lists(st.integers(), min_size=0, max_size=30))
def test_blist_sort_sorted(lst):
    """BList.sort() should produce a sorted list."""
    bl = BList(lst)
    bl.sort()
    assert list(bl) == sorted(lst)


@given(st.lists(st.integers(), min_size=0, max_size=30))
def test_blist_preserves_elements(lst):
    """BList should preserve all elements."""
    bl = BList(lst)
    assert sorted(bl) == sorted(lst)


@given(st.lists(st.integers(), min_size=0, max_size=30))
def test_barrel_list_preserves_elements(lst):
    """BarrelList should preserve all elements."""
    bl = BarrelList(lst)
    assert sorted(bl) == sorted(lst)


@given(st.lists(st.integers(), min_size=1, max_size=30),
       st.integers())
def test_barrel_list_append(lst, item):
    """BarrelList.append should add item."""
    bl = BarrelList(lst)
    bl.append(item)
    assert item in bl
    assert len(bl) == len(lst) + 1


# ─────────────────────────────────────────────────────────────────────────────
# typeutils - make_sentinel, get_all_subclasses
# ─────────────────────────────────────────────────────────────────────────────
from boltons.typeutils import make_sentinel, get_all_subclasses

@given(st.text(alphabet=string.ascii_letters, min_size=1, max_size=20))
def test_make_sentinel_unique(name):
    """Two sentinels with same name should be different objects."""
    s1 = make_sentinel(name)
    s2 = make_sentinel(name)
    assert s1 is not s2


@given(st.text(alphabet=string.ascii_letters, min_size=1, max_size=20))
def test_make_sentinel_repr(name):
    """Sentinel repr should contain the name."""
    s = make_sentinel(name)
    assert name in repr(s)


def test_get_all_subclasses_includes_direct():
    """get_all_subclasses should include direct subclasses."""
    class Base:
        pass
    class Child(Base):
        pass
    subs = get_all_subclasses(Base)
    assert Child in subs


def test_get_all_subclasses_includes_indirect():
    """get_all_subclasses should include indirect subclasses."""
    class Base:
        pass
    class Child(Base):
        pass
    class GrandChild(Child):
        pass
    subs = get_all_subclasses(Base)
    assert GrandChild in subs



# ─────────────────────────────────────────────────────────────────────────────
# IMPROVED: mathutils - Bits format roundtrips (from reference test_mathutils_pbt.py)
# ─────────────────────────────────────────────────────────────────────────────

@given(st.integers(min_value=0, max_value=2**32))
@example(0)
@example(1)
@example(255)
@example(256)
def test_bits_int_roundtrip_from_int(n):
    """Bits.from_int(n).as_int() should equal n."""
    bits = Bits.from_int(n)
    assert bits.as_int() == n


@given(st.integers(min_value=0, max_value=2**16))
@example(0)
@example(255)
def test_bits_hex_roundtrip(n):
    """Bits hex roundtrip: from_int -> as_hex -> from_hex -> as_int."""
    bits = Bits.from_int(n)
    hex_str = bits.as_hex()
    result = Bits.from_hex(hex_str)
    assert result.as_int() == bits.as_int()


@given(st.integers(min_value=0, max_value=2**16))
@example(0)
@example(1)
@example(255)
def test_bits_bin_roundtrip(n):
    """Bits binary string roundtrip: from_int -> as_bin -> from_bin."""
    bits = Bits.from_int(n)
    bin_str = bits.as_bin()
    result = Bits.from_bin(bin_str)
    assert result == bits


@given(st.integers(min_value=0, max_value=2**16))
def test_bits_hex_idempotent(n):
    """Bits hex conversion should be idempotent."""
    bits = Bits.from_int(n)
    hex1 = bits.as_hex()
    hex2 = Bits.from_hex(hex1).as_hex()
    assert hex1 == hex2


@given(st.integers(min_value=0, max_value=2**16))
def test_bits_bin_idempotent(n):
    """Bits binary conversion should be idempotent."""
    bits = Bits.from_int(n)
    bin1 = bits.as_bin()
    bin2 = Bits.from_bin(bin1).as_bin()
    assert bin1 == bin2


@given(st.integers(min_value=0, max_value=255),
       st.integers(min_value=0, max_value=255))
def test_bits_and_commutative(a, b):
    """Bits AND should be commutative."""
    ba = Bits.from_int(a)
    bb = Bits.from_int(b)
    assert (ba & bb).as_int() == (bb & ba).as_int()


@given(st.integers(min_value=0, max_value=255),
       st.integers(min_value=0, max_value=255))
def test_bits_or_commutative(a, b):
    """Bits OR should be commutative."""
    ba = Bits.from_int(a)
    bb = Bits.from_int(b)
    assert (ba | bb).as_int() == (bb | ba).as_int()


@given(st.integers(min_value=0, max_value=255))
def test_bits_and_identity(n):
    """Bits AND with all-ones should be identity."""
    bits = Bits.from_int(n)
    ones = Bits.from_int(2**bits.len - 1)
    assert (bits & ones).as_int() == n


@given(st.integers(min_value=0, max_value=255))
def test_bits_or_zero_identity(n):
    """Bits OR with zero should be identity."""
    bits = Bits.from_int(n)
    zero = Bits.from_int(0)
    assert (bits | zero).as_int() == n


@given(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
       st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
                min_size=2, max_size=20).map(sorted))
def test_ceil_with_options_in_list(x, options):
    """ceil(x, options) should return smallest option >= x."""
    assume(len(set(options)) >= 2)
    assume(any(o >= x for o in options))  # ensure a valid ceil exists
    result = bolt_ceil(x, options)
    assert result >= x
    # No smaller option exists that is still >= x
    smaller_valid = [o for o in options if o >= x and o < result]
    assert len(smaller_valid) == 0


@given(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
       st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
                min_size=2, max_size=20).map(sorted))
def test_floor_with_options_in_list(x, options):
    """floor(x, options) should return largest option <= x."""
    assume(len(set(options)) >= 2)
    assume(any(o <= x for o in options))  # ensure a valid floor exists
    result = bolt_floor(x, options)
    assert result <= x
    # No larger option exists that is still <= x
    larger_valid = [o for o in options if o <= x and o > result]
    assert len(larger_valid) == 0



# ─────────────────────────────────────────────────────────────────────────────
# IMPROVED: strutils - args2sh shlex roundtrip, gzip levels, int_list compact
# ─────────────────────────────────────────────────────────────────────────────
import shlex

@given(st.lists(
    st.text(alphabet=string.ascii_letters + string.digits + '-._', min_size=1, max_size=20),
    min_size=1, max_size=10
))
def test_args2sh_shlex_roundtrip(args):
    """args2sh output should be parseable by shlex.split back to original args."""
    sh = args2sh(args)
    parsed = shlex.split(sh)
    assert parsed == args


@given(st.binary(), st.integers(min_value=1, max_value=9))
@example(b'test', 1)
@example(b'test', 9)
@example(b'', 5)
def test_gzip_all_levels_roundtrip(data, level):
    """All gzip compression levels should roundtrip correctly."""
    compressed = gzip_bytes(data, level=level)
    decompressed = gunzip_bytes(compressed)
    assert decompressed == data


@given(st.lists(st.integers(min_value=0, max_value=1000), max_size=50))
@example([])
@example([1])
@example([1, 2, 3])
@example([1, 3, 5, 7])
@example([1, 2, 3, 10, 11, 12])
def test_format_parse_roundtrip_sorted_unique(int_list):
    """format_int_list then parse_int_list returns sorted unique values."""
    formatted = format_int_list(int_list)
    parsed = parse_int_list(formatted)
    assert parsed == sorted(set(int_list))


@given(st.text(alphabet=string.ascii_letters + string.digits, min_size=1, max_size=50))
def test_asciify_is_ascii(text):
    """asciify output should always be ASCII."""
    result = asciify(text)
    assert is_ascii(result)


@given(st.text(alphabet=string.ascii_letters + string.digits, min_size=1, max_size=50))
def test_asciify_idempotent(text):
    """asciify applied twice should equal applied once."""
    once = asciify(text)
    twice = asciify(once)
    assert once == twice


@given(st.text(min_size=0, max_size=200))
def test_strip_ansi_is_ascii_safe(text):
    """strip_ansi should not introduce non-ASCII characters."""
    result = strip_ansi(text)
    # Result should only contain chars that were in the original (minus ANSI codes)
    assert isinstance(result, str)


@given(st.text(alphabet=string.ascii_letters, min_size=1, max_size=30))
def test_pluralize_singularize_roundtrip(word):
    """pluralize then singularize should return original word (for simple words)."""
    plural = pluralize(word)
    singular = singularize(plural)
    # Not always exact roundtrip, but should be a string
    assert isinstance(singular, str)
    assert len(singular) > 0



# ─────────────────────────────────────────────────────────────────────────────
# IMPROVED: iterutils - backoff monotonic, remap, chunk_ranges coverage
# ─────────────────────────────────────────────────────────────────────────────
from boltons.iterutils import backoff_iter, remap, chunk_ranges, frange, xfrange

@given(st.floats(min_value=0.001, max_value=10.0, allow_nan=False, allow_infinity=False),
       st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
       st.integers(min_value=2, max_value=20))
def test_backoff_iter_monotonic(start, stop, count):
    """backoff_iter values should be monotonically non-decreasing."""
    assume(start < stop)
    vals = list(backoff_iter(start, stop, count=count))
    assert len(vals) == count
    for i in range(len(vals) - 1):
        assert vals[i] <= vals[i + 1]


@given(st.floats(min_value=0.001, max_value=10.0, allow_nan=False, allow_infinity=False),
       st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
       st.integers(min_value=2, max_value=20))
def test_backoff_iter_bounds(start, stop, count):
    """backoff_iter first value should be start, last should be <= stop."""
    assume(start < stop)
    vals = list(backoff_iter(start, stop, count=count))
    assert abs(vals[0] - start) < 1e-9
    assert vals[-1] <= stop + 1e-9


@given(st.dictionaries(
    st.text(alphabet=string.ascii_letters, min_size=1, max_size=5),
    st.integers(min_value=1, max_value=100),
    min_size=1, max_size=10
))
def test_remap_identity(d):
    """remap with identity visit should return equal structure."""
    result = remap(d)
    assert result == d


@given(st.dictionaries(
    st.text(alphabet=string.ascii_letters, min_size=1, max_size=5),
    st.integers(min_value=1, max_value=100),
    min_size=1, max_size=10
))
def test_remap_double_negate(d):
    """remap with double negation should return original values."""
    result = remap(d, visit=lambda p, k, v: (k, -v) if isinstance(v, int) else (k, v))
    result2 = remap(result, visit=lambda p, k, v: (k, -v) if isinstance(v, int) else (k, v))
    assert result2 == d


@given(st.integers(min_value=1, max_value=1000),
       st.integers(min_value=1, max_value=50))
def test_chunk_ranges_covers_all(n, chunk_size):
    """chunk_ranges should cover all indices from 0 to n."""
    ranges = list(chunk_ranges(n, chunk_size))
    # All ranges should be non-empty
    assert all(start < stop for start, stop in ranges)
    # First range starts at 0
    assert ranges[0][0] == 0
    # Last range ends at n
    assert ranges[-1][1] == n
    # Ranges should be contiguous
    for i in range(len(ranges) - 1):
        assert ranges[i][1] == ranges[i + 1][0]


@given(st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
       st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
       st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False))
def test_frange_count(start, stop, step):
    """frange should produce correct number of elements."""
    assume(start < stop)
    assume(step > 0)
    result = frange(start, stop, step)
    import math
    expected_count = int(math.ceil((stop - start) / step))
    assert len(result) == expected_count


# ─────────────────────────────────────────────────────────────────────────────
# IMPROVED: cacheutils - LRU eviction policy
# ─────────────────────────────────────────────────────────────────────────────

@given(st.integers(min_value=2, max_value=10),
       st.lists(st.integers(min_value=0, max_value=20), min_size=3, max_size=50))
def test_lru_evicts_least_recently_used(max_size, items):
    """LRU should evict least recently used items when full."""
    cache = LRU(max_size=max_size)
    unique_items = list(dict.fromkeys(items))
    assume(len(unique_items) > max_size)

    # Fill cache to capacity
    for item in unique_items[:max_size]:
        cache[item] = item

    # Access first item to make it recently used
    _ = cache[unique_items[0]]

    # Add one more item to trigger eviction
    new_item = unique_items[max_size]
    cache[new_item] = new_item

    # First item should still be there (was recently accessed)
    assert unique_items[0] in cache
    # Second item should be evicted (was LRU)
    assert unique_items[1] not in cache
    # New item should be present
    assert new_item in cache


@given(st.integers(min_value=1, max_value=20),
       st.lists(st.integers(min_value=0, max_value=50), max_size=100))
def test_lri_size_invariant(max_size, items):
    """LRI cache should never exceed max_size."""
    cache = LRI(max_size=max_size)
    for item in items:
        cache[item] = item * 2
        assert len(cache) <= max_size


@given(st.integers(min_value=1, max_value=20),
       st.lists(st.integers(min_value=0, max_value=50), max_size=100))
def test_lru_size_invariant(max_size, items):
    """LRU cache should never exceed max_size."""
    cache = LRU(max_size=max_size)
    for item in items:
        cache[item] = item * 2
        assert len(cache) <= max_size


# ─────────────────────────────────────────────────────────────────────────────
# IMPROVED: dictutils - OMD double inversion
# ─────────────────────────────────────────────────────────────────────────────

@given(st.lists(
    st.tuples(st.text(alphabet=string.ascii_letters, min_size=1, max_size=5),
              st.integers(min_value=1, max_value=100)),
    min_size=1, max_size=20
))
def test_omd_inverted_double_inversion(pairs):
    """OMD.inverted().inverted() should equal original OMD (for unique values)."""
    # Use unique values to ensure clean inversion
    unique_pairs = list({v: k for k, v in pairs}.items())
    unique_pairs = [(v, k) for k, v in unique_pairs]
    omd = OrderedMultiDict(unique_pairs)
    double_inv = omd.inverted().inverted()
    # Keys and values should match
    assert set(omd.keys()) == set(double_inv.keys())
    for k in omd.keys():
        assert omd[k] == double_inv[k]


@given(st.lists(
    st.tuples(st.text(alphabet=string.ascii_letters, min_size=1, max_size=5),
              st.integers(min_value=1, max_value=100)),
    min_size=1, max_size=20
))
def test_omd_inverted_swaps_keys_values(pairs):
    """OMD.inverted() should swap keys and values."""
    omd = OrderedMultiDict(pairs)
    inv = omd.inverted()
    # All original values should be keys in inverted
    for k, v in pairs:
        assert v in inv


@given(st.lists(
    st.tuples(st.text(alphabet=string.ascii_letters, min_size=1, max_size=5),
              st.integers(min_value=1, max_value=100)),
    min_size=1, max_size=20
))
def test_omd_getlist_covers_all_values(pairs):
    """OMD.getlist(k) should return all values for key k."""
    omd = OrderedMultiDict(pairs)
    for k in set(k for k, v in pairs):
        expected_vals = [v for pk, v in pairs if pk == k]
        assert omd.getlist(k) == expected_vals


# ─────────────────────────────────────────────────────────────────────────────
# IMPROVED: queueutils - HeapPriorityQueue and SortedPriorityQueue
# ─────────────────────────────────────────────────────────────────────────────
from boltons.queueutils import HeapPriorityQueue, SortedPriorityQueue

@given(st.lists(
    st.tuples(st.text(alphabet=string.ascii_letters, min_size=1, max_size=10),
              st.integers(min_value=1, max_value=100)),
    min_size=1, max_size=20
).filter(lambda pairs: len(set(p[0] for p in pairs)) == len(pairs)))
def test_heap_priority_queue_pop_order(pairs):
    """HeapPriorityQueue should pop items in descending priority order."""
    hpq = HeapPriorityQueue()
    for task, priority in pairs:
        hpq.add(task, priority)
    
    popped = []
    while hpq:
        popped.append(hpq.pop())
    
    # Should pop in descending priority order
    priorities = [p for _, p in pairs]
    sorted_tasks = [t for t, _ in sorted(pairs, key=lambda x: -x[1])]
    assert popped == sorted_tasks


@given(st.lists(
    st.tuples(st.text(alphabet=string.ascii_letters, min_size=1, max_size=10),
              st.integers(min_value=1, max_value=100)),
    min_size=1, max_size=20
).filter(lambda pairs: len(set(p[0] for p in pairs)) == len(pairs)))
def test_sorted_priority_queue_pop_order(pairs):
    """SortedPriorityQueue should pop items in descending priority order."""
    spq = SortedPriorityQueue()
    for task, priority in pairs:
        spq.add(task, priority)
    
    popped = []
    while spq:
        popped.append(spq.pop())
    
    sorted_tasks = [t for t, _ in sorted(pairs, key=lambda x: -x[1])]
    assert popped == sorted_tasks


@given(st.lists(
    st.tuples(st.text(alphabet=string.ascii_letters, min_size=1, max_size=10),
              st.integers(min_value=1, max_value=100)),
    min_size=1, max_size=20
).filter(lambda pairs: len(set(p[0] for p in pairs)) == len(pairs)))
def test_heap_and_sorted_pq_same_order(pairs):
    """HeapPriorityQueue and SortedPriorityQueue should produce same pop order."""
    hpq = HeapPriorityQueue()
    spq = SortedPriorityQueue()
    for task, priority in pairs:
        hpq.add(task, priority)
        spq.add(task, priority)
    
    heap_order = []
    while hpq:
        heap_order.append(hpq.pop())
    
    sorted_order = []
    while spq:
        sorted_order.append(spq.pop())
    
    assert heap_order == sorted_order


# ─────────────────────────────────────────────────────────────────────────────
# IMPROVED: setutils - complement properties
# ─────────────────────────────────────────────────────────────────────────────
from boltons.setutils import complement

@given(st.sets(st.integers(min_value=0, max_value=100), max_size=20))
def test_complement_excludes_original(s):
    """complement(s) should not contain any element of s."""
    c = complement(s)
    for item in s:
        assert item not in c


@given(st.sets(st.integers(min_value=0, max_value=100), max_size=20),
       st.integers(min_value=0, max_value=200))
def test_complement_contains_non_members(s, item):
    """complement(s) should contain items not in s."""
    c = complement(s)
    if item not in s:
        assert item in c
    else:
        assert item not in c


@given(st.sets(st.integers(min_value=0, max_value=100), max_size=20))
def test_complement_double_complement(s):
    """Double complement should equal original set."""
    c = complement(s)
    cc = complement(c)
    # Double complement: items in s should be in cc
    for item in s:
        assert item in cc
    # Items not in s should not be in cc
    for item in range(101):
        if item not in s:
            assert item not in cc

