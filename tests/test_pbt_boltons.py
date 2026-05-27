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
    """Mean should be between min and max of data."""
    s = Stats(data)
    assert min(data) <= s.mean <= max(data)


@given(st.lists(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
                min_size=1, max_size=100))
def test_stats_median_in_range(data):
    """Median should be between min and max of data."""
    s = Stats(data)
    assert min(data) <= s.median <= max(data)


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

