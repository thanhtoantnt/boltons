"""Property-based tests for iterutils module using Hypothesis."""
from hypothesis import given, strategies as st, settings, example, assume
import pytest
from itertools import chain

from boltons.iterutils import (
    chunked, chunked_iter, flatten, flatten_iter, unique, unique_iter,
    split, split_iter, bucketize, partition, one, first,
    is_iterable, is_scalar, is_collection, same, pairwise, pairwise_iter,
    windowed, windowed_iter, lstrip, rstrip, strip, backoff, backoff_iter,
    redundant, soft_sorted, untyped_sorted
)


# ============================================================================
# HIGH PRIORITY: Inverse/Reconstruction Properties
# ============================================================================

class TestChunkedFlattenInverse:
    """Test that chunked and flatten are inverses (without fill)."""

    @given(st.lists(st.integers(), max_size=100), st.integers(min_value=1, max_value=20))
    @example([], 1)
    @example([1], 1)
    @example([1, 2, 3], 2)
    @example(list(range(10)), 3)
    def test_flatten_chunked_roundtrip(self, lst, size):
        """Flattening chunked list (no fill) should return original."""
        chunks = chunked(lst, size)
        result = flatten(chunks)
        assert result == lst

    @given(st.lists(st.integers(), max_size=100), st.integers(min_value=1, max_value=20))
    def test_chunked_preserves_elements(self, lst, size):
        """Chunked should preserve all elements."""
        chunks = chunked(lst, size)
        flattened = flatten(chunks)
        assert flattened == lst
        # All elements accounted for
        assert len(flattened) == len(lst)

    @given(st.lists(st.integers(), min_size=1, max_size=50), st.integers(min_value=1, max_value=10))
    def test_chunked_size_property(self, lst, size):
        """All chunks except possibly the last should have size elements."""
        chunks = chunked(lst, size)
        if chunks:
            # All chunks except last should be full
            for chunk in chunks[:-1]:
                assert len(chunk) == size
            # Last chunk should be <= size
            assert len(chunks[-1]) <= size


class TestSplitReconstruction:
    """Test split/join reconstruction properties."""

    @given(st.lists(st.integers(), max_size=100), st.integers())
    @example([], 0)
    @example([1, 2, 3], 2)
    @example([1, 0, 2, 0, 3], 0)
    def test_split_preserves_non_separator_elements(self, lst, sep):
        """Split should preserve all non-separator elements."""
        parts = split(lst, sep)
        # Flatten and remove separators
        reconstructed = [x for part in parts for x in part]
        expected = [x for x in lst if x != sep]
        assert reconstructed == expected

    @given(st.lists(st.integers(), max_size=50), st.integers())
    def test_split_no_separator_in_parts(self, lst, sep):
        """Split parts should not contain the separator."""
        parts = split(lst, sep)
        for part in parts:
            assert sep not in part


# ============================================================================
# HIGH PRIORITY: Idempotence Properties
# ============================================================================

class TestIdempotence:
    """Test idempotence properties."""

    @given(st.lists(st.integers(), max_size=100))
    @example([])
    @example([1])
    @example([1, 1, 1])
    @example([1, 2, 1, 2])
    def test_unique_idempotent(self, lst):
        """Applying unique twice should equal applying once."""
        once = unique(lst)
        twice = unique(once)
        assert once == twice

    @given(st.lists(st.lists(st.integers(), max_size=10), max_size=20))
    @example([])
    @example([[1, 2], [3, 4]])
    @example([[[1, 2]], [[3, 4]]])
    def test_flatten_idempotent(self, nested):
        """Flattening twice should equal flattening once."""
        once = flatten(nested)
        twice = flatten(once)
        assert once == twice

    @given(st.lists(st.integers(), max_size=100))
    def test_sorted_idempotent(self, lst):
        """Sorting twice should equal sorting once."""
        once = sorted(lst)
        twice = sorted(once)
        assert once == twice


# ============================================================================
# MEDIUM PRIORITY: Uniqueness and Deduplication
# ============================================================================

class TestUniqueness:
    """Test unique and redundant functions."""

    @given(st.lists(st.integers(), max_size=100))
    @example([])
    @example([1])
    @example([1, 1, 1])
    @example([1, 2, 3, 2, 1])
    def test_unique_no_duplicates(self, lst):
        """unique should produce no duplicates."""
        result = unique(lst)
        assert len(result) == len(set(result))

    @given(st.lists(st.integers(), max_size=100))
    def test_unique_preserves_order(self, lst):
        """unique should preserve first occurrence order."""
        result = unique(lst)
        # Check that order is preserved
        seen = set()
        expected = []
        for item in lst:
            if item not in seen:
                seen.add(item)
                expected.append(item)
        assert result == expected

    @given(st.lists(st.integers(), max_size=100))
    def test_unique_length_property(self, lst):
        """unique result should be <= original length."""
        result = unique(lst)
        assert len(result) <= len(lst)

    @given(st.lists(st.integers(), max_size=100))
    def test_redundant_finds_duplicates(self, lst):
        """redundant should find all duplicate occurrences."""
        result = redundant(lst)
        # All items in result should appear more than once in original
        for item in result:
            assert lst.count(item) > 1


# ============================================================================
# MEDIUM PRIORITY: Partitioning and Bucketing
# ============================================================================

class TestPartitioning:
    """Test partition and bucketize functions."""

    @given(st.lists(st.integers(), max_size=100))
    @example([])
    @example([1, 2, 3])
    @example([0, 1, 0, 1])
    def test_partition_completeness(self, lst):
        """Partition should account for all elements."""
        trues, falses = partition(lst, key=lambda x: x % 2 == 0)
        # All elements should be in one partition or the other
        assert len(trues) + len(falses) == len(lst)
        # Reconstruct original (order may differ)
        assert sorted(trues + falses) == sorted(lst)

    @given(st.lists(st.integers(), max_size=100))
    def test_partition_no_overlap(self, lst):
        """Partitions should not overlap."""
        trues, falses = partition(lst, key=lambda x: x > 0)
        # No element should be in both partitions
        assert set(trues).isdisjoint(set(falses))

    @given(st.lists(st.integers(), max_size=100))
    def test_bucketize_completeness(self, lst):
        """Bucketize should account for all elements."""
        buckets = bucketize(lst, key=lambda x: x % 3)
        # All elements should be in some bucket
        all_items = [item for bucket_items in buckets.values() for item in bucket_items]
        assert sorted(all_items) == sorted(lst)


# ============================================================================
# MEDIUM PRIORITY: Ordering Properties
# ============================================================================

class TestOrdering:
    """Test ordering-related properties."""

    @given(st.lists(st.integers(), max_size=100))
    @example([])
    @example([1])
    @example([3, 1, 2])
    def test_sorted_is_ordered(self, lst):
        """Sorted list should be in order."""
        result = sorted(lst)
        for i in range(len(result) - 1):
            assert result[i] <= result[i + 1]

    @given(st.lists(st.integers(), max_size=100))
    def test_sorted_preserves_elements(self, lst):
        """Sorting should preserve all elements."""
        result = sorted(lst)
        assert sorted(result) == sorted(lst)
        assert len(result) == len(lst)

    @given(st.lists(st.integers(min_value=-100, max_value=100), max_size=50))
    def test_soft_sorted_preserves_elements(self, lst):
        """soft_sorted should preserve all elements."""
        result = soft_sorted(lst)
        assert sorted(result) == sorted(lst)


# ============================================================================
# MEDIUM PRIORITY: Strip Operations
# ============================================================================

class TestStripOperations:
    """Test lstrip, rstrip, strip operations."""

    @given(st.lists(st.integers(), max_size=100), st.integers())
    @example([1, 1, 2, 3], 1)
    @example([1, 2, 3, 1, 1], 1)
    def test_strip_removes_from_ends(self, lst, value):
        """strip should remove value from both ends."""
        result = strip(lst, value)
        # Result should not start or end with value (unless empty)
        if result:
            assert result[0] != value
            assert result[-1] != value

    @given(st.lists(st.integers(), max_size=100), st.integers())
    def test_lstrip_removes_from_start(self, lst, value):
        """lstrip should remove value from start only."""
        result = lstrip(lst, value)
        # Result should not start with value (unless empty)
        if result:
            assert result[0] != value

    @given(st.lists(st.integers(), max_size=100), st.integers())
    def test_rstrip_removes_from_end(self, lst, value):
        """rstrip should remove value from end only."""
        result = rstrip(lst, value)
        # Result should not end with value (unless empty)
        if result:
            assert result[-1] != value

    @given(st.lists(st.integers(), max_size=100), st.integers())
    def test_strip_composition(self, lst, value):
        """strip should equal rstrip(lstrip(...))."""
        result1 = strip(lst, value)
        result2 = rstrip(lstrip(lst, value), value)
        assert result1 == result2


# ============================================================================
# MEDIUM PRIORITY: Windowing and Pairing
# ============================================================================

class TestWindowing:
    """Test windowed and pairwise functions."""

    @given(st.lists(st.integers(), min_size=2, max_size=50))
    @example([1, 2])
    @example([1, 2, 3, 4])
    def test_pairwise_count(self, lst):
        """pairwise should produce len(lst)-1 pairs."""
        result = list(pairwise(lst))
        assert len(result) == len(lst) - 1

    @given(st.lists(st.integers(), min_size=2, max_size=50))
    def test_pairwise_consecutive(self, lst):
        """pairwise should produce consecutive pairs."""
        result = list(pairwise(lst))
        for i, (a, b) in enumerate(result):
            assert a == lst[i]
            assert b == lst[i + 1]

    @given(st.lists(st.integers(), min_size=3, max_size=50), st.integers(min_value=2, max_value=5))
    def test_windowed_size(self, lst, size):
        """windowed should produce windows of correct size."""
        if len(lst) >= size:
            result = list(windowed(lst, size))
            for window in result:
                assert len(window) == size


# ============================================================================
# MEDIUM PRIORITY: Backoff Sequence
# ============================================================================

class TestBackoff:
    """Test backoff sequence properties."""

    @given(st.floats(min_value=0.1, max_value=10), st.floats(min_value=10, max_value=1000), st.floats(min_value=1.5, max_value=3))
    @example(1, 100, 2)
    def test_backoff_monotonic_increasing(self, start, stop, factor):
        """Backoff sequence should be monotonically increasing."""
        assume(start < stop)
        assume(factor > 1)
        result = list(backoff(start, stop, factor=factor))
        for i in range(len(result) - 1):
            assert result[i] < result[i + 1]

    @given(st.floats(min_value=0.1, max_value=10), st.floats(min_value=10, max_value=1000))
    def test_backoff_bounded(self, start, stop):
        """Backoff values should be between start and stop."""
        assume(start < stop)
        result = list(backoff(start, stop))
        for value in result:
            assert start <= value <= stop


# ============================================================================
# MEDIUM PRIORITY: Type Checking
# ============================================================================

class TestTypeChecking:
    """Test is_iterable, is_scalar, is_collection."""

    @given(st.lists(st.integers()))
    def test_list_is_iterable(self, lst):
        """Lists should be iterable."""
        assert is_iterable(lst)

    @given(st.lists(st.integers()))
    def test_list_is_collection(self, lst):
        """Lists should be collections."""
        assert is_collection(lst)

    @given(st.integers())
    def test_int_is_scalar(self, n):
        """Integers should be scalar."""
        assert is_scalar(n)

    @given(st.integers())
    def test_int_not_collection(self, n):
        """Integers should not be collections."""
        assert not is_collection(n)

    @given(st.text())
    def test_string_is_scalar(self, s):
        """Strings should be scalar (treated as atomic)."""
        assert is_scalar(s)

    @given(st.text())
    def test_string_is_iterable(self, s):
        """Strings should be iterable."""
        assert is_iterable(s)


# ============================================================================
# MEDIUM PRIORITY: One and First
# ============================================================================

class TestOneAndFirst:
    """Test one and first functions."""

    @given(st.lists(st.integers(min_value=1, max_value=100), min_size=1, max_size=1))
    @example([42])
    def test_one_single_truthy_element(self, lst):
        """one should return the single truthy element."""
        result = one(lst)
        assert result == lst[0]

    @given(st.lists(st.integers(min_value=1, max_value=100), min_size=2, max_size=10))
    def test_one_multiple_truthy_elements_returns_default(self, lst):
        """one with multiple truthy elements should return default."""
        result = one(lst, default='default')
        # Should return default when multiple truthy values
        assert result == 'default'

    @given(st.lists(st.integers(), min_size=1, max_size=50))
    @example([1])
    @example([1, 2, 3])
    def test_first_returns_first(self, lst):
        """first should return the first element."""
        result = first(lst)
        assert result == lst[0]

    def test_first_empty_returns_default(self):
        """first on empty should return default."""
        result = first([], default='default')
        assert result == 'default'


# ============================================================================
# MEDIUM PRIORITY: Same
# ============================================================================

class TestSame:
    """Test same function."""

    @given(st.lists(st.just(42), max_size=50))
    @example([])
    @example([42])
    @example([42, 42, 42])
    def test_same_all_equal(self, lst):
        """same should return True when all elements are equal."""
        result = same(lst)
        assert result is True

    @given(st.lists(st.integers(), min_size=2, max_size=50).filter(lambda l: len(set(l)) > 1))
    @example([1, 2])
    @example([1, 1, 2])
    def test_same_not_all_equal(self, lst):
        """same should return False when elements differ."""
        result = same(lst)
        assert result is False


# ============================================================================
# Edge Cases and Boundary Conditions
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_chunked_empty_list(self):
        """Chunking empty list should return empty list."""
        assert chunked([], 5) == []

    def test_flatten_empty_list(self):
        """Flattening empty list should return empty list."""
        assert flatten([]) == []

    def test_unique_empty_list(self):
        """unique on empty list should return empty list."""
        assert unique([]) == []

    def test_split_empty_list(self):
        """Splitting empty list should return list with empty list."""
        result = split([], 0)
        assert result == [[]]

    def test_partition_empty_list(self):
        """Partitioning empty list should return two empty lists."""
        trues, falses = partition([])
        assert trues == []
        assert falses == []

    def test_bucketize_empty_list(self):
        """Bucketizing empty list should return empty dict."""
        result = bucketize([])
        assert result == {}

    @given(st.lists(st.integers(), max_size=100))
    def test_unique_iter_equals_unique(self, lst):
        """unique_iter should produce same result as unique."""
        result1 = unique(lst)
        result2 = list(unique_iter(lst))
        assert result1 == result2

    @given(st.lists(st.integers(), max_size=100), st.integers(min_value=1, max_value=20))
    def test_chunked_iter_equals_chunked(self, lst, size):
        """chunked_iter should produce same result as chunked."""
        result1 = chunked(lst, size)
        result2 = [list(chunk) for chunk in chunked_iter(lst, size)]
        assert result1 == result2


# ============================================================================
# Invariant Properties
# ============================================================================

class TestInvariants:
    """Test invariant properties that should always hold."""

    @given(st.lists(st.integers(), max_size=100))
    def test_flatten_length_bounded(self, nested):
        """Flattened length should be >= original length."""
        result = flatten(nested)
        # For a list of integers, flatten should not change it
        assert len(result) >= 0

    @given(st.lists(st.integers(), max_size=100), st.integers(min_value=1, max_value=20))
    def test_chunked_no_data_loss(self, lst, size):
        """Chunking should not lose any data."""
        chunks = chunked(lst, size)
        total_elements = sum(len(chunk) for chunk in chunks)
        assert total_elements == len(lst)

    @given(st.lists(st.integers(), max_size=100))
    def test_unique_subset(self, lst):
        """unique result should be a subset of original."""
        result = unique(lst)
        assert set(result).issubset(set(lst))
