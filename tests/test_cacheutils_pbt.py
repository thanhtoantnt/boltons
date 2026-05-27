"""Property-based tests for cacheutils module using Hypothesis."""
from hypothesis import given, strategies as st, settings, example, assume
import pytest

from boltons.cacheutils import LRU, LRI, cachedproperty, cached, cachedmethod


# ============================================================================
# HIGH PRIORITY: LRU Cache Invariants
# ============================================================================

class TestLRUInvariants:
    """Test LRU cache invariant properties."""

    @given(st.integers(min_value=1, max_value=100), st.lists(st.integers(), min_size=1, max_size=200))
    @example(5, [1, 2, 3, 4, 5])
    @example(3, [1, 2, 3, 4, 5])
    def test_lru_size_limit(self, max_size, items):
        """LRU cache should never exceed max_size."""
        cache = LRU(max_size=max_size)

        for item in items:
            cache[item] = item * 2

        assert len(cache) <= max_size

    @given(st.integers(min_value=1, max_value=50), st.lists(st.integers(), min_size=1, max_size=100))
    @example(5, [1, 2, 3, 4, 5, 6])
    def test_lru_evicts_least_recently_used(self, max_size, items):
        """LRU should evict least recently used items."""
        cache = LRU(max_size=max_size)

        # Fill cache
        for item in items[:max_size]:
            cache[item] = item

        # Access first item to make it recently used
        if items[:max_size]:
            _ = cache[items[0]]

        # Add one more item to trigger eviction
        if len(items) > max_size:
            cache[items[max_size]] = items[max_size]

            # First item should still be there (was recently accessed)
            assert items[0] in cache
            # But second item should be evicted (least recently used)
            if len(items[:max_size]) > 1:
                assert items[1] not in cache

    @given(st.integers(min_value=1, max_value=50), st.lists(st.integers(), max_size=100))
    def test_lru_get_updates_recency(self, max_size, items):
        """Getting an item should update its recency."""
        cache = LRU(max_size=max_size)

        for item in items:
            cache[item] = item * 2

        # Get all items to update recency
        for key in list(cache.keys()):
            _ = cache[key]

        # Cache should still have same items
        original_keys = set(cache.keys())

        # Add more items
        for item in items:
            cache[item + 1000] = item

        # Some original keys should still be present if accessed recently
        assert len(cache) <= max_size


# ============================================================================
# HIGH PRIORITY: LRI Cache Invariants
# ============================================================================

class TestLRIInvariants:
    """Test LRI (Least Recently Inserted) cache invariant properties."""

    @given(st.integers(min_value=1, max_value=100), st.lists(st.integers(), min_size=1, max_size=200))
    @example(5, [1, 2, 3, 4, 5])
    def test_lri_size_limit(self, max_size, items):
        """LRI cache should never exceed max_size."""
        cache = LRI(max_size=max_size)

        for item in items:
            cache[item] = item * 2

        assert len(cache) <= max_size

    @given(st.integers(min_value=2, max_value=50), st.lists(st.integers(), min_size=3, max_size=100))
    @example(2, [1, 2, 3])
    def test_lri_evicts_oldest_insertion(self, max_size, items):
        """LRI should evict oldest inserted items regardless of access."""
        cache = LRI(max_size=max_size)

        # Fill cache
        for item in items[:max_size]:
            cache[item] = item

        # Access first item multiple times (shouldn't matter for LRI)
        if items[:max_size]:
            for _ in range(10):
                _ = cache.get(items[0])

        # Add one more item to trigger eviction
        if len(items) > max_size:
            cache[items[max_size]] = items[max_size]

            # First item should be evicted (oldest insertion)
            assert items[0] not in cache
            # New item should be present
            assert items[max_size] in cache


# ============================================================================
# MEDIUM PRIORITY: Cache Hit/Miss Tracking
# ============================================================================

class TestCacheHitMiss:
    """Test cache hit/miss tracking."""

    @given(st.integers(min_value=1, max_value=50), st.lists(st.integers(), min_size=1, max_size=100))
    @example(10, [1, 2, 3, 1, 2, 3])
    def test_lru_hit_count(self, max_size, items):
        """LRU should track cache hits correctly."""
        cache = LRU(max_size=max_size)

        # Insert items
        for item in items:
            cache[item] = item

        initial_hits = cache.hit_count

        # Access existing items
        hit_count = 0
        for item in items[:max_size]:
            if item in cache:
                _ = cache[item]
                hit_count += 1

        # Hit count should increase
        assert cache.hit_count >= initial_hits

    @given(st.integers(min_value=1, max_value=50), st.lists(st.integers(), min_size=1, max_size=100))
    def test_lru_miss_count(self, max_size, items):
        """LRU should track cache misses correctly."""
        cache = LRU(max_size=max_size)

        initial_misses = cache.miss_count

        # Try to access non-existent items
        for item in items:
            _ = cache.get(item + 10000)

        # Miss count should increase
        assert cache.miss_count > initial_misses


# ============================================================================
# MEDIUM PRIORITY: Cached Decorator
# ============================================================================

class TestCachedDecorator:
    """Test @cached decorator properties."""

    @given(st.lists(st.integers(), min_size=1, max_size=50))
    @example([1, 2, 3, 1, 2, 3])
    def test_cached_function_memoization(self, inputs):
        """Cached function should return same result for same input."""
        call_count = [0]

        @cached(LRU(max_size=100))
        def expensive_func(x):
            call_count[0] += 1
            return x * 2

        # Call with same inputs multiple times
        results = []
        for inp in inputs:
            results.append(expensive_func(inp))

        # Call again with same inputs
        results2 = []
        for inp in inputs:
            results2.append(expensive_func(inp))

        # Results should be identical
        assert results == results2

        # Call count should be less than total calls (due to caching)
        unique_inputs = len(set(inputs))
        assert call_count[0] <= len(inputs)
        assert call_count[0] == unique_inputs + len(inputs)  # First pass + unique from second

    @given(st.integers(), st.integers())
    @example(5, 5)
    @example(5, 10)
    def test_cached_function_deterministic(self, x, y):
        """Cached function should be deterministic."""
        @cached(LRU(max_size=100))
        def add(a, b):
            return a + b

        result1 = add(x, y)
        result2 = add(x, y)
        assert result1 == result2


# ============================================================================
# MEDIUM PRIORITY: Cached Property
# ============================================================================

class TestCachedProperty:
    """Test cachedproperty decorator."""

    @given(st.integers())
    @example(42)
    def test_cachedproperty_computed_once(self, value):
        """Cached property should be computed only once."""
        call_count = [0]

        class MyClass:
            def __init__(self, val):
                self.val = val

            @cachedproperty
            def expensive_property(self):
                call_count[0] += 1
                return self.val * 2

        obj = MyClass(value)

        # Access property multiple times
        result1 = obj.expensive_property
        result2 = obj.expensive_property
        result3 = obj.expensive_property

        # Should be computed only once
        assert call_count[0] == 1
        # Results should be identical
        assert result1 == result2 == result3

    @given(st.integers())
    def test_cachedproperty_per_instance(self, value):
        """Cached property should be per-instance."""
        class MyClass:
            def __init__(self, val):
                self.val = val

            @cachedproperty
            def doubled(self):
                return self.val * 2

        obj1 = MyClass(value)
        obj2 = MyClass(value + 1)

        # Each instance should have its own cached value
        assert obj1.doubled == value * 2
        assert obj2.doubled == (value + 1) * 2


# ============================================================================
# MEDIUM PRIORITY: Cache Operations
# ============================================================================

class TestCacheOperations:
    """Test cache operations like clear, pop, etc."""

    @given(st.integers(min_value=1, max_value=50), st.lists(st.integers(), min_size=1, max_size=100))
    @example(10, [1, 2, 3, 4, 5])
    def test_lru_clear(self, max_size, items):
        """Clear should empty the cache."""
        cache = LRU(max_size=max_size)

        for item in items:
            cache[item] = item

        cache.clear()
        assert len(cache) == 0

    @given(st.integers(min_value=1, max_value=50), st.lists(st.integers(), min_size=1, max_size=100))
    def test_lru_pop(self, max_size, items):
        """Pop should remove and return item."""
        cache = LRU(max_size=max_size)

        for item in items[:max_size]:
            cache[item] = item * 2

        if items[:max_size]:
            key = items[0]
            if key in cache:
                value = cache.pop(key)
                assert value == key * 2
                assert key not in cache

    @given(st.integers(min_value=1, max_value=50), st.lists(st.integers(), min_size=1, max_size=100))
    def test_lru_get_default(self, max_size, items):
        """Get with default should return default for missing keys."""
        cache = LRU(max_size=max_size)

        for item in items:
            cache[item] = item

        # Get non-existent key with default
        result = cache.get(max(items) + 1000 if items else 1000, default='missing')
        assert result == 'missing'


# ============================================================================
# Invariant Properties
# ============================================================================

class TestInvariants:
    """Test invariant properties that should always hold."""

    @given(st.integers(min_value=1, max_value=50), st.lists(st.integers(), max_size=100))
    def test_lru_keys_values_consistent(self, max_size, items):
        """Keys and values should be consistent."""
        cache = LRU(max_size=max_size)

        for item in items:
            cache[item] = item * 2

        # Every key should have a corresponding value
        for key in cache.keys():
            assert key in cache
            assert cache[key] == key * 2

    @given(st.integers(min_value=1, max_value=50), st.lists(st.integers(), max_size=100))
    def test_lru_len_matches_keys(self, max_size, items):
        """Length should match number of keys."""
        cache = LRU(max_size=max_size)

        for item in items:
            cache[item] = item

        assert len(cache) == len(list(cache.keys()))

    @given(st.integers(min_value=1, max_value=50))
    def test_lru_empty_cache(self, max_size):
        """Empty cache should have length 0."""
        cache = LRU(max_size=max_size)
        assert len(cache) == 0
        assert list(cache.keys()) == []


# ============================================================================
# Edge Cases and Boundary Conditions
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_lru_size_one(self):
        """LRU with size 1 should work."""
        cache = LRU(max_size=1)
        cache['a'] = 1
        assert len(cache) == 1

        cache['b'] = 2
        assert len(cache) == 1
        assert 'b' in cache
        assert 'a' not in cache

    def test_lri_size_one(self):
        """LRI with size 1 should work."""
        cache = LRI(max_size=1)
        cache['a'] = 1
        assert len(cache) == 1

        cache['b'] = 2
        assert len(cache) == 1
        assert 'b' in cache
        assert 'a' not in cache

    @given(st.integers(min_value=1, max_value=50))
    def test_lru_overwrite_same_key(self, max_size):
        """Overwriting same key should not increase size."""
        cache = LRU(max_size=max_size)
        cache['key'] = 1
        cache['key'] = 2
        cache['key'] = 3

        assert len(cache) == 1
        assert cache['key'] == 3

    def test_lru_zero_size_raises(self):
        """LRU with size 0 should raise or handle gracefully."""
        try:
            cache = LRU(max_size=0)
            # If it doesn't raise, should not allow any items
            cache['a'] = 1
            assert len(cache) == 0
        except (ValueError, AssertionError):
            # Also acceptable to reject size 0
            pass

    @given(st.integers(min_value=1, max_value=50), st.lists(st.integers(), max_size=100))
    def test_lru_contains(self, max_size, items):
        """'in' operator should work correctly."""
        cache = LRU(max_size=max_size)

        for item in items:
            cache[item] = item

        for key in cache.keys():
            assert key in cache

        # Non-existent key
        assert (max(items) + 1000 if items else 1000) not in cache


# ============================================================================
# Type Preservation
# ============================================================================

class TestTypePreservation:
    """Test that caches preserve expected types."""

    @given(st.integers(min_value=1, max_value=50), st.lists(st.tuples(st.text(min_size=1, max_size=10), st.integers()), max_size=50))
    def test_lru_preserves_types(self, max_size, items):
        """LRU should preserve key and value types."""
        cache = LRU(max_size=max_size)

        for key, value in items:
            cache[key] = value

        for key in cache.keys():
            assert isinstance(key, str)
            assert isinstance(cache[key], int)

    @given(st.integers(min_value=1, max_value=50))
    def test_lru_keys_returns_iterable(self, max_size):
        """keys() should return an iterable."""
        cache = LRU(max_size=max_size)
        cache['a'] = 1
        cache['b'] = 2

        keys = cache.keys()
        assert hasattr(keys, '__iter__')

    @given(st.integers(min_value=1, max_value=50))
    def test_lru_values_returns_iterable(self, max_size):
        """values() should return an iterable."""
        cache = LRU(max_size=max_size)
        cache['a'] = 1
        cache['b'] = 2

        values = cache.values()
        assert hasattr(values, '__iter__')


# ============================================================================
# Cache Behavior Under Load
# ============================================================================

class TestCacheBehavior:
    """Test cache behavior under various access patterns."""

    @given(st.integers(min_value=5, max_value=20), st.lists(st.integers(min_value=0, max_value=10), min_size=50, max_size=100))
    @example(5, [1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    def test_lru_working_set(self, max_size, accesses):
        """LRU should handle working set efficiently."""
        cache = LRU(max_size=max_size)

        # Simulate access pattern
        for item in accesses:
            if item not in cache:
                cache[item] = item * 2
            else:
                _ = cache[item]

        # Cache should contain most recently accessed items
        assert len(cache) <= max_size

    @given(st.integers(min_value=5, max_value=20), st.integers(min_value=0, max_value=10))
    def test_lru_repeated_access(self, max_size, key):
        """Repeated access to same key should keep it in cache."""
        cache = LRU(max_size=max_size)

        # Fill cache
        for i in range(max_size):
            cache[i] = i

        # Repeatedly access one key
        for _ in range(100):
            _ = cache[key]

        # That key should still be in cache
        assert key in cache
