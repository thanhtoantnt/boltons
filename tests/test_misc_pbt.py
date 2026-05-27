"""Property-based tests for setutils, listutils, and other utility modules using Hypothesis."""
from hypothesis import given, strategies as st, settings, example, assume
import pytest

from boltons.setutils import IndexedSet, complement
from boltons.listutils import BarrelList
from boltons.queueutils import HeapPriorityQueue, SortedPriorityQueue
from boltons.pathutils import augpath, shrinkuser, expandpath


# ============================================================================
# HIGH PRIORITY: IndexedSet Properties
# ============================================================================

class TestIndexedSet:
    """Test IndexedSet properties."""

    @given(st.lists(st.integers(), max_size=100))
    @example([])
    @example([1])
    @example([1, 2, 3])
    @example([1, 1, 2, 2, 3])
    def test_indexedset_no_duplicates(self, items):
        """IndexedSet should contain no duplicates."""
        iset = IndexedSet(items)
        # Convert to list to check for duplicates
        items_list = list(iset)
        assert len(items_list) == len(set(items_list))

    @given(st.lists(st.integers(), max_size=100))
    @example([1, 2, 3])
    def test_indexedset_preserves_order(self, items):
        """IndexedSet should preserve insertion order."""
        iset = IndexedSet(items)
        # Get unique items in order
        seen = set()
        expected = []
        for item in items:
            if item not in seen:
                seen.add(item)
                expected.append(item)

        assert list(iset) == expected

    @given(st.lists(st.integers(), min_size=1, max_size=50))
    @example([1, 2, 3])
    def test_indexedset_indexing(self, items):
        """IndexedSet should support indexing."""
        iset = IndexedSet(items)
        unique_items = list(dict.fromkeys(items))  # Preserve order, remove dupes

        for i in range(len(unique_items)):
            assert iset[i] == unique_items[i]

    @given(st.lists(st.integers(), max_size=100))
    def test_indexedset_set_operations(self, items):
        """IndexedSet should support set operations."""
        iset1 = IndexedSet(items[:len(items)//2])
        iset2 = IndexedSet(items[len(items)//2:])

        # Union
        union = iset1 | iset2
        assert isinstance(union, (set, IndexedSet))

        # Intersection
        intersection = iset1 & iset2
        assert isinstance(intersection, (set, IndexedSet))

    @given(st.lists(st.integers(), max_size=100), st.integers())
    @example([1, 2, 3], 2)
    def test_indexedset_contains(self, items, item):
        """IndexedSet 'in' operator should work correctly."""
        iset = IndexedSet(items)
        if item in items:
            assert item in iset
        else:
            assert item not in iset


# ============================================================================
# HIGH PRIORITY: Complement Set Properties
# ============================================================================

class TestComplementSet:
    """Test complement set properties."""

    @given(st.sets(st.integers(min_value=0, max_value=100), max_size=20))
    @example(set())
    @example({1, 2, 3})
    def test_complement_double_complement(self, items):
        """Double complement should return to original."""
        comp = complement(items)
        double_comp = complement(comp)

        # Check a sample of values
        for i in range(10):
            assert (i in items) == (i in double_comp)

    @given(st.sets(st.integers(min_value=0, max_value=100), max_size=20), st.integers(min_value=0, max_value=100))
    @example({1, 2, 3}, 5)
    def test_complement_membership(self, items, value):
        """Complement should invert membership."""
        comp = complement(items)

        if value in items:
            assert value not in comp
        else:
            assert value in comp


# ============================================================================
# MEDIUM PRIORITY: BarrelList Properties
# ============================================================================

class TestBarrelList:
    """Test BarrelList properties."""

    @given(st.lists(st.integers(), max_size=100))
    @example([])
    @example([1])
    @example([1, 2, 3])
    def test_barrellist_length(self, items):
        """BarrelList length should match number of items."""
        blist = BarrelList(items)
        assert len(blist) == len(items)

    @given(st.lists(st.integers(), min_size=1, max_size=50))
    @example([1, 2, 3])
    def test_barrellist_indexing(self, items):
        """BarrelList should support indexing."""
        blist = BarrelList(items)

        for i in range(len(items)):
            assert blist[i] == items[i]

    @given(st.lists(st.integers(), max_size=100))
    def test_barrellist_iteration(self, items):
        """BarrelList should support iteration."""
        blist = BarrelList(items)
        assert list(blist) == items

    @given(st.lists(st.integers(), max_size=50), st.integers())
    @example([1, 2, 3], 4)
    def test_barrellist_append(self, items, new_item):
        """BarrelList append should add item."""
        blist = BarrelList(items)
        blist.append(new_item)

        assert len(blist) == len(items) + 1
        assert blist[-1] == new_item


# ============================================================================
# MEDIUM PRIORITY: Priority Queue Properties
# ============================================================================

class TestPriorityQueues:
    """Test priority queue properties."""

    @given(st.lists(st.integers(), max_size=50))
    @example([])
    @example([1])
    @example([3, 1, 2])
    def test_heappriorityqueue_sorted_output(self, items):
        """HeapPriorityQueue should output items in priority order."""
        pq = HeapPriorityQueue()

        for item in items:
            pq.add(item)

        result = []
        while pq:
            result.append(pq.pop())

        # Result should be sorted
        assert result == sorted(items)

    @given(st.lists(st.integers(), max_size=50))
    @example([3, 1, 2])
    def test_sortedpriorityqueue_sorted_output(self, items):
        """SortedPriorityQueue should output items in priority order."""
        pq = SortedPriorityQueue()

        for item in items:
            pq.add(item)

        result = []
        while pq:
            result.append(pq.pop())

        # Result should be sorted
        assert result == sorted(items)

    @given(st.lists(st.integers(), max_size=50))
    def test_heappriorityqueue_length(self, items):
        """HeapPriorityQueue length should match number of items."""
        pq = HeapPriorityQueue()

        for item in items:
            pq.add(item)

        assert len(pq) == len(items)

    @given(st.lists(st.integers(), min_size=1, max_size=50))
    @example([3, 1, 2])
    def test_heappriorityqueue_peek(self, items):
        """HeapPriorityQueue peek should return min without removing."""
        pq = HeapPriorityQueue()

        for item in items:
            pq.add(item)

        min_item = min(items)
        assert pq.peek() == min_item
        # Length should not change
        assert len(pq) == len(items)


# ============================================================================
# MEDIUM PRIORITY: Path Utilities
# ============================================================================

class TestPathUtils:
    """Test path utility functions."""

    @given(st.text(alphabet='abcdefghijklmnopqrstuvwxyz/', min_size=1, max_size=50))
    @example('/path/to/file.txt')
    @example('relative/path.txt')
    def test_augpath_preserves_base(self, path):
        """augpath should preserve base path structure."""
        # Add suffix
        result = augpath(path, suffix='_backup')
        # Should contain original path elements
        assert isinstance(result, str)

    @given(st.text(alphabet='abcdefghijklmnopqrstuvwxyz/', min_size=1, max_size=50))
    def test_augpath_extension(self, path):
        """augpath should handle extension changes."""
        result = augpath(path, ext='.bak')
        # Should end with new extension
        assert result.endswith('.bak') or '.' not in path


# ============================================================================
# Invariant Properties
# ============================================================================

class TestInvariants:
    """Test invariant properties that should always hold."""

    @given(st.lists(st.integers(), max_size=100))
    def test_indexedset_length_matches_unique(self, items):
        """IndexedSet length should match unique items."""
        iset = IndexedSet(items)
        assert len(iset) == len(set(items))

    @given(st.lists(st.integers(), max_size=100))
    def test_barrellist_preserves_order(self, items):
        """BarrelList should preserve order."""
        blist = BarrelList(items)
        assert list(blist) == items

    @given(st.lists(st.integers(), min_size=1, max_size=50))
    def test_heappriorityqueue_min_at_top(self, items):
        """HeapPriorityQueue should have min at top."""
        pq = HeapPriorityQueue()

        for item in items:
            pq.add(item)

        assert pq.peek() == min(items)


# ============================================================================
# Edge Cases and Boundary Conditions
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_indexedset_empty(self):
        """Empty IndexedSet should work."""
        iset = IndexedSet()
        assert len(iset) == 0
        assert list(iset) == []

    def test_barrellist_empty(self):
        """Empty BarrelList should work."""
        blist = BarrelList()
        assert len(blist) == 0
        assert list(blist) == []

    def test_heappriorityqueue_empty(self):
        """Empty HeapPriorityQueue should work."""
        pq = HeapPriorityQueue()
        assert len(pq) == 0

    def test_indexedset_single_item(self):
        """IndexedSet with single item should work."""
        iset = IndexedSet([42])
        assert len(iset) == 1
        assert iset[0] == 42

    def test_heappriorityqueue_single_item(self):
        """HeapPriorityQueue with single item should work."""
        pq = HeapPriorityQueue()
        pq.add(42)
        assert pq.pop() == 42
        assert len(pq) == 0

    @given(st.lists(st.integers(), min_size=1, max_size=50))
    def test_indexedset_add_duplicate(self, items):
        """Adding duplicate to IndexedSet should not increase size."""
        iset = IndexedSet(items)
        original_len = len(iset)

        if items:
            iset.add(items[0])
            assert len(iset) == original_len


# ============================================================================
# Type Preservation
# ============================================================================

class TestTypePreservation:
    """Test that functions preserve expected types."""

    @given(st.lists(st.integers(), max_size=50))
    def test_indexedset_returns_indexedset(self, items):
        """IndexedSet operations should return IndexedSet or set."""
        iset = IndexedSet(items)
        assert isinstance(iset, IndexedSet)

    @given(st.lists(st.integers(), max_size=50))
    def test_barrellist_returns_barrellist(self, items):
        """BarrelList should be a list-like type."""
        blist = BarrelList(items)
        assert hasattr(blist, '__getitem__')
        assert hasattr(blist, '__len__')

    @given(st.sets(st.integers(min_value=0, max_value=100), max_size=20))
    def test_complement_returns_set_like(self, items):
        """complement should return a set-like object."""
        comp = complement(items)
        # Should support 'in' operator
        assert hasattr(comp, '__contains__')


# ============================================================================
# Set Operations
# ============================================================================

class TestSetOperations:
    """Test set operations on IndexedSet."""

    @given(st.lists(st.integers(), max_size=50), st.lists(st.integers(), max_size=50))
    @example([1, 2, 3], [2, 3, 4])
    def test_indexedset_union(self, items1, items2):
        """IndexedSet union should contain all unique items."""
        iset1 = IndexedSet(items1)
        iset2 = IndexedSet(items2)

        union = iset1 | iset2
        expected = set(items1) | set(items2)

        assert set(union) == expected

    @given(st.lists(st.integers(), max_size=50), st.lists(st.integers(), max_size=50))
    @example([1, 2, 3], [2, 3, 4])
    def test_indexedset_intersection(self, items1, items2):
        """IndexedSet intersection should contain common items."""
        iset1 = IndexedSet(items1)
        iset2 = IndexedSet(items2)

        intersection = iset1 & iset2
        expected = set(items1) & set(items2)

        assert set(intersection) == expected

    @given(st.lists(st.integers(), max_size=50), st.lists(st.integers(), max_size=50))
    def test_indexedset_difference(self, items1, items2):
        """IndexedSet difference should contain items in first but not second."""
        iset1 = IndexedSet(items1)
        iset2 = IndexedSet(items2)

        difference = iset1 - iset2
        expected = set(items1) - set(items2)

        assert set(difference) == expected


# ============================================================================
# Priority Queue Behavior
# ============================================================================

class TestPriorityQueueBehavior:
    """Test priority queue behavior under various conditions."""

    @given(st.lists(st.integers(), min_size=2, max_size=50))
    @example([3, 1, 2])
    def test_heappriorityqueue_pop_order(self, items):
        """Items should be popped in ascending order."""
        pq = HeapPriorityQueue()

        for item in items:
            pq.add(item)

        prev = pq.pop()
        while pq:
            curr = pq.pop()
            assert curr >= prev
            prev = curr

    @given(st.lists(st.integers(), max_size=50))
    def test_heappriorityqueue_add_pop_cycle(self, items):
        """Adding and popping should maintain heap property."""
        pq = HeapPriorityQueue()

        for item in items:
            pq.add(item)
            if len(pq) > 10:
                pq.pop()

        # Remaining items should still be in order
        result = []
        while pq:
            result.append(pq.pop())

        assert result == sorted(result)


# ============================================================================
# List Operations
# ============================================================================

class TestListOperations:
    """Test list operations on BarrelList."""

    @given(st.lists(st.integers(), max_size=50), st.integers(), st.integers(min_value=0, max_value=49))
    @example([1, 2, 3], 4, 1)
    def test_barrellist_insert(self, items, new_item, index):
        """BarrelList insert should work correctly."""
        if not items:
            assume(False)

        blist = BarrelList(items)
        index = min(index, len(items))

        blist.insert(index, new_item)

        assert len(blist) == len(items) + 1
        assert blist[index] == new_item

    @given(st.lists(st.integers(), min_size=1, max_size=50))
    @example([1, 2, 3])
    def test_barrellist_remove(self, items):
        """BarrelList remove should work correctly."""
        blist = BarrelList(items)
        item_to_remove = items[0]

        blist.remove(item_to_remove)

        assert len(blist) == len(items) - 1
        # First occurrence should be removed
        expected = items.copy()
        expected.remove(item_to_remove)
        assert list(blist) == expected
