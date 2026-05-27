"""Property-based tests for dictutils module using Hypothesis."""
from hypothesis import given, strategies as st, settings, example, assume
import pytest

from boltons.dictutils import OrderedMultiDict, OMD, OneToOne, ManyToMany, subdict, FrozenDict


# ============================================================================
# HIGH PRIORITY: OrderedMultiDict Inversion Roundtrip
# ============================================================================

class TestOMDInversion:
    """Test OrderedMultiDict.inverted() roundtrip property."""

    @given(st.lists(st.tuples(st.text(min_size=1, max_size=10), st.integers()), max_size=20))
    @example([])
    @example([('a', 1)])
    @example([('a', 1), ('b', 2)])
    @example([('a', 1), ('a', 2)])
    def test_omd_double_inversion_roundtrip(self, items):
        """Inverting twice should return to original structure."""
        omd = OrderedMultiDict(items)
        inverted = omd.inverted()
        double_inverted = inverted.inverted()

        # Should have same items (order may differ for multi-values)
        assert sorted(omd.items(multi=True)) == sorted(double_inverted.items(multi=True))

    @given(st.lists(st.tuples(st.text(min_size=1, max_size=10), st.integers()), max_size=20))
    def test_omd_inversion_preserves_data(self, items):
        """Inversion should preserve all data."""
        omd = OrderedMultiDict(items)
        inverted = omd.inverted()

        # All original values should be keys in inverted
        original_values = [v for k, v in omd.items(multi=True)]
        inverted_keys = list(inverted.keys())

        # Check that data is preserved (as multiset)
        assert sorted(original_values) == sorted(inverted_keys)


# ============================================================================
# HIGH PRIORITY: OneToOne Bidirectional Mapping
# ============================================================================

class TestOneToOneBijection:
    """Test OneToOne bidirectional mapping properties."""

    @given(st.dictionaries(st.text(min_size=1, max_size=10), st.integers(), max_size=20))
    @example({})
    @example({'a': 1})
    @example({'a': 1, 'b': 2})
    def test_onetoone_bijection_property(self, mapping):
        """OneToOne should maintain bijection: oto[k]=v implies oto.inv[v]=k."""
        oto = OneToOne(mapping)

        for key, value in oto.items():
            # Forward mapping
            assert oto[key] == value
            # Reverse mapping
            assert oto.inv[value] == key

    @given(st.dictionaries(st.text(min_size=1, max_size=10), st.integers(), max_size=20))
    def test_onetoone_double_inversion(self, mapping):
        """Double inversion should return to original."""
        oto = OneToOne(mapping)
        double_inv = oto.inv.inv

        assert dict(oto) == dict(double_inv)

    @given(st.text(min_size=1, max_size=10), st.integers())
    @example('key', 42)
    def test_onetoone_set_updates_inverse(self, key, value):
        """Setting a value should update the inverse mapping."""
        oto = OneToOne()
        oto[key] = value

        assert oto[key] == value
        assert oto.inv[value] == key


# ============================================================================
# MEDIUM PRIORITY: OrderedMultiDict Operations
# ============================================================================

class TestOMDOperations:
    """Test OrderedMultiDict operations and invariants."""

    @given(st.lists(st.tuples(st.text(min_size=1, max_size=10), st.integers()), max_size=20))
    @example([])
    @example([('a', 1)])
    def test_omd_add_preserves_order(self, items):
        """Adding items should preserve insertion order."""
        omd = OrderedMultiDict()
        for key, value in items:
            omd.add(key, value)

        # Items should be in insertion order
        result_items = omd.items(multi=True)
        assert result_items == items

    @given(st.lists(st.tuples(st.text(min_size=1, max_size=10), st.integers()), min_size=1, max_size=20))
    def test_omd_getlist_returns_all_values(self, items):
        """getlist should return all values for a key."""
        omd = OrderedMultiDict(items)

        # Group items by key
        key_values = {}
        for key, value in items:
            if key not in key_values:
                key_values[key] = []
            key_values[key].append(value)

        # Check that getlist returns all values
        for key, expected_values in key_values.items():
            assert omd.getlist(key) == expected_values

    @given(st.lists(st.tuples(st.text(min_size=1, max_size=10), st.integers()), max_size=20))
    def test_omd_get_returns_last_value(self, items):
        """get should return the most recent value for a key."""
        if not items:
            assume(False)

        omd = OrderedMultiDict(items)

        # Find last value for each key
        last_values = {}
        for key, value in items:
            last_values[key] = value

        for key, expected_value in last_values.items():
            assert omd.get(key) == expected_value

    @given(st.lists(st.tuples(st.text(min_size=1, max_size=10), st.integers()), max_size=20))
    def test_omd_len_counts_unique_keys(self, items):
        """len should count unique keys, not total items."""
        omd = OrderedMultiDict(items)
        unique_keys = set(k for k, v in items)
        assert len(omd) == len(unique_keys)

    @given(st.lists(st.tuples(st.text(min_size=1, max_size=10), st.integers()), max_size=20))
    def test_omd_keys_unique(self, items):
        """keys() should return unique keys."""
        omd = OrderedMultiDict(items)
        keys = list(omd.keys())
        # Keys should be unique
        assert len(keys) == len(set(keys))


# ============================================================================
# MEDIUM PRIORITY: FrozenDict Immutability
# ============================================================================

class TestFrozenDictImmutability:
    """Test FrozenDict immutability properties."""

    @given(st.dictionaries(st.text(min_size=1, max_size=10), st.integers(), max_size=20))
    @example({})
    @example({'a': 1})
    def test_frozendict_immutable(self, mapping):
        """FrozenDict should be immutable."""
        fd = FrozenDict(mapping)

        # Should not allow item assignment
        with pytest.raises((TypeError, AttributeError)):
            fd['new_key'] = 42

    @given(st.dictionaries(st.text(min_size=1, max_size=10), st.integers(), max_size=20))
    def test_frozendict_hashable(self, mapping):
        """FrozenDict should be hashable."""
        fd = FrozenDict(mapping)
        # Should be able to hash it
        hash(fd)

        # Should be able to use as dict key
        d = {fd: 'value'}
        assert d[fd] == 'value'

    @given(st.dictionaries(st.text(min_size=1, max_size=10), st.integers(), max_size=20))
    def test_frozendict_equality(self, mapping):
        """FrozenDict equality should work correctly."""
        fd1 = FrozenDict(mapping)
        fd2 = FrozenDict(mapping)
        assert fd1 == fd2

        # Equal FrozenDicts should have equal hashes
        assert hash(fd1) == hash(fd2)


# ============================================================================
# MEDIUM PRIORITY: Subdict
# ============================================================================

class TestSubdict:
    """Test subdict function."""

    @given(st.dictionaries(st.text(min_size=1, max_size=10), st.integers(), min_size=1, max_size=20),
           st.lists(st.text(min_size=1, max_size=10), max_size=10))
    @example({'a': 1, 'b': 2}, ['a'])
    @example({'a': 1, 'b': 2, 'c': 3}, ['a', 'c'])
    def test_subdict_contains_only_specified_keys(self, mapping, keys):
        """subdict should contain only specified keys."""
        result = subdict(mapping, keys)

        # Result should only have keys that are in both mapping and keys
        expected_keys = set(keys) & set(mapping.keys())
        assert set(result.keys()) == expected_keys

    @given(st.dictionaries(st.text(min_size=1, max_size=10), st.integers(), min_size=1, max_size=20))
    def test_subdict_preserves_values(self, mapping):
        """subdict should preserve values for selected keys."""
        keys = list(mapping.keys())[:len(mapping) // 2] if mapping else []
        result = subdict(mapping, keys)

        for key in result:
            assert result[key] == mapping[key]

    @given(st.dictionaries(st.text(min_size=1, max_size=10), st.integers(), max_size=20))
    def test_subdict_empty_keys(self, mapping):
        """subdict with empty keys should return empty dict."""
        result = subdict(mapping, [])
        assert result == {}


# ============================================================================
# MEDIUM PRIORITY: ManyToMany
# ============================================================================

class TestManyToMany:
    """Test ManyToMany mapping."""

    @given(st.lists(st.tuples(st.text(min_size=1, max_size=10), st.integers()), max_size=20))
    @example([])
    @example([('a', 1)])
    @example([('a', 1), ('a', 2)])
    def test_manytomany_bidirectional(self, items):
        """ManyToMany should maintain bidirectional mapping."""
        mtm = ManyToMany(items)

        # Check forward mapping
        for key, value in items:
            assert value in mtm[key]

        # Check reverse mapping
        for key, value in items:
            assert key in mtm.inv[value]

    @given(st.lists(st.tuples(st.text(min_size=1, max_size=10), st.integers()), max_size=20))
    def test_manytomany_double_inversion(self, items):
        """Double inversion should preserve structure."""
        mtm = ManyToMany(items)
        double_inv = mtm.inv.inv

        # Should have same items
        assert sorted(mtm.items(multi=True)) == sorted(double_inv.items(multi=True))


# ============================================================================
# Invariant Properties
# ============================================================================

class TestInvariants:
    """Test invariant properties that should always hold."""

    @given(st.lists(st.tuples(st.text(min_size=1, max_size=10), st.integers()), max_size=20))
    def test_omd_items_multi_complete(self, items):
        """items(multi=True) should return all items."""
        omd = OrderedMultiDict(items)
        result = omd.items(multi=True)

        # Should have same number of items
        assert len(result) == len(items)

    @given(st.lists(st.tuples(st.text(min_size=1, max_size=10), st.integers()), max_size=20))
    def test_omd_items_single_unique_keys(self, items):
        """items(multi=False) should have unique keys."""
        omd = OrderedMultiDict(items)
        result = omd.items(multi=False)

        keys = [k for k, v in result]
        assert len(keys) == len(set(keys))

    @given(st.dictionaries(st.text(min_size=1, max_size=10), st.integers(), max_size=20))
    def test_onetoone_no_duplicate_values(self, mapping):
        """OneToOne should not allow duplicate values."""
        oto = OneToOne(mapping)
        values = list(oto.values())
        # All values should be unique
        assert len(values) == len(set(values))

    @given(st.dictionaries(st.text(min_size=1, max_size=10), st.integers(), max_size=20))
    def test_frozendict_read_only(self, mapping):
        """FrozenDict should allow reads but not writes."""
        fd = FrozenDict(mapping)

        # Should allow reads
        for key, value in mapping.items():
            assert fd[key] == value

        # Should not allow writes
        with pytest.raises((TypeError, AttributeError)):
            fd['new'] = 123


# ============================================================================
# Edge Cases and Boundary Conditions
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_omd_empty(self):
        """Empty OMD should work."""
        omd = OrderedMultiDict()
        assert len(omd) == 0
        assert list(omd.items()) == []

    def test_onetoone_empty(self):
        """Empty OneToOne should work."""
        oto = OneToOne()
        assert len(oto) == 0
        assert len(oto.inv) == 0

    def test_frozendict_empty(self):
        """Empty FrozenDict should work."""
        fd = FrozenDict()
        assert len(fd) == 0
        assert dict(fd) == {}

    def test_subdict_nonexistent_keys(self):
        """subdict with nonexistent keys should ignore them."""
        d = {'a': 1, 'b': 2}
        result = subdict(d, ['a', 'c', 'd'])
        assert result == {'a': 1}

    def test_omd_pop_removes_all_values(self):
        """pop should remove all values for a key."""
        omd = OrderedMultiDict([('a', 1), ('a', 2), ('b', 3)])
        omd.pop('a')
        assert 'a' not in omd
        assert omd.getlist('a') == []

    def test_omd_add_to_existing_key(self):
        """add should append to existing key."""
        omd = OrderedMultiDict([('a', 1)])
        omd.add('a', 2)
        assert omd.getlist('a') == [1, 2]

    @given(st.text(min_size=1, max_size=10), st.integers(), st.integers())
    def test_onetoone_overwrite_updates_inverse(self, key, value1, value2):
        """Overwriting a value should update inverse correctly."""
        if value1 == value2:
            assume(False)

        oto = OneToOne()
        oto[key] = value1
        assert oto.inv[value1] == key

        oto[key] = value2
        assert oto.inv[value2] == key
        # Old value should no longer map back
        assert value1 not in oto.inv


# ============================================================================
# Type Preservation
# ============================================================================

class TestTypePreservation:
    """Test that functions preserve expected types."""

    @given(st.lists(st.tuples(st.text(min_size=1, max_size=10), st.integers()), max_size=20))
    def test_omd_inverted_returns_omd(self, items):
        """inverted() should return an OMD."""
        omd = OrderedMultiDict(items)
        inverted = omd.inverted()
        assert isinstance(inverted, OrderedMultiDict)

    @given(st.dictionaries(st.text(min_size=1, max_size=10), st.integers(), max_size=20))
    def test_onetoone_inv_returns_onetoone(self, mapping):
        """inv should return a OneToOne."""
        oto = OneToOne(mapping)
        assert isinstance(oto.inv, OneToOne)

    @given(st.dictionaries(st.text(min_size=1, max_size=10), st.integers(), max_size=20))
    def test_subdict_returns_dict(self, mapping):
        """subdict should return a dict."""
        keys = list(mapping.keys())[:len(mapping) // 2] if mapping else []
        result = subdict(mapping, keys)
        assert isinstance(result, dict)

    @given(st.lists(st.tuples(st.text(min_size=1, max_size=10), st.integers()), max_size=20))
    def test_omd_getlist_returns_list(self, items):
        """getlist should return a list."""
        omd = OrderedMultiDict(items)
        if items:
            key = items[0][0]
            result = omd.getlist(key)
            assert isinstance(result, list)


# ============================================================================
# Ordering Properties
# ============================================================================

class TestOrdering:
    """Test ordering properties of OrderedMultiDict."""

    @given(st.lists(st.tuples(st.text(min_size=1, max_size=10), st.integers()), min_size=1, max_size=20))
    @example([('a', 1), ('b', 2), ('c', 3)])
    def test_omd_preserves_insertion_order(self, items):
        """OMD should preserve insertion order."""
        omd = OrderedMultiDict(items)
        result_items = omd.items(multi=True)
        assert result_items == items

    @given(st.lists(st.tuples(st.text(min_size=1, max_size=10), st.integers()), min_size=2, max_size=20))
    def test_omd_reversed(self, items):
        """reversed(omd) should reverse key order."""
        omd = OrderedMultiDict(items)
        keys = list(omd.keys())
        reversed_keys = list(reversed(omd))
        assert reversed_keys == list(reversed(keys))
