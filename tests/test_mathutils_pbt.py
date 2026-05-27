"""Property-based tests for mathutils module using Hypothesis."""
from hypothesis import given, strategies as st, settings, example, assume
import pytest
import math

from boltons.mathutils import clamp, ceil, floor, Bits


# ============================================================================
# HIGH PRIORITY: Bits Roundtrip Properties
# ============================================================================

class TestBitsRoundtrip:
    """Test Bits conversion roundtrip properties."""

    @given(st.integers(min_value=0, max_value=2**32))
    @example(0)
    @example(1)
    @example(255)
    @example(256)
    def test_bits_int_roundtrip(self, n):
        """Converting to int and back should preserve value."""
        bits = Bits.from_int(n)
        result = bits.as_int()
        assert result == n

    @given(st.integers(min_value=0, max_value=2**16))
    @example(0)
    @example(255)
    def test_bits_hex_roundtrip(self, n):
        """Converting to hex and back should preserve value."""
        bits = Bits.from_int(n)
        hex_str = bits.as_hex()
        result = Bits.from_hex(hex_str)
        # Values should match (length may differ due to padding)
        assert result.as_int() == bits.as_int()

    @given(st.integers(min_value=0, max_value=2**16))
    @example(0)
    @example(1)
    @example(255)
    def test_bits_bin_roundtrip(self, n):
        """Converting to binary string and back should preserve value."""
        bits = Bits.from_int(n)
        bin_str = bits.as_bin()
        result = Bits.from_bin(bin_str)
        assert result == bits

    @given(st.integers(min_value=0, max_value=2**16))
    def test_bits_bytes_roundtrip(self, n):
        """Converting to bytes and back should preserve value."""
        bits = Bits.from_int(n)
        bytes_val = bits.as_bytes()
        result = Bits.from_bytes(bytes_val)
        # Values should match (length may differ due to byte alignment)
        assert result.as_int() == bits.as_int()

    @given(st.lists(st.booleans(), min_size=1, max_size=32))
    @example([True])
    @example([False])
    @example([True, False, True])
    def test_bits_list_roundtrip(self, bool_list):
        """Converting to list and back should preserve value."""
        bits = Bits.from_list(bool_list)
        result_list = bits.as_list()
        assert result_list == bool_list


# ============================================================================
# HIGH PRIORITY: Clamp Properties
# ============================================================================

class TestClampProperties:
    """Test clamp function properties."""

    @given(st.floats(allow_nan=False, allow_infinity=False),
           st.floats(allow_nan=False, allow_infinity=False),
           st.floats(allow_nan=False, allow_infinity=False))
    @example(5.0, 0.0, 10.0)
    @example(-5.0, 0.0, 10.0)
    @example(15.0, 0.0, 10.0)
    def test_clamp_bounds(self, x, a, b):
        """Clamped value should be within bounds."""
        if a > b:
            a, b = b, a  # Ensure lower <= upper
        result = clamp(x, a, b)
        assert a <= result <= b

    @given(st.floats(allow_nan=False, allow_infinity=False),
           st.floats(allow_nan=False, allow_infinity=False),
           st.floats(allow_nan=False, allow_infinity=False))
    def test_clamp_identity_when_in_range(self, x, a, b):
        """Clamp should return x when x is already in range."""
        if a > b:
            a, b = b, a
        if a <= x <= b:
            assert clamp(x, a, b) == x

    @given(st.floats(allow_nan=False, allow_infinity=False),
           st.floats(allow_nan=False, allow_infinity=False))
    def test_clamp_lower_bound(self, x, lower):
        """Clamp with only lower bound should be >= lower."""
        result = clamp(x, lower=lower)
        assert result >= lower

    @given(st.floats(allow_nan=False, allow_infinity=False),
           st.floats(allow_nan=False, allow_infinity=False))
    def test_clamp_upper_bound(self, x, upper):
        """Clamp with only upper bound should be <= upper."""
        result = clamp(x, upper=upper)
        assert result <= upper

    @given(st.floats(allow_nan=False, allow_infinity=False),
           st.floats(allow_nan=False, allow_infinity=False),
           st.floats(allow_nan=False, allow_infinity=False))
    def test_clamp_idempotent(self, x, a, b):
        """Clamping twice should equal clamping once."""
        if a > b:
            a, b = b, a
        once = clamp(x, a, b)
        twice = clamp(once, a, b)
        assert once == twice

    @given(st.floats(allow_nan=False, allow_infinity=False),
           st.floats(allow_nan=False, allow_infinity=False))
    def test_clamp_invalid_bounds_raises(self, a, b):
        """Clamp should raise when upper < lower."""
        if a > b:
            with pytest.raises(ValueError):
                clamp(5, lower=a, upper=b)


# ============================================================================
# MEDIUM PRIORITY: Ceil and Floor Properties
# ============================================================================

class TestCeilFloorProperties:
    """Test ceil and floor function properties."""

    @given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6))
    @example(0.0)
    @example(1.5)
    @example(-1.5)
    def test_ceil_greater_or_equal(self, x):
        """Ceiling should be >= input."""
        result = ceil(x)
        assert result >= x

    @given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6))
    def test_floor_less_or_equal(self, x):
        """Floor should be <= input."""
        result = floor(x)
        assert result <= x

    @given(st.integers(min_value=-1000, max_value=1000))
    @example(0)
    @example(42)
    def test_ceil_integer_identity(self, n):
        """Ceiling of integer should be the integer itself."""
        assert ceil(n) == n

    @given(st.integers(min_value=-1000, max_value=1000))
    def test_floor_integer_identity(self, n):
        """Floor of integer should be the integer itself."""
        assert floor(n) == n

    @given(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6))
    def test_ceil_floor_difference(self, x):
        """Difference between ceil and floor should be 0 or 1."""
        c = ceil(x)
        f = floor(x)
        diff = c - f
        assert diff in (0, 1)

    @given(st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=100))
    def test_ceil_with_options(self, x):
        """Ceil with options should return value from options >= x."""
        options = [0, 10, 20, 30, 40, 50, 100]
        try:
            result = ceil(x, options=options)
            assert result in options
            assert result >= x
            # Should be the smallest option >= x
            valid_options = [opt for opt in options if opt >= x]
            assert result == min(valid_options)
        except ValueError:
            # x is greater than all options
            assert x > max(options)

    @given(st.floats(allow_nan=False, allow_infinity=False, min_value=0, max_value=100))
    def test_floor_with_options(self, x):
        """Floor with options should return value from options <= x."""
        options = [0, 10, 20, 30, 40, 50, 100]
        try:
            result = floor(x, options=options)
            assert result in options
            assert result <= x
            # Should be the largest option <= x
            valid_options = [opt for opt in options if opt <= x]
            assert result == max(valid_options)
        except ValueError:
            # x is less than all options
            assert x < min(options)

    @given(st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100))
    def test_ceil_idempotent(self, x):
        """Ceiling twice should equal ceiling once."""
        once = ceil(x)
        twice = ceil(once)
        assert once == twice

    @given(st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100))
    def test_floor_idempotent(self, x):
        """Floor twice should equal floor once."""
        once = floor(x)
        twice = floor(once)
        assert once == twice


# ============================================================================
# MEDIUM PRIORITY: Bits Invariants
# ============================================================================

class TestBitsInvariants:
    """Test Bits invariant properties."""

    @given(st.integers(min_value=0, max_value=2**16))
    @example(0)
    @example(1)
    def test_bits_length_consistent(self, n):
        """Bits length should match binary representation."""
        bits = Bits.from_int(n)
        bin_str = bits.as_bin()
        assert len(bits) == len(bin_str)

    @given(st.integers(min_value=0, max_value=2**16))
    def test_bits_non_negative(self, n):
        """Bits value should always be non-negative."""
        bits = Bits.from_int(n)
        assert bits.as_int() >= 0

    @given(st.integers(min_value=0, max_value=255))
    def test_bits_hex_format(self, n):
        """Hex representation should be valid hex."""
        bits = Bits.from_int(n)
        hex_str = bits.as_hex()
        # Should be valid hex (no 0x prefix from as_hex)
        int(hex_str, 16)  # Should not raise

    @given(st.integers(min_value=0, max_value=255))
    def test_bits_bin_format(self, n):
        """Binary representation should be valid binary."""
        bits = Bits.from_int(n)
        bin_str = bits.as_bin()
        # Should be valid binary (only 0s and 1s)
        assert all(c in '01' for c in bin_str)

    @given(st.integers(min_value=1, max_value=2**16))
    def test_bits_indexing(self, n):
        """Bits indexing should match binary string."""
        bits = Bits.from_int(n)
        bin_str = bits.as_bin()
        for i in range(len(bits)):
            assert bits[i] == (bin_str[i] == '1')


# ============================================================================
# MEDIUM PRIORITY: Bits Operations
# ============================================================================

class TestBitsOperations:
    """Test Bits bitwise operations."""

    @given(st.integers(min_value=0, max_value=255), st.integers(min_value=0, max_value=255))
    @example(0, 0)
    @example(255, 0)
    @example(255, 255)
    def test_bits_or_commutative(self, a, b):
        """Bitwise OR should be commutative."""
        bits_a = Bits.from_int(a)
        bits_b = Bits.from_int(b)
        result1 = bits_a | bits_b
        result2 = bits_b | bits_a
        assert result1 == result2

    @given(st.integers(min_value=0, max_value=255), st.integers(min_value=0, max_value=255))
    def test_bits_and_commutative(self, a, b):
        """Bitwise AND should be commutative."""
        bits_a = Bits.from_int(a)
        bits_b = Bits.from_int(b)
        result1 = bits_a & bits_b
        result2 = bits_b & bits_a
        assert result1 == result2

    @given(st.integers(min_value=0, max_value=255))
    @example(0)
    @example(255)
    def test_bits_or_identity(self, n):
        """OR with zero should be identity."""
        bits = Bits.from_int(n)
        zero = Bits.from_int(0, len_=bits.len)
        result = bits | zero
        assert result.as_int() == n

    @given(st.integers(min_value=0, max_value=255))
    def test_bits_and_identity(self, n):
        """AND with all ones should be identity."""
        bits = Bits.from_int(n)
        all_ones = Bits.from_int(2**bits.len - 1, len_=bits.len)
        result = bits & all_ones
        assert result.as_int() == n

    @given(st.integers(min_value=1, max_value=255), st.integers(min_value=0, max_value=8))
    @example(1, 0)
    @example(1, 1)
    def test_bits_shift_left_increases_length(self, n, shift):
        """Left shift should increase length."""
        bits = Bits.from_int(n)
        shifted = bits << shift
        assert len(shifted) == len(bits) + shift

    @given(st.integers(min_value=1, max_value=255), st.integers(min_value=0, max_value=4))
    def test_bits_shift_right_decreases_length(self, n, shift):
        """Right shift should decrease length."""
        bits = Bits.from_int(n)
        if shift < len(bits):
            shifted = bits >> shift
            assert len(shifted) == len(bits) - shift


# ============================================================================
# MEDIUM PRIORITY: Bits Equality
# ============================================================================

class TestBitsEquality:
    """Test Bits equality properties."""

    @given(st.integers(min_value=0, max_value=255))
    @example(0)
    @example(42)
    def test_bits_equality_reflexive(self, n):
        """Bits should equal itself."""
        bits = Bits.from_int(n)
        assert bits == bits

    @given(st.integers(min_value=0, max_value=255))
    def test_bits_equality_symmetric(self, n):
        """Bits equality should be symmetric."""
        bits1 = Bits.from_int(n)
        bits2 = Bits.from_int(n)
        assert bits1 == bits2
        assert bits2 == bits1

    @given(st.integers(min_value=0, max_value=255))
    def test_bits_hash_consistent(self, n):
        """Equal Bits should have equal hashes."""
        bits1 = Bits.from_int(n)
        bits2 = Bits.from_int(n)
        if bits1 == bits2:
            assert hash(bits1) == hash(bits2)


# ============================================================================
# Edge Cases and Boundary Conditions
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_clamp_equal_bounds(self):
        """Clamp with equal bounds should return that value."""
        assert clamp(5, 3, 3) == 3
        assert clamp(1, 3, 3) == 3

    def test_ceil_zero(self):
        """Ceiling of zero should be zero."""
        assert ceil(0) == 0

    def test_floor_zero(self):
        """Floor of zero should be zero."""
        assert floor(0) == 0

    def test_bits_zero(self):
        """Bits from zero should work."""
        bits = Bits.from_int(0)
        assert bits.as_int() == 0
        assert bits.as_bin() == '0'

    def test_bits_one(self):
        """Bits from one should work."""
        bits = Bits.from_int(1)
        assert bits.as_int() == 1
        assert bits.as_bin() == '1'

    def test_bits_empty_list(self):
        """Bits from empty list should raise or handle gracefully."""
        try:
            bits = Bits.from_list([])
            # If it doesn't raise, should be zero
            assert bits.as_int() == 0
        except (ValueError, IndexError):
            # Also acceptable to reject empty list
            pass

    def test_bits_negative_raises(self):
        """Bits should reject negative values."""
        with pytest.raises(ValueError):
            Bits.from_int(-1)

    @given(st.floats(min_value=0, max_value=100))
    def test_ceil_floor_options_empty_raises(self, x):
        """Ceil/floor with empty options should raise."""
        with pytest.raises((ValueError, IndexError)):
            ceil(x, options=[])
        with pytest.raises((ValueError, IndexError)):
            floor(x, options=[])


# ============================================================================
# Type Preservation
# ============================================================================

class TestTypePreservation:
    """Test that functions preserve expected types."""

    @given(st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100),
           st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100),
           st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100))
    def test_clamp_preserves_type(self, x, a, b):
        """Clamp should preserve numeric type."""
        if a > b:
            a, b = b, a
        result = clamp(x, a, b)
        assert isinstance(result, (int, float))

    @given(st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100))
    def test_ceil_returns_int(self, x):
        """Ceil should return int."""
        result = ceil(x)
        assert isinstance(result, int)

    @given(st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100))
    def test_floor_returns_int(self, x):
        """Floor should return int."""
        result = floor(x)
        assert isinstance(result, int)

    @given(st.integers(min_value=0, max_value=255))
    def test_bits_as_int_returns_int(self, n):
        """Bits.as_int() should return int."""
        bits = Bits.from_int(n)
        result = bits.as_int()
        assert isinstance(result, int)

    @given(st.integers(min_value=0, max_value=255))
    def test_bits_as_hex_returns_str(self, n):
        """Bits.as_hex() should return str."""
        bits = Bits.from_int(n)
        result = bits.as_hex()
        assert isinstance(result, str)

    @given(st.integers(min_value=0, max_value=255))
    def test_bits_as_bin_returns_str(self, n):
        """Bits.as_bin() should return str."""
        bits = Bits.from_int(n)
        result = bits.as_bin()
        assert isinstance(result, str)

    @given(st.integers(min_value=0, max_value=255))
    def test_bits_as_bytes_returns_bytes(self, n):
        """Bits.as_bytes() should return bytes."""
        bits = Bits.from_int(n)
        result = bits.as_bytes()
        assert isinstance(result, bytes)

    @given(st.integers(min_value=0, max_value=255))
    def test_bits_as_list_returns_list(self, n):
        """Bits.as_list() should return list of bools."""
        bits = Bits.from_int(n)
        result = bits.as_list()
        assert isinstance(result, list)
        assert all(isinstance(b, bool) for b in result)
