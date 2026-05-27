# Property-Based Testing for Boltons Library

## Summary

This project adds comprehensive property-based testing (PBT) coverage to the boltons utility library using Hypothesis. Property-based tests automatically generate hundreds of test cases to verify that functions satisfy mathematical properties like roundtrips, idempotence, and invariants.

## Test Coverage

### Test Files Created

1. **test_strutils_pbt.py** (38 tests)
   - Gzip/gunzip roundtrip properties
   - Integer list format/parse roundtrip
   - Shell escaping roundtrip with shlex
   - Camel/underscore case conversion
   - Idempotence: slugify, asciify, strip_ansi, html2text
   - Invariants: ordinalize format, bytes2human monotonicity
   - Pluralization properties

2. **test_iterutils_pbt.py** (46 tests)
   - Chunked/flatten inverse relationship
   - Split/join reconstruction
   - Unique idempotence and no-duplicates property
   - Partition completeness and no-overlap
   - Sorted ordering properties
   - Strip operations (lstrip, rstrip, strip)
   - Windowing and pairwise operations
   - Backoff monotonic increasing

3. **test_mathutils_pbt.py** (47 tests)
   - Bits format conversion roundtrips (hex, bin, bytes, list)
   - Clamp bounds and idempotence
   - Ceil/floor properties with options
   - Bits bitwise operations (commutativity, identity)
   - Bits equality and hashing

4. **test_urlutils_pbt.py** (44 tests)
   - URL parsing/serialization roundtrip
   - Quote/unquote roundtrip for path, query, fragment, userinfo
   - Quote idempotence
   - URL normalization properties
   - URL component extraction

5. **test_dictutils_pbt.py** (40 tests)
   - OrderedMultiDict double inversion roundtrip
   - OneToOne bijection properties
   - OMD operations preserve order and data
   - FrozenDict immutability and hashability
   - Subdict key filtering
   - ManyToMany bidirectional mapping

6. **test_cacheutils_pbt.py** (37 tests)
   - LRU/LRI cache size limits
   - LRU eviction policy (least recently used)
   - LRI eviction policy (least recently inserted)
   - Cache hit/miss tracking
   - Cached decorator memoization
   - Cached property computed once

7. **test_misc_pbt.py** (27 tests)
   - IndexedSet no duplicates and order preservation
   - Complement double complement property
   - BarrelList operations
   - Priority queue sorted output
   - Path utilities

## Test Results

**Total: 279 tests**
- **Passing: 257 (92%)**
- **Failing: 22 (8%)**

### Key Properties Tested

#### HIGH PRIORITY - Roundtrip Properties
- ✅ `gunzip_bytes(gzip_bytes(x)) == x` - Perfect compression roundtrip
- ✅ `parse_int_list(format_int_list(lst)) == sorted(set(lst))` - Integer list roundtrip
- ✅ `shlex.split(args2sh(args)) == args` - Shell escaping roundtrip
- ✅ `unquote(quote_path_part(s)) == s` - URL encoding roundtrip
- ✅ `Bits.from_hex(b.as_hex()).as_int() == b.as_int()` - Bits format conversions
- ✅ `omd.inverted().inverted() == omd` - OrderedMultiDict double inversion

#### MEDIUM PRIORITY - Idempotence Properties
- ✅ `slugify(slugify(s)) == slugify(s)` - Slugify idempotence
- ✅ `unique(unique(x)) == unique(x)` - Unique idempotence
- ✅ `clamp(clamp(x, a, b), a, b) == clamp(x, a, b)` - Clamp idempotence
- ✅ `flatten(flatten(x)) == flatten(x)` - Flatten idempotence

#### MEDIUM PRIORITY - Invariant Properties
- ✅ `lower <= clamp(x, lower, upper) <= upper` - Clamp bounds
- ✅ `ceil(x) >= x` and `floor(x) <= x` - Ceil/floor properties
- ✅ `len(unique(x)) <= len(x)` - Unique reduces size
- ✅ `len(cache) <= max_size` - Cache size limits
- ✅ `oto[k] == v implies oto.inv[v] == k` - OneToOne bijection

## Known Failing Tests

Some tests fail due to edge cases or API behavior that differs from initial assumptions:

1. **Cache eviction tests** - LRU/LRI eviction behavior has subtle edge cases
2. **URL property tests** - URL parsing has complex normalization rules
3. **Priority queue tests** - Need to handle duplicate priorities
4. **First/one functions** - Behavior with falsy values needs refinement

These failures represent opportunities to either:
- Fix the tests to match actual API behavior
- Discover potential bugs in the implementation
- Document edge cases more clearly

## Benefits of Property-Based Testing

1. **Broader Coverage**: Each test runs 100+ generated examples automatically
2. **Edge Case Discovery**: Hypothesis finds corner cases developers miss
3. **Regression Prevention**: Properties ensure behavior stays consistent
4. **Documentation**: Properties serve as executable specifications
5. **Confidence**: Mathematical properties provide stronger guarantees than examples

## Running the Tests

```bash
# Install hypothesis
pip install hypothesis

# Run all property-based tests
pytest tests/test_*_pbt.py -v

# Run specific module tests
pytest tests/test_strutils_pbt.py -v

# Run with more examples (thorough)
pytest tests/test_strutils_pbt.py --hypothesis-seed=0 -v
```

## Future Work

1. Fix remaining 22 failing tests
2. Add more properties for:
   - timeutils (datetime parsing/formatting)
   - jsonutils (JSON serialization)
   - funcutils (function wrapping)
   - tbutils (traceback formatting)
3. Increase test examples for CI (currently using default 100)
4. Add stateful testing for cache eviction policies
5. Add performance properties (e.g., compression ratio bounds)

## Conclusion

This property-based testing suite significantly enhances the boltons library's test coverage by verifying mathematical properties that hold across thousands of generated inputs. The 257 passing tests provide strong confidence in the correctness of core utilities, while the 22 failing tests highlight areas for further investigation and refinement.
