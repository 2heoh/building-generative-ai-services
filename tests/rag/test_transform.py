import json
import pytest
from rag.transform import chunk

@pytest.fixture(scope="module")
def tokens():
    return [1, 2, 3, 4, 5]

def test_chunking_with_size_2_success(tokens):
    result = chunk(tokens, chunk_size=2)
    
    assert result == [[1, 2], [3, 4], [5]]

def test_chunking_with_size_5_success(tokens):
    result = chunk(tokens, chunk_size=5)
    
    assert result == [[1, 2, 3, 4, 5]]    


@pytest.mark.parametrize("tokens, chunk_size, expected", [ 
    ([1, 2, 3, 4, 5], 2, [[1, 2], [3, 4], [5]]), # valid
    ([1, 2, 3, 4, 5], 3, [[1, 2, 3], [4, 5]]), # valid
    ([1, 2, 3, 4, 5], 1, [[1], [2], [3], [4], [5]]),  # valid
    ([], 3, []), # valid/empty input
    ([1, 2, 3], 5, [[1, 2, 3]]),   # boundary input
    ([1, 2, 3, 4, 5], 0, "ValueError"), # invalid (chunk_size <= 0)
    ([1, 2, 3, 4, 5], -1, "ValueError"), # invalid (chunk_size <= 0)
    (
        list(range(10000)), 1000, [list(range(i, i + 1000)) # huge data
        for i in range(0, 10000, 1000)]
    )
])
def test_token_chunking(tokens, chunk_size, expected): 
    if expected == "ValueError":
        with pytest.raises(ValueError):
            chunk(tokens, chunk_size)
    else:
        assert chunk(tokens, chunk_size) == expected


# Load test data at module level for use in parametrize decorator
with open('tests/rag/test_data.json') as f:
    _test_data = json.load(f)

@pytest.mark.parametrize("case", _test_data)
def test_token_chunking_from_json(case):
    tokens = case.get("tokens")
    chunk_size = case.get("chunk_size")
    expected = case.get("expected")
    
    if expected == "ValueError":
        with pytest.raises(ValueError):
            chunk(tokens, chunk_size)
    else:
        assert chunk(tokens, chunk_size) == expected
