import pytest
def calculate_recall(expected: list[int], retrieved: list[int]) -> int: 
    true_positives = len(set(expected) & set(retrieved))
    return true_positives / len(expected)


def calculate_precision(expected: list[int], retrieved: list[int]) -> int: 
    true_positives = len(set(expected) & set(retrieved))
    return true_positives / len(retrieved)


@pytest.mark.integration
@pytest.mark.parametrize("query_vector, expected_ids", [ 
    ([0.1, 0.2, 0.3, 0.4], [1, 2, 3]),
    ([0.2, 0.3, 0.4, 0.5], [2, 1, 3]),
    ([0.3, 0.4, 0.5, 0.6], [3, 2, 1]),
])
def test_retrieval_subsystem(db_client, query_vector, expected_ids): 
    response = db_client.query_points( 
        collection_name="test",
        query=query_vector,
        limit=3
    )

    retrieved_ids = [point.id for point in response.points]
    recall = calculate_recall(expected_ids, retrieved_ids)
    precision = calculate_precision(expected_ids, retrieved_ids)

    assert recall >= 0.66 
    assert precision >= 0.66 