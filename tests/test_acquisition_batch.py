from src.acquisition_batch import AcquisitionBatch


def test_acquisition_batch():
    indices = [1, 2, 3]
    scores = [0.1, 0.2, 0.3]
    original_scores = None
    acq_b = AcquisitionBatch(indices, scores, original_scores)
    assert acq_b is not None
    assert acq_b.indices == indices
    assert acq_b.scores == scores
    assert acq_b.orignal_scores == original_scores
