import pytest

from video_eval.qwen import validate_answer


def valid():
    return {"dimension_id": "D1.7", "applicable": True, "score": 4,
            "confidence": .8, "evidence": [{"start": 0., "end": 1., "observation": "visible"}],
            "reason": "ok", "needs_review": False}


def test_accepts_valid_answer():
    assert validate_answer(valid(), "D1.7", 2.)["score"] == 4


def test_rejects_wrong_dimension_score_and_timestamp():
    answer = valid()
    answer["dimension_id"] = "D2.4"
    with pytest.raises(ValueError):
        validate_answer(answer, "D1.7", 2.)
    answer = valid()
    answer["score"] = 3.5
    with pytest.raises(ValueError):
        validate_answer(answer, "D1.7", 2.)
    answer = valid()
    answer["evidence"][0]["end"] = 5.
    with pytest.raises(ValueError):
        validate_answer(answer, "D1.7", 2.)


def test_non_applicable_must_be_scoreless():
    answer = valid()
    answer.update(applicable=False, score=None, evidence=[])
    assert validate_answer(answer, "D1.7", 2.)["score"] is None
