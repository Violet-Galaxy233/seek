from app.domain import AnswerValue, QuestionAnswer
from app.fixtures import build_demo_service


def _scores(result):
    return {candidate.album.id: candidate.score for candidate in result.candidates}


def test_search_returns_twenty_ranked_candidates_and_a_question() -> None:
    result = build_demo_service().search("红色背景，有一只小皮鞋")

    assert len(result.candidates) == 20
    assert result.question is not None
    assert all(candidate.matching_clues for candidate in result.candidates)
    assert all(candidate.conflicting_clues for candidate in result.candidates)
    assert [item.score for item in result.candidates] == sorted(
        (item.score for item in result.candidates), reverse=True
    )


def test_yes_answer_changes_ranking_and_question_is_not_repeated() -> None:
    service = build_demo_service()
    initial = service.search("红色背景，有小皮鞋")
    assert initial.question is not None

    answered = service.search(
        "红色背景，有小皮鞋",
        answers=(QuestionAnswer(initial.question.id, AnswerValue.YES),),
    )

    assert _scores(answered) != _scores(initial)
    assert answered.question is None or answered.question.id != initial.question.id


def test_uncertain_answer_does_not_change_scores() -> None:
    service = build_demo_service()
    initial = service.search("红色背景，有小皮鞋")
    assert initial.question is not None

    answered = service.search(
        "红色背景，有小皮鞋",
        answers=(QuestionAnswer(initial.question.id, AnswerValue.UNCERTAIN),),
    )

    assert _scores(answered) == _scores(initial)


def test_search_preserves_raw_memory_separately() -> None:
    result = build_demo_service().search("  红色背景，  有鞋  ")

    assert result.clue.raw_text == "  红色背景，  有鞋  "
    assert result.clue.normalized_text == "红色背景， 有鞋"

