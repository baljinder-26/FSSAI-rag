from app.answer import LLMAnswer, answer_question


def rows(retriever, q):
    return [e for e in retriever.retrieve(q) if e.kind == "row"]


def test_exact_category_and_parameter_lookup(retriever):
    (e,) = rows(retriever, "What is the maximum limit of lead in tea?")
    assert "Metal: Lead | Food: Tea | Limit: 5.0 on dry matter basis" in e.text
    assert e.page == 6


def test_same_food_different_parameter_gives_different_limit(retriever):
    (lead,) = rows(retriever, "lead in tea")
    (copper,) = rows(retriever, "copper in tea")
    assert "Limit: 5.0" in lead.text and lead.page == 6
    assert "Limit: 150" in copper.text and copper.page == 7


def test_food_missing_under_that_metal_is_not_returned(retriever):
    # Rice appears under Cadmium only, so "lead in rice" must have no table evidence.
    assert retriever.retrieve("lead in rice") == []
    (cd,) = rows(retriever, "cadmium in polished rice")
    assert "Limit: 0.4" in cd.text


def test_condition_changes_the_meaning_of_the_limit(retriever):
    (e,) = rows(retriever, "copper in infant foods")
    assert "15 (But not less than 2.8)" in e.text
    (e,) = rows(retriever, "lead in tea")
    assert "on dry matter basis" in e.text


def test_question_combining_table_and_prose(retriever):
    evidence = retriever.retrieve("What is the lead limit in tea and what is a crop contaminant?")
    assert {e.kind for e in evidence} == {"row", "prose"}

    def fake_llm(question, evidence_text):
        ids = [line[1:3] for line in evidence_text.splitlines() if line.startswith("[E")]
        return LLMAnswer(answer="Lead in tea: 5.0 on dry matter basis (page 6). "
                                "A crop contaminant is a substance not intentionally added to food.",
                         evidence_ids=ids)

    resp = answer_question("What is the lead limit in tea and what is a crop contaminant?",
                           retriever, fake_llm)
    assert [s.page for s in resp.sources] == [6, 1]
