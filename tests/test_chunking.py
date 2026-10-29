from app.chunking import record_to_chunk
from tests.conftest import find


def test_row_chunk_keeps_headers_category_unit_condition_and_source(parsed):
    records, _ = parsed
    (tea,) = find(records, "Lead", "Tea")
    text = record_to_chunk(tea).text
    assert "Metal contaminant: Lead" in text
    assert "Article of food: Tea" in text
    assert "5.0 on dry matter basis" in text
    assert "mg/kg or mg/L (ppm)" in text
    assert "Condition: on dry matter basis" in text
    assert "Regulation 2.1.1(2)" in text and "1. Lead" in text
    assert "Compendium_Contaminants_Regulations.pdf, page 6" in text


def test_every_row_is_validated_and_has_metadata(parsed):
    records, _ = parsed
    assert len(records) > 300
    assert not [r for r in records if r.needs_review]
    assert all(r.limit_value is not None and r.page and r.table and r.document for r in records)


def test_row_split_across_pages_is_joined(parsed):
    records, _ = parsed
    cadmium = [r for r in records if r.parameter == "Cadmium" and r.food.startswith("Cereal grains")]
    assert len(cadmium) == 1
    assert "Quinoa" in cadmium[0].food and cadmium[0].page == 11 and cadmium[0].page_end == 12


def test_prose_is_split_by_clause_not_size(parsed):
    _, chunks = parsed
    prose = [c for c in chunks if c.kind == "prose"]
    assert any("Crop contaminant" in c.text and "1.2.1" in c.text for c in prose)
    assert all(c.page == 1 for c in prose)
