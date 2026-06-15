from pathlib import Path

from nano_tox_agent.literature import load_literature, select_literature


EXPECTED_TITLES = {
    "The eNanoMapper database for nanomaterial safety information",
    "Predicting Cytotoxicity of Nanoparticles: A Meta-Analysis Using Machine Learning",
    "Application of Machine Learning in Nanotoxicology: A Critical Review and Perspective",
    "caNanoLab: data sharing to expedite the use of nanotechnology in biomedicine",
    "AI and Machine Learning Approaches for Predicting Nanoparticles Toxicity: The Critical Role of Physiochemical Properties",
    "Smart Drug-Delivery Systems for Cancer Nanotherapy",
}

ALLOWED_URL_PREFIXES = (
    "https://doi.org/",
    "https://www.beilstein-journals.org/",
    "https://cananolab.cancer.gov/",
    "https://arxiv.org/",
)


def test_load_literature_reads_entries():
    entries = load_literature(Path("literature/literature_base.json"))
    assert len(entries) == 6
    assert entries[0]["title"] == "The eNanoMapper database for nanomaterial safety information"
    assert "data_source" in entries[0]["used_for"]


def test_select_literature_filters_by_use():
    entries = load_literature(Path("literature/literature_base.json"))
    selected = select_literature(entries, "feature_explanation")
    titles = {entry["title"] for entry in selected}
    assert "AI and Machine Learning Approaches for Predicting Nanoparticles Toxicity: The Critical Role of Physiochemical Properties" in titles
    assert "Smart Drug-Delivery Systems for Cancer Nanotherapy" not in titles


def test_cananolab_literature_is_deferred_context_not_training_source():
    entries = load_literature(Path("literature/literature_base.json"))
    entry = next(
        item
        for item in entries
        if item["title"] == "caNanoLab: data sharing to expedite the use of nanotechnology in biomedicine"
    )
    assert "data_source" not in entry["used_for"]
    assert "deferred_source_cross_check" in entry["used_for"]
    assert "后续" in entry["data_relevance"]


def test_literature_entries_have_traceable_urls():
    entries = load_literature(Path("literature/literature_base.json"))
    for entry in entries:
        assert entry["url"].startswith(ALLOWED_URL_PREFIXES)
        assert entry["url"] != "https://pubs.acs.org/"


def test_literature_entries_have_complete_summaries():
    entries = load_literature(Path("literature/literature_base.json"))
    assert {entry["title"] for entry in entries} == EXPECTED_TITLES
    for entry in entries:
        assert len(entry["full_summary"]) >= 120
        assert entry["method_relevance"].strip()
        assert entry["data_relevance"].strip()
        assert entry["limitations"].strip()
        assert len(entry["key_points"]) >= 3
