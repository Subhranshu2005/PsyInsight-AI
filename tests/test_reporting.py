import pandas as pd

from psyinsight.reporting import ResearchReportGenerator


def test_markdown_report_contains_sections():
    report = ResearchReportGenerator(title="Test Study")
    report.add_section("Data Validation", {"rows": 100})
    report.add_section("Descriptive Stats", pd.DataFrame({"mean": [1.0]}))
    markdown = report.to_markdown()
    assert "Test Study" in markdown
    assert "Data Validation" in markdown
    assert "Descriptive Stats" in markdown


def test_html_report_is_valid_shape():
    report = ResearchReportGenerator(title="Test Study")
    report.add_section("Section A", {"x": 1})
    html = report.to_html()
    assert html.startswith("<!DOCTYPE html>")
    assert "Section A" in html


def test_json_roundtrip():
    report = ResearchReportGenerator(title="Test Study")
    report.add_section("Section A", {"x": 1})
    payload = report.to_dict()
    assert payload["title"] == "Test Study"
    assert payload["sections"][0]["heading"] == "Section A"


def test_from_pipeline_builds_expected_sections():
    report = ResearchReportGenerator.from_pipeline(
        title="Pipeline Study",
        validation_report={"rows": 10},
        descriptive_stats={"mean": 1},
        ml_report={"accuracy": 0.9},
    )
    headings = [s["heading"] for s in report.sections]
    assert "Data Validation" in headings
    assert "Machine Learning Results" in headings


def test_file_export(tmp_path):
    report = ResearchReportGenerator(title="File Study")
    report.add_section("Section A", {"x": 1})
    md_path = tmp_path / "report.md"
    report.to_markdown(str(md_path))
    assert md_path.exists()
