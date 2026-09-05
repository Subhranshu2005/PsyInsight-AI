"""
PsyInsight AI — Report Generator module
==========================================

Assembles outputs from every other module into a single research report
(Markdown / HTML / JSON).

    from psyinsight.reporting import ResearchReportGenerator
"""

from .report_builder import ResearchReportGenerator

__all__ = ["ResearchReportGenerator"]
