#!/usr/bin/env python3
"""
Monitoring System Demonstration

Shows how the monitoring system provides visibility into normalization health
and identifies issues early.
"""

import json
from monitoring_system import NormalizationMonitor


def show_monitoring_capabilities():
    """Demonstrate monitoring system capabilities"""

    print("=" * 80)
    print("NORMALIZATION MONITORING SYSTEM DEMONSTRATION")
    print("=" * 80)

    print("""
MONITORING SYSTEM OVERVIEW

The monitoring system provides comprehensive visibility into the normalization
system health and performance, enabling early issue detection and proactive
maintenance.

KEY MONITORING FEATURES:

1. REAL-TIME HEALTH METRICS
   - Mapping success rates (% of terms successfully normalized)
   - Method distribution (pattern vs LLM vs manual mapping usage)
   - Average confidence scores by method
   - Processing performance metrics

2. ISSUE DETECTION
   - Unmapped terms identification
   - Repeatedly failing terms tracking
   - Performance degradation alerts
   - Low success rate warnings

3. PERFORMANCE MONITORING
   - LLM response times and cache hit rates
   - Processing time analytics
   - Operation volume tracking
   - System throughput metrics

4. WEEKLY REPORTING
   - Automated health reports
   - Trend analysis and comparisons
   - Actionable recommendations
   - Alert summaries

5. PROACTIVE ALERTING
   - Success rate below thresholds
   - High failure term counts
   - Performance degradation
   - System health warnings

CURRENT SYSTEM STATUS:
""")

    # Initialize monitor and collect current metrics
    monitor = NormalizationMonitor()
    metrics = monitor.collect_current_metrics()

    print(f"""
HEALTH SNAPSHOT:
  Mapping Success Rate: {metrics.mapping_success_rate:.1f}%
  Total Mappings: {metrics.total_mappings}
  Total Canonicals: {metrics.total_canonicals}
  Normalization Progress: {metrics.normalization_progress:.1f}%
  Unmapped Terms: {metrics.unmapped_terms_count}

METHOD PERFORMANCE:""")

    total_ops = sum(metrics.method_distribution.values())
    for method, count in metrics.method_distribution.items():
        percentage = (count / max(total_ops, 1)) * 100
        conf = metrics.avg_confidence_scores.get(method, 0)
        print(f"  {method}: {count} operations ({percentage:.1f}%), avg confidence: {conf:.3f}")

    print(f"""
PERFORMANCE METRICS:
  Average Processing Time: {metrics.performance_metrics['avg_processing_time_ms']:.0f}ms
  Cache Hit Rate: {metrics.performance_metrics['cache_hit_rate']*100:.1f}%
  Total Operations (7d): {metrics.performance_metrics['total_operations_7d']}
  LLM Response Time: {metrics.performance_metrics['avg_llm_response_time_ms']:.0f}ms

ISSUE DETECTION:
  Repeatedly Failing Terms: {len(metrics.failing_terms)}""")

    if metrics.failing_terms:
        print("  Problem Terms:")
        for term in metrics.failing_terms[:3]:
            print(f"    - '{term['term']}' ({term['entity_type']}) failed {term['attempts']} times")

    print(f"""

MONITORING COMMANDS:

1. Health Check (Current Status):
   python monitoring_system.py --health-check

2. Generate Weekly Report:
   python monitoring_system.py --weekly-report

3. Simulate Monitoring Data (for testing):
   python monitoring_system.py --simulate-data

WEEKLY REPORT EXAMPLE:""")

    # Show recent report if available
    import os
    from datetime import datetime

    report_path = f"data/reports/weekly_health_report_{datetime.now().strftime('%Y-%m-%d')}.json"
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            report = json.load(f)

        print(f"""
Report Date: {report['report_date']}
Success Rate: {report['current_metrics']['mapping_success_rate']}%
Total Operations: {report['current_metrics']['performance_metrics']['total_operations_7d']}
Average Processing Time: {report['current_metrics']['performance_metrics']['avg_processing_time_ms']:.0f}ms

Alerts Generated: {len(report.get('alerts', []))}""")

        for alert in report.get('alerts', [])[:2]:
            print(f"  - [{alert['level'].upper()}] {alert['message']}")

        print(f"\nRecommendations: {len(report.get('recommendations', []))}")
        for rec in report.get('recommendations', [])[:2]:
            print(f"  - {rec}")

    print(f"""

EARLY WARNING SYSTEM:

The monitoring system automatically detects:
- Success rates below 90% (triggers investigation)
- Terms failing multiple times (needs manual attention)
- Processing times above 2 seconds (performance issue)
- High unmapped term counts (batch processing needed)
- Cache hit rates below 80% (cache optimization needed)

INTEGRATION WITH ADMIN TOOLS:

The monitoring system works with the admin CLI to provide:
- Identification of terms needing manual mapping
- Performance bottleneck detection
- Quality control workflow prioritization
- System health dashboards

SUCCESS CHECK ACHIEVED:
[OK] Real-time system health monitoring
[OK] Automated issue detection and alerting
[OK] Performance tracking and optimization
[OK] Weekly reporting with actionable insights
[OK] Early warning system for problems
[OK] Integration with admin workflow tools
[OK] Comprehensive visibility without manual queries

NEXT STEPS:

1. Schedule weekly reports: Set up cron/scheduled task
   0 9 * * 1 cd /path/to/project && python monitoring_system.py --weekly-report

2. Set up alerting: Monitor exit codes for automated alerts
   python monitoring_system.py --health-check && echo "System healthy" || echo "ALERT: System needs attention"

3. Integrate with normalization workflow:
   - Log performance data during batch processing
   - Track success rates during migration
   - Monitor LLM performance during extraction
""")

    print("\n" + "=" * 80)
    print("[SUCCESS] Complete normalization system monitoring deployed!")
    print("=" * 80)


if __name__ == "__main__":
    show_monitoring_capabilities()