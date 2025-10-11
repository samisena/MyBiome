# Temperature Experiment Report

**Date**: 2025-10-11 22:20:49
**Model**: qwen3:14b
**Paper**: 41038680

## Paper Details
**Title**: Modulation of gene expression by traditional Asian antidiabetic nutraceuticals: A review of potential effects.
**Abstract Length**: 1936 characters

## Performance Summary

| Temperature | Time (s) | Interventions | Field Completeness (%) | Avg Mechanism Length | Study Fields | JSON Size |
|-------------|----------|---------------|------------------------|----------------------|--------------|-----------|
| 0.00 | 192.4 | 5 | 58.3 | 7.6 | 6 | 3161 |
| 0.15 | 251.9 | 6 | 58.3 | 7.3 | 6 | 3583 |
| 0.30 | 224.6 | 6 | 58.3 | 9.3 | 6 | 3803 |

## Detailed Analysis

### Temperature: 0
- **Extraction Time**: 192.41 seconds
- **Total Interventions**: 5
- **Field Completeness**: 58.3%
- **Average Mechanism Length**: 7.6 words
- **Study Fields Present**: 6
- **JSON Response Size**: 3161 characters

**Sample Extraction** (first intervention):
```json
{
  "intervention_name": "Galohgor (Indonesian herbal beverage)",
  "dosage": null,
  "duration": null,
  "frequency": null,
  "intensity": null,
  "administration_route": "oral",
  "mechanism": "modulation of insulin signaling pathways and antioxidant activity",
  "correlation_type": "positive",
  "correlation_strength": "moderate",
  "delivery_method": "oral",
  "adverse_effects": null,
  "extraction_confidence": "medium"
}
```

### Temperature: 0.15
- **Extraction Time**: 251.94 seconds
- **Total Interventions**: 6
- **Field Completeness**: 58.3%
- **Average Mechanism Length**: 7.3 words
- **Study Fields Present**: 6
- **JSON Response Size**: 3583 characters

**Sample Extraction** (first intervention):
```json
{
  "intervention_name": "Galohgor",
  "dosage": null,
  "duration": null,
  "frequency": null,
  "intensity": null,
  "administration_route": "oral",
  "mechanism": "modulation of insulin signaling pathway and antioxidant activity",
  "correlation_type": "positive",
  "correlation_strength": "moderate",
  "delivery_method": "oral",
  "adverse_effects": null,
  "extraction_confidence": "high"
}
```

### Temperature: 0.3
- **Extraction Time**: 224.57 seconds
- **Total Interventions**: 6
- **Field Completeness**: 58.3%
- **Average Mechanism Length**: 9.3 words
- **Study Fields Present**: 6
- **JSON Response Size**: 3803 characters

**Sample Extraction** (first intervention):
```json
{
  "intervention_name": "Galohgor herbal beverage",
  "dosage": null,
  "duration": null,
  "frequency": null,
  "intensity": null,
  "administration_route": "oral",
  "mechanism": "modulation of oxidative stress and glucose metabolism pathways",
  "correlation_type": "positive",
  "correlation_strength": "moderate",
  "delivery_method": "herbal",
  "adverse_effects": null,
  "extraction_confidence": "high"
}
```

## Recommendations

- **Fastest extraction**: Temperature 0 (192.4s)
- **Highest quality** (field completeness): Temperature 0 (58.3%)
- **Most interventions extracted**: Temperature 0.15 (6 interventions)

**Overall Assessment**:
- Temperature **0** provides the best balance of speed and quality