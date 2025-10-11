# Temperature Experiment Report

**Date**: 2025-10-11 22:38:40
**Model**: qwen3:14b
**Paper**: 41038680

## Paper Details
**Title**: Modulation of gene expression by traditional Asian antidiabetic nutraceuticals: A review of potential effects.
**Abstract Length**: 1936 characters

## Performance Summary

| Temperature | Time (s) | Interventions | Field Completeness (%) | Avg Mechanism Length | Study Fields | JSON Size |
|-------------|----------|---------------|------------------------|----------------------|--------------|-----------|
| 0.30 | 217.5 | 6 | 50.0 | 5.7 | 6 | 3254 |
| 0.40 | 212.5 | 6 | 50.0 | 7.7 | 6 | 3706 |
| 0.50 | 256.7 | 4 | 50.0 | 8.2 | 6 | 2719 |

## Detailed Analysis

### Temperature: 0.3
- **Extraction Time**: 217.53 seconds
- **Total Interventions**: 6
- **Field Completeness**: 50.0%
- **Average Mechanism Length**: 5.7 words
- **Study Fields Present**: 6
- **JSON Response Size**: 3254 characters

**Sample Extraction** (first intervention):
```json
{
  "intervention_name": "Galohgor herbal beverage",
  "dosage": null,
  "duration": null,
  "frequency": null,
  "intensity": null,
  "administration_route": null,
  "mechanism": "modulates antioxidant enzyme gene expression",
  "correlation_type": "positive",
  "correlation_strength": "moderate",
  "delivery_method": "oral",
  "adverse_effects": null,
  "extraction_confidence": "medium"
}
```

### Temperature: 0.4
- **Extraction Time**: 212.53 seconds
- **Total Interventions**: 6
- **Field Completeness**: 50.0%
- **Average Mechanism Length**: 7.7 words
- **Study Fields Present**: 6
- **JSON Response Size**: 3706 characters

**Sample Extraction** (first intervention):
```json
{
  "intervention_name": "Galohgor herbal beverage",
  "dosage": null,
  "duration": null,
  "frequency": null,
  "intensity": null,
  "administration_route": null,
  "mechanism": "modulation of insulin signaling and antioxidant pathways",
  "correlation_type": "positive",
  "correlation_strength": "moderate",
  "delivery_method": "oral",
  "adverse_effects": null,
  "extraction_confidence": "high"
}
```

### Temperature: 0.5
- **Extraction Time**: 256.67 seconds
- **Total Interventions**: 4
- **Field Completeness**: 50.0%
- **Average Mechanism Length**: 8.2 words
- **Study Fields Present**: 6
- **JSON Response Size**: 2719 characters

**Sample Extraction** (first intervention):
```json
{
  "intervention_name": "Galohgor (Indonesian herbal beverage)",
  "dosage": null,
  "duration": null,
  "frequency": null,
  "intensity": null,
  "administration_route": null,
  "mechanism": "modulation of insulin signaling pathway genes",
  "correlation_type": "positive",
  "correlation_strength": "moderate",
  "delivery_method": "oral",
  "adverse_effects": null,
  "extraction_confidence": "medium"
}
```

## Recommendations

- **Fastest extraction**: Temperature 0.4 (212.5s)
- **Highest quality** (field completeness): Temperature 0.3 (50.0%)
- **Most interventions extracted**: Temperature 0.3 (6 interventions)

**Overall Assessment**:
- Temperature **0.3** recommended for quality (despite -5.0s slower)
- Temperature **0.4** recommended for speed (if quality difference is minimal)