"""
Test script for new hierarchical extraction prompt.
Tests the new two-layer format with study-level and intervention-level fields.
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from back_end.src.data.config import setup_logging
from back_end.src.data.api_clients import get_llm_client
from back_end.src.data.utils import parse_json_safely
from back_end.src.data_collection.database_manager import database_manager

logger = setup_logging(__name__, 'test_hierarchical_prompt.log')


NEW_HIERARCHICAL_PROMPT = """You are a biomedical research expert extracting structured intervention data from medical literature.

TASK: Extract interventions from the paper as a JSON array. Return ONLY the JSON array—no markdown, no explanations.

OUTPUT FORMAT (CRITICAL):
Your response MUST start with [ and end with ]. NO explanations, NO markdown code blocks (```), NO commentary.
WRONG: ```json[{{...}}]``` or "Here are the conditions: [{{...}}]"
CORRECT: [{{...}}]

FORMAT:
[{{
  "health_condition": "primary condition targeted (NOT underlying disease)",
  "study_focus": ["research question 1", "research question 2"],
  "measured_metrics": ["specific measurement 1", "specific measurement 2"],
  "findings": ["key result 1", "key result 2"],
  "study_location": "string or null",
  "publisher": "journal name or null",
  "sample_size": number or null,
  "study_duration": "string or null",
  "study_type": "string or null",
  "population_details": "string or null",
  "interventions": [{{
    "intervention_name": "specific name",
    "dosage": "string or null",
    "duration": "string or null",
    "frequency": "string or null",
    "intensity": "string or null",
    "administration_route": "string or null",
    "mechanism": "concise HOW it works (3-10 words)",
    "correlation_type": "positive|negative|neutral|inconclusive",
    "correlation_strength": "very strong|strong|moderate|weak|very weak|null",
    "delivery_method": "oral|injection|topical|behavioral|etc or null",
    "adverse_effects": "string or null",
    "extraction_confidence": "very high|high|medium|low|very low"
  }}]
}}]

KEY DISTINCTIONS:
- study_focus = WHAT they studied (research questions)
- measured_metrics = HOW they measured it (tools/instruments)
- findings = WHAT they found (results with data when available)

RULES:
1. One entry per condition studied
2. Multiple interventions for same condition → same entry, multiple objects in interventions array
3. Multiple conditions studied → separate entries
4. health_condition = specific complication, NOT underlying disease (e.g., "diabetes foot" not "diabetes")
5. mechanism = biological/behavioral pathway, NOT intervention name repeated
6. Extract 2-5 key findings with quantitative data when available
7. Return [] if no interventions found

EXAMPLES:

Paper: "Family-centered training for diabetes foot prevention in Jodhpur"
Abstract: "RCT with 54 diabetic patients per group. Intervention included family training for foot care. Knowledge scores: 13.4±1.2 vs 9.9±2.7 (p<0.001). Practice scores: 7.9±1.4 vs 6.2±1.3 (p<0.001). Foot ulcers: 0 vs 4 (8%)."

[{{
  "health_condition": "diabetes foot",
  "study_focus": ["foot care knowledge improvement", "foot ulcer prevention"],
  "measured_metrics": ["foot care knowledge scores", "practice scores", "foot ulcer incidence"],
  "findings": ["knowledge scores higher in intervention group (13.4±1.2 vs 9.9±2.7, p<0.001)", "practice scores higher (7.9±1.4 vs 6.2±1.3, p<0.001)", "zero ulcers in intervention vs 4 (8%) in control"],
  "study_location": "India",
  "publisher": null,
  "sample_size": 54,
  "study_duration": "9 months",
  "study_type": "randomized controlled trial",
  "population_details": "diabetic patients aged 18-60 and family members",
  "interventions": [{{
    "intervention_name": "family-centered foot care training",
    "dosage": null,
    "duration": "9 months",
    "frequency": null,
    "intensity": null,
    "administration_route": null,
    "mechanism": "improved adherence through family education",
    "correlation_type": "positive",
    "correlation_strength": "strong",
    "delivery_method": "behavioral",
    "adverse_effects": null,
    "extraction_confidence": "very high"
  }}]
}}]

---

Paper: "Probiotics for IBS: 8-week RCT"
Abstract: "120 IBS patients received Lactobacillus plantarum 10^9 CFU daily or placebo. Pain scores improved significantly (p<0.001). Microbiota analysis showed increased beneficial bacteria."

[{{
  "health_condition": "irritable bowel syndrome",
  "study_focus": ["probiotic efficacy on IBS symptoms", "gut microbiota changes"],
  "measured_metrics": ["abdominal pain scores", "gut microbiota composition", "inflammatory markers"],
  "findings": ["pain scores improved significantly (p<0.001)", "increased beneficial bacteria", "reduced inflammatory markers"],
  "study_location": null,
  "publisher": null,
  "sample_size": 120,
  "study_duration": "8 weeks",
  "study_type": "randomized controlled trial",
  "population_details": "IBS patients",
  "interventions": [{{
    "intervention_name": "Lactobacillus plantarum probiotic",
    "dosage": "10^9 CFU daily",
    "duration": "8 weeks",
    "frequency": "daily",
    "intensity": null,
    "administration_route": "oral",
    "mechanism": "gut microbiome modulation",
    "correlation_type": "positive",
    "correlation_strength": "strong",
    "delivery_method": "oral",
    "adverse_effects": null,
    "extraction_confidence": "very high"
  }}]
}}]

---

Paper: "Exercise vs diet vs combined for type 2 diabetes"
Abstract: "200 patients (50 per group). HbA1c reduction: exercise 0.8% (p<0.01), diet 1.1% (p<0.001), combined 1.5% (p<0.001). Weight loss: -3.1kg, -4.5kg, -8.2kg respectively."

[{{
  "health_condition": "type 2 diabetes",
  "study_focus": ["glycemic control efficacy", "weight management"],
  "measured_metrics": ["HbA1c levels", "body weight", "quality of life scores"],
  "findings": ["exercise reduced HbA1c by 0.8% (p<0.01)", "diet reduced HbA1c by 1.1% (p<0.001)", "combined reduced HbA1c by 1.5% (p<0.001)", "combined achieved greatest weight loss (-8.2kg)"],
  "study_location": null,
  "publisher": "Diabetes Care",
  "sample_size": 200,
  "study_duration": "12 months",
  "study_type": "randomized controlled trial",
  "population_details": "adults with type 2 diabetes",
  "interventions": [
    {{
      "intervention_name": "aerobic exercise",
      "dosage": null,
      "duration": "12 months",
      "frequency": null,
      "intensity": "moderate",
      "administration_route": null,
      "mechanism": "improved insulin sensitivity",
      "correlation_type": "positive",
      "correlation_strength": "moderate",
      "delivery_method": "behavioral",
      "adverse_effects": null,
      "extraction_confidence": "very high"
    }},
    {{
      "intervention_name": "Mediterranean diet",
      "dosage": null,
      "duration": "12 months",
      "frequency": "daily",
      "intensity": null,
      "administration_route": null,
      "mechanism": "reduced inflammation and improved metabolism",
      "correlation_type": "positive",
      "correlation_strength": "strong",
      "delivery_method": "dietary",
      "adverse_effects": null,
      "extraction_confidence": "very high"
    }},
    {{
      "intervention_name": "combined exercise and diet",
      "dosage": null,
      "duration": "12 months",
      "frequency": "daily",
      "intensity": "moderate",
      "administration_route": null,
      "mechanism": "synergistic metabolic improvement",
      "correlation_type": "positive",
      "correlation_strength": "very strong",
      "delivery_method": "behavioral",
      "adverse_effects": null,
      "extraction_confidence": "very high"
    }}
  ]
}}]

PAPER:
{paper_content}
"""


SYSTEM_MESSAGE = "Provide only the final JSON output without showing your reasoning process or using <think> tags. Start your response immediately with the [ character."


def create_prompt_for_paper(paper):
    """Create the full prompt for a paper."""
    content_sections = []
    content_sections.append(f"Title: {paper['title']}")
    content_sections.append(f"Abstract: {paper['abstract']}")

    paper_content = "\n\n".join(content_sections)
    return NEW_HIERARCHICAL_PROMPT.format(paper_content=paper_content)


def flatten_hierarchical_to_interventions(hierarchical_data, paper_pmid):
    """
    Convert hierarchical format to flat intervention records for database.

    Input format:
    [{
      "health_condition": "...",
      "study_focus": [...],
      "measured_metrics": [...],
      "findings": [...],
      "study_location": "...",
      "publisher": "...",
      "sample_size": ...,
      "study_duration": "...",
      "study_type": "...",
      "population_details": "...",
      "interventions": [...]
    }]

    Output format:
    [{
      "intervention_name": "...",
      "health_condition": "...",
      "study_focus": [...],  # NEW
      "measured_metrics": [...],  # NEW
      "findings": [...],  # NEW
      "study_location": "...",  # NEW
      "publisher": "...",  # NEW
      ...all other intervention fields...
    }]
    """
    flat_interventions = []

    for condition_entry in hierarchical_data:
        # Extract study-level fields
        health_condition = condition_entry.get('health_condition')
        study_focus = condition_entry.get('study_focus', [])
        measured_metrics = condition_entry.get('measured_metrics', [])
        findings = condition_entry.get('findings', [])
        study_location = condition_entry.get('study_location')
        publisher = condition_entry.get('publisher')
        sample_size = condition_entry.get('sample_size')
        study_duration = condition_entry.get('study_duration')
        study_type = condition_entry.get('study_type')
        population_details = condition_entry.get('population_details')

        # Extract intervention-level array
        interventions = condition_entry.get('interventions', [])

        # Flatten: create one record per intervention, copying study-level fields
        for intervention in interventions:
            flat_record = {
                # Intervention-level fields
                'intervention_name': intervention.get('intervention_name'),
                'dosage': intervention.get('dosage'),
                'duration': intervention.get('duration'),
                'frequency': intervention.get('frequency'),
                'intensity': intervention.get('intensity'),
                'administration_route': intervention.get('administration_route'),
                'mechanism': intervention.get('mechanism'),
                'correlation_type': intervention.get('correlation_type'),
                'correlation_strength': intervention.get('correlation_strength'),
                'delivery_method': intervention.get('delivery_method'),
                'adverse_effects': intervention.get('adverse_effects'),
                'extraction_confidence': intervention.get('extraction_confidence'),

                # Study-level fields (copied to each intervention)
                'health_condition': health_condition,
                'study_focus': study_focus,  # NEW
                'measured_metrics': measured_metrics,  # NEW
                'findings': findings,  # NEW
                'study_location': study_location,  # NEW
                'publisher': publisher,  # NEW
                'sample_size': sample_size,
                'study_duration': study_duration,
                'study_type': study_type,
                'population_details': population_details,

                # Metadata
                'paper_id': paper_pmid
            }

            flat_interventions.append(flat_record)

    return flat_interventions


def test_prompt_on_paper(paper, client):
    """Test the new prompt on a single paper."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing paper PMID: {paper['pmid']}")
    logger.info(f"Title: {paper['title'][:100]}...")
    logger.info(f"{'='*80}\n")

    # Create prompt
    prompt = create_prompt_for_paper(paper)

    # Call LLM
    logger.info("Calling LLM with new hierarchical prompt...")
    response = client.generate(
        prompt=prompt,
        temperature=0.3,
        max_tokens=8000,
        system_message=SYSTEM_MESSAGE
    )

    response_text = response.get('content', '')

    # Show raw response
    logger.info(f"\nRaw LLM Response:\n{response_text}\n")

    # Parse JSON
    logger.info("Parsing JSON response...")
    try:
        hierarchical_data = parse_json_safely(response_text, paper['pmid'])

        if not hierarchical_data:
            logger.error("Failed to parse JSON or empty response")
            return None

        logger.info(f"Successfully parsed {len(hierarchical_data)} condition entries")

        # Display hierarchical structure
        for i, entry in enumerate(hierarchical_data, 1):
            logger.info(f"\n--- Condition Entry {i} ---")
            logger.info(f"  health_condition: {entry.get('health_condition')}")
            logger.info(f"  study_focus: {entry.get('study_focus')}")
            logger.info(f"  measured_metrics: {entry.get('measured_metrics')}")
            logger.info(f"  findings: {entry.get('findings')}")
            logger.info(f"  study_location: {entry.get('study_location')}")
            logger.info(f"  publisher: {entry.get('publisher')}")
            logger.info(f"  sample_size: {entry.get('sample_size')}")
            logger.info(f"  study_duration: {entry.get('study_duration')}")
            logger.info(f"  study_type: {entry.get('study_type')}")
            logger.info(f"  population_details: {entry.get('population_details')}")
            logger.info(f"  interventions: {len(entry.get('interventions', []))} intervention(s)")

            for j, intervention in enumerate(entry.get('interventions', []), 1):
                logger.info(f"    [{j}] {intervention.get('intervention_name')} - {intervention.get('mechanism')}")

        # Flatten to database format
        logger.info("\n\nFlattening to database format...")
        flat_interventions = flatten_hierarchical_to_interventions(hierarchical_data, paper['pmid'])
        logger.info(f"Created {len(flat_interventions)} flat intervention records")

        # Display flat records
        for i, record in enumerate(flat_interventions, 1):
            logger.info(f"\n--- Flat Record {i} ---")
            logger.info(f"  intervention_name: {record.get('intervention_name')}")
            logger.info(f"  health_condition: {record.get('health_condition')}")
            logger.info(f"  mechanism: {record.get('mechanism')}")
            logger.info(f"  study_focus: {record.get('study_focus')}")
            logger.info(f"  measured_metrics: {record.get('measured_metrics')}")
            logger.info(f"  findings: {record.get('findings')}")
            logger.info(f"  study_location: {record.get('study_location')}")
            logger.info(f"  publisher: {record.get('publisher')}")

        return {
            'hierarchical': hierarchical_data,
            'flat': flat_interventions
        }

    except Exception as e:
        logger.error(f"Error processing response: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Test the new hierarchical prompt on sample papers."""
    print("Starting hierarchical prompt test")
    logger.info("Starting hierarchical prompt test\n")

    # Get LLM client
    print("Initializing LLM client (qwen3:14b)...")
    logger.info("Initializing LLM client (qwen3:14b)...")
    client = get_llm_client('qwen3:14b')
    print("LLM client initialized")

    # Get sample papers from database
    print("Fetching sample papers from database...")
    logger.info("Fetching sample papers from database...")
    papers = database_manager.get_all_papers(limit=1)  # Test on just 1 paper

    if not papers:
        print("No papers found in database")
        logger.error("No papers found in database")
        return

    print(f"Found {len(papers)} paper to test")
    logger.info(f"Found {len(papers)} paper to test\n")

    # Test on each paper
    results = []
    for paper in papers:
        print(f"\nTesting paper: {paper['pmid']}")
        result = test_prompt_on_paper(paper, client)
        if result:
            results.append({
                'pmid': paper['pmid'],
                'title': paper['title'],
                'result': result
            })
            print(f"[OK] Successfully extracted {len(result['flat'])} interventions")

        logger.info("\n" + "="*80 + "\n")

    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    logger.info(f"\n{'='*80}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Papers tested: {len(papers)}")
    logger.info(f"Successful extractions: {len(results)}")
    logger.info(f"Failed extractions: {len(papers) - len(results)}")

    if results:
        total_flat_interventions = sum(len(r['result']['flat']) for r in results)
        print(f"Papers tested: {len(papers)}")
        print(f"Successful extractions: {len(results)}")
        print(f"Total flat interventions created: {total_flat_interventions}")
        print(f"\n[SUCCESS] New hierarchical prompt is working correctly!")
        logger.info(f"Total flat interventions created: {total_flat_interventions}")
        logger.info(f"\n[SUCCESS] New hierarchical prompt is working correctly!")
    else:
        print("[FAIL] All extractions failed - prompt needs debugging")
        logger.error("[FAIL] All extractions failed - prompt needs debugging")


if __name__ == '__main__':
    main()
