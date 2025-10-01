# MyBiome Pipeline - Complete Data Flow Analysis

**Generated:** 2025-10-01
**Purpose:** Comprehensive trace of execution path and data flow through the entire MyBiome research pipeline

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Entry Point Analysis](#entry-point-analysis)
3. [Phase 1: Collection](#phase-1-collection)
4. [Phase 2: Processing](#phase-2-processing)
5. [Phase 3: Deduplication](#phase-3-deduplication)
6. [Data Structures](#data-structures)
7. [Database Schema](#database-schema)
8. [Function Call Chains](#function-call-chains)
9. [Error Handling Paths](#error-handling-paths)

---

## Executive Summary

The MyBiome pipeline follows a **three-phase batch architecture** for processing biomedical research papers:

1. **Collection Phase**: Parallel paper collection from PubMed for 60 medical conditions
2. **Processing Phase**: Sequential dual-model LLM extraction (gemma2:9b → qwen2.5:14b)
3. **Deduplication Phase**: LLM-based entity merging and canonical mapping

**Key Characteristics:**
- Fully resumable with JSON session persistence
- Processes ALL unprocessed items in each phase
- No condition-by-condition iteration (pure batch mode)
- Quality gates between phases
- Comprehensive error recovery

---

## Entry Point Analysis

### Command Line Execution
```bash
python -m back_end.src.orchestration.batch_medical_rotation --papers-per-condition 10
```

### File: `batch_medical_rotation.py`

#### Entry Function: `main()` (Lines 532-606)
```python
def main():
    # 1. Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--papers-per-condition', type=int, default=10)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--start-phase', choices=['collection', 'processing', 'deduplication'])
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()

    # 2. Create pipeline instance
    pipeline = BatchMedicalRotationPipeline()

    # 3. Execute based on args
    if args.status:
        status = pipeline.get_status()
        print(json.dumps(status, indent=2))
    else:
        result = pipeline.run_batch_pipeline(
            papers_per_condition=args.papers_per_condition,
            resume=args.resume,
            start_phase=args.start_phase
        )
```

#### Pipeline Initialization: `BatchMedicalRotationPipeline.__init__()` (Lines 109-127)
```python
def __init__(self):
    # Initialize phase orchestrators
    self.paper_collector = RotationPaperCollector()
    self.llm_processor = RotationLLMProcessor()
    self.dedup_integrator = RotationDeduplicationIntegrator()

    # Session management
    self.session_file = Path(config.data_root) / "batch_session.json"
    self.current_session: Optional[BatchSession] = None

    # Signal handling for graceful shutdown
    signal.signal(signal.SIGINT, self._signal_handler)
    signal.signal(signal.SIGTERM, self._signal_handler)
```

**Data Structures Created:**
- `BatchSession`: Tracks pipeline state, statistics, and phase completion
- Session file path: `back_end/data/batch_session.json`

---

## Phase 1: Collection

### Orchestrator: `batch_medical_rotation.py`

#### Entry: `run_batch_pipeline()` → `_run_collection_phase()` (Lines 351-394)

```python
def _run_collection_phase(self, session: BatchSession) -> Dict[str, Any]:
    # 1. Call batch collector
    collection_result = self.paper_collector.collect_all_conditions_batch(
        papers_per_condition=session.papers_per_condition,
        min_year=2015
    )

    # 2. Update session statistics
    session.collection_result = {
        'total_conditions': 60,
        'successful_conditions': collection_result.successful_conditions,
        'failed_conditions': collection_result.failed_conditions,
        'total_papers_collected': collection_result.total_papers_collected,
        'collection_time_seconds': collection_result.total_collection_time_seconds,
        'quality_gate_passed': collection_result.success
    }

    # 3. Quality gate check (80% success rate required)
    if not collection_result.success:
        return {'success': False, 'error': collection_result.error}

    # 4. Mark phase complete
    session.collection_completed = True
    session.current_phase = BatchPhase.PROCESSING
```

### Collector: `rotation_paper_collector.py`

#### Entry: `collect_all_conditions_batch()` (Lines 89-200)

```python
def collect_all_conditions_batch(self, papers_per_condition: int = 10,
                                min_year: int = 2015) -> BatchCollectionResult:
    # 1. Get all 60 medical conditions
    all_conditions = self.get_all_conditions()  # Returns list of 60 conditions

    # 2. Parallel collection using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Submit all condition collection tasks
        future_to_condition = {
            executor.submit(
                self._collect_single_condition_without_s2,
                condition, papers_per_condition, min_year, max_year
            ): condition for condition in all_conditions
        }

        # 3. Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_condition):
            result = future.result()
            if result['success']:
                successful_conditions += 1
                total_papers_collected += result['papers_collected']

    # 4. Quality gate check
    success_rate = (successful_conditions / 60) * 100
    quality_threshold = 80.0

    return BatchCollectionResult(
        total_conditions=60,
        successful_conditions=successful_conditions,
        failed_conditions=failed_conditions,
        total_papers_collected=total_papers_collected,
        success=success_rate >= quality_threshold
    )
```

#### Per-Condition Collection: `_collect_single_condition_without_s2()` (Lines 202-253)

```python
def _collect_single_condition_without_s2(self, condition: str, target_count: int,
                                       min_year: int, max_year: Optional[int]):
    # 1. Call with retry logic
    collection_result = self._collect_with_retry(
        condition, target_count, min_year, max_year
    )

    return {
        'success': True,
        'condition': condition,
        'papers_collected': collection_result['papers_collected'],
        'total_papers': collection_result['papers_collected']
    }
```

#### Retry Logic: `_collect_with_retry()` (Lines 257-312)

```python
def _collect_with_retry(self, condition: str, needed_papers: int,
                       min_year: int, max_year: Optional[int]) -> Dict[str, Any]:
    for attempt in range(self.max_retries):  # 3 attempts
        try:
            # Call PubMed collector (S2 disabled in batch mode)
            result = self.pubmed_collector.collect_interventions_by_condition(
                condition=condition,
                min_year=min_year,
                max_year=max_year,
                max_results=needed_papers,
                include_fulltext=True,
                use_interleaved_s2=False  # CRITICAL: S2 disabled
            )

            papers_collected = result.get('paper_count', 0)
            if papers_collected > 0:
                return {
                    'papers_collected': papers_collected,
                    'pubmed_stats': result.get('pubmed_stats', {})
                }
```

### PubMed Collector: `pubmed_collector.py`

#### Entry: `collect_interventions_by_condition()` (Lines 116-160)

```python
def collect_interventions_by_condition(self, condition: str, min_year: int = 2010,
                                      max_year: Optional[int] = None,
                                      max_results: int = 100,
                                      include_fulltext: bool = True,
                                      use_interleaved_s2: bool = True):
    # 1. Build enhanced search query
    query = self._build_intervention_query(condition, include_study_filter=True)

    # 2. Route based on S2 setting
    if use_interleaved_s2:
        return self._collect_with_interleaved_s2(...)  # Not used in batch
    else:
        return self._collect_traditional_batch(...)  # Used in batch mode
```

#### Traditional Batch Collection: `_collect_traditional_batch()` (Lines 411-524)

```python
def _collect_traditional_batch(self, condition: str, query: str, min_year: int,
                             max_year: Optional[int], max_results: int, include_fulltext: bool):
    for attempt in range(max_search_attempts):  # Up to 5 attempts
        # 1. Search PubMed
        pmid_list = self._search_papers_with_offset(query, min_year, current_search_size, max_year, search_offset)

        # 2. Filter existing papers
        new_pmids = self._filter_existing_papers(pmid_list)

        # 3. Fetch metadata
        metadata_file = self.fetch_papers_metadata(pmids_to_process)

        # 4. Parse XML and store papers
        batch_papers = self.parser.parse_metadata_file(str(metadata_file))
        all_new_papers.extend(batch_papers)

        # 5. Keep XML file for verification (cleanup after success)
        metadata_files.append(str(metadata_file))

    # 6. Retrieve fulltext if requested
    if include_fulltext:
        fulltext_stats = self._process_fulltext_batch(all_new_papers)

    # 7. Clean up XML files
    for xml_file in metadata_files:
        Path(xml_file).unlink()

    return {
        'condition': condition,
        'paper_count': len(all_new_papers),
        'papers_stored': len(all_new_papers),
        'status': 'success'
    }
```

### Paper Parser: `paper_parser.py`

#### Entry: `parse_metadata_file()` (Lines 38-101)

```python
def parse_metadata_file(self, file_path: str, batch_size: int = 50) -> List[Dict]:
    # 1. Parse XML
    tree = ET.parse(file_path)
    root = tree.getroot()
    articles = root.findall(".//PubmedArticle")

    # 2. Process in batches
    batches = batch_process(articles, batch_size)

    for batch in batches:
        batch_papers = []
        for article in batch:
            paper = self._parse_single_article(article)
            if paper:
                batch_papers.append(paper)

        # 3. Insert batch to database
        batch_inserted, batch_skipped = self._insert_papers_batch(batch_papers)
        all_papers.extend(batch_papers)

    return all_papers
```

#### Single Article Parsing: `_parse_single_article()` (Lines 103-172)

```python
def _parse_single_article(self, article: ET.Element) -> Optional[Dict]:
    # Extract metadata
    pmid = article.find(".//PMID").text.strip()
    title = article.find(".//ArticleTitle").text.strip()
    abstract = self._extract_abstract(article_meta)
    journal = self._extract_journal_info(article_meta)
    publication_date = self._extract_publication_date(article_meta)
    doi = self._extract_doi(article)
    pmc_id = self._extract_pmc_id(article)
    keywords = self._extract_keywords(article)

    return {
        'pmid': pmid,
        'title': title,
        'abstract': abstract,
        'journal': journal,
        'publication_date': publication_date,
        'doi': doi,
        'pmc_id': pmc_id,
        'keywords': keywords,
        'has_fulltext': False,
        'fulltext_source': None,
        'fulltext_path': None,
        'discovery_source': 'pubmed'
    }
```

### Database Manager: `database_manager.py`

#### Entry: `insert_paper()` (Lines 505-557)

```python
def insert_paper(self, paper: Dict) -> bool:
    # 1. Validate paper data
    validation_result = validation_manager.validate_paper(paper)
    if not validation_result.is_valid:
        raise ValueError(f"Paper validation failed")
    validated_paper = validation_result.cleaned_data

    # 2. Insert to database
    with self.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR IGNORE INTO papers
            (pmid, title, abstract, journal, publication_date, doi, pmc_id,
             keywords, has_fulltext, fulltext_source, fulltext_path,
             processing_status, discovery_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            validated_paper['pmid'],
            validated_paper['title'],
            validated_paper['abstract'],
            # ... all fields
            'pending',  # processing_status
            'pubmed'    # discovery_source
        ))

        was_new = cursor.rowcount > 0
        conn.commit()

        # 3. Cleanup temporary files if successful
        if was_new:
            self._cleanup_paper_files(validated_paper['pmid'])

        return was_new
```

**Collection Phase Output:**
- Papers inserted into `papers` table with `processing_status = 'pending'`
- Session updated with collection statistics
- XML metadata files deleted after successful database insertion

---

## Phase 2: Processing

### Orchestrator: `batch_medical_rotation.py`

#### Entry: `run_batch_pipeline()` → `_run_processing_phase()` (Lines 396-440)

```python
def _run_processing_phase(self, session: BatchSession) -> Dict[str, Any]:
    # 1. Call batch LLM processor
    processing_result = self.llm_processor.process_all_papers_batch()

    # 2. Update session statistics
    session.processing_result = {
        'total_papers_found': processing_result.get('total_papers_found', 0),
        'papers_processed': processing_result.get('papers_processed', 0),
        'papers_failed': processing_result.get('papers_failed', 0),
        'interventions_extracted': processing_result.get('interventions_extracted', 0),
        'processing_time_seconds': processing_result.get('processing_time_seconds', 0),
        'success_rate': processing_result.get('success_rate', 0),
        'model_statistics': processing_result.get('model_statistics', {})
    }

    # 3. Mark phase complete
    session.processing_completed = True
    session.current_phase = BatchPhase.DEDUPLICATION
```

### LLM Processor: `rotation_llm_processor.py`

#### Entry: `process_all_papers_batch()` (Lines 318-419)

```python
def process_all_papers_batch(self, batch_size: Optional[int] = None) -> Dict[str, Any]:
    # 1. Get all unprocessed papers
    unprocessed_papers = self._get_all_unprocessed_papers()

    if not unprocessed_papers:
        return {'success': True, 'papers_processed': 0}

    # 2. Thermal safety check
    is_safe, thermal_status = self.thermal_monitor.is_thermal_safe()
    if not is_safe:
        self.thermal_monitor.wait_for_cooling()

    # 3. Determine optimal batch size
    if batch_size is None:
        batch_size = self.get_optimal_batch_size()  # GPU memory optimized

    # 4. Process using dual-model analyzer
    processing_result = self.dual_analyzer.process_papers_batch(
        papers=unprocessed_papers,
        save_to_db=True,
        batch_size=batch_size
    )

    # 5. Compile results
    return {
        'success': True,
        'total_papers_found': len(unprocessed_papers),
        'papers_processed': processing_result['successful_papers'],
        'papers_failed': len(processing_result['failed_papers']),
        'interventions_extracted': processing_result['total_interventions'],
        'model_statistics': processing_result['model_statistics'],
        'interventions_by_category': processing_result['interventions_by_category']
    }
```

#### Get Unprocessed Papers: `_get_all_unprocessed_papers()` (Lines 421-429)

```python
def _get_all_unprocessed_papers(self) -> List[Dict[str, Any]]:
    # Delegate to dual_model_analyzer
    return self.dual_analyzer.get_unprocessed_papers()
```

### Dual Model Analyzer: `dual_model_analyzer.py`

#### Entry: `get_unprocessed_papers()` (Lines 460-495)

```python
def get_unprocessed_papers(self, limit: Optional[int] = None) -> List[Dict]:
    # Get papers NOT processed by ANY of our models
    with self.repository_mgr.db_manager.get_connection() as conn:
        cursor = conn.cursor()

        model_names = list(self.models.keys())  # ['gemma2:9b', 'qwen2.5:14b']
        placeholders = ','.join(['?' for _ in model_names])

        query = f'''
            SELECT DISTINCT p.*
            FROM papers p
            WHERE p.abstract IS NOT NULL
              AND p.abstract != ''
              AND (p.processing_status IS NULL OR p.processing_status != 'failed')
              AND p.pmid NOT IN (
                  SELECT DISTINCT paper_id
                  FROM interventions
                  WHERE extraction_model IN ({placeholders})
              )
            ORDER BY
                COALESCE(p.influence_score, 0) DESC,
                COALESCE(p.citation_count, 0) DESC,
                p.publication_date DESC
        '''

        cursor.execute(query, model_names)
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
```

**Key Logic**: A paper is "unprocessed" if it has NO interventions from ANY model. This ensures each paper is processed exactly once by both models.

#### Entry: `process_papers_batch()` (Lines 346-441)

```python
def process_papers_batch(self, papers: List[Dict], save_to_db: bool = True,
                       batch_size: int = None) -> Dict[str, Any]:
    # 1. Optimize batch size for GPU
    if batch_size is None:
        batch_size = self.gpu_optimization.get('optimal_batch_size', 3)
    optimized_batch_size = self._optimize_batch_size_for_memory(batch_size)

    # 2. Process in batches
    batches = batch_process(papers, optimized_batch_size)

    for batch in batches:
        for paper in batch:
            # 3. Extract interventions with dual models
            results = self.extract_interventions(paper)

            # 4. Save to database
            if save_to_db and results.get('total_interventions', 0) > 0:
                self._save_interventions_batch(results.get('interventions', []))

            # Rate limiting
            time.sleep(1.0)

    return {
        'total_papers': len(papers),
        'successful_papers': total_processed - len(failed_papers),
        'failed_papers': failed_papers,
        'total_interventions': total_interventions,
        'interventions_by_category': category_counts,
        'model_statistics': model_stats
    }
```

#### Single Paper Extraction: `extract_interventions()` (Lines 243-311)

```python
def extract_interventions(self, paper: Dict) -> Dict[str, Any]:
    pmid = paper.get('pmid', 'unknown')

    # 1. Validate input
    if not paper.get('abstract') or not paper.get('title'):
        return {'pmid': pmid, 'models': {}, 'total_interventions': 0}

    # 2. Update processing status
    self.repository_mgr.papers.update_processing_status(pmid, 'processing')

    # 3. Run both models sequentially
    model_results = {}
    all_interventions = []

    for model_name in ['gemma2:9b', 'qwen2.5:14b']:
        result = self.extract_with_single_model(paper, model_name)

        # 4. Validate and enhance interventions
        if result.interventions and not result.error:
            validated_interventions = self._validate_and_enhance_interventions(
                result.interventions, paper, model_name
            )
            result.interventions = validated_interventions
            all_interventions.extend(validated_interventions)

        model_results[model_name] = result
        time.sleep(0.5)  # Delay between models

    # 5. Return raw results (no consensus yet)
    return {
        'pmid': pmid,
        'models': model_results,
        'total_interventions': len(all_interventions),
        'interventions': all_interventions,  # All raw extractions
        'consensus_processed': False  # Flag for deduplication phase
    }
```

#### Single Model Extraction: `extract_with_single_model()` (Lines 180-241)

```python
@handle_llm_errors("extract with single model", max_retries=3)
def extract_with_single_model(self, paper: Dict, model_name: str) -> ModelResult:
    pmid = paper.get('pmid', 'unknown')
    start_time = time.time()

    try:
        model_config = self.models[model_name]
        client = model_config['client']

        # 1. Create prompt
        prompt = self.prompt_service.create_extraction_prompt(paper)

        # 2. Calculate dynamic max_tokens
        dynamic_max_tokens = self._calculate_dynamic_max_tokens(prompt, model_config)

        # 3. Call LLM
        system_message = self.prompt_service.create_system_message()
        full_prompt = f"{system_message}\n\n{prompt}"

        response = client.generate(
            prompt=full_prompt,
            temperature=0.3,
            max_tokens=dynamic_max_tokens
        )

        # 4. Parse JSON response
        response_text = response.get('content', '')
        interventions = parse_json_safely(response_text, f"{pmid}_{model_name}")

        return ModelResult(
            model_name=model_name,
            interventions=interventions,
            extraction_time=time.time() - start_time
        )
```

#### Validation and Enhancement: `_validate_and_enhance_interventions()` (Lines 313-343)

```python
def _validate_and_enhance_interventions(self, interventions: List[Dict],
                                      paper: Dict, model_name: str) -> List[Dict]:
    validated = []

    for intervention in interventions:
        try:
            # 1. Add metadata
            intervention['paper_id'] = paper['pmid']
            intervention['extraction_model'] = model_name

            # 2. Validate using category validator
            validated_intervention = self.validator.validate_intervention(intervention)
            validated.append(validated_intervention)

        except Exception as e:
            continue  # Skip invalid interventions

    return validated
```

#### Save to Database: `_save_interventions_batch()` (Lines 443-457)

```python
def _save_interventions_batch(self, interventions: List[Dict]):
    for intervention in interventions:
        try:
            # Add flag for later consensus processing
            intervention['consensus_processed'] = False

            # Use standard insertion (no normalization yet)
            success = self.repository_mgr.interventions.insert_intervention(intervention)

        except Exception as e:
            logger.error(f"Error saving raw intervention: {e}")
```

### Database Manager: `database_manager.py`

#### Entry: `insert_intervention()` (Lines 604-662)

```python
def insert_intervention(self, intervention: Dict) -> bool:
    # 1. Validate intervention data
    validated_intervention = category_validator.validate_intervention(intervention)

    # 2. Insert to database
    with self.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO interventions
            (paper_id, intervention_category, intervention_name, intervention_details,
             health_condition, correlation_type, correlation_strength, confidence_score,
             sample_size, study_duration, study_type, population_details,
             supporting_quote, delivery_method, severity, adverse_effects, cost_category,
             extraction_model, validation_status, consensus_confidence, model_agreement,
             models_used, raw_extraction_count, models_contributing)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            validated_intervention['paper_id'],
            validated_intervention['intervention_category'],
            validated_intervention['intervention_name'],
            json.dumps(validated_intervention.get('intervention_details', {})),
            validated_intervention['health_condition'],
            validated_intervention['correlation_type'],
            validated_intervention.get('correlation_strength'),
            validated_intervention.get('confidence_score'),
            validated_intervention.get('sample_size'),
            validated_intervention.get('study_duration'),
            validated_intervention.get('study_type'),
            validated_intervention.get('population_details'),
            validated_intervention.get('supporting_quote'),
            validated_intervention.get('delivery_method'),
            validated_intervention.get('severity'),
            validated_intervention.get('adverse_effects'),
            validated_intervention.get('cost_category'),
            validated_intervention.get('extraction_model', 'consensus'),
            'pending',
            validated_intervention.get('consensus_confidence'),
            validated_intervention.get('model_agreement'),
            validated_intervention.get('models_used'),
            validated_intervention.get('raw_extraction_count', 1),
            json.dumps(validated_intervention.get('models_contributing', []))
        ))

        was_new = cursor.rowcount > 0
        conn.commit()
        return was_new
```

**Processing Phase Output:**
- Interventions inserted into `interventions` table
- Each paper generates 2 sets of extractions (one per model)
- `extraction_model` field indicates which model extracted it
- `consensus_processed = False` flag indicates needs deduplication
- Papers marked as `processing_status = 'processed'`

---

## Phase 3: Deduplication

### Orchestrator: `batch_medical_rotation.py`

#### Entry: `run_batch_pipeline()` → `_run_deduplication_phase()` (Lines 442-494)

```python
def _run_deduplication_phase(self, session: BatchSession) -> Dict[str, Any]:
    # 1. Run global batch deduplication
    deduplication_result = self.dedup_integrator.deduplicate_all_data_batch()

    # 2. Update session with results
    total_processed = deduplication_result.get('interventions_processed', 0)
    total_merged = deduplication_result.get('total_merged', 0)

    session.deduplication_result = {
        'total_interventions_processed': total_processed,
        'deduplicated_interventions': total_merged,
        'entities_before': total_processed,
        'entities_after': total_processed - total_merged,
        'entities_merged': total_merged,
        'deduplication_rate': (total_merged / total_processed * 100),
        'duplicate_groups_found': deduplication_result.get('duplicate_groups_found', 0),
        'papers_processed': deduplication_result.get('papers_processed', 0),
        'method': 'llm_comprehensive_deduplication'
    }

    # 3. Mark phase complete
    session.deduplication_completed = True
    session.current_phase = BatchPhase.COMPLETED
```

### Deduplication Integrator: `rotation_deduplication_integrator.py`

#### Entry: `deduplicate_all_data_batch()` (Lines 119-189)

```python
def deduplicate_all_data_batch(self) -> Dict[str, Any]:
    # 1. Get comprehensive pre-deduplication stats
    pre_stats = self._get_comprehensive_unprocessed_stats()

    # 2. Run comprehensive deduplication with retry
    dedup_result = self._run_comprehensive_deduplication_with_retry()

    # 3. Get post-deduplication stats
    post_stats = self._get_comprehensive_unprocessed_stats()

    # 4. Calculate metrics
    interventions_processed = (
        pre_stats['unprocessed_interventions'] - post_stats['unprocessed_interventions']
    )

    return {
        'success': True,
        'interventions_before_processing': pre_stats['unprocessed_interventions'],
        'interventions_after_processing': post_stats['unprocessed_interventions'],
        'interventions_processed': interventions_processed,
        'canonical_mappings_created': pre_stats['missing_canonical_mapping'] - post_stats['missing_canonical_mapping'],
        'normalizations_completed': pre_stats['missing_normalization'] - post_stats['missing_normalization']
    }
```

#### Comprehensive Deduplication: `_run_comprehensive_deduplication_with_retry()` (Lines 255-357)

```python
def _run_comprehensive_deduplication_with_retry(self) -> Dict[str, Any]:
    for attempt in range(self.max_retries + 1):
        try:
            # 1. Create batch processor
            processor = create_batch_processor()

            # 2. Check if batch_deduplicate_entities method exists
            if hasattr(processor, 'batch_deduplicate_entities'):
                dedup_result = processor.batch_deduplicate_entities()
                return dedup_result
            else:
                # Fallback: comprehensive processing
                # Get all unprocessed interventions
                cursor.execute("""
                    SELECT id, intervention_name, health_condition, paper_id
                    FROM interventions
                    WHERE intervention_canonical_id IS NULL
                       OR normalized IS NULL OR normalized = 0
                       OR consensus_confidence IS NULL
                """)
                unprocessed_interventions = [dict(row) for row in cursor.fetchall()]

                # Process in batches
                for batch in batches:
                    for intervention in batch:
                        # Normalize intervention name
                        normalized_name = processor.get_or_compute_normalized_term(
                            intervention['intervention_name'], 'intervention'
                        )

                        # Find or create canonical entity
                        canonical_entity = processor.find_canonical_by_name(
                            normalized_name, 'intervention'
                        )

                        if not canonical_entity:
                            canonical_id = processor.create_canonical_entity(
                                normalized_name, 'intervention'
                            )
                        else:
                            canonical_id = canonical_entity['id']

                        # Update intervention
                        cursor.execute("""
                            UPDATE interventions
                            SET intervention_canonical_id = ?,
                                normalized = 1,
                                consensus_confidence = COALESCE(consensus_confidence, 0.8)
                            WHERE id = ?
                        """, (canonical_id, intervention['id']))
```

### Batch Entity Processor: `batch_entity_processor.py`

#### Entry: `batch_deduplicate_entities()` (Lines 290-445)

```python
def batch_deduplicate_entities(self) -> Dict[str, Any]:
    # 1. Get ALL interventions from database
    cursor.execute("""
        SELECT i.id, i.intervention_name, i.health_condition, i.paper_id,
               i.correlation_type, i.correlation_strength, i.confidence_score,
               i.extraction_model, i.models_used, i.consensus_confidence
        FROM interventions i
        ORDER BY i.paper_id, i.intervention_name
    """)
    all_interventions = [dict(row) for row in cursor.fetchall()]

    # 2. Group interventions by paper
    interventions_by_paper = defaultdict(list)
    for intervention in all_interventions:
        interventions_by_paper[intervention['paper_id']].append(intervention)

    # PHASE 1: Within-paper deduplication
    for paper_id, paper_interventions in interventions_by_paper.items():
        if len(paper_interventions) < 2:
            continue  # No duplicates possible

        # Detect duplicate groups
        duplicate_groups = self.duplicate_detector.detect_same_paper_duplicates(paper_interventions)

        for group in duplicate_groups:
            if len(group) > 1:
                # Merge duplicates using LLM analysis
                merged_intervention = self.duplicate_detector.merge_duplicate_group(group, paper_info)

                # Update database
                self._update_intervention_with_merge(merged_intervention, group)

                total_merged += len(group) - 1

    # PHASE 2: Cross-paper semantic deduplication
    unique_interventions = {}
    for intervention in all_interventions:
        name = intervention['intervention_name'].lower().strip()
        if name not in unique_interventions:
            unique_interventions[name] = []
        unique_interventions[name].append(intervention)

    # Process ALL intervention names in batches
    intervention_names = list(unique_interventions.keys())
    batch_size = 20

    for i in range(0, len(intervention_names), batch_size):
        batch_names = intervention_names[i:i + batch_size]

        # Use LLM to identify semantic duplicates
        llm_analysis = self.get_llm_duplicate_analysis(batch_names)

        # Process LLM-identified duplicates
        batch_merged = self._process_llm_duplicate_analysis(llm_analysis, unique_interventions)
        total_merged += batch_merged

    return {
        'total_merged': total_merged,
        'interventions_processed': len(all_interventions),
        'duplicate_groups_found': total_duplicate_groups,
        'papers_processed': len(interventions_by_paper),
        'method': 'llm_comprehensive_deduplication',
        'phases_completed': ['within_paper_deduplication', 'cross_paper_semantic_analysis']
    }
```

#### Within-Paper Duplicate Detection: (via DuplicateDetector)

```python
def detect_same_paper_duplicates(self, interventions: List[Dict]) -> List[List[Dict]]:
    # 1. Normalize all terms in batch
    normalized_terms = self.batch_normalize_consensus_terms(interventions)

    # 2. Group by normalized (intervention, condition) pair
    groups = defaultdict(list)
    for intervention in interventions:
        norm_intervention = normalized_terms['interventions'].get(
            intervention['intervention_name'], intervention['intervention_name']
        )
        norm_condition = normalized_terms['conditions'].get(
            intervention['health_condition'], intervention['health_condition']
        )

        key = (norm_intervention.lower(), norm_condition.lower())
        groups[key].append(intervention)

    # 3. Return groups with 2+ members (duplicates)
    return [group for group in groups.values() if len(group) >= 1]
```

#### Merge Duplicate Group:

```python
def merge_duplicate_group(self, group: List[Dict], paper: Dict) -> Dict:
    # 1. Select LLM consensus wording if available
    if len(group) > 1:
        intervention_names = [i['intervention_name'] for i in group]
        condition_names = [i['health_condition'] for i in group]

        # Use LLM to choose best wording
        consensus_intervention = self.llm_processor.choose_consensus_wording(
            intervention_names, 'intervention'
        )
        consensus_condition = self.llm_processor.choose_consensus_wording(
            condition_names, 'condition'
        )
    else:
        # Single intervention, use as-is
        consensus_intervention = group[0]['intervention_name']
        consensus_condition = group[0]['health_condition']

    # 2. Merge confidence scores
    extraction_conf, study_conf, consensus_conf = self._merge_dual_confidence(group)

    # 3. Track contributing models
    models_used = set()
    for intervention in group:
        model = intervention.get('extraction_model', 'unknown')
        models_used.add(model)

    # 4. Create merged intervention
    return {
        'intervention_name': consensus_intervention,
        'health_condition': consensus_condition,
        'models_used': ','.join(sorted(models_used)),
        'consensus_confidence': consensus_conf,
        'correlation_strength': sum(i.get('correlation_strength', 0) for i in group) / len(group),
        'confidence_score': extraction_conf,
        'raw_extraction_count': len(group)
    }
```

#### Update Database with Merge: `_update_intervention_with_merge()` (Lines 447-482)

```python
def _update_intervention_with_merge(self, merged_intervention: Dict, original_group: List[Dict]):
    # 1. Keep first intervention, update with merged data
    primary_id = original_group[0]['id']

    cursor.execute("""
        UPDATE interventions
        SET intervention_name = ?,
            models_used = ?,
            consensus_confidence = ?,
            correlation_strength = ?,
            confidence_score = ?,
            normalized = 1
        WHERE id = ?
    """, (
        merged_intervention.get('intervention_name'),
        merged_intervention.get('models_used', 'dual_consensus'),
        merged_intervention.get('consensus_confidence', 0.9),
        merged_intervention.get('correlation_strength'),
        merged_intervention.get('confidence_score'),
        primary_id
    ))

    # 2. DELETE duplicate interventions
    if len(original_group) > 1:
        duplicate_ids = [intervention['id'] for intervention in original_group[1:]]
        placeholders = ','.join(['?' for _ in duplicate_ids])
        cursor.execute(f"DELETE FROM interventions WHERE id IN ({placeholders})", duplicate_ids)

    self.db.commit()
```

**Deduplication Phase Output:**
- Duplicate interventions DELETED from database
- Remaining interventions updated with:
  - `models_used`: Combined model names
  - `consensus_confidence`: Merged confidence
  - `normalized = 1`: Flag indicating processed
  - Consensus wording from LLM

---

## Data Structures

### Key Data Classes

#### BatchSession (batch_medical_rotation.py)
```python
@dataclass
class BatchSession:
    session_id: str
    papers_per_condition: int
    current_phase: BatchPhase
    iteration_number: int
    start_time: str

    # Phase completion tracking
    collection_completed: bool = False
    processing_completed: bool = False
    deduplication_completed: bool = False

    # Statistics
    total_papers_collected: int = 0
    total_papers_processed: int = 0
    total_interventions_extracted: int = 0
    total_duplicates_removed: int = 0

    # Phase results
    collection_result: Optional[Dict[str, Any]] = None
    processing_result: Optional[Dict[str, Any]] = None
    deduplication_result: Optional[Dict[str, Any]] = None
```

#### BatchCollectionResult (rotation_paper_collector.py)
```python
@dataclass
class BatchCollectionResult:
    total_conditions: int = 0
    successful_conditions: int = 0
    failed_conditions: int = 0
    total_papers_collected: int = 0
    total_collection_time_seconds: float = 0.0
    conditions_results: List[Dict[str, Any]] = None
    success: bool = False
    error: Optional[str] = None
```

#### ModelResult (dual_model_analyzer.py)
```python
@dataclass
class ModelResult:
    model_name: str
    interventions: List[Dict]
    extraction_time: float
    error: Optional[str] = None
```

### Paper Dictionary Structure
```python
{
    'pmid': str,              # PubMed ID
    'title': str,             # Paper title
    'abstract': str,          # Abstract text
    'journal': str,           # Journal name
    'publication_date': str,  # YYYY-MM-DD format
    'doi': str,               # DOI
    'pmc_id': str,            # PubMed Central ID
    'keywords': List[str],    # Keywords/MeSH terms
    'has_fulltext': bool,     # Fulltext available
    'fulltext_source': str,   # Source (pmc, unpaywall)
    'fulltext_path': str,     # Path to fulltext
    'processing_status': str, # pending, processing, processed, failed
    'discovery_source': str   # pubmed, semantic_scholar
}
```

### Intervention Dictionary Structure
```python
{
    'paper_id': str,                    # PMID of source paper
    'intervention_category': str,       # Category (exercise, diet, supplement, etc.)
    'intervention_name': str,           # Name of intervention
    'intervention_details': Dict,       # Category-specific details (JSON)
    'health_condition': str,            # Condition being treated
    'correlation_type': str,            # positive, negative, neutral, inconclusive
    'correlation_strength': float,      # 0.0-1.0
    'confidence_score': float,          # Extraction confidence
    'sample_size': int,                 # Study sample size
    'study_duration': str,              # Duration text
    'study_type': str,                  # RCT, observational, etc.
    'population_details': str,          # Population description
    'supporting_quote': str,            # Quote from paper
    'extraction_model': str,            # Model that extracted (gemma2:9b or qwen2.5:14b)
    'models_used': str,                 # Combined models after dedup
    'consensus_confidence': float,      # Consensus confidence after dedup
    'normalized': bool,                 # Whether processed by deduplication
    'intervention_canonical_id': int,   # Canonical entity ID
    'condition_canonical_id': int,      # Canonical condition ID
}
```

---

## Database Schema

### papers Table
```sql
CREATE TABLE papers (
    pmid TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT,
    journal TEXT,
    publication_date TEXT,
    doi TEXT,
    pmc_id TEXT,
    keywords TEXT,  -- JSON array
    has_fulltext BOOLEAN DEFAULT FALSE,
    fulltext_source TEXT,
    fulltext_path TEXT,
    processing_status TEXT DEFAULT 'pending',  -- pending, processing, processed, failed
    discovery_source TEXT DEFAULT 'pubmed',

    -- Semantic Scholar fields (not used in batch pipeline)
    s2_paper_id TEXT,
    influence_score REAL,
    citation_count INTEGER,
    tldr TEXT,
    s2_embedding TEXT,
    s2_processed BOOLEAN DEFAULT FALSE,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

### interventions Table
```sql
CREATE TABLE interventions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT NOT NULL,
    intervention_category TEXT NOT NULL,
    intervention_name TEXT NOT NULL,
    intervention_details TEXT,  -- JSON
    health_condition TEXT NOT NULL,
    correlation_type TEXT CHECK(correlation_type IN ('positive', 'negative', 'neutral', 'inconclusive')),
    correlation_strength REAL CHECK(correlation_strength >= 0 AND correlation_strength <= 1),
    confidence_score REAL CHECK(confidence_score >= 0 AND confidence_score <= 1),

    -- Dual confidence metrics
    extraction_confidence REAL,
    study_confidence REAL,

    -- Study details
    sample_size INTEGER,
    study_duration TEXT,
    study_type TEXT,
    population_details TEXT,
    supporting_quote TEXT,

    -- Extraction tracking
    extraction_model TEXT NOT NULL,
    extraction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Validation tracking
    validation_status TEXT DEFAULT 'pending',
    validation_issues TEXT,

    -- Consensus tracking (after deduplication)
    consensus_confidence REAL,
    model_agreement TEXT,
    models_used TEXT,
    raw_extraction_count INTEGER DEFAULT 1,

    -- Canonical entity mapping
    intervention_canonical_id INTEGER,
    condition_canonical_id INTEGER,
    normalized BOOLEAN DEFAULT 0,

    FOREIGN KEY (paper_id) REFERENCES papers(pmid) ON DELETE CASCADE,
    UNIQUE(paper_id, intervention_category, intervention_name, health_condition)
)
```

### canonical_entities Table
```sql
CREATE TABLE canonical_entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,  -- intervention, condition, side_effect
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(canonical_name, entity_type)
)
```

### entity_mappings Table
```sql
CREATE TABLE entity_mappings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_text TEXT NOT NULL,
    canonical_id INTEGER NOT NULL,
    entity_type TEXT NOT NULL,
    confidence_score REAL,
    mapping_method TEXT,  -- exact_canonical, normalized, llm_semantic, manual
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (canonical_id) REFERENCES canonical_entities(id),
    UNIQUE(raw_text, entity_type)
)
```

---

## Function Call Chains

### Complete Pipeline Execution Chain

```
main()
└── BatchMedicalRotationPipeline.__init__()
    ├── RotationPaperCollector.__init__()
    ├── RotationLLMProcessor.__init__()
    └── RotationDeduplicationIntegrator.__init__()

└── run_batch_pipeline()
    ├── create_new_session() or load_existing_session()
    │   └── _save_session() → writes JSON to back_end/data/batch_session.json
    │
    ├── _run_collection_phase()
    │   └── RotationPaperCollector.collect_all_conditions_batch()
    │       ├── get_all_conditions() → returns 60 conditions from config
    │       ├── ThreadPoolExecutor.submit() × 60 (max_workers=2)
    │       │   └── _collect_single_condition_without_s2()
    │       │       └── _collect_with_retry()
    │       │           └── PubMedCollector.collect_interventions_by_condition()
    │       │               ├── _build_intervention_query()
    │       │               └── _collect_traditional_batch()
    │       │                   ├── _search_papers_with_offset()
    │       │                   │   └── pubmed_client.search_papers()
    │       │                   ├── _filter_existing_papers()
    │       │                   │   └── database_manager.get_connection()
    │       │                   ├── fetch_papers_metadata()
    │       │                   │   └── pubmed_client.fetch_papers()
    │       │                   ├── PubmedParser.parse_metadata_file()
    │       │                   │   ├── _parse_single_article()
    │       │                   │   └── _insert_papers_batch()
    │       │                   │       └── database_manager.insert_paper()
    │       │                   │           ├── validation_manager.validate_paper()
    │       │                   │           └── conn.execute(INSERT INTO papers...)
    │       │                   └── _process_fulltext_batch()
    │       │                       └── fulltext_retriever.process_papers_batch()
    │       └── Quality gate check (80% success rate)
    │
    ├── _run_processing_phase()
    │   └── RotationLLMProcessor.process_all_papers_batch()
    │       ├── _get_all_unprocessed_papers()
    │       │   └── DualModelAnalyzer.get_unprocessed_papers()
    │       │       └── database query: papers NOT IN (interventions.paper_id)
    │       ├── thermal_monitor.is_thermal_safe()
    │       ├── get_optimal_batch_size()
    │       └── DualModelAnalyzer.process_papers_batch()
    │           └── batch_process(papers, batch_size)
    │               └── for each paper:
    │                   └── extract_interventions()
    │                       ├── update_processing_status(pmid, 'processing')
    │                       ├── for model in ['gemma2:9b', 'qwen2.5:14b']:
    │                       │   └── extract_with_single_model()
    │                       │       ├── prompt_service.create_extraction_prompt()
    │                       │       ├── _calculate_dynamic_max_tokens()
    │                       │       ├── client.generate()
    │                       │       ├── parse_json_safely()
    │                       │       └── _validate_and_enhance_interventions()
    │                       │           └── category_validator.validate_intervention()
    │                       └── _save_interventions_batch()
    │                           └── repository_mgr.interventions.insert_intervention()
    │                               └── database_manager.insert_intervention()
    │                                   └── conn.execute(INSERT OR REPLACE INTO interventions...)
    │
    └── _run_deduplication_phase()
        └── RotationDeduplicationIntegrator.deduplicate_all_data_batch()
            ├── _get_comprehensive_unprocessed_stats()
            │   └── database queries for unprocessed counts
            ├── _run_comprehensive_deduplication_with_retry()
            │   └── create_batch_processor()
            │       └── BatchEntityProcessor.__init__()
            │           ├── EntityRepository.__init__()
            │           ├── LLMProcessor.__init__()
            │           └── DuplicateDetector.__init__()
            │       └── batch_deduplicate_entities()
            │           ├── SELECT all interventions, GROUP BY paper_id
            │           ├── PHASE 1: Within-paper deduplication
            │           │   └── for each paper with 2+ interventions:
            │           │       ├── DuplicateDetector.detect_same_paper_duplicates()
            │           │       │   ├── batch_normalize_consensus_terms()
            │           │       │   └── group by (normalized_intervention, normalized_condition)
            │           │       ├── DuplicateDetector.merge_duplicate_group()
            │           │       │   ├── LLMProcessor.choose_consensus_wording()
            │           │       │   └── _merge_dual_confidence()
            │           │       └── _update_intervention_with_merge()
            │           │           ├── UPDATE interventions SET ... WHERE id = primary_id
            │           │           └── DELETE FROM interventions WHERE id IN (duplicate_ids)
            │           └── PHASE 2: Cross-paper semantic deduplication
            │               └── group interventions by intervention_name
            │                   └── batch_process(intervention_names, batch_size=20)
            │                       ├── get_llm_duplicate_analysis()
            │                       │   └── LLMProcessor.get_llm_duplicate_analysis()
            │                       │       └── client.generate(semantic_similarity_prompt)
            │                       └── _process_llm_duplicate_analysis()
            │                           └── for each duplicate_group:
            │                               └── merge_duplicate_group()
            │                                   └── _update_intervention_with_merge()
            └── _get_comprehensive_unprocessed_stats()
                └── compare before/after statistics
```

---

## Error Handling Paths

### Collection Phase Errors

**1. PubMed API Failure:**
```python
# rotation_paper_collector.py::_collect_with_retry()
for attempt in range(self.max_retries):  # 3 attempts
    try:
        result = self.pubmed_collector.collect_interventions_by_condition(...)
        if papers_collected > 0:
            return result
    except Exception as e:
        logger.warning(f"Collection attempt {attempt + 1} failed: {e}")
        if attempt < self.max_retries - 1:
            time.sleep(self.retry_delays[attempt])  # [30, 60, 120] seconds
```

**Recovery**: Retry with exponential backoff, mark condition as failed if all retries exhausted

**2. XML Parsing Error:**
```python
# paper_parser.py::parse_metadata_file()
try:
    tree = ET.parse(file_path)
    root = tree.getroot()
except ET.ParseError as e:
    logger.error(f"XML parsing error: {e}")
    return []  # Skip this batch
```

**Recovery**: Skip malformed XML file, continue with other batches

**3. Database Insertion Error:**
```python
# database_manager.py::insert_paper()
try:
    cursor.execute(INSERT OR IGNORE INTO papers...)
    conn.commit()
except Exception as e:
    logger.error(f"Error inserting paper: {e}")
    conn.rollback()
    return False
```

**Recovery**: Rollback transaction, mark paper as failed, continue processing

**4. Quality Gate Failure:**
```python
# rotation_paper_collector.py::collect_all_conditions_batch()
success_rate = (successful_conditions / 60) * 100
if success_rate < 80.0:
    return BatchCollectionResult(
        success=False,
        error=f"Quality gate failed: {success_rate:.1f}% below 80% threshold"
    )
```

**Recovery**: Pipeline stops, session saved at collection phase, can be resumed

### Processing Phase Errors

**1. LLM Generation Failure:**
```python
# dual_model_analyzer.py::extract_with_single_model()
@handle_llm_errors("extract with single model", max_retries=3)
def extract_with_single_model(self, paper: Dict, model_name: str):
    try:
        response = client.generate(...)
        return ModelResult(...)
    except Exception as e:
        return ModelResult(
            model_name=model_name,
            interventions=[],
            extraction_time=extraction_time,
            error=str(e)
        )
```

**Recovery**: Retry up to 3 times with decorator, return empty result if all fail

**2. JSON Parsing Error:**
```python
# utils.py::parse_json_safely()
def parse_json_safely(text: str, context: str = "unknown"):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from text
        match = re.search(r'\{.*\}|\[.*\]', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return []
```

**Recovery**: Attempt to extract JSON from text, return empty list if extraction fails

**3. Validation Error:**
```python
# category_validators.py::validate_intervention()
def validate_intervention(self, intervention: Dict) -> Dict:
    if not intervention.get('intervention_name'):
        raise CategoryValidationError("Missing intervention_name")

    if intervention.get('intervention_category') not in valid_categories:
        raise CategoryValidationError("Invalid category")

    return validated_intervention
```

**Recovery**: Skip invalid intervention, log error, continue with next

**4. Thermal Protection:**
```python
# rotation_llm_processor.py::process_all_papers_batch()
is_safe, thermal_status = self.thermal_monitor.is_thermal_safe()
if not is_safe:
    self.thermal_monitor.wait_for_cooling()  # Block until GPU temp < 75°C
```

**Recovery**: Wait for GPU to cool down, then resume processing

### Deduplication Phase Errors

**1. LLM Analysis Failure:**
```python
# batch_entity_processor.py::batch_deduplicate_entities()
try:
    llm_analysis = self.get_llm_duplicate_analysis(batch_names)
    batch_merged = self._process_llm_duplicate_analysis(llm_analysis, unique_interventions)
except Exception as e:
    logger.warning(f"LLM analysis failed: {e}")
    continue  # Skip this batch, continue with next
```

**Recovery**: Skip failed batch, continue with remaining batches

**2. Database Update Failure:**
```python
# batch_entity_processor.py::_update_intervention_with_merge()
try:
    cursor.execute("UPDATE interventions SET ... WHERE id = ?", ...)
    cursor.execute(f"DELETE FROM interventions WHERE id IN (...)", ...)
    self.db.commit()
except Exception as e:
    logger.error(f"Failed to update merged intervention: {e}")
    self.db.rollback()
```

**Recovery**: Rollback transaction, intervention remains unmerged, logged for manual review

**3. Retry Mechanism:**
```python
# rotation_deduplication_integrator.py::_run_comprehensive_deduplication_with_retry()
for attempt in range(self.max_retries + 1):  # 3 attempts total
    try:
        processor = create_batch_processor()
        dedup_result = processor.batch_deduplicate_entities()
        return dedup_result
    except Exception as e:
        if attempt < self.max_retries:
            time.sleep(self.retry_delays[attempt])  # [30, 60] seconds
        else:
            raise DeduplicationError(f"Failed after {self.max_retries + 1} attempts")
```

**Recovery**: Retry with exponential backoff, fail entire phase if all attempts exhausted

### Session Recovery

**Graceful Shutdown:**
```python
# batch_medical_rotation.py::_signal_handler()
def _signal_handler(self, signum, frame):
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    self.shutdown_requested = True
```

**Session Persistence:**
```python
# batch_medical_rotation.py::_save_session()
def _save_session(self):
    data = {
        'session_id': self.current_session.session_id,
        'current_phase': self.current_session.current_phase.value,
        'collection_completed': self.current_session.collection_completed,
        'processing_completed': self.current_session.processing_completed,
        'deduplication_completed': self.current_session.deduplication_completed,
        # ... all statistics
    }
    with open(self.session_file, 'w') as f:
        json.dump(data, f, indent=2)
```

**Resume Workflow:**
```bash
python batch_medical_rotation.py --resume --start-phase processing
```

```python
# batch_medical_rotation.py::run_batch_pipeline()
if resume:
    session = self.load_existing_session()
    if start_phase:
        session.current_phase = BatchPhase(start_phase)

# Skip completed phases
if session.current_phase == BatchPhase.COLLECTION and not session.collection_completed:
    self._run_collection_phase(session)

if session.current_phase == BatchPhase.PROCESSING and not session.processing_completed:
    self._run_processing_phase(session)

if session.current_phase == BatchPhase.DEDUPLICATION and not session.deduplication_completed:
    self._run_deduplication_phase(session)
```

---

## Key Design Patterns

### 1. **Batch Processing Architecture**
- No condition-by-condition iteration
- ALL unprocessed items processed in each phase
- Quality gates between phases
- Natural breakpoints for recovery

### 2. **Database-Driven State**
- Query for unprocessed items: `papers.processing_status = 'pending'`
- Query for unprocessed interventions: `interventions.normalized = 0`
- Stateless processing: can resume from any point

### 3. **Dual-Model Sequential Processing**
- gemma2:9b → qwen2.5:14b (sequential, not parallel)
- Optimized for 8GB VRAM
- Each model creates separate intervention records
- Deduplication phase merges duplicates

### 4. **Three-Phase Separation**
- Collection: PubMed → papers table
- Processing: papers → interventions table (duplicates)
- Deduplication: interventions → interventions table (deduplicated)

### 5. **Comprehensive Error Recovery**
- Retry with exponential backoff
- Circuit breaker patterns
- Transaction rollback
- Session persistence
- Graceful shutdown

---

## Performance Characteristics

### Collection Phase
- **Parallelism**: 2 workers (ThreadPoolExecutor)
- **Rate Limiting**: 0.5s between papers, 2.0s between batches
- **Throughput**: ~50-100 papers/minute (depends on PubMed API)
- **Memory**: Low (streaming XML parsing)

### Processing Phase
- **Parallelism**: Sequential (one model at a time)
- **Batch Size**: 5 papers (GPU-optimized)
- **Rate Limiting**: 0.5s between models, 1.0s between papers, 2.0s between batches
- **Throughput**: ~10-20 papers/hour (dual-model analysis)
- **Memory**: 8GB VRAM (gemma2:9b: ~4GB, qwen2.5:14b: ~5GB)
- **Thermal Protection**: Max 85°C, cooling to 75°C

### Deduplication Phase
- **Parallelism**: Single-threaded (database operations)
- **Batch Size**: 20 interventions (LLM analysis)
- **Throughput**: ~100-200 interventions/minute
- **Memory**: Low (batch processing)

---

## Summary

The MyBiome pipeline is a **sophisticated three-phase batch processing system** that:

1. **Collects** papers from PubMed in parallel (2 workers, 60 conditions)
2. **Processes** papers with dual LLMs sequentially (gemma2:9b → qwen2.5:14b)
3. **Deduplicates** interventions using LLM-based semantic analysis

**Key Strengths:**
- Fully resumable at phase boundaries
- Comprehensive error handling with retry logic
- GPU-optimized for 8GB VRAM constraints
- Database-driven state management
- Quality gates prevent propagation of bad data

**Critical Data Flow:**
- PubMed API → XML → Parser → papers table → LLM → interventions table (duplicates) → Deduplication → interventions table (deduplicated)

**Session Management:**
- JSON file tracks phase completion and statistics
- Can resume from any phase
- Graceful shutdown support

This architecture enables **reliable, scalable processing** of large volumes of biomedical research papers with sophisticated entity resolution and duplicate detection.