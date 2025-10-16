-- Enhanced Database Schema for MyBiome Research Platform
-- Supports both LLM extraction results and comprehensive data mining outputs

-- ================================================================
-- EXISTING CORE TABLES (already implemented)
-- ================================================================

-- Papers table (already exists)
-- interventions table (already exists)
-- intervention_categories table (already exists)

-- ================================================================
-- NEW DATA MINING TABLES
-- ================================================================

-- 1. KNOWLEDGE GRAPH STORAGE
-- Stores nodes and edges for the medical knowledge graph
CREATE TABLE IF NOT EXISTS knowledge_graph_nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT UNIQUE NOT NULL,              -- Unique identifier for the node
    node_type TEXT NOT NULL,                   -- 'intervention', 'condition', 'study'
    node_name TEXT NOT NULL,                   -- Display name
    node_data TEXT,                            -- JSON object with node properties
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS knowledge_graph_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    edge_id TEXT UNIQUE NOT NULL,              -- Unique identifier for the edge
    source_node_id TEXT NOT NULL,              -- Source node ID
    target_node_id TEXT NOT NULL,              -- Target node ID
    edge_type TEXT NOT NULL,                   -- 'treats', 'causes', 'correlates', 'prevents'
    edge_weight REAL DEFAULT 1.0,              -- Relationship strength
    evidence_type TEXT NOT NULL,               -- 'positive', 'negative', 'neutral', 'unsure'
    confidence REAL CHECK(confidence >= 0 AND confidence <= 1),

    -- Study evidence details
    study_id TEXT,                             -- Reference to paper PMID
    study_title TEXT,
    sample_size INTEGER,
    study_design TEXT,                         -- 'RCT', 'observational', 'meta-analysis', etc.
    publication_year INTEGER,
    journal TEXT,
    doi TEXT,
    effect_size REAL,
    p_value REAL,

    -- Mechanism-based edge fields (Phase 3c integration)
    mechanism_cluster_id INTEGER,             -- FK to mechanism_clusters
    mechanism_canonical_name TEXT,            -- Mechanism IS the edge label
    mechanism_text_raw TEXT,                  -- Original mechanism description
    mechanism_similarity_score REAL,          -- Similarity to cluster centroid

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    generation_model TEXT,                     -- Which model/process created this edge
    generation_version TEXT,                   -- Version of the model/process

    FOREIGN KEY (source_node_id) REFERENCES knowledge_graph_nodes(node_id),
    FOREIGN KEY (target_node_id) REFERENCES knowledge_graph_nodes(node_id),
    FOREIGN KEY (study_id) REFERENCES papers(pmid),
    FOREIGN KEY (mechanism_cluster_id) REFERENCES mechanism_clusters(cluster_id)
);

-- 2. BAYESIAN SCORING RESULTS
-- Stores statistical analysis results for intervention-condition pairs
CREATE TABLE IF NOT EXISTS bayesian_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    intervention_name TEXT NOT NULL,
    condition_name TEXT NOT NULL,

    -- Bayesian analysis results
    posterior_mean REAL,                       -- Bayesian posterior mean
    posterior_variance REAL,                   -- Bayesian posterior variance
    credible_interval_lower REAL,              -- 95% credible interval lower bound
    credible_interval_upper REAL,              -- 95% credible interval upper bound
    bayes_factor REAL,                         -- Evidence strength compared to null hypothesis

    -- Evidence counts
    positive_evidence_count INTEGER DEFAULT 0,
    negative_evidence_count INTEGER DEFAULT 0,
    neutral_evidence_count INTEGER DEFAULT 0,
    total_studies INTEGER DEFAULT 0,

    -- Prior parameters
    alpha_prior REAL DEFAULT 1.0,
    beta_prior REAL DEFAULT 1.0,

    -- Innovation penalty solution
    innovation_penalty_adjusted BOOLEAN DEFAULT FALSE,
    confidence_adjusted_score REAL,            -- Score after adjusting for innovation penalty

    -- Metadata
    analysis_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    analysis_model TEXT,                       -- Which Bayesian model was used
    data_snapshot_id TEXT,                     -- ID of data state when analysis was run

    UNIQUE(intervention_name, condition_name, analysis_model, data_snapshot_id)
);

-- 3. TREATMENT RECOMMENDATIONS
-- Stores AI-generated treatment recommendations per condition
CREATE TABLE IF NOT EXISTS treatment_recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    condition_name TEXT NOT NULL,

    -- Recommendation details
    recommended_intervention TEXT NOT NULL,
    recommendation_rank INTEGER,               -- 1st choice, 2nd choice, etc.
    confidence_score REAL CHECK(confidence_score >= 0 AND confidence_score <= 1),
    evidence_strength TEXT,                    -- 'strong', 'moderate', 'weak', 'insufficient'

    -- Supporting evidence
    supporting_studies_count INTEGER,
    average_effect_size REAL,
    consistency_score REAL,                    -- How consistent is evidence across studies

    -- Recommendation reasoning
    recommendation_rationale TEXT,             -- AI explanation for recommendation
    contraindications TEXT,                    -- Known contraindications or warnings
    optimal_dosage TEXT,                       -- Recommended dosage/implementation
    duration_recommendation TEXT,              -- Recommended treatment duration

    -- Risk assessment
    safety_profile TEXT,                       -- 'high', 'moderate', 'low', 'unknown'
    side_effects TEXT,                         -- JSON array of known side effects
    interaction_warnings TEXT,                 -- Drug/supplement interactions

    -- Target population
    population_specificity TEXT,               -- Who this recommendation is for
    age_restrictions TEXT,
    comorbidity_considerations TEXT,

    -- Metadata
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    generation_model TEXT,                     -- AI model that generated recommendation
    model_version TEXT,
    data_version TEXT,                         -- Version of underlying data
    human_reviewed BOOLEAN DEFAULT FALSE,
    reviewer_notes TEXT,

    UNIQUE(condition_name, recommended_intervention, generation_model, data_version)
);

-- 4. RESEARCH GAPS ANALYSIS
-- Identifies under-researched areas and opportunities
CREATE TABLE IF NOT EXISTS research_gaps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    gap_type TEXT NOT NULL,                    -- 'intervention_gap', 'condition_gap', 'population_gap', 'methodology_gap'
    gap_name TEXT NOT NULL,
    gap_description TEXT,

    -- Gap specifics
    intervention_name TEXT,                    -- If intervention-specific gap
    condition_name TEXT,                       -- If condition-specific gap
    population_demographic TEXT,               -- If population-specific gap

    -- Gap severity
    priority_score REAL CHECK(priority_score >= 0 AND priority_score <= 10),
    urgency_level TEXT CHECK(urgency_level IN ('low', 'medium', 'high', 'critical')),

    -- Gap metrics
    current_study_count INTEGER,               -- Number of existing studies
    ideal_study_count INTEGER,                 -- Number of studies needed
    evidence_quality_score REAL,              -- Quality of existing evidence

    -- Research recommendations
    suggested_study_design TEXT,               -- 'RCT', 'longitudinal', 'meta-analysis', etc.
    suggested_sample_size INTEGER,
    estimated_cost_category TEXT CHECK(estimated_cost_category IN ('low', 'medium', 'high')),
    potential_impact_score REAL,               -- Expected impact if gap filled

    -- Innovation potential
    innovation_opportunity TEXT,               -- Description of innovation potential
    technology_requirements TEXT,              -- Tech needed to address gap
    collaboration_opportunities TEXT,          -- Suggested partnerships

    -- Metadata
    identified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    identification_method TEXT,                -- How gap was identified
    analysis_model TEXT,
    validation_status TEXT DEFAULT 'pending' CHECK(validation_status IN ('pending', 'validated', 'disputed', 'resolved')),

    UNIQUE(gap_type, gap_name, intervention_name, condition_name)
);

-- 5. INNOVATION TRACKING
-- Tracks emerging treatments and research trends
CREATE TABLE IF NOT EXISTS innovation_tracking (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    innovation_type TEXT NOT NULL,             -- 'emerging_treatment', 'new_mechanism', 'novel_combination', 'technology_advancement'
    innovation_name TEXT NOT NULL,
    innovation_description TEXT,

    -- Innovation details
    intervention_category TEXT,
    target_conditions TEXT,                    -- JSON array of conditions it may treat
    mechanism_of_action TEXT,
    development_stage TEXT,                    -- 'preclinical', 'phase1', 'phase2', 'phase3', 'approved', 'market'

    -- Trend analysis
    first_mentioned_date DATE,
    mention_frequency_trend TEXT,              -- JSON array of mention counts over time
    research_momentum_score REAL,             -- How quickly research is progressing

    -- Evidence tracking
    supporting_papers_count INTEGER DEFAULT 0,
    preliminary_efficacy_signals TEXT,         -- Early signs of efficacy
    safety_profile_emerging TEXT,              -- Emerging safety data

    -- Commercial potential
    patent_status TEXT,
    estimated_market_potential TEXT,
    regulatory_pathway TEXT,
    key_researchers TEXT,                      -- JSON array of leading researchers
    key_institutions TEXT,                     -- JSON array of leading institutions

    -- Innovation metrics
    novelty_score REAL CHECK(novelty_score >= 0 AND novelty_score <= 10),
    potential_impact_score REAL CHECK(potential_impact_score >= 0 AND potential_impact_score <= 10),
    adoption_likelihood REAL CHECK(adoption_likelihood >= 0 AND adoption_likelihood <= 1),

    -- Metadata
    tracked_since TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    tracking_model TEXT,
    data_sources TEXT,                         -- JSON array of data sources

    UNIQUE(innovation_name, innovation_type)
);

-- 6. BIOLOGICAL PATTERNS ANALYSIS
-- Stores discovered patterns in biological data
CREATE TABLE IF NOT EXISTS biological_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern_type TEXT NOT NULL,                -- 'pathway', 'biomarker', 'genetic', 'microbiome', 'metabolic'
    pattern_name TEXT NOT NULL,
    pattern_description TEXT,

    -- Pattern specifics
    biological_system TEXT,                    -- 'cardiovascular', 'digestive', 'immune', etc.
    molecular_targets TEXT,                    -- JSON array of molecular targets
    biomarkers_involved TEXT,                  -- JSON array of relevant biomarkers

    -- Pattern evidence
    supporting_interventions TEXT,             -- JSON array of interventions that affect this pattern
    affected_conditions TEXT,                 -- JSON array of conditions influenced by this pattern
    mechanism_pathway TEXT,                   -- Biological pathway description

    -- Pattern strength
    pattern_strength REAL CHECK(pattern_strength >= 0 AND pattern_strength <= 1),
    evidence_consistency REAL CHECK(evidence_consistency >= 0 AND evidence_consistency <= 1),
    replication_studies_count INTEGER,

    -- Clinical relevance
    clinical_actionability TEXT,              -- How actionable this pattern is clinically
    therapeutic_targets TEXT,                 -- JSON array of potential therapeutic targets
    diagnostic_potential TEXT,                -- Potential for diagnostic applications

    -- Discovery details
    discovery_method TEXT,                    -- How pattern was discovered
    statistical_significance REAL,
    effect_size REAL,
    confidence_interval TEXT,

    -- Metadata
    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    discovery_model TEXT,
    validation_status TEXT DEFAULT 'pending' CHECK(validation_status IN ('pending', 'validated', 'disputed', 'refuted')),

    UNIQUE(pattern_name, pattern_type, discovery_model)
);

-- 7. CONDITION SIMILARITY MAPPING
-- Maps similarities between different health conditions
CREATE TABLE IF NOT EXISTS condition_similarities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    condition_a TEXT NOT NULL,
    condition_b TEXT NOT NULL,

    -- Similarity metrics
    overall_similarity_score REAL CHECK(overall_similarity_score >= 0 AND overall_similarity_score <= 1),
    symptom_similarity REAL CHECK(symptom_similarity >= 0 AND symptom_similarity <= 1),
    treatment_similarity REAL CHECK(treatment_similarity >= 0 AND treatment_similarity <= 1),
    pathway_similarity REAL CHECK(pathway_similarity >= 0 AND pathway_similarity <= 1),
    genetic_similarity REAL CHECK(genetic_similarity >= 0 AND genetic_similarity <= 1),

    -- Shared elements
    shared_interventions TEXT,                 -- JSON array of common interventions
    shared_symptoms TEXT,                      -- JSON array of common symptoms
    shared_biomarkers TEXT,                    -- JSON array of common biomarkers
    shared_pathways TEXT,                      -- JSON array of common biological pathways

    -- Clinical insights
    treatment_transferability_score REAL,     -- How likely treatments transfer between conditions
    comorbidity_likelihood REAL,              -- How often these conditions co-occur
    differential_diagnosis_difficulty REAL,    -- How hard to distinguish between them

    -- Supporting evidence
    similarity_basis TEXT,                     -- Description of why they're similar
    evidence_papers_count INTEGER,
    cross_treatment_studies_count INTEGER,

    -- Metadata
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    calculation_method TEXT,
    data_version TEXT,

    UNIQUE(condition_a, condition_b, calculation_method, data_version),
    CHECK(condition_a < condition_b)  -- Ensure consistent ordering
);

-- 8. INTERVENTION COMBINATION ANALYSIS
-- Analyzes synergies and interactions between interventions
CREATE TABLE IF NOT EXISTS intervention_combinations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    combination_id TEXT UNIQUE NOT NULL,      -- Unique identifier for this combination
    intervention_names TEXT NOT NULL,         -- JSON array of intervention names
    target_condition TEXT NOT NULL,

    -- Combination metrics
    synergy_score REAL,                       -- How much better together than apart
    interaction_type TEXT,                    -- 'synergistic', 'additive', 'antagonistic', 'unknown'
    combined_efficacy REAL,                   -- Overall efficacy when combined

    -- Individual vs combined
    individual_efficacies TEXT,               -- JSON array of individual efficacy scores
    combination_improvement REAL,             -- How much better combined vs best individual
    statistical_significance REAL,

    -- Safety profile
    interaction_safety TEXT,                  -- 'safe', 'caution', 'contraindicated', 'unknown'
    known_interactions TEXT,                  -- JSON array of known interactions
    side_effect_profile TEXT,                 -- How side effects change when combined

    -- Evidence base
    combination_studies_count INTEGER,
    evidence_quality_score REAL,
    replication_studies_count INTEGER,

    -- Clinical recommendations
    dosage_adjustments TEXT,                  -- Recommended dosage modifications
    timing_recommendations TEXT,              -- How to time the interventions
    monitoring_requirements TEXT,             -- What to monitor when using combination

    -- Population specificity
    optimal_populations TEXT,                 -- JSON array of populations where combination works best
    contraindicated_populations TEXT,         -- JSON array of populations to avoid

    -- Metadata
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    analysis_method TEXT,
    evidence_cutoff_date DATE,

    UNIQUE(combination_id, target_condition, analysis_method)
);

-- 9. FAILED INTERVENTIONS ANALYSIS
-- Learns from failed treatments to avoid repetition and understand mechanisms
CREATE TABLE IF NOT EXISTS failed_interventions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    intervention_name TEXT NOT NULL,
    target_condition TEXT NOT NULL,

    -- Failure characteristics
    failure_type TEXT NOT NULL,               -- 'ineffective', 'harmful', 'intolerable', 'impractical'
    failure_description TEXT,
    failure_mechanism TEXT,                   -- Why it failed (biological mechanism)

    -- Failure evidence
    negative_studies_count INTEGER,
    harmful_effects_reported TEXT,             -- JSON array of reported harms
    discontinuation_rate REAL,                -- Rate of people stopping treatment

    -- Failure patterns
    population_specific_failure BOOLEAN,      -- Does it fail only in certain populations?
    failure_populations TEXT,                 -- JSON array of populations where it fails
    dosage_related_failure BOOLEAN,           -- Is failure related to dosage?
    timing_related_failure BOOLEAN,           -- Is failure related to timing?

    -- Learning opportunities
    alternative_approaches TEXT,              -- JSON array of alternative approaches
    mechanism_insights TEXT,                  -- What we learned about disease mechanism
    biomarker_predictions TEXT,               -- Biomarkers that might predict failure

    -- Prevention recommendations
    screening_recommendations TEXT,           -- How to screen patients before treatment
    monitoring_recommendations TEXT,          -- How to monitor during treatment
    discontinuation_criteria TEXT,            -- When to stop treatment

    -- Research implications
    research_questions_generated TEXT,        -- JSON array of research questions this raises
    hypothesis_refinements TEXT,              -- How this changes our hypotheses

    -- Metadata
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    analysis_depth TEXT,                      -- 'basic', 'comprehensive', 'meta-analysis'
    evidence_sources TEXT,                    -- JSON array of evidence sources

    UNIQUE(intervention_name, target_condition, failure_type)
);

-- 10. DATA MINING SESSIONS
-- Tracks data mining orchestrator runs and their results
CREATE TABLE IF NOT EXISTS data_mining_sessions (
    session_id TEXT PRIMARY KEY,
    session_type TEXT NOT NULL,               -- 'full_pipeline', 'partial', 'update'

    -- Session configuration
    config_snapshot TEXT,                     -- JSON of configuration used
    data_version TEXT,                        -- Version/hash of input data
    models_used TEXT,                         -- JSON array of models/algorithms used

    -- Session metrics
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    duration_seconds INTEGER,
    records_processed INTEGER,
    insights_generated INTEGER,

    -- Results summary
    results_summary TEXT,                     -- JSON summary of key findings
    quality_metrics TEXT,                     -- JSON of quality assessment metrics
    validation_results TEXT,                  -- JSON of validation test results

    -- Output tracking
    output_files TEXT,                        -- JSON array of generated files
    database_changes TEXT,                    -- JSON summary of database changes

    -- Status tracking
    status TEXT DEFAULT 'running' CHECK(status IN ('running', 'completed', 'failed', 'interrupted')),
    error_message TEXT,
    warnings_count INTEGER DEFAULT 0,

    -- Metadata
    orchestrator_version TEXT,
    environment_info TEXT,                    -- JSON of system environment info
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ================================================================
-- INDEXES FOR PERFORMANCE
-- ================================================================

-- Knowledge Graph indexes
CREATE INDEX IF NOT EXISTS idx_kg_nodes_type ON knowledge_graph_nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_kg_nodes_name ON knowledge_graph_nodes(node_name);
CREATE INDEX IF NOT EXISTS idx_kg_edges_source ON knowledge_graph_edges(source_node_id);
CREATE INDEX IF NOT EXISTS idx_kg_edges_target ON knowledge_graph_edges(target_node_id);
CREATE INDEX IF NOT EXISTS idx_kg_edges_type ON knowledge_graph_edges(edge_type);
CREATE INDEX IF NOT EXISTS idx_kg_edges_study ON knowledge_graph_edges(study_id);
CREATE INDEX IF NOT EXISTS idx_kg_edges_mechanism ON knowledge_graph_edges(mechanism_cluster_id);
CREATE INDEX IF NOT EXISTS idx_kg_edges_mechanism_name ON knowledge_graph_edges(mechanism_canonical_name);

-- Bayesian scores indexes
CREATE INDEX IF NOT EXISTS idx_bayesian_intervention ON bayesian_scores(intervention_name);
CREATE INDEX IF NOT EXISTS idx_bayesian_condition ON bayesian_scores(condition_name);
CREATE INDEX IF NOT EXISTS idx_bayesian_combo ON bayesian_scores(intervention_name, condition_name);
CREATE INDEX IF NOT EXISTS idx_bayesian_timestamp ON bayesian_scores(analysis_timestamp);

-- Treatment recommendations indexes
CREATE INDEX IF NOT EXISTS idx_recommendations_condition ON treatment_recommendations(condition_name);
CREATE INDEX IF NOT EXISTS idx_recommendations_intervention ON treatment_recommendations(recommended_intervention);
CREATE INDEX IF NOT EXISTS idx_recommendations_rank ON treatment_recommendations(recommendation_rank);
CREATE INDEX IF NOT EXISTS idx_recommendations_confidence ON treatment_recommendations(confidence_score);

-- Research gaps indexes
CREATE INDEX IF NOT EXISTS idx_gaps_type ON research_gaps(gap_type);
CREATE INDEX IF NOT EXISTS idx_gaps_priority ON research_gaps(priority_score);
CREATE INDEX IF NOT EXISTS idx_gaps_intervention ON research_gaps(intervention_name);
CREATE INDEX IF NOT EXISTS idx_gaps_condition ON research_gaps(condition_name);

-- Innovation tracking indexes
CREATE INDEX IF NOT EXISTS idx_innovation_type ON innovation_tracking(innovation_type);
CREATE INDEX IF NOT EXISTS idx_innovation_stage ON innovation_tracking(development_stage);
CREATE INDEX IF NOT EXISTS idx_innovation_impact ON innovation_tracking(potential_impact_score);

-- Biological patterns indexes
CREATE INDEX IF NOT EXISTS idx_patterns_type ON biological_patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_patterns_system ON biological_patterns(biological_system);
CREATE INDEX IF NOT EXISTS idx_patterns_strength ON biological_patterns(pattern_strength);

-- Condition similarities indexes
CREATE INDEX IF NOT EXISTS idx_similarities_a ON condition_similarities(condition_a);
CREATE INDEX IF NOT EXISTS idx_similarities_b ON condition_similarities(condition_b);
CREATE INDEX IF NOT EXISTS idx_similarities_score ON condition_similarities(overall_similarity_score);

-- Intervention combinations indexes
CREATE INDEX IF NOT EXISTS idx_combinations_condition ON intervention_combinations(target_condition);
CREATE INDEX IF NOT EXISTS idx_combinations_synergy ON intervention_combinations(synergy_score);
CREATE INDEX IF NOT EXISTS idx_combinations_type ON intervention_combinations(interaction_type);

-- Failed interventions indexes
CREATE INDEX IF NOT EXISTS idx_failed_intervention ON failed_interventions(intervention_name);
CREATE INDEX IF NOT EXISTS idx_failed_condition ON failed_interventions(target_condition);
CREATE INDEX IF NOT EXISTS idx_failed_type ON failed_interventions(failure_type);

-- Data mining sessions indexes
CREATE INDEX IF NOT EXISTS idx_sessions_start ON data_mining_sessions(start_time);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON data_mining_sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_type ON data_mining_sessions(session_type);

-- ================================================================
-- TRIGGERS FOR AUTOMATIC TIMESTAMP UPDATES
-- ================================================================

-- Update timestamps on record changes
CREATE TRIGGER IF NOT EXISTS update_kg_nodes_timestamp
AFTER UPDATE ON knowledge_graph_nodes
BEGIN
    UPDATE knowledge_graph_nodes SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_innovation_timestamp
AFTER UPDATE ON innovation_tracking
BEGIN
    UPDATE innovation_tracking SET last_updated = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- ================================================================
-- VIEWS FOR COMMON QUERIES
-- ================================================================

-- Comprehensive intervention view combining multiple data sources
CREATE VIEW IF NOT EXISTS intervention_insights AS
SELECT
    i.intervention_name,
    i.health_condition,
    i.correlation_type,
    i.confidence_score,
    COUNT(*) as study_count,
    AVG(i.confidence_score) as avg_confidence,
    bs.posterior_mean,
    bs.bayes_factor,
    tr.recommendation_rank,
    tr.evidence_strength,
    ic.synergy_score,
    GROUP_CONCAT(DISTINCT i.extraction_model) as models_used
FROM interventions i
LEFT JOIN bayesian_scores bs ON i.intervention_name = bs.intervention_name
    AND i.health_condition = bs.condition_name
LEFT JOIN treatment_recommendations tr ON i.intervention_name = tr.recommended_intervention
    AND i.health_condition = tr.condition_name
LEFT JOIN intervention_combinations ic ON i.health_condition = ic.target_condition
    AND json_extract(ic.intervention_names, '$[0]') = i.intervention_name
GROUP BY i.intervention_name, i.health_condition;

-- Research opportunity view
CREATE VIEW IF NOT EXISTS research_opportunities AS
SELECT
    rg.gap_name,
    rg.gap_type,
    rg.priority_score,
    rg.intervention_name,
    rg.condition_name,
    rg.potential_impact_score,
    it.innovation_name,
    it.novelty_score,
    it.development_stage
FROM research_gaps rg
LEFT JOIN innovation_tracking it ON rg.intervention_name = it.innovation_name
WHERE rg.priority_score > 5.0 OR it.novelty_score > 7.0
ORDER BY rg.priority_score DESC, it.novelty_score DESC;