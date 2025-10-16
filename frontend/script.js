// Global variables
let interventionsData = null;
let dataTable = null;

// DOM elements
const elements = {
    loading: document.getElementById('loading'),
    error: document.getElementById('error'),
    errorDetails: document.getElementById('error-details'),
    summary: document.getElementById('summary'),
    filters: document.getElementById('filters'),
    tableSection: document.getElementById('table-section'),
    topPerformers: document.getElementById('top-performers'),
    lastUpdated: document.getElementById('last-updated')
};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    loadData();
    setupFilters();
});

// Load interventions data from JSON file
async function loadData() {
    try {
        showElement(elements.loading);
        hideElement(elements.error);

        const response = await fetch('data/interventions.json');
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        interventionsData = await response.json();

        hideElement(elements.loading);
        populateSummaryStats();
        populateFilterOptions();
        initializeDataTable();
        populateTopPerformers();
        updateLastUpdated();

        showElement(elements.summary);
        showElement(elements.filters);
        showElement(elements.tableSection);
        showElement(elements.topPerformers);

    } catch (error) {
        console.error('Error loading data:', error);
        hideElement(elements.loading);
        elements.errorDetails.textContent = `Failed to load data: ${error.message}`;
        showElement(elements.error);
    }
}

// Populate summary statistics
function populateSummaryStats() {
    const meta = interventionsData.metadata;

    document.getElementById('total-interventions').textContent = meta.total_interventions.toLocaleString();
    document.getElementById('unique-interventions').textContent = meta.unique_interventions.toLocaleString();
    document.getElementById('unique-conditions').textContent = meta.unique_conditions.toLocaleString();
    document.getElementById('unique-papers').textContent = meta.unique_papers.toLocaleString();
    document.getElementById('canonical-groups').textContent = (meta.canonical_groups || 0).toLocaleString();
    document.getElementById('semantic-relationships').textContent = (meta.total_relationships || 0).toLocaleString();
    document.getElementById('multi-category-interventions').textContent = (meta.multi_category_interventions || 0).toLocaleString();
    document.getElementById('high-scoring-interventions').textContent = (meta.high_scoring_interventions || 0).toLocaleString();
    document.getElementById('positive-correlations').textContent = meta.positive_correlations.toLocaleString();
    document.getElementById('negative-correlations').textContent = meta.negative_correlations.toLocaleString();
}

// Populate filter options dynamically
function populateFilterOptions() {
    const categoryFilter = document.getElementById('category-filter');
    const conditionCategoryFilter = document.getElementById('condition-category-filter');
    const functionalCategoryFilter = document.getElementById('functional-category-filter');
    const therapeuticCategoryFilter = document.getElementById('therapeutic-category-filter');

    // Populate intervention categories
    const interventionCategories = Object.keys(interventionsData.metadata.intervention_categories).sort();
    interventionCategories.forEach(category => {
        const option = document.createElement('option');
        option.value = category;
        option.textContent = category.charAt(0).toUpperCase() + category.slice(1);
        categoryFilter.appendChild(option);
    });

    // Populate condition categories
    const conditionCategories = Object.keys(interventionsData.metadata.condition_categories).sort();
    conditionCategories.forEach(category => {
        const option = document.createElement('option');
        option.value = category;
        option.textContent = category.charAt(0).toUpperCase() + category.slice(1);
        conditionCategoryFilter.appendChild(option);
    });

    // Populate functional categories (multi-category)
    if (interventionsData.metadata.multi_category_stats && interventionsData.metadata.multi_category_stats.functional) {
        const functionalCategories = Object.keys(interventionsData.metadata.multi_category_stats.functional).sort();
        functionalCategories.forEach(category => {
            const option = document.createElement('option');
            option.value = category;
            option.textContent = category.charAt(0).toUpperCase() + category.slice(1).replace(/_/g, ' ');
            functionalCategoryFilter.appendChild(option);
        });
    }

    // Populate therapeutic categories (multi-category)
    if (interventionsData.metadata.multi_category_stats && interventionsData.metadata.multi_category_stats.therapeutic) {
        const therapeuticCategories = Object.keys(interventionsData.metadata.multi_category_stats.therapeutic).sort();
        therapeuticCategories.forEach(category => {
            const option = document.createElement('option');
            option.value = category;
            option.textContent = category.charAt(0).toUpperCase() + category.slice(1).replace(/_/g, ' ');
            therapeuticCategoryFilter.appendChild(option);
        });
    }
}

// Initialize DataTables
function initializeDataTable() {
    // Debug: Check data quality
    const hasMechanism = interventionsData.interventions.filter(i => i.mechanism_canonical_names && i.mechanism_canonical_names.length > 0).length;
    const hasStudyConfidence = interventionsData.interventions.filter(i => i.correlation.study_confidence !== null && i.correlation.study_confidence !== undefined).length;
    console.log(`Data Quality Check:`);
    console.log(`  - ${hasMechanism}/${interventionsData.interventions.length} have mechanism canonical names (${(hasMechanism/interventionsData.interventions.length*100).toFixed(1)}%)`);
    console.log(`  - ${hasStudyConfidence}/${interventionsData.interventions.length} have study confidence values (${(hasStudyConfidence/interventionsData.interventions.length*100).toFixed(1)}%)`);

    // Debug first intervention
    console.log('Sample intervention data:', interventionsData.interventions[0]);
    console.log('Sample mechanism canonical names:', interventionsData.interventions[0].mechanism_canonical_names);

    const tableData = interventionsData.interventions.map(intervention => [
        formatCanonicalGroup(intervention.intervention.hierarchy),  // Canonical Group
        formatCategory(intervention.intervention.categories || intervention.intervention.category),  // Category
        formatMechanism(intervention.mechanism_canonical_names),  // Mechanism (now uses canonical names from Phase 3c)
        formatConditionName(intervention.condition),  // Health Condition
        formatCategory(intervention.condition.categories || intervention.condition.category),  // Condition Category
        formatCorrelationType(intervention.correlation.type),  // Correlation
        formatBayesianScore(intervention.bayesian_scoring),  // Bayesian Score
        formatSampleSize(intervention.study.sample_size),  // Sample Size
        intervention.study.type || 'N/A',  // Study Type
        formatPaperLink(intervention.paper),  // Paper
        formatDetailsButton(intervention)  // Details
    ]);

    dataTable = $('#interventions-table').DataTable({
        data: tableData,
        pageLength: 25,
        responsive: true,
        autoWidth: false,
        order: [[6, 'desc']], // Sort by Bayesian score (descending)
        buttons: [
            'csv', 'excel'
        ],
        dom: 'Bfrtip',
        columnDefs: [
            {
                targets: [5, 9],  // Correlation and Paper columns are not orderable
                orderable: false
            },
            {
                targets: [0],  // Canonical Group
                width: '150px'
            },
            {
                targets: [1],  // Category
                width: '120px'
            },
            {
                targets: [2],  // Mechanism column
                width: '300px',
                className: 'mechanism-column'
            },
            {
                targets: [3],  // Health Condition
                width: '200px'
            },
            {
                targets: [4],  // Condition Category
                width: '120px'
            },
            {
                targets: [5],  // Correlation
                width: '110px'
            },
            {
                targets: [6],  // Bayesian Score
                width: '130px'
            },
            {
                targets: [7],  // Sample Size
                width: '100px'
            },
            {
                targets: [8],  // Study Type
                width: '120px'
            },
            {
                targets: [9],  // Paper
                width: '150px'
            },
            {
                targets: [10],  // Details
                width: '80px'
            }
        ],
        language: {
            search: "Search:",
            lengthMenu: "Show _MENU_ interventions per page",
            info: "Showing _START_ to _END_ of _TOTAL_ interventions",
            paginate: {
                previous: "← Previous",
                next: "Next →"
            }
        }
    });
}

// Format intervention name (with canonical name if available)
function formatInterventionName(intervention) {
    if (intervention.canonical_name && intervention.canonical_name !== intervention.name) {
        return `${intervention.canonical_name}<br><small>(${intervention.name})</small>`;
    }
    return intervention.name;
}

// Format condition name (with canonical name if available)
function formatConditionName(condition) {
    if (condition.canonical_name && condition.canonical_name !== condition.name) {
        return `${condition.canonical_name}<br><small>(${condition.name})</small>`;
    }
    return condition.name;
}

// Format category (supports multi-category display)
function formatCategory(categoryData) {
    // Handle legacy single-category format
    if (typeof categoryData === 'string') {
        if (!categoryData) return 'N/A';
        return `<span class="category-badge">${categoryData}</span>`;
    }

    // Handle multi-category format (categories object)
    if (!categoryData || typeof categoryData !== 'object') return 'N/A';

    const badges = [];

    // Primary category
    if (categoryData.primary && categoryData.primary.length > 0) {
        categoryData.primary.forEach(cat => {
            badges.push(`<span class="category-badge badge-primary-category" title="Primary: ${cat}">${cat}</span>`);
        });
    }

    // Functional categories
    if (categoryData.functional && categoryData.functional.length > 0) {
        categoryData.functional.forEach(cat => {
            badges.push(`<span class="category-badge badge-functional" title="Functional: ${cat}">${cat}</span>`);
        });
    }

    // Therapeutic categories
    if (categoryData.therapeutic && categoryData.therapeutic.length > 0) {
        categoryData.therapeutic.forEach(cat => {
            badges.push(`<span class="category-badge badge-therapeutic" title="Therapeutic: ${cat}">${cat}</span>`);
        });
    }

    // System categories (for conditions)
    if (categoryData.system && categoryData.system.length > 0) {
        categoryData.system.forEach(cat => {
            badges.push(`<span class="category-badge badge-system" title="System: ${cat}">${cat}</span>`);
        });
    }

    // Pathway categories (for mechanisms)
    if (categoryData.pathway && categoryData.pathway.length > 0) {
        categoryData.pathway.forEach(cat => {
            badges.push(`<span class="category-badge badge-pathway" title="Pathway: ${cat}">${cat}</span>`);
        });
    }

    // Target categories (for mechanisms)
    if (categoryData.target && categoryData.target.length > 0) {
        categoryData.target.forEach(cat => {
            badges.push(`<span class="category-badge badge-target" title="Target: ${cat}">${cat}</span>`);
        });
    }

    // Comorbidity categories (for conditions)
    if (categoryData.comorbidity && categoryData.comorbidity.length > 0) {
        categoryData.comorbidity.forEach(cat => {
            badges.push(`<span class="category-badge badge-comorbidity" title="Comorbidity: ${cat}">${cat}</span>`);
        });
    }

    if (badges.length === 0) return 'N/A';

    return `<div class="category-badges">${badges.join(' ')}</div>`;
}

// Format canonical group from hierarchical data
function formatCanonicalGroup(hierarchy) {
    if (!hierarchy || !hierarchy.layer_1_canonical) return 'N/A';
    return `<span class="canonical-group">${hierarchy.layer_1_canonical}</span>`;
}

// Format mechanism of action (now displays canonical names from Phase 3c)
function formatMechanism(mechanismCanonicalNames) {
    // mechanismCanonicalNames is now an array of canonical names from Phase 3c
    if (!mechanismCanonicalNames || mechanismCanonicalNames.length === 0) {
        return '<span class="mechanism-none">Not specified</span>';
    }

    // Display each mechanism on a separate line
    const mechanismItems = mechanismCanonicalNames.map(name =>
        `<div class="mechanism-item">${name}</div>`
    ).join('');

    return `<div class="mechanism-list">${mechanismItems}</div>`;
}

// Format outcome type with health impact color coding
function formatCorrelationType(type) {
    const classes = {
        'improves': 'outcome-improves',
        'worsens': 'outcome-worsens',
        'no_effect': 'outcome-no-effect',
        'inconclusive': 'outcome-inconclusive',
        // Legacy backward compatibility
        'positive': 'outcome-improves',
        'negative': 'outcome-worsens',
        'neutral': 'outcome-no-effect'
    };

    const labels = {
        'improves': 'Improves',
        'worsens': 'Worsens',
        'no_effect': 'No Effect',
        'inconclusive': 'Inconclusive',
        // Legacy
        'positive': 'Improves',
        'negative': 'Worsens',
        'neutral': 'No Effect'
    };

    const className = classes[type] || 'outcome-no-effect';
    const label = labels[type] || type || 'N/A';
    return `<span class="outcome-badge ${className}">${label}</span>`;
}

// Helper function to get strength label
function getStrengthLabel(strength) {
    if (!strength && strength !== 0) return 'N/A';

    if (strength >= 0.75) return 'Very Strong';
    if (strength >= 0.5) return 'Strong';
    if (strength >= 0.25) return 'Weak';
    if (strength >= 0) return 'Very Weak';
    return 'N/A';
}

// Format strength as categorical label
function formatStrengthBar(strength) {
    if (!strength && strength !== 0) return 'N/A';

    // Map numeric values to categorical labels
    let label = getStrengthLabel(strength);
    let className = 'strength-unknown';

    if (strength >= 0.75) {
        className = 'strength-very-strong';
    } else if (strength >= 0.5) {
        className = 'strength-strong';
    } else if (strength >= 0.25) {
        className = 'strength-weak';
    } else if (strength >= 0) {
        className = 'strength-very-weak';
    }

    return `<span class="strength-label ${className}">${label}</span>`;
}

// Format Bayesian score (Phase 4b)
function formatBayesianScore(bayesian) {
    if (!bayesian || !bayesian.score) return '<span class="bayesian-na">N/A</span>';

    const score = bayesian.score;
    const percentage = Math.round(score * 100);

    // Color code by score
    let className = 'bayesian-low';
    if (score >= 0.7) className = 'bayesian-high';
    else if (score >= 0.5) className = 'bayesian-medium';

    const evidenceTitle = `Positive: ${bayesian.positive_evidence || 0}, Negative: ${bayesian.negative_evidence || 0}, Neutral: ${bayesian.neutral_evidence || 0}`;

    return `
        <div class="bayesian-score ${className}" title="${evidenceTitle}">
            <div class="score-value">${percentage}%</div>
            <div class="score-bar" style="width: ${percentage}%"></div>
            <div class="evidence-counts">${bayesian.total_studies || 0} studies</div>
        </div>
    `;
}

// Format confidence as progress bar
function formatConfidenceBar(confidence) {
    if (!confidence && confidence !== 0) return 'N/A';

    const percentage = Math.round(confidence * 100);
    return `
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: ${percentage}%"></div>
            <div class="bar-text">${percentage}%</div>
        </div>
    `;
}

// Format sample size
function formatSampleSize(size) {
    if (!size) return 'Not specified';
    return size.toLocaleString();
}

// Format paper link
function formatPaperLink(paper) {
    if (!paper.pubmed_id) return 'N/A';

    const title = paper.title ? (paper.title.length > 50 ?
        paper.title.substring(0, 47) + '...' : paper.title) : 'View Paper';

    const url = `https://pubmed.ncbi.nlm.nih.gov/${paper.pubmed_id}/`;

    return `<a href="${url}" target="_blank" rel="noopener noreferrer" class="paper-link" title="${paper.title || 'View on PubMed'}">${title}</a>`;
}

// Format details button
function formatDetailsButton(intervention) {
    return `<button class="details-btn" onclick="showDetails(${intervention.id})" title="View detailed information">Details</button>`;
}

// Show detailed information modal
function showDetails(interventionId) {
    const intervention = interventionsData.interventions.find(i => i.id === interventionId);
    if (!intervention) return;

    const hierarchyInfo = intervention.intervention.hierarchy || {};
    const conditionHierarchyInfo = intervention.condition.hierarchy || {};

    // Format categories (multi-category support)
    function formatCategoriesForModal(entity) {
        if (entity.categories) {
            const parts = [];
            if (entity.categories.primary && entity.categories.primary.length > 0) {
                parts.push(`Primary: ${entity.categories.primary.join(', ')}`);
            }
            if (entity.categories.functional && entity.categories.functional.length > 0) {
                parts.push(`Functional: ${entity.categories.functional.join(', ')}`);
            }
            if (entity.categories.therapeutic && entity.categories.therapeutic.length > 0) {
                parts.push(`Therapeutic: ${entity.categories.therapeutic.join(', ')}`);
            }
            if (entity.categories.system && entity.categories.system.length > 0) {
                parts.push(`System: ${entity.categories.system.join(', ')}`);
            }
            if (entity.categories.comorbidity && entity.categories.comorbidity.length > 0) {
                parts.push(`Comorbidity: ${entity.categories.comorbidity.join(', ')}`);
            }
            return parts.length > 0 ? parts.join('\n  ') : (entity.category || 'N/A');
        }
        return entity.category || 'N/A';
    }

    const interventionCategories = formatCategoriesForModal(intervention.intervention);
    const conditionCategories = formatCategoriesForModal(intervention.condition);

    const details = `
Intervention: ${intervention.intervention.canonical_name || intervention.intervention.name}
Categories:
  ${interventionCategories}
Details: ${intervention.intervention.details || 'N/A'}
Delivery Method: ${intervention.intervention.delivery_method || 'N/A'}

Hierarchical Classification:
- Category (L0): ${hierarchyInfo.layer_0_category || 'N/A'}
- Canonical Group (L1): ${hierarchyInfo.layer_1_canonical || 'N/A'}
- Specific Variant (L2): ${hierarchyInfo.layer_2_variant || 'N/A'}
- Dosage/Details (L3): ${hierarchyInfo.layer_3_detail || 'N/A'}

Mechanism of Action (Canonical Names from Phase 3c):
${intervention.mechanism_canonical_names && intervention.mechanism_canonical_names.length > 0
    ? intervention.mechanism_canonical_names.map(name => `- ${name}`).join('\n')
    : 'Not specified'}

Raw Mechanism Text:
${intervention.mechanism || 'Not specified'}

Health Condition: ${intervention.condition.canonical_name || intervention.condition.name}
Categories:
  ${conditionCategories}
Severity: ${intervention.condition.severity || 'N/A'}

Condition Hierarchy:
- Category (L0): ${conditionHierarchyInfo.layer_0_category || 'N/A'}
- Canonical Group (L1): ${conditionHierarchyInfo.layer_1_canonical || 'N/A'}
- Specific Variant (L2): ${conditionHierarchyInfo.layer_2_variant || 'N/A'}
- Details (L3): ${conditionHierarchyInfo.layer_3_detail || 'N/A'}

Health Impact: ${{'improves': 'Improves', 'worsens': 'Worsens', 'no_effect': 'No Effect', 'inconclusive': 'Inconclusive', 'positive': 'Improves', 'negative': 'Worsens', 'neutral': 'No Effect'}[intervention.correlation.type] || intervention.correlation.type || 'N/A'}
Study Confidence: ${intervention.correlation.study_confidence || 'N/A'}

Bayesian Evidence Score (Phase 4b):
${intervention.bayesian_scoring ? `- Posterior Mean: ${(intervention.bayesian_scoring.score * 100).toFixed(1)}%
- Conservative Score (10th percentile): ${(intervention.bayesian_scoring.conservative_score * 100).toFixed(1)}%
- Evidence Counts: ${intervention.bayesian_scoring.positive_evidence || 0} positive, ${intervention.bayesian_scoring.negative_evidence || 0} negative, ${intervention.bayesian_scoring.neutral_evidence || 0} neutral
- Total Studies: ${intervention.bayesian_scoring.total_studies || 0}
- Bayes Factor: ${intervention.bayesian_scoring.bayes_factor ? intervention.bayesian_scoring.bayes_factor.toFixed(2) : 'N/A'}` : 'Not available (requires Phase 4b scoring)'}

Study Details:
- Sample Size: ${intervention.study.sample_size || 'Not specified'}
- Study Type: ${intervention.study.type || 'N/A'}
- Duration: ${intervention.study.duration || 'N/A'}
- Population: ${intervention.study.population || 'N/A'}
- Adverse Effects: ${intervention.study.adverse_effects || 'None reported'}
- Cost Category: ${intervention.study.cost_category || 'N/A'}

Paper Information:
- Title: ${intervention.paper.title || 'N/A'}
- Journal: ${intervention.paper.journal || 'N/A'}
- PubMed ID: ${intervention.paper.pubmed_id || 'N/A'}
- DOI: ${intervention.paper.doi || 'N/A'}
- Publication Date: ${intervention.paper.publication_date || 'N/A'}

Supporting Quote:
"${intervention.supporting_quote || 'No quote available'}"

Extraction Model: ${intervention.extraction_model || 'N/A'}
Extracted: ${intervention.extraction_timestamp ? new Date(intervention.extraction_timestamp).toLocaleString() : 'N/A'}
    `;

    alert(details); // Simple modal - could be enhanced with a proper modal library
}

// Setup filter functionality
function setupFilters() {
    const categoryFilter = document.getElementById('category-filter');
    const conditionCategoryFilter = document.getElementById('condition-category-filter');
    const functionalCategoryFilter = document.getElementById('functional-category-filter');
    const therapeuticCategoryFilter = document.getElementById('therapeutic-category-filter');
    const correlationFilter = document.getElementById('correlation-filter');
    const confidenceFilter = document.getElementById('confidence-filter');
    const confidenceValue = document.getElementById('confidence-value');
    const clearFiltersBtn = document.getElementById('clear-filters');

    // Update confidence value display
    confidenceFilter.addEventListener('input', function() {
        confidenceValue.textContent = parseFloat(this.value).toFixed(1);
        applyFilters();
    });

    // Apply filters on change
    categoryFilter.addEventListener('change', applyFilters);
    conditionCategoryFilter.addEventListener('change', applyFilters);
    functionalCategoryFilter.addEventListener('change', applyFilters);
    therapeuticCategoryFilter.addEventListener('change', applyFilters);
    correlationFilter.addEventListener('change', applyFilters);

    // Clear all filters
    clearFiltersBtn.addEventListener('click', function() {
        categoryFilter.value = '';
        conditionCategoryFilter.value = '';
        functionalCategoryFilter.value = '';
        therapeuticCategoryFilter.value = '';
        correlationFilter.value = '';
        confidenceFilter.value = '0';
        confidenceValue.textContent = '0.0';
        applyFilters();
    });
}

// Apply current filter settings
function applyFilters() {
    if (!dataTable) return;

    const categoryFilter = document.getElementById('category-filter').value;
    const conditionCategoryFilter = document.getElementById('condition-category-filter').value;
    const functionalCategoryFilter = document.getElementById('functional-category-filter').value;
    const therapeuticCategoryFilter = document.getElementById('therapeutic-category-filter').value;
    const correlationFilter = document.getElementById('correlation-filter').value;
    const confidenceFilter = parseFloat(document.getElementById('confidence-filter').value);

    // Clear existing search
    dataTable.search('').columns().search('').draw();

    // Apply filters using DataTables search API
    $.fn.dataTable.ext.search.push(
        function(settings, data, dataIndex) {
            const intervention = interventionsData.interventions[dataIndex];

            // Helper function to check if entity has category (supports multi-category)
            function hasCategory(entity, categoryValue, categoryType = 'primary') {
                // Check legacy single-category format
                if (entity.category === categoryValue) return true;

                // Check multi-category format
                if (entity.categories && entity.categories[categoryType]) {
                    return entity.categories[categoryType].includes(categoryValue);
                }

                return false;
            }

            // Primary intervention category filter
            if (categoryFilter && !hasCategory(intervention.intervention, categoryFilter, 'primary')) {
                return false;
            }

            // Condition category filter
            if (conditionCategoryFilter && !hasCategory(intervention.condition, conditionCategoryFilter, 'primary')) {
                return false;
            }

            // Functional category filter (multi-category)
            if (functionalCategoryFilter && !hasCategory(intervention.intervention, functionalCategoryFilter, 'functional')) {
                return false;
            }

            // Therapeutic category filter (multi-category)
            if (therapeuticCategoryFilter && !hasCategory(intervention.intervention, therapeuticCategoryFilter, 'therapeutic')) {
                return false;
            }

            // Correlation type filter
            if (correlationFilter && intervention.correlation.type !== correlationFilter) {
                return false;
            }

            // Study confidence filter (if available)
            const studyConf = intervention.correlation.study_confidence || 0;
            if (studyConf < confidenceFilter) {
                return false;
            }

            return true;
        }
    );

    dataTable.draw();

    // Remove the custom filter after drawing
    $.fn.dataTable.ext.search.pop();
}

// Populate top performers section
function populateTopPerformers() {
    const topInterventionsList = document.getElementById('top-interventions-list');
    const topConditionsList = document.getElementById('top-conditions-list');

    // Top interventions
    interventionsData.top_performers.interventions.forEach(intervention => {
        const li = document.createElement('li');
        li.innerHTML = `
            <span>${intervention.name}</span>
            <span class="count-badge">${intervention.count} (${intervention.paper_count} papers)</span>
        `;
        topInterventionsList.appendChild(li);
    });

    // Top conditions
    interventionsData.top_performers.conditions.forEach(condition => {
        const li = document.createElement('li');
        li.innerHTML = `
            <span>${condition.name}</span>
            <span class="count-badge">${condition.count} (${condition.paper_count} papers)</span>
        `;
        topConditionsList.appendChild(li);
    });
}

// Update last updated timestamp
function updateLastUpdated() {
    if (interventionsData.metadata.generated_at) {
        const date = new Date(interventionsData.metadata.generated_at);
        elements.lastUpdated.textContent = date.toLocaleString();
    }
}

// Helper functions
function showElement(element) {
    if (element) {
        element.style.display = 'block';
    }
}

function hideElement(element) {
    if (element) {
        element.style.display = 'none';
    }
}

// Add keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + K to focus search
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        const searchInput = document.querySelector('.dataTables_filter input');
        if (searchInput) {
            searchInput.focus();
        }
    }
});

// Export functions for testing
window.interventionsApp = {
    loadData,
    showDetails,
    applyFilters
};
