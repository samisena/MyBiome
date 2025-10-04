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
    document.getElementById('canonical-entities').textContent = meta.canonical_entities.toLocaleString();
    document.getElementById('positive-correlations').textContent = meta.positive_correlations.toLocaleString();
    document.getElementById('negative-correlations').textContent = meta.negative_correlations.toLocaleString();
}

// Populate filter options dynamically
function populateFilterOptions() {
    const categoryFilter = document.getElementById('category-filter');
    const conditionCategoryFilter = document.getElementById('condition-category-filter');

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
}

// Initialize DataTables
function initializeDataTable() {
    const tableData = interventionsData.interventions.map(intervention => [
        formatInterventionName(intervention.intervention),
        formatCategory(intervention.intervention.category),
        formatConditionName(intervention.condition),
        formatCategory(intervention.condition.category),
        formatCorrelationType(intervention.correlation.type),
        formatStrengthBar(intervention.correlation.strength),
        formatConfidenceBar(intervention.correlation.extraction_confidence),
        formatSampleSize(intervention.study.sample_size),
        intervention.study.type || 'N/A',
        formatPaperLink(intervention.paper),
        formatDetailsButton(intervention)
    ]);

    dataTable = $('#interventions-table').DataTable({
        data: tableData,
        pageLength: 25,
        responsive: true,
        order: [[6, 'desc'], [5, 'desc']], // Sort by confidence, then strength
        buttons: [
            'csv', 'excel'
        ],
        dom: 'Bfrtip',
        columnDefs: [
            {
                targets: [4, 5, 6, 10],
                orderable: false
            },
            {
                targets: [5, 6],
                width: '100px'
            },
            {
                targets: [10],
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
        return `<strong>${intervention.canonical_name}</strong><br><small>(${intervention.name})</small>`;
    }
    return intervention.name;
}

// Format condition name (with canonical name if available)
function formatConditionName(condition) {
    if (condition.canonical_name && condition.canonical_name !== condition.name) {
        return `<strong>${condition.canonical_name}</strong><br><small>(${condition.name})</small>`;
    }
    return condition.name;
}

// Format category
function formatCategory(category) {
    if (!category) return 'N/A';
    return `<span class="category-badge">${category}</span>`;
}

// Format correlation type with color coding
function formatCorrelationType(type) {
    const classes = {
        'positive': 'correlation-positive',
        'negative': 'correlation-negative',
        'neutral': 'correlation-neutral',
        'inconclusive': 'correlation-inconclusive'
    };

    const className = classes[type] || 'correlation-neutral';
    return `<span class="correlation-badge ${className}">${type || 'N/A'}</span>`;
}

// Format strength as progress bar
function formatStrengthBar(strength) {
    if (!strength && strength !== 0) return 'N/A';

    const percentage = Math.round(strength * 100);
    return `
        <div class="strength-bar">
            <div class="strength-fill" style="width: ${percentage}%"></div>
            <div class="bar-text">${percentage}%</div>
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

    const details = `
Intervention: ${intervention.intervention.canonical_name || intervention.intervention.name}
Category: ${intervention.intervention.category || 'N/A'}
Details: ${intervention.intervention.details || 'N/A'}
Delivery Method: ${intervention.intervention.delivery_method || 'N/A'}

Health Condition: ${intervention.condition.canonical_name || intervention.condition.name}
Condition Category: ${intervention.condition.category || 'N/A'}
Severity: ${intervention.condition.severity || 'N/A'}

Correlation Type: ${intervention.correlation.type || 'N/A'}
Correlation Strength: ${intervention.correlation.strength || 'N/A'}
Extraction Confidence: ${intervention.correlation.extraction_confidence || 'N/A'}
Study Confidence: ${intervention.correlation.study_confidence || 'N/A'}

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
    correlationFilter.addEventListener('change', applyFilters);

    // Clear all filters
    clearFiltersBtn.addEventListener('click', function() {
        categoryFilter.value = '';
        conditionCategoryFilter.value = '';
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
    const correlationFilter = document.getElementById('correlation-filter').value;
    const confidenceFilter = parseFloat(document.getElementById('confidence-filter').value);

    // Clear existing search
    dataTable.search('').columns().search('').draw();

    // Apply filters using DataTables search API
    $.fn.dataTable.ext.search.push(
        function(settings, data, dataIndex) {
            const intervention = interventionsData.interventions[dataIndex];

            // Category filter
            if (categoryFilter && intervention.intervention.category !== categoryFilter) {
                return false;
            }

            // Condition category filter
            if (conditionCategoryFilter && intervention.condition.category !== conditionCategoryFilter) {
                return false;
            }

            // Correlation type filter
            if (correlationFilter && intervention.correlation.type !== correlationFilter) {
                return false;
            }

            // Confidence filter
            if (intervention.correlation.extraction_confidence < confidenceFilter) {
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
