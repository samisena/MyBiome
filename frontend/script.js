// Global variables
let correlationsData = null;
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

// Load correlations data from JSON file
async function loadData() {
    try {
        showElement(elements.loading);
        hideElement(elements.error);
        
        const response = await fetch('data/correlations.json');
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        correlationsData = await response.json();
        
        hideElement(elements.loading);
        populateSummaryStats();
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
    const stats = correlationsData.summary_stats;
    
    document.getElementById('total-correlations').textContent = stats.total_correlations.toLocaleString();
    document.getElementById('unique-strains').textContent = stats.unique_strains.toLocaleString();
    document.getElementById('unique-conditions').textContent = stats.unique_conditions.toLocaleString();
    document.getElementById('unique-papers').textContent = stats.unique_papers.toLocaleString();
    document.getElementById('positive-correlations').textContent = stats.positive_correlations.toLocaleString();
    document.getElementById('negative-correlations').textContent = stats.negative_correlations.toLocaleString();
}

// Initialize DataTables
function initializeDataTable() {
    const tableData = correlationsData.correlations.map(correlation => [
        correlation.probiotic_strain,
        correlation.health_condition,
        formatCorrelationType(correlation.correlation_type),
        formatStrengthBar(correlation.correlation_strength),
        formatConfidenceBar(correlation.confidence_score),
        formatSampleSize(correlation.sample_size),
        correlation.study_type,
        formatPaperLink(correlation.paper),
        correlation.paper.journal || 'N/A',
        formatValidationStatus(correlation.validation_status),
        formatDetailsButton(correlation)
    ]);

    dataTable = $('#correlations-table').DataTable({
        data: tableData,
        pageLength: 25,
        responsive: true,
        order: [[4, 'desc'], [3, 'desc']], // Sort by confidence, then strength
        buttons: [
            'csv', 'excel'
        ],
        dom: 'Bfrtip',
        columnDefs: [
            { 
                targets: [2, 3, 4, 9, 10], 
                orderable: false 
            },
            {
                targets: [3, 4],
                width: '100px'
            },
            {
                targets: [10],
                width: '80px'
            }
        ],
        language: {
            search: "🔍 Search:",
            lengthMenu: "Show _MENU_ correlations per page",
            info: "Showing _START_ to _END_ of _TOTAL_ correlations",
            paginate: {
                previous: "← Previous",
                next: "Next →"
            }
        },
        initComplete: function() {
            // Add custom search functionality
            this.api().columns().every(function() {
                const column = this;
                const header = $(column.header());
                
                // Skip certain columns from individual filtering
                if (header.text() === 'Details' || header.text() === 'Correlation' || 
                    header.text() === 'Strength' || header.text() === 'Confidence') {
                    return;
                }
            });
        }
    });
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
    return `<span class="correlation-badge ${className}">${type}</span>`;
}

// Format strength as progress bar
function formatStrengthBar(strength) {
    if (!strength) return 'N/A';
    
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
    if (!confidence) return 'N/A';
    
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
    if (!paper.pmid) return 'N/A';
    
    const title = paper.title ? (paper.title.length > 50 ? 
        paper.title.substring(0, 47) + '...' : paper.title) : 'View Paper';
    
    return `<a href="${paper.pubmed_url}" target="_blank" rel="noopener noreferrer" class="paper-link" title="${paper.title}">${title}</a>`;
}

// Format validation status
function formatValidationStatus(status) {
    const classes = {
        'pending': 'validation-pending',
        'verified': 'validation-verified',
        'conflicted': 'validation-conflicted',
        'failed': 'validation-failed'
    };
    
    const className = classes[status] || 'validation-pending';
    return `<span class="validation-badge ${className}">${status}</span>`;
}

// Format details button
function formatDetailsButton(correlation) {
    return `<button class="details-btn" onclick="showDetails(${correlation.id})" title="View detailed information">Details</button>`;
}

// Show detailed information modal
function showDetails(correlationId) {
    const correlation = correlationsData.correlations.find(c => c.id === correlationId);
    if (!correlation) return;
    
    const details = `
        Probiotic Strain: ${correlation.probiotic_strain}
        Health Condition: ${correlation.health_condition}
        Correlation Type: ${correlation.correlation_type}
        Correlation Strength: ${correlation.correlation_strength || 'N/A'}
        Confidence Score: ${correlation.confidence_score || 'N/A'}
        
        Study Details:
        - Sample Size: ${correlation.sample_size || 'Not specified'}
        - Study Type: ${correlation.study_type}
        - Effect Size: ${correlation.effect_size}
        - Dosage: ${correlation.dosage}
        - Duration: ${correlation.study_duration}
        - Population: ${correlation.population_details}
        
        Paper Information:
        - Title: ${correlation.paper.title || 'N/A'}
        - Journal: ${correlation.paper.journal || 'N/A'}
        - PMID: ${correlation.paper.pmid || 'N/A'}
        - Publication Date: ${correlation.paper.publication_date || 'N/A'}
        
        Supporting Quote:
        "${correlation.supporting_quote || 'No quote available'}"
        
        Validation Status: ${correlation.validation_status}
    `;
    
    alert(details); // Simple modal - could be enhanced with a proper modal library
}

// Setup filter functionality
function setupFilters() {
    const correlationFilter = document.getElementById('correlation-filter');
    const confidenceFilter = document.getElementById('confidence-filter');
    const confidenceValue = document.getElementById('confidence-value');
    const validationFilter = document.getElementById('validation-filter');
    const clearFiltersBtn = document.getElementById('clear-filters');
    
    // Update confidence value display
    confidenceFilter.addEventListener('input', function() {
        confidenceValue.textContent = parseFloat(this.value).toFixed(1);
        applyFilters();
    });
    
    // Apply filters on change
    correlationFilter.addEventListener('change', applyFilters);
    validationFilter.addEventListener('change', applyFilters);
    
    // Clear all filters
    clearFiltersBtn.addEventListener('click', function() {
        correlationFilter.value = '';
        confidenceFilter.value = '0';
        confidenceValue.textContent = '0.0';
        validationFilter.value = '';
        applyFilters();
    });
}

// Apply current filter settings
function applyFilters() {
    if (!dataTable) return;
    
    const correlationFilter = document.getElementById('correlation-filter').value;
    const confidenceFilter = parseFloat(document.getElementById('confidence-filter').value);
    const validationFilter = document.getElementById('validation-filter').value;
    
    // Clear existing search
    dataTable.search('').columns().search('').draw();
    
    // Apply filters using DataTables search API
    dataTable.search(function(settings, data, dataIndex) {
        const correlation = correlationsData.correlations[dataIndex];
        
        // Correlation type filter
        if (correlationFilter && correlation.correlation_type !== correlationFilter) {
            return false;
        }
        
        // Confidence filter
        if (correlation.confidence_score < confidenceFilter) {
            return false;
        }
        
        // Validation status filter
        if (validationFilter && correlation.validation_status !== validationFilter) {
            return false;
        }
        
        return true;
    }).draw();
}

// Populate top performers section
function populateTopPerformers() {
    const topStrainsList = document.getElementById('top-strains-list');
    const topConditionsList = document.getElementById('top-conditions-list');
    
    // Top strains
    correlationsData.top_strains.forEach(strain => {
        const li = document.createElement('li');
        li.innerHTML = `
            <span>${strain.probiotic_strain}</span>
            <span class="count-badge">${strain.count}</span>
        `;
        topStrainsList.appendChild(li);
    });
    
    // Top conditions
    correlationsData.top_conditions.forEach(condition => {
        const li = document.createElement('li');
        li.innerHTML = `
            <span>${condition.health_condition}</span>
            <span class="count-badge">${condition.count}</span>
        `;
        topConditionsList.appendChild(li);
    });
}

// Update last updated timestamp
function updateLastUpdated() {
    if (correlationsData.export_timestamp) {
        const date = new Date(correlationsData.export_timestamp);
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
window.correlationsApp = {
    loadData,
    showDetails,
    applyFilters
};