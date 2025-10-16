# MyBiome Medical Knowledge Network Visualization

**Experimental prototype** - Interactive force-directed graph visualization of Phase 4a knowledge graph data.

## Overview

This is an experimental visualization showing the medical knowledge network as an interactive cosmic-themed graph with:
- **895 nodes**: 524 interventions (orange) + 371 conditions (blue)
- **628 edges**: Treatment relationships colored by evidence type
- **362 mechanisms**: Biological/behavioral pathways shown as edge labels

## Quick Start

### Step 1: Generate Data

```bash
# From project root
python -m back_end.src.utils.export_network_visualization_data
```

This exports the knowledge graph from the database to `data/network_graph.json` (345 KB).

### Step 2: Open Visualization

Open `index.html` in your web browser:
- **Chrome/Edge**: Recommended for best performance
- **Firefox**: Full support
- **Safari**: Full support

**Note**: Must be opened via HTTP server or file:// protocol to load JSON data.

### Simple HTTP Server (if needed)

```bash
# Python 3
cd frontend_network_viz_experiment
python -m http.server 8000

# Then open: http://localhost:8000
```

## Features

### Visual Design

- **Dark cosmic theme**: Deep black background (#0a0a0a) with glowing nodes
- **Node types**:
  - **Interventions**: Orange glow (#ff9800)
  - **Conditions**: Blue/cyan glow (#00bcd4)
- **Edge colors**:
  - **Green**: Positive evidence (78% of edges)
  - **Red**: Negative evidence (9% of edges)
  - **Gray**: Neutral evidence (13% of edges)
- **Node sizing**: Scaled by cluster size (interventions) or connection count (conditions)
- **Edge thickness**: Proportional to confidence score

### Interactive Features

1. **Drag nodes**: Click and drag individual nodes to reposition
2. **Zoom/Pan**:
   - Mouse wheel to zoom in/out
   - Click and drag canvas to pan
3. **Node hover**: Highlights connected edges and shows tooltip
4. **Edge hover**: Shows mechanism of action and study details
5. **Search**: Find nodes by name
6. **Filters**:
   - Intervention categories (13 types)
   - Evidence types (positive/negative/neutral)
   - Confidence threshold slider
7. **Controls**:
   - Reset filters button
   - Center view button

### Sidebar Controls

- **Search box**: Filter nodes by name
- **Category checkboxes**: Show/hide intervention types
- **Evidence checkboxes**: Filter by positive/negative/neutral
- **Confidence slider**: Hide low-confidence relationships
- **Statistics**: Real-time counts of visible/total nodes and edges
- **Legend**: Visual key for colors and symbols

## Data Structure

The visualization loads from `data/network_graph.json`:

```json
{
  "nodes": [
    {
      "id": "acetaminophen",
      "name": "Acetaminophen",
      "type": "intervention",
      "category": "medication",
      "cluster_size": 1,
      "evidence_count": 3
    },
    {
      "id": "condition-name",
      "name": "Condition Name",
      "type": "condition"
    }
  ],
  "links": [
    {
      "source": "acetaminophen",
      "target": "condition-name",
      "mechanism": "reduced inflammation",
      "effect": "positive",
      "confidence": 0.65,
      "study_id": "12345678"
    }
  ],
  "metadata": { ... }
}
```

## Technology Stack

- **D3.js v7**: Force-directed graph layout and interactions
- **Vanilla JavaScript**: No framework dependencies
- **CSS3**: Glowing effects and styling
- **Single HTML file**: Self-contained and portable

## Performance

- **Load time**: <3 seconds for 895 nodes + 628 edges
- **Rendering**: SVG-based (good for <2000 nodes)
- **Interactions**: Smooth at 30+ FPS
- **Memory**: ~50 MB typical usage

## Known Issues / Future Improvements

### Current Limitations

- Text labels visible at all zoom levels (can be cluttered when zoomed out)
- No edge bundling (edges can overlap in dense areas)
- Node labels truncated at 20 characters
- Simulation can take 5-10 seconds to stabilize on load

### Potential Enhancements

1. **Level-of-detail rendering**: Hide labels when zoomed out
2. **Edge bundling**: Group parallel edges for clarity
3. **Canvas fallback**: For >1000 nodes, use Canvas instead of SVG
4. **Layout options**: Hierarchical, circular, or clustered layouts
5. **Mechanism clustering**: Visual grouping of nodes by shared mechanisms
6. **Export functionality**: Save graph as image or filtered data
7. **Animation controls**: Pause/play simulation, adjust speed
8. **Multi-select**: Select multiple nodes to analyze relationships
9. **Path finding**: Highlight paths between two nodes
10. **Time filtering**: Filter by publication year

## Development Workflow

### Iteration Cycle

1. Edit `index.html`
2. Refresh browser (Ctrl+F5 to bypass cache)
3. Test changes
4. Repeat

### Re-export Data

If database changes:
```bash
python -m back_end.src.utils.export_network_visualization_data
```

### Design Experiments

The single-file structure makes it easy to:
- Copy `index.html` to `index_v2.html` for A/B testing
- Test different color schemes
- Try alternative layouts
- Experiment with interaction patterns

## Merging to Production

When the design is finalized:

1. **Copy visualization**:
   ```bash
   cp index.html ../frontend/network.html
   ```

2. **Update data path** in `network.html`:
   ```javascript
   // Change: 'data/network_graph.json'
   // To:     'data/network_graph.json' (same path works)
   ```

3. **Update export script** to output to production location:
   ```python
   OUTPUT_PATH = PROJECT_ROOT / "frontend" / "data" / "network_graph.json"
   ```

4. **Add navigation link** in `frontend/index.html`:
   ```html
   <a href="network.html" class="nav-link">Network Visualization</a>
   ```

5. **Update main README** with visualization documentation

## Browser Compatibility

- **Chrome 90+**: Full support ✅
- **Edge 90+**: Full support ✅
- **Firefox 88+**: Full support ✅
- **Safari 14+**: Full support ✅
- **Mobile**: Touch-enabled drag, pinch-zoom supported 📱

## File Inventory

```
frontend_network_viz_experiment/
├── index.html           (main visualization - 500 lines)
├── data/
│   └── network_graph.json  (exported graph data - 345 KB)
└── README.md            (this file)
```

## Credits

- **Data**: Phase 4a Knowledge Graph (MyBiome Research Pipeline)
- **Visualization**: D3.js v7 force-directed layout
- **Design**: Cosmic dark theme inspired by network visualization best practices

## License

Part of the MyBiome Research Platform - Internal use only.

---

**Last Updated**: October 16, 2025
**Status**: Experimental prototype - ready for testing and iteration
