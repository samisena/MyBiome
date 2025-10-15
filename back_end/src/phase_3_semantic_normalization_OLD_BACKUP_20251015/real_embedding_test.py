"""
Phase 3.6 Test with Real nomic-embed-text Embeddings

Uses Ollama's nomic-embed-text model to generate semantic embeddings
for mechanism texts, then runs HDBSCAN clustering.
"""

import sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import requests
import time
from typing import List, Tuple, Dict

# Check if hdbscan is available
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("[ERROR] hdbscan not installed!")
    print("Install with: pip install hdbscan")
    exit(1)

# Configuration
DB_PATH = "c:/Users/samis/Desktop/MyBiome/back_end/data/processed/intervention_research.db"
OLLAMA_API_URL = "http://localhost:11434/api/embeddings"
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_DIM = 768

def check_ollama_connection():
    """Check if Ollama is running and model is available."""
    print("\n[Checking Ollama connection...]")

    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={"model": EMBEDDING_MODEL, "prompt": "test"},
            timeout=5
        )

        if response.status_code == 200:
            print(f"  Ollama API: Connected")
            print(f"  Model: {EMBEDDING_MODEL} available")
            return True
        else:
            print(f"  [ERROR] Ollama returned status {response.status_code}")
            print(f"  Response: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"  [ERROR] Cannot connect to Ollama at {OLLAMA_API_URL}")
        print(f"  Make sure Ollama is running: ollama serve")
        return False
    except Exception as e:
        print(f"  [ERROR] Connection check failed: {e}")
        return False

def generate_embeddings_batch(texts: List[str], batch_size: int = 10) -> np.ndarray:
    """Generate embeddings for texts using Ollama API in batches."""
    print(f"\n[3/6] Generating real embeddings using {EMBEDDING_MODEL}...")
    print(f"  Processing {len(texts)} mechanisms in batches of {batch_size}")

    embeddings = []
    failed_count = 0

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(texts) + batch_size - 1) // batch_size

        if batch_num % 10 == 0 or batch_num == 1:
            print(f"  Processing batch {batch_num}/{total_batches}...")

        for text in batch:
            try:
                response = requests.post(
                    OLLAMA_API_URL,
                    json={"model": EMBEDDING_MODEL, "prompt": text},
                    timeout=30
                )

                if response.status_code == 200:
                    embedding = response.json().get('embedding', [])
                    if len(embedding) == EMBEDDING_DIM:
                        embeddings.append(embedding)
                    else:
                        print(f"  [WARNING] Invalid embedding dimension: {len(embedding)}")
                        embeddings.append([0.0] * EMBEDDING_DIM)
                        failed_count += 1
                else:
                    print(f"  [WARNING] API error {response.status_code} for text: {text[:50]}...")
                    embeddings.append([0.0] * EMBEDDING_DIM)
                    failed_count += 1

            except Exception as e:
                print(f"  [WARNING] Failed to embed text: {text[:50]}... - {e}")
                embeddings.append([0.0] * EMBEDDING_DIM)
                failed_count += 1

        # Small delay to avoid overwhelming Ollama
        time.sleep(0.1)

    embeddings_array = np.array(embeddings, dtype=np.float32)

    # Normalize embeddings (unit vectors)
    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    embeddings_array = embeddings_array / (norms + 1e-9)

    print(f"\n  Generated {len(embeddings_array)} embeddings (shape: {embeddings_array.shape})")
    if failed_count > 0:
        print(f"  [WARNING] {failed_count} embeddings failed (using zero vectors)")

    return embeddings_array

def save_embeddings_cache(mechanisms: List[str], embeddings: np.ndarray, cache_path: str):
    """Save embeddings to cache file for reuse."""
    print(f"\n  Saving embeddings cache to {cache_path}...")

    cache_data = {
        'mechanisms': mechanisms,
        'embeddings': embeddings.tolist(),
        'model': EMBEDDING_MODEL,
        'dimension': EMBEDDING_DIM,
        'timestamp': datetime.now().isoformat()
    }

    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)

    with open(cache_path, 'w') as f:
        json.dump(cache_data, f)

    print(f"  Cache saved successfully")

def load_embeddings_cache(cache_path: str) -> Tuple[List[str], np.ndarray]:
    """Load embeddings from cache file."""
    print(f"\n  Loading embeddings cache from {cache_path}...")

    try:
        with open(cache_path, 'r') as f:
            cache_data = json.load(f)

        mechanisms = cache_data['mechanisms']
        embeddings = np.array(cache_data['embeddings'], dtype=np.float32)

        print(f"  Cache loaded: {len(mechanisms)} mechanisms")
        print(f"  Model: {cache_data.get('model', 'unknown')}")
        print(f"  Cached at: {cache_data.get('timestamp', 'unknown')}")

        return mechanisms, embeddings

    except FileNotFoundError:
        print(f"  Cache file not found")
        return None, None
    except Exception as e:
        print(f"  [ERROR] Failed to load cache: {e}")
        return None, None

def test_database_connection():
    """Test database connection and check for mechanisms."""
    print("="*80)
    print("PHASE 3.6 MECHANISM CLUSTERING - REAL EMBEDDINGS TEST")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("[1/6] Testing database connection...")

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='interventions'
        """)

        if not cursor.fetchone():
            print("[ERROR] interventions table not found!")
            return None

        print("  Database connection successful")
        print("  interventions table exists")

        return conn

    except Exception as e:
        print(f"[ERROR] Database connection failed: {e}")
        return None

def load_mechanisms(conn):
    """Load mechanism texts from database."""
    print("\n[2/6] Loading mechanisms from database...")

    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT mechanism
        FROM interventions
        WHERE mechanism IS NOT NULL
          AND mechanism != ''
          AND mechanism != 'N/A'
        ORDER BY mechanism
    """)

    mechanisms = [row[0] for row in cursor.fetchall()]

    print(f"  Loaded {len(mechanisms)} unique mechanism texts")

    if len(mechanisms) == 0:
        print("[ERROR] No valid mechanisms found!")
        return None

    # Show sample
    print("\n  Sample mechanisms (first 5):")
    for i, mech in enumerate(mechanisms[:5], 1):
        print(f"    {i}. {mech[:80]}{'...' if len(mech) > 80 else ''}")

    return mechanisms

def run_hdbscan_clustering(embeddings, min_cluster_size=5, min_samples=3,
                          cluster_selection_epsilon=0.0):
    """Run HDBSCAN clustering with specified hyperparameters."""
    print("\n[4/6] Running HDBSCAN clustering...")

    print(f"  Hyperparameters:")
    print(f"    min_cluster_size: {min_cluster_size}")
    print(f"    min_samples: {min_samples}")
    print(f"    cluster_selection_epsilon: {cluster_selection_epsilon}")

    # Run clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric='euclidean',
        cluster_selection_method='eom'
    )

    print("\n  Fitting HDBSCAN...")
    start_time = time.time()
    cluster_labels = clusterer.fit_predict(embeddings)
    elapsed = time.time() - start_time

    # Analyze results
    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    num_noise = list(cluster_labels).count(-1)
    singleton_percentage = (num_noise / len(cluster_labels)) * 100

    print(f"\n  Clustering complete (took {elapsed:.1f}s):")
    print(f"    Total mechanisms: {len(cluster_labels)}")
    print(f"    Clusters discovered: {num_clusters}")
    print(f"    Noise points (unclustered): {num_noise}")
    print(f"    Singleton percentage: {singleton_percentage:.1f}%")

    # Quality metrics
    if num_clusters > 1:
        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

        # Filter out noise points for quality metrics
        non_noise_mask = cluster_labels != -1
        if np.sum(non_noise_mask) > 0:
            silhouette = silhouette_score(embeddings[non_noise_mask], cluster_labels[non_noise_mask])
            davies_bouldin = davies_bouldin_score(embeddings[non_noise_mask], cluster_labels[non_noise_mask])
            calinski_harabasz = calinski_harabasz_score(embeddings[non_noise_mask], cluster_labels[non_noise_mask])

            print(f"\n  Quality metrics:")
            print(f"    Silhouette score: {silhouette:.3f} (target: >0.40)")
            print(f"    Davies-Bouldin index: {davies_bouldin:.3f} (target: <1.0)")
            print(f"    Calinski-Harabasz score: {calinski_harabasz:.1f} (higher = better)")

            return cluster_labels, clusterer, {
                'silhouette': silhouette,
                'davies_bouldin': davies_bouldin,
                'calinski_harabasz': calinski_harabasz,
                'num_clusters': num_clusters,
                'singleton_percentage': singleton_percentage
            }

    return cluster_labels, clusterer, {
        'num_clusters': num_clusters,
        'singleton_percentage': singleton_percentage
    }

def analyze_clusters(mechanisms, cluster_labels):
    """Analyze cluster composition."""
    print("\n[5/6] Analyzing cluster composition...")

    # Group mechanisms by cluster
    cluster_dict = {}
    for mechanism, label in zip(mechanisms, cluster_labels):
        if label not in cluster_dict:
            cluster_dict[label] = []
        cluster_dict[label].append(mechanism)

    # Sort by size (excluding noise cluster -1)
    clusters_sorted = sorted(
        [(k, v) for k, v in cluster_dict.items() if k != -1],
        key=lambda x: len(x[1]),
        reverse=True
    )

    # Cluster size distribution
    sizes = [len(v) for k, v in clusters_sorted]
    if sizes:
        avg_size = sum(sizes) / len(sizes)
        max_size = max(sizes)
        min_size = min(sizes)

        print(f"\n  Cluster size distribution:")
        print(f"    Average cluster size: {avg_size:.2f}")
        print(f"    Largest cluster: {max_size} members")
        print(f"    Smallest cluster: {min_size} members")

        # Size buckets
        buckets = {
            "Tiny (1-2)": sum(1 for s in sizes if s <= 2),
            "Small (3-5)": sum(1 for s in sizes if 3 <= s <= 5),
            "Medium (6-10)": sum(1 for s in sizes if 6 <= s <= 10),
            "Large (11-20)": sum(1 for s in sizes if 11 <= s <= 20),
            "Very Large (>20)": sum(1 for s in sizes if s > 20)
        }

        print("\n  Size buckets:")
        for bucket, count in buckets.items():
            percentage = (count / len(sizes)) * 100
            print(f"    {bucket}: {count} clusters ({percentage:.1f}%)")

    return clusters_sorted

def display_sample_clusters(clusters_sorted, max_display=10, max_members=7):
    """Display sample clusters for manual quality assessment."""
    print(f"\n[6/6] Sample clusters for manual quality assessment (top {max_display})...")
    print("="*80)

    for i, (cluster_id, members) in enumerate(clusters_sorted[:max_display], 1):
        print(f"\nCluster {cluster_id} ({len(members)} members):")
        print("-"*80)

        for j, mechanism in enumerate(members[:max_members], 1):
            print(f"  {j}. {mechanism[:75]}{'...' if len(mechanism) > 75 else ''}")

        if len(members) > max_members:
            print(f"  ... and {len(members) - max_members} more members")

    if len(clusters_sorted) > max_display:
        print(f"\n... and {len(clusters_sorted) - max_display} more clusters")

def print_summary(metrics, use_real_embeddings=True):
    """Print test summary."""
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    if metrics:
        silhouette = metrics.get('silhouette', 0)
        davies_bouldin = metrics.get('davies_bouldin', 0)
        num_clusters = metrics.get('num_clusters', 0)
        singleton_pct = metrics.get('singleton_percentage', 0)

        print("\nClustering Results:")
        print(f"  Clusters discovered: {num_clusters}")
        print(f"  Singleton percentage: {singleton_pct:.1f}%")

        if silhouette > 0:
            print("\nQuality Assessment:")
            if silhouette > 0.40 and davies_bouldin < 1.0:
                print("  [PASS] Clustering quality meets target thresholds")
            else:
                print("  [REVIEW NEEDED] Clustering quality below targets")
                if silhouette <= 0.40:
                    print(f"    - Silhouette score {silhouette:.3f} is below target (>0.40)")
                if davies_bouldin >= 1.0:
                    print(f"    - Davies-Bouldin index {davies_bouldin:.3f} is above target (<1.0)")

    if use_real_embeddings:
        print("\nEmbeddings:")
        print(f"  Using REAL {EMBEDDING_MODEL} embeddings")
        print(f"  These are semantic vectors that capture meaning")
    else:
        print("\nImportant Notes:")
        print("  1. This test uses MOCK EMBEDDINGS (random vectors)")
        print("  2. Real production system must use nomic-embed-text embeddings")

    print("\nManual Quality Check:")
    print("  Review the sample clusters above and assess:")
    print("  - Do mechanisms in the same cluster share similar concepts?")
    print("  - Are there obvious misclassifications?")
    print("  - Would canonical names be meaningful for these clusters?")

    print("\nNext Steps:")
    if use_real_embeddings:
        if metrics.get('silhouette', 0) < 0.40 or singleton_pct > 30:
            print("  1. Run hyperparameter experiments to optimize clustering")
            print("  2. Adjust min_cluster_size, min_samples, or epsilon")
            print("  3. Once quality improves: Proceed to Phases 6-8")
        else:
            print("  1. Quality looks good - proceed to hyperparameter fine-tuning")
            print("  2. Then integrate into main pipeline (Phases 6-8)")
    else:
        print("  1. Generate real embeddings using nomic-embed-text")
        print("  2. Re-run clustering with semantic embeddings")
        print("  3. Assess quality and optimize hyperparameters")

    print("\n" + "="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

def main():
    """Run Phase 3.6 test with real embeddings."""

    # Check Ollama connection
    if not check_ollama_connection():
        print("\n[ERROR] Cannot proceed without Ollama connection")
        print("Start Ollama with: ollama serve")
        print("Then pull the model: ollama pull nomic-embed-text")
        return

    # Step 1: Test database
    conn = test_database_connection()
    if not conn:
        return

    # Step 2: Load mechanisms
    mechanisms = load_mechanisms(conn)
    if not mechanisms:
        conn.close()
        return

    # Step 3: Generate or load embeddings
    cache_path = "back_end/data/semantic_normalization_cache/mechanism_embeddings_nomic.json"

    cached_mechanisms, cached_embeddings = load_embeddings_cache(cache_path)

    if cached_mechanisms is not None and cached_mechanisms == mechanisms:
        print("  Using cached embeddings (mechanisms match)")
        embeddings = cached_embeddings
    else:
        print("  Cache miss - generating new embeddings")
        embeddings = generate_embeddings_batch(mechanisms, batch_size=10)
        save_embeddings_cache(mechanisms, embeddings, cache_path)

    # Step 4: Run clustering
    cluster_labels, clusterer, metrics = run_hdbscan_clustering(embeddings)

    # Step 5: Analyze clusters
    clusters_sorted = analyze_clusters(mechanisms, cluster_labels)

    # Step 6: Display sample clusters
    display_sample_clusters(clusters_sorted)

    # Summary
    print_summary(metrics, use_real_embeddings=True)

    # Cleanup
    conn.close()

if __name__ == "__main__":
    main()
