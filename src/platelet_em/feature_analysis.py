"""Quantitative morphological feature analysis and activation grading.

This module operates on the annotated images produced by the Phase-1
pre-annotation pipeline. It extracts the granule contours (marked in blue) and
derives a battery of spatial-statistics descriptors that characterize the
platelet's activation state:

* Pairwise and nearest-neighbour distance metrics.
* DBSCAN clustering of granule centroids.
* Proximity / grouping analysis.
* Spatial-distribution indices, including the Clark-Evans index.

These descriptors feed a transparent, rule-based scoring scheme that grades the
platelet as ``UNACTIVATED``, ``PARTIALLY_ACTIVATED``, or ``ACTIVATED`` with an
associated confidence. The same morphological descriptors are intended to be
fused with CNN embeddings in the deep-learning grading stage (Phase 2).

.. note::
   This rule-based analyzer is the interpretable baseline / feature source. The
   learned multi-class grader (Grades 0-3) described in the accompanying paper
   lives under :mod:`platelet_em.deep_learning` and will be released separately.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance_matrix
import pandas as pd
from collections import Counter
import seaborn as sns


class AlphaGranuleActivationAnalyzer:
    def __init__(self, image_path):
        """
        Initialize the analyzer with the annotated platelet image

        Args:
            image_path: Path to the annotated platelet image
        """
        self.image = cv2.imread(image_path)
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.alpha_granules = []
        self.centroids = []
        self.activation_metrics = {}

    def extract_alpha_granules(self, color_channel='blue'):
        """
        Extract alpha granule contours from the blue channel
        Assumes alpha granules are marked in blue color
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        if color_channel == 'blue':
            # Define blue color range for alpha granules
            lower_blue = np.array([100, 50, 50])
            upper_blue = np.array([130, 255, 255])
            mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area (remove noise)
        min_area = 100  # Adjust based on your image scale
        max_area = 50000  # Adjust based on your image scale

        filtered_contours = []
        centroids = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                filtered_contours.append(contour)

                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centroids.append([cx, cy])

        self.alpha_granules = filtered_contours
        self.centroids = np.array(centroids)

        print(f"Found {len(self.alpha_granules)} alpha granules")
        return len(self.alpha_granules)

    def calculate_distance_metrics(self):
        """Calculate various distance-based metrics"""
        if len(self.centroids) < 2:
            return {"error": "Not enough granules for analysis"}

        # Calculate pairwise distances
        distances = pdist(self.centroids)
        distance_matrix = squareform(distances)

        # Remove diagonal (distance to self = 0)
        np.fill_diagonal(distance_matrix, np.inf)

        # Nearest neighbor distances
        nearest_distances = np.min(distance_matrix, axis=1)

        # Average nearest neighbor distance
        avg_nearest_distance = np.mean(nearest_distances)

        # Overall statistics
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)

        return {
            'avg_nearest_distance': avg_nearest_distance,
            'mean_all_distances': mean_distance,
            'std_distances': std_distance,
            'nearest_distances': nearest_distances,
            'distance_matrix': distance_matrix
        }

    def perform_clustering_analysis(self, eps_ratio=0.3):
        """
        Perform DBSCAN clustering on alpha granules

        Args:
            eps_ratio: Ratio of average nearest neighbor distance to use as eps
        """
        if len(self.centroids) < 2:
            return {"error": "Not enough granules for clustering"}

        # Calculate eps based on average nearest neighbor distance
        distance_metrics = self.calculate_distance_metrics()
        eps = distance_metrics['avg_nearest_distance'] * eps_ratio

        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=2).fit(self.centroids)
        labels = clustering.labels_

        # Number of clusters (excluding noise points labeled as -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        # Calculate clustering metrics
        clustered_granules = len(labels) - n_noise
        clustering_percentage = (clustered_granules / len(labels)) * 100

        # Cluster sizes
        cluster_sizes = []
        if n_clusters > 0:
            cluster_counter = Counter([label for label in labels if label != -1])
            cluster_sizes = list(cluster_counter.values())

        return {
            'labels': labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'clustered_granules': clustered_granules,
            'clustering_percentage': clustering_percentage,
            'cluster_sizes': cluster_sizes,
            'eps_used': eps
        }

    def calculate_proximity_analysis(self, proximity_threshold_ratio=2.0):
        """
        Analyze granules based on proximity to neighbors

        Args:
            proximity_threshold_ratio: Multiple of average granule size to use as threshold
        """
        if len(self.centroids) < 2:
            return {"error": "Not enough granules for analysis"}

        # Estimate average granule size from contour areas
        areas = [cv2.contourArea(contour) for contour in self.alpha_granules]
        avg_area = np.mean(areas)
        avg_radius = np.sqrt(avg_area / np.pi)

        proximity_threshold = avg_radius * proximity_threshold_ratio

        # Calculate distance matrix
        distance_metrics = self.calculate_distance_metrics()
        distance_matrix = distance_metrics['distance_matrix']

        # Count neighbors within threshold for each granule
        neighbor_counts = []
        for i in range(len(self.centroids)):
            neighbors = np.sum(distance_matrix[i] < proximity_threshold)
            neighbor_counts.append(neighbors)

        # Calculate metrics
        avg_neighbors = np.mean(neighbor_counts)
        isolated_granules = np.sum(np.array(neighbor_counts) == 0)
        grouped_granules = len(neighbor_counts) - isolated_granules
        grouping_percentage = (grouped_granules / len(neighbor_counts)) * 100

        return {
            'proximity_threshold': proximity_threshold,
            'neighbor_counts': neighbor_counts,
            'avg_neighbors': avg_neighbors,
            'isolated_granules': isolated_granules,
            'grouped_granules': grouped_granules,
            'grouping_percentage': grouping_percentage
        }

    def calculate_spatial_distribution_index(self):
        """
        Calculate various spatial distribution indices
        """
        if len(self.centroids) < 3:
            return {"error": "Not enough granules for spatial analysis"}

        # Calculate variance-to-mean ratio for x and y coordinates
        x_coords = self.centroids[:, 0]
        y_coords = self.centroids[:, 1]

        x_vmr = np.var(x_coords) / np.mean(x_coords) if np.mean(x_coords) > 0 else 0
        y_vmr = np.var(y_coords) / np.mean(y_coords) if np.mean(y_coords) > 0 else 0

        # Calculate nearest neighbor distances
        distance_metrics = self.calculate_distance_metrics()
        nearest_distances = distance_metrics['nearest_distances']

        # Clark-Evans index (ratio of observed to expected nearest neighbor distance)
        # Expected distance for random distribution = 0.5 / sqrt(density)
        image_area = self.image.shape[0] * self.image.shape[1]
        density = len(self.centroids) / image_area
        expected_distance = 0.5 / np.sqrt(density)
        clark_evans_index = np.mean(nearest_distances) / expected_distance

        return {
            'x_variance_to_mean_ratio': x_vmr,
            'y_variance_to_mean_ratio': y_vmr,
            'clark_evans_index': clark_evans_index,
            'spatial_distribution_score': (x_vmr + y_vmr) / 2
        }

    def classify_activation_state(self):
        """
        Classify the activation state based on multiple metrics
        """
        # Perform all analyses
        clustering_results = self.perform_clustering_analysis()
        proximity_results = self.calculate_proximity_analysis()
        spatial_results = self.calculate_spatial_distribution_index()
        distance_metrics = self.calculate_distance_metrics()

        if any('error' in result for result in [clustering_results, proximity_results, spatial_results]):
            return {"classification": "INSUFFICIENT_DATA", "confidence": 0.0}

        # Define activation criteria and scoring
        activation_score = 0
        criteria_met = []

        # Criterion 1: Clustering percentage (weight: 0.3)
        clustering_pct = clustering_results['clustering_percentage']
        if clustering_pct > 60:
            activation_score += 0.3
            criteria_met.append(f"High clustering: {clustering_pct:.1f}%")
        elif clustering_pct > 40:
            activation_score += 0.15
            criteria_met.append(f"Moderate clustering: {clustering_pct:.1f}%")

        # Criterion 2: Grouping percentage (weight: 0.25)
        grouping_pct = proximity_results['grouping_percentage']
        if grouping_pct > 70:
            activation_score += 0.25
            criteria_met.append(f"High grouping: {grouping_pct:.1f}%")
        elif grouping_pct > 50:
            activation_score += 0.125
            criteria_met.append(f"Moderate grouping: {grouping_pct:.1f}%")

        # Criterion 3: Average neighbors (weight: 0.2)
        avg_neighbors = proximity_results['avg_neighbors']
        if avg_neighbors > 2:
            activation_score += 0.2
            criteria_met.append(f"High neighbor count: {avg_neighbors:.1f}")
        elif avg_neighbors > 1:
            activation_score += 0.1
            criteria_met.append(f"Moderate neighbor count: {avg_neighbors:.1f}")

        # Criterion 4: Clark-Evans index (weight: 0.15)
        clark_evans = spatial_results['clark_evans_index']
        if clark_evans < 0.8:  # Less than random = clustered
            activation_score += 0.15
            criteria_met.append(f"Clustered distribution (CE: {clark_evans:.2f})")
        elif clark_evans < 1.0:
            activation_score += 0.075

        # Criterion 5: Number of clusters relative to total granules (weight: 0.1)
        n_clusters = clustering_results['n_clusters']
        n_granules = len(self.centroids)
        cluster_ratio = n_clusters / n_granules if n_granules > 0 else 1

        if cluster_ratio < 0.5:  # Few clusters relative to granules = more clustering
            activation_score += 0.1
            criteria_met.append(f"Few clusters relative to granules: {cluster_ratio:.2f}")
        elif cluster_ratio < 0.7:
            activation_score += 0.05

        # Classification based on activation score
        if activation_score >= 0.7:
            classification = "ACTIVATED"
            confidence = min(activation_score, 1.0)
        elif activation_score >= 0.4:
            classification = "PARTIALLY_ACTIVATED"
            confidence = activation_score
        else:
            classification = "UNACTIVATED"
            confidence = 1.0 - activation_score

        # Store comprehensive results
        self.activation_metrics = {
            'classification': classification,
            'confidence': confidence,
            'activation_score': activation_score,
            'criteria_met': criteria_met,
            'clustering_results': clustering_results,
            'proximity_results': proximity_results,
            'spatial_results': spatial_results,
            'distance_metrics': distance_metrics,
            'n_granules': n_granules
        }

        return self.activation_metrics

    def visualize_analysis(self, save_path=None):
        """
        Create comprehensive visualization of the analysis
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Alpha Granule Activation Analysis', fontsize=16, fontweight='bold')

        # Original image with granules marked
        axes[0, 0].imshow(self.image_rgb)
        axes[0, 0].scatter(self.centroids[:, 0], self.centroids[:, 1],
                           c='red', s=50, alpha=0.7, marker='x')
        axes[0, 0].set_title(f'Alpha Granules Detected (n={len(self.centroids)})')
        axes[0, 0].axis('off')

        # Clustering visualization
        if 'clustering_results' in self.activation_metrics:
            labels = self.activation_metrics['clustering_results']['labels']
            unique_labels = set(labels)
            colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

            axes[0, 1].imshow(self.image_rgb, alpha=0.3)
            for k, col in zip(unique_labels, colors):
                if k == -1:
                    col = [0, 0, 0, 1]  # Black for noise

                class_member_mask = (labels == k)
                xy = self.centroids[class_member_mask]
                axes[0, 1].scatter(xy[:, 0], xy[:, 1], c=[col], s=100, alpha=0.8)

            axes[0, 1].set_title(f'DBSCAN Clustering\n'
                                 f'{self.activation_metrics["clustering_results"]["n_clusters"]} clusters')
            axes[0, 1].axis('off')

        # Distance distribution
        if 'distance_metrics' in self.activation_metrics:
            nearest_distances = self.activation_metrics['distance_metrics']['nearest_distances']
            axes[0, 2].hist(nearest_distances, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 2].axvline(np.mean(nearest_distances), color='red', linestyle='--',
                               label=f'Mean: {np.mean(nearest_distances):.1f}')
            axes[0, 2].set_title('Nearest Neighbor Distances')
            axes[0, 2].set_xlabel('Distance (pixels)')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].legend()

        # Activation metrics summary
        axes[1, 0].axis('off')
        if hasattr(self, 'activation_metrics') and self.activation_metrics:
            metrics_text = f"""
CLASSIFICATION: {self.activation_metrics['classification']}
Confidence: {self.activation_metrics['confidence']:.2f}
Activation Score: {self.activation_metrics['activation_score']:.2f}

Key Metrics:
• Clustering %: {self.activation_metrics['clustering_results']['clustering_percentage']:.1f}%
• Grouping %: {self.activation_metrics['proximity_results']['grouping_percentage']:.1f}%
• Avg Neighbors: {self.activation_metrics['proximity_results']['avg_neighbors']:.1f}
• Clark-Evans Index: {self.activation_metrics['spatial_results']['clark_evans_index']:.2f}

Criteria Met:
{chr(10).join(['• ' + criterion for criterion in self.activation_metrics['criteria_met']])}
            """
            axes[1, 0].text(0.05, 0.95, metrics_text, transform=axes[1, 0].transAxes,
                            fontsize=10, verticalalignment='top', fontfamily='monospace',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

        # Cluster size distribution
        if 'clustering_results' in self.activation_metrics:
            cluster_sizes = self.activation_metrics['clustering_results']['cluster_sizes']
            if cluster_sizes:
                axes[1, 1].bar(range(len(cluster_sizes)), cluster_sizes, color='lightcoral')
                axes[1, 1].set_title('Cluster Size Distribution')
                axes[1, 1].set_xlabel('Cluster ID')
                axes[1, 1].set_ylabel('Number of Granules')
            else:
                axes[1, 1].text(0.5, 0.5, 'No clusters detected',
                                ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Cluster Size Distribution')

        # Neighbor count distribution
        if 'proximity_results' in self.activation_metrics:
            neighbor_counts = self.activation_metrics['proximity_results']['neighbor_counts']
            axes[1, 2].hist(neighbor_counts, bins=range(max(neighbor_counts) + 2),
                            alpha=0.7, color='lightgreen', edgecolor='black')
            axes[1, 2].set_title('Neighbor Count Distribution')
            axes[1, 2].set_xlabel('Number of Neighbors')
            axes[1, 2].set_ylabel('Frequency')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")

        plt.show()

    def generate_report(self):
        """Generate a comprehensive text report"""
        if not hasattr(self, 'activation_metrics') or not self.activation_metrics:
            self.classify_activation_state()

        report = f"""
========================================
ALPHA GRANULE ACTIVATION ANALYSIS REPORT
========================================

CLASSIFICATION: {self.activation_metrics['classification']}
Confidence Level: {self.activation_metrics['confidence']:.2f} ({self.activation_metrics['confidence']*100:.1f}%)
Overall Activation Score: {self.activation_metrics['activation_score']:.3f}

GRANULE COUNT: {self.activation_metrics['n_granules']}

CLUSTERING ANALYSIS:
• Number of clusters: {self.activation_metrics['clustering_results']['n_clusters']}
• Granules in clusters: {self.activation_metrics['clustering_results']['clustered_granules']} ({self.activation_metrics['clustering_results']['clustering_percentage']:.1f}%)
• Isolated granules: {self.activation_metrics['clustering_results']['n_noise']}
• Average cluster size: {np.mean(self.activation_metrics['clustering_results']['cluster_sizes']) if self.activation_metrics['clustering_results']['cluster_sizes'] else 0:.1f}

PROXIMITY ANALYSIS:
• Granules with neighbors: {self.activation_metrics['proximity_results']['grouped_granules']} ({self.activation_metrics['proximity_results']['grouping_percentage']:.1f}%)
• Isolated granules: {self.activation_metrics['proximity_results']['isolated_granules']}
• Average neighbors per granule: {self.activation_metrics['proximity_results']['avg_neighbors']:.2f}
• Proximity threshold used: {self.activation_metrics['proximity_results']['proximity_threshold']:.1f} pixels

SPATIAL DISTRIBUTION:
• Clark-Evans Index: {self.activation_metrics['spatial_results']['clark_evans_index']:.3f}
  ({'Clustered' if self.activation_metrics['spatial_results']['clark_evans_index'] < 1.0 else 'Random/Dispersed'})
• Spatial distribution score: {self.activation_metrics['spatial_results']['spatial_distribution_score']:.3f}

DISTANCE METRICS:
• Average nearest neighbor distance: {self.activation_metrics['distance_metrics']['avg_nearest_distance']:.1f} pixels
• Mean all pairwise distances: {self.activation_metrics['distance_metrics']['mean_all_distances']:.1f} pixels

ACTIVATION CRITERIA MET:
{chr(10).join(['• ' + criterion for criterion in self.activation_metrics['criteria_met']])}

INTERPRETATION:
        """

        if self.activation_metrics['classification'] == 'ACTIVATED':
            report += """
The alpha granules show ACTIVATED state characteristics:
- High degree of clustering/grouping
- Granules are concentrated in specific areas
- Low spatial dispersion
- Multiple granules in close proximity
This suggests the platelet is in an activated state, likely in response to stimulation.
            """
        elif self.activation_metrics['classification'] == 'PARTIALLY_ACTIVATED':
            report += """
The alpha granules show PARTIALLY ACTIVATED state characteristics:
- Moderate clustering with some dispersed granules
- Mixed pattern of grouped and isolated granules
- Intermediate spatial organization
This suggests the platelet may be in early stages of activation or partially stimulated.
            """
        else:
            report += """
The alpha granules show UNACTIVATED state characteristics:
- Low degree of clustering
- Granules are more uniformly distributed
- High spatial dispersion
- Most granules are isolated
This suggests the platelet is in a resting, unactivated state.
            """

        report += "\n========================================\n"

        return report


def analyze_platelet_activation(image_path, visualization_save_path=None):
    """
    Complete analysis workflow for platelet activation state

    Args:
        image_path: Path to annotated platelet image
        visualization_save_path: Optional path to save visualization

    Returns:
        Dictionary containing analysis results
    """
    # Initialize analyzer
    analyzer = AlphaGranuleActivationAnalyzer(image_path)

    # Extract alpha granules
    n_granules = analyzer.extract_alpha_granules()

    if n_granules < 3:
        print("Warning: Too few alpha granules detected for reliable analysis")
        return {"error": "Insufficient granules for analysis"}

    # Perform classification
    results = analyzer.classify_activation_state()

    # Generate visualization
    analyzer.visualize_analysis(visualization_save_path)

    # Print report
    print(analyzer.generate_report())

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Rule-based platelet activation analysis from an annotated image."
    )
    parser.add_argument("image_path", help="Path to the annotated platelet image.")
    parser.add_argument(
        "-s", "--save", default="activation_analysis_results.png",
        help="Path to save the analysis visualization figure.",
    )
    args = parser.parse_args()

    results = analyze_platelet_activation(
        image_path=args.image_path,
        visualization_save_path=args.save,
    )

    print(f"\nFinal Classification: {results.get('classification', 'ERROR')}")
    print(f"Confidence: {results.get('confidence', 0):.2f}")
