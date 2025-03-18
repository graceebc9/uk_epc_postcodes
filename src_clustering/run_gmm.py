
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os
from datetime import datetime


def run_gmm_clustering(
    dfgeo_full, 
    cols, 
    input_name = 'LAD21NM', 
    run_name=None, 
    output_dir=None, 
    n_components_range=range(2, 18),
    exclude_lads=None,


):
    """
    Run GMM clustering analysis on geographic data and save results to specified directory.
    
    Args:
        dfgeo_full (DataFrame): Input geodataframe with features and geometry
        cols (list): List of column names to use as features for clustering
        input_name": name of col with identifying feature of inputs 
        run_name (str, optional): Name for this clustering run. Defaults to timestamp if None.
        output_dir (str, optional): Directory to save outputs. Created if doesn't exist.
        n_components_range (range, optional): Range of cluster numbers to evaluate.
        exclude_lads (list, optional): List of unit names to exclude from analysis, (must be members of input_name)
        name_mapping (dict, optional): Mapping of LAD names to more readable names.
        
    Returns:
        dict: Results containing best model, cluster assignments, and evaluation metrics
    """
    # Create run name if not provided
    if run_name is None:
        run_name = f"gmm_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Set up output directory
    if output_dir is None:
        output_dir = "clustering_results"
    
    # Create full output path for this run
    run_output_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_output_dir, exist_ok=True)
    
    print(f"Starting GMM clustering run: {run_name}")
    print(f"Results will be saved to: {run_output_dir}")
    
    # Create log file for this run
    log_path = os.path.join(run_output_dir, "run_log.txt")
    with open(log_path, "w") as log_file:
        log_file.write(f"GMM Clustering Run: {run_name}\n")
        log_file.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Features used: {cols}\n\n")
    
    # Helper function to log both to console and file
    def log_message(message):
        print(message)
        with open(log_path, "a") as log_file:
            log_file.write(message + "\n")
    
    # Data preparation
    log_message("Preparing data...")
    df_geo = dfgeo_full[cols + ['geometry', input_name]].dropna()
    
    # Exclude specified LADs if any
    if exclude_lads:
        for lad in exclude_lads:
            df_geo = df_geo[df_geo[input_name] != lad]
        log_message(f"Excluded LADs: {exclude_lads}")
    
    # Extract the features for clustering
    X = df_geo[cols].values
    
    # Scale the data (important for GMM)
    log_message("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Function to evaluate GMM for different numbers of components
    def evaluate_gmm(X_scaled, n_components_range):
        results = {}
        
        for n_components in n_components_range:
            log_message(f"Evaluating GMM with {n_components} components")
            
            # Fit GMM model
            gmm = GaussianMixture(
                n_components=n_components, 
                random_state=42,
                n_init=10,  # Multiple initializations to find better solution
                covariance_type='full',
                max_iter=200
            )
            
            # Fit and predict
            gmm.fit(X_scaled)
            labels = gmm.predict(X_scaled)
            
            # Collect metrics
            results[n_components] = {
                'bic': gmm.bic(X_scaled),
                'aic': gmm.aic(X_scaled),
                'log_likelihood': gmm.score(X_scaled) * X_scaled.shape[0],
                'gmm': gmm,
                'labels': labels
            }
            
            # Silhouette score can only be calculated if n_components > 1
            if n_components > 1:
                results[n_components]['silhouette'] = silhouette_score(X_scaled, labels)
                results[n_components]['davies_bouldin'] = davies_bouldin_score(X_scaled, labels)
            else:
                results[n_components]['silhouette'] = 0
                results[n_components]['davies_bouldin'] = float('inf')  # Lower is better
        
        return results
    
    # Evaluate GMM
    log_message(f"Evaluating models for {len(n_components_range)} different cluster numbers...")
    gmm_results = evaluate_gmm(X_scaled, n_components_range)
    
    # Extract results for plotting
    bic = []
    aic = []
    silhouette_scores = []
    davies_bouldin_scores = []
    log_likelihood = []
    
    for n_components in n_components_range:
        bic.append(gmm_results[n_components]['bic'])
        aic.append(gmm_results[n_components]['aic'])
        silhouette_scores.append(gmm_results[n_components]['silhouette'])
        davies_bouldin_scores.append(gmm_results[n_components]['davies_bouldin'])
        log_likelihood.append(gmm_results[n_components]['log_likelihood'])
    
    # Create a grid of plots for evaluation metrics
    log_message("Generating evaluation metrics plots...")
    plt.figure(figsize=(20, 15))
    gs = GridSpec(3, 2, figure=plt.gcf())
    
    # Plot BIC and AIC
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(n_components_range, bic, 'o-', label='BIC', color='blue')
    ax1.plot(n_components_range, aic, 'o-', label='AIC', color='red')
    ax1.set_xlabel('Number of components')
    ax1.set_ylabel('Information Criterion')
    ax1.set_title('BIC and AIC (lower is better)')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot Silhouette Score
    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(n_components_range, silhouette_scores, 'o-', color='green')
    ax2.set_xlabel('Number of components')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score (higher is better)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Plot Davies-Bouldin Score
    ax3 = plt.subplot(gs[1, 0])
    ax3.plot(n_components_range, davies_bouldin_scores, 'o-', color='purple')
    ax3.set_xlabel('Number of components')
    ax3.set_ylabel('Davies-Bouldin Score')
    ax3.set_title('Davies-Bouldin Score (lower is better)')
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # Plot Log-Likelihood
    ax4 = plt.subplot(gs[1, 1])
    ax4.plot(n_components_range, log_likelihood, 'o-', color='orange')
    ax4.set_xlabel('Number of components')
    ax4.set_ylabel('Log-Likelihood')
    ax4.set_title('Log-Likelihood (higher is better)')
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    # Find best number of components according to different metrics
    best_bic = n_components_range[np.argmin(bic)]
    best_aic = n_components_range[np.argmin(aic)]
    best_silhouette = n_components_range[np.argmax(silhouette_scores)]
    best_davies_bouldin = n_components_range[np.argmin(davies_bouldin_scores)]
    
    # PCA for visualization
    log_message("Performing PCA for visualization...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Plot explained variance ratio
    ax5 = plt.subplot(gs[2, 0])
    explained_variance = pca.explained_variance_ratio_
    ax5.bar(range(1, len(explained_variance) + 1), explained_variance)
    ax5.set_xlabel('Principal Component')
    ax5.set_ylabel('Explained Variance Ratio')
    ax5.set_title('PCA Explained Variance')
    ax5.grid(True, linestyle='--', alpha=0.7)
    
    # Log summary of best models
    log_message(f"Best number of components according to BIC: {best_bic}")
    log_message(f"Best number of components according to AIC: {best_aic}")
    log_message(f"Best number of components according to Silhouette Score: {best_silhouette}")
    log_message(f"Best number of components according to Davies-Bouldin Score: {best_davies_bouldin}")
    
    # Get consensus on best number of components using a voting mechanism
    votes = {}
    for metric, best in zip(['BIC', 'AIC', 'Silhouette', 'Davies-Bouldin'], 
                            [best_bic, best_aic, best_silhouette, best_davies_bouldin]):
        if best not in votes:
            votes[best] = []
        votes[best].append(metric)
    
    # Display the voting results
    log_message("\nVoting results for best number of components:")
    for n_components, metrics in votes.items():
        log_message(f"{n_components} components: {len(metrics)} votes from {', '.join(metrics)}")
    
    # Choose the number of components that received the most votes
    best_consensus = max(votes.items(), key=lambda x: len(x[1]))[0]
    log_message(f"\nBest consensus number of components: {best_consensus}")
    
    # Use best model according to consensus
    best_gmm = gmm_results[best_consensus]['gmm']
    best_labels = gmm_results[best_consensus]['labels']
    
    # Plot the best model with PCA projection
    ax6 = plt.subplot(gs[2, 1])
    scatter = ax6.scatter(X_pca[:, 0], X_pca[:, 1], c=best_labels, cmap='viridis', alpha=0.8)
    ax6.set_xlabel('PCA Component 1')
    ax6.set_ylabel('PCA Component 2')
    ax6.set_title(f'GMM Clustering with {best_consensus} Components')
    plt.colorbar(scatter, ax=ax6, label='Cluster')
    
    plt.tight_layout()
    eval_plot_path = os.path.join(run_output_dir, 'gmm_evaluation.png')
    plt.savefig(eval_plot_path, dpi=300, bbox_inches='tight')
    log_message(f"Saved evaluation metrics plot to {eval_plot_path}")
    plt.close()
    
    # Visualize selected cluster models
    log_message("Generating visualizations for selected cluster models...")
    for n_components in [2, 3, 4, 5]:
        if n_components in n_components_range:
            labels = gmm_results[n_components]['labels']
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.8, s=50)
            plt.colorbar(scatter, label='Cluster')
            
            # Add centroids if available (not directly available from GMM but can be computed)
            means = gmm_results[n_components]['gmm'].means_
            if means.shape[1] > 2:
                means_pca = pca.transform(means)
                plt.scatter(means_pca[:, 0], means_pca[:, 1], c='red', marker='X', s=200, 
                           edgecolor='black', label='Cluster Centers')
            
            plt.title(f'GMM with {n_components} Components')
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            cluster_plot_path = os.path.join(run_output_dir, f'gmm_cluster_{n_components}.png')
            plt.savefig(cluster_plot_path, dpi=300, bbox_inches='tight')
            log_message(f"Saved {n_components}-cluster model plot to {cluster_plot_path}")
            plt.close()
    
    # Feature importance analysis for the best model
    log_message("Performing feature importance analysis for the best model...")
    if best_consensus > 0:
        # Add cluster labels to the original dataframe
        df_geo['cluster'] = best_labels
        
        # Save cluster assignments to CSV
        cluster_df = df_geo[[input_name, 'cluster'] + cols]
        cluster_csv_path = os.path.join(run_output_dir, 'cluster_assignments.csv')
        cluster_df.to_csv(cluster_csv_path, index=False)
        log_message(f"Saved cluster assignments to {cluster_csv_path}")
        
        # Analyze feature distributions across clusters
        plt.figure(figsize=(18, 12))
        for i, feature in enumerate(cols):
            plt.subplot(2, 4, i+1)
            sns.boxplot(x='cluster', y=feature, data=df_geo)
            plt.title(f'Distribution of {feature} by Cluster')
            plt.xlabel('Cluster')
            plt.ylabel(feature)
            plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        feature_dist_path = os.path.join(run_output_dir, 'feature_distribution_by_cluster.png')
        plt.savefig(feature_dist_path, dpi=300, bbox_inches='tight')
        log_message(f"Saved feature distribution plot to {feature_dist_path}")
        plt.close()
        
        # Create a profile of each cluster
        cluster_profiles = df_geo.groupby('cluster')[cols].mean()
        
        # Save cluster profiles to CSV
        profiles_csv_path = os.path.join(run_output_dir, 'cluster_profiles.csv')
        cluster_profiles.to_csv(profiles_csv_path)
        log_message(f"Saved cluster profiles to {profiles_csv_path}")
        
        # Normalize to show relative importance
        cluster_profiles_norm = (cluster_profiles - cluster_profiles.mean()) / cluster_profiles.std()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(cluster_profiles_norm, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
        plt.title('Normalized Cluster Profiles')
        
        profiles_plot_path = os.path.join(run_output_dir, 'cluster_profiles.png')
        plt.savefig(profiles_plot_path, dpi=300, bbox_inches='tight')
        log_message(f"Saved cluster profiles plot to {profiles_plot_path}")
        plt.close()
        
        # Print and log cluster sizes
        log_message("\nCluster sizes:")
        cluster_sizes = df_geo['cluster'].value_counts().sort_index()
        cluster_size_data = []
        for cluster, size in cluster_sizes.items():
            percentage = size/len(df_geo)*100
            log_message(f"Cluster {cluster}: {size} samples ({percentage:.2f}%)")
            cluster_size_data.append({
                'cluster': cluster,
                'size': size,
                'percentage': percentage
            })
        
        # Save cluster sizes to CSV
        sizes_df = pd.DataFrame(cluster_size_data)
        sizes_csv_path = os.path.join(run_output_dir, 'cluster_sizes.csv')
        sizes_df.to_csv(sizes_csv_path, index=False)
        log_message(f"Saved cluster size information to {sizes_csv_path}")
        
        # Plot geographic distribution
        plt.figure(figsize=(10, 8))
        geo_plot = df_geo.plot('cluster', cmap='Paired', legend=True, ax=plt.gca())
        plt.title(f'Region Clustering with {best_consensus} Clusters')
        plt.tight_layout()
        
        geo_plot_path = os.path.join(run_output_dir, 'geographic_clusters.png')
        plt.savefig(geo_plot_path, dpi=300, bbox_inches='tight')
        log_message(f"Saved geographic clustering plot to {geo_plot_path}")
        plt.close()
    
    # Create a summary of the run results
    summary = {
        'run_name': run_name,
        'output_dir': run_output_dir,
        'best_consensus_clusters': best_consensus,
        'best_models': {
            'BIC': best_bic,
            'AIC': best_aic,
            'Silhouette': best_silhouette,
            'Davies-Bouldin': best_davies_bouldin
        },
        'metrics': {
            'bic': bic,
            'aic': aic,
            'silhouette_scores': silhouette_scores,
            'davies_bouldin_scores': davies_bouldin_scores,
            'log_likelihood': log_likelihood
        },
        'best_model': best_gmm,
        'best_labels': best_labels,
        'features_used': cols,
        'pca': pca,
        'scaler': scaler,
        'df_geo': df_geo  # With cluster assignments
    }
    
    log_message(f"\nGMM clustering run '{run_name}' completed successfully!")
    return summary

