import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import json

class TS2VecMSMAnalyzer:
    """
    Analyzer for TS2Vec-MSM experimental results.
    Compares performance across different configurations and tasks.
    """
    
    def __init__(self, results_dir='training'):
        self.results_dir = results_dir
        self.results = {}
        
    def load_results(self):
        """Load all experimental results from the training directory."""
        result_dirs = glob.glob(os.path.join(self.results_dir, '*'))
        
        for result_dir in result_dirs:
            if os.path.isdir(result_dir):
                try:
                    # Extract experiment info from directory name
                    dir_name = os.path.basename(result_dir)
                    parts = dir_name.split('__')
                    if len(parts) >= 2:
                        dataset = parts[0]
                        run_info = parts[1].split('_')[0]
                        
                        # Load evaluation results
                        eval_file = os.path.join(result_dir, 'eval_res.npy')
                        if os.path.exists(eval_file):
                            eval_res = np.load(eval_file, allow_pickle=True).item()
                            
                            # Store results
                            if dataset not in self.results:
                                self.results[dataset] = {}
                            self.results[dataset][run_info] = eval_res
                            
                except Exception as e:
                    print(f"Error loading {result_dir}: {e}")
    
    def compare_configurations(self, dataset, task_type):
        """
        Compare different configurations for a specific dataset and task.
        """
        if dataset not in self.results:
            print(f"No results found for dataset: {dataset}")
            return
        
        configurations = ['contrastive', 'msm', 'hybrid_static', 'hybrid_dynamic']
        
        print(f"\n=== {dataset.upper()} - {task_type.upper()} RESULTS ===")
        print("-" * 60)
        
        results_df = pd.DataFrame()
        
        for config in configurations:
            if config in self.results[dataset]:
                result = self.results[dataset][config]
                
                if task_type == 'classification':
                    # Extract accuracy
                    if 'ours' in result:
                        accuracy = result['ours']['acc']
                        print(f"{config:15}: Accuracy = {accuracy:.4f}")
                        results_df.loc[config, 'Accuracy'] = accuracy
                        
                elif task_type == 'forecasting':
                    # Extract MSE/MAE for different prediction lengths
                    if 'ours' in result:
                        for pred_len in result['ours']:
                            mse = result['ours'][pred_len]['raw']['MSE']
                            mae = result['ours'][pred_len]['raw']['MAE']
                            print(f"{config:15} (len={pred_len}): MSE={mse:.4f}, MAE={mae:.4f}")
                            results_df.loc[config, f'MSE_{pred_len}'] = mse
                            results_df.loc[config, f'MAE_{pred_len}'] = mae
                            
                elif task_type == 'anomaly_detection':
                    # Extract AUROC and other metrics
                    if 'ours' in result:
                        for metric in result['ours']:
                            value = result['ours'][metric]
                            print(f"{config:15}: {metric} = {value:.4f}")
                            results_df.loc[config, metric] = value
        
        return results_df
    
    def plot_ablation_results(self, dataset, task_type, metric='Accuracy'):
        """
        Plot ablation study results showing the effect of different λ values.
        """
        if dataset not in self.results:
            return
        
        configurations = ['contrastive', 'msm', 'hybrid_static', 'hybrid_dynamic']
        lambda_values = [0.0, 1.0, 0.5, 0.5]  # Corresponding λ values
        
        results = []
        labels = []
        
        for i, config in enumerate(configurations):
            if config in self.results[dataset]:
                result = self.results[dataset][config]
                
                if task_type == 'classification' and 'ours' in result:
                    results.append(result['ours']['acc'])
                    labels.append(f"{config}\n(λ={lambda_values[i]})")
                elif task_type == 'forecasting' and 'ours' in result:
                    # Use average MSE across all prediction lengths
                    mse_values = [result['ours'][pred_len]['raw']['MSE'] for pred_len in result['ours']]
                    results.append(np.mean(mse_values))
                    labels.append(f"{config}\n(λ={lambda_values[i]})")
        
        if results:
            plt.figure(figsize=(10, 6))
            bars = plt.bar(range(len(results)), results, color=['blue', 'red', 'green', 'purple'])
            plt.xticks(range(len(results)), labels, rotation=0)
            plt.ylabel(metric)
            plt.title(f'{dataset} - {task_type.replace("_", " ").title()} - Ablation Study')
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, results):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{value:.4f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(f'experiments/{dataset}_{task_type}_ablation.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def generate_paper_table(self):
        """
        Generate a comprehensive table for the research paper.
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE RESULTS TABLE FOR PAPER")
        print("="*80)
        
        table_data = []
        
        for dataset in self.results:
            for config in self.results[dataset]:
                result = self.results[dataset][config]
                
                row = {
                    'Dataset': dataset,
                    'Configuration': config,
                }
                
                # Add metrics based on available results
                if 'ours' in result:
                    if isinstance(result['ours'], dict):
                        if 'acc' in result['ours']:  # Classification
                            row['Accuracy'] = f"{result['ours']['acc']:.4f}"
                        else:  # Forecasting or Anomaly Detection
                            for key, value in result['ours'].items():
                                if isinstance(value, dict) and 'MSE' in value:
                                    row[f'MSE_{key}'] = f"{value['MSE']:.4f}"
                                    row[f'MAE_{key}'] = f"{value['MAE']:.4f}"
                                elif isinstance(value, (int, float)):
                                    row[key] = f"{value:.4f}"
                
                table_data.append(row)
        
        # Create DataFrame and display
        df = pd.DataFrame(table_data)
        print(df.to_string(index=False))
        
        # Save to CSV
        df.to_csv('experiments/comprehensive_results.csv', index=False)
        print(f"\nResults saved to experiments/comprehensive_results.csv")
        
        return df

# Usage example
if __name__ == "__main__":
    analyzer = TS2VecMSMAnalyzer()
    analyzer.load_results()
    
    # Generate comprehensive results
    results_df = analyzer.generate_paper_table()
    
    # Example: Compare configurations for a specific dataset
    # analyzer.compare_configurations('ETTh1', 'forecasting')
    # analyzer.plot_ablation_results('ETTh1', 'forecasting')
