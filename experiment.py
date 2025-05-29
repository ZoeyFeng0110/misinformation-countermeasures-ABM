import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import itertools
from pathlib import Path
import pickle

# Import your modules
from data_preprocessing import load_event_data, preprocess_event_data
from model import MisinformationModel

class AutomatedExperimentFramework:
    """Automated experiment framework - redesigned version"""
    
    def __init__(self, data_path, output_dir="experiment_results"):
        """
        Initialize experiment framework
        
        Args:
            data_path: Data folder path
            output_dir: Experiment results output directory
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load and preprocess data
        print("Loading data...")
        self.event_data = load_event_data(data_path)
        self.preprocessed_data = preprocess_event_data(self.event_data)
        print(f"Data loading completed: {len(self.preprocessed_data['users'])} users")
        
        # Experiment results storage
        self.results = []
        self.experiment_log = []
        
    def run_all_experiments(self):
        """Run all experiments"""
        print("Starting automated experiments...")
        
        # Experiment 1: Baseline comparison (without keynode strategy)
        print("\nExperiment 1: Strategy baseline comparison")
        baseline_results = self.experiment_1_baseline_comparison()
        
        # Experiment 2: Strategy parameter sensitivity analysis under different key node definitions
        print("\nExperiment 2: Strategy parameter sensitivity based on different key node definitions")
        keynode_definition_results = self.experiment_2_keynode_definition_impact()
        
        # Experiment 3: Optimal parameter combination strategy testing
        print("\nExperiment 3: Optimal parameter combination strategies")
        combination_results = self.experiment_3_optimal_combination_strategies()
        
        # Experiment 4: Temporal evolution analysis
        print("\nExperiment 4: Temporal evolution analysis")
        temporal_results = self.experiment_4_temporal_analysis()
        
        # Generate comprehensive report
        print("\nGenerating experiment report...")
        self.generate_comprehensive_report()
        
        print(f"All experiments completed! Results saved in: {self.output_dir}")
        
    def experiment_1_baseline_comparison(self):
        """Experiment 1: Baseline strategy comparison (without keynode activation strategy)"""
        
        strategies = {
            'baseline': {
                'key_node_enabled': False,
                'fact_check_enabled': False,
                'early_warning_enabled': False
            },
            'fact_check_only': {
                'key_node_enabled': False,  # keynode not activated, only used as definition
                'key_node_followers': 1111,  # default key node definition
                'fact_check_enabled': True,
                'fact_check_threshold': 50,
                'fact_check_delay': 30,
                'early_warning_enabled': False
            },
            'early_warning_only': {
                'key_node_enabled': False,  # keynode not activated, only used as definition
                'key_node_followers': 1111,  # default key node definition
                'fact_check_enabled': False,
                'early_warning_enabled': True,
                'early_warning_threshold': 10,
                'early_warning_coverage': 0.3
            }
        }
        
        results = []
        for strategy_name, settings in strategies.items():
            print(f"  Testing strategy: {strategy_name}")
            strategy_results = self.run_strategy_experiment(
                strategy_name, settings, runs=1, steps=200
            )
            results.extend(strategy_results)
            
        # Save results
        baseline_df = pd.DataFrame(results)
        baseline_df.to_csv(self.output_dir / "experiment_1_baseline_comparison.csv", index=False)
        
        # Generate comparison chart
        self.plot_baseline_comparison(baseline_df)
        
        return results
    
    def experiment_2_keynode_definition_impact(self):
        """Experiment 2: Impact of different key node definitions on strategy effectiveness"""
        
        print("  Testing impact of different key node definitions...")
        
        # Different key node definitions (follower thresholds)
        keynode_definitions = [146, 402, 1111, 5000]
        
        # 2.1 Fact check strategy parameter sensitivity under different key node definitions
        print("    Fact check strategy parameter sensitivity")
        factcheck_results = self.sensitivity_factcheck_with_keynode_definitions(keynode_definitions)
        
        # 2.2 Early warning strategy parameter sensitivity under different key node definitions
        print("    Early warning strategy parameter sensitivity")
        earlywarning_results = self.sensitivity_earlywarning_with_keynode_definitions(keynode_definitions)
        
        # Merge all results
        all_results = factcheck_results + earlywarning_results
        
        # Save results
        sensitivity_df = pd.DataFrame(all_results)
        sensitivity_df.to_csv(self.output_dir / "experiment_2_keynode_definition_impact.csv", index=False)
        
        # Generate visualization
        self.plot_keynode_definition_impact(sensitivity_df)
        
        return all_results
    
    def sensitivity_factcheck_with_keynode_definitions(self, keynode_definitions):
        """Fact check strategy parameter sensitivity under different key node definitions"""
        
        # Fact check parameter space
        activation_thresholds = [10, 30, 50]
        delays = [10, 30, 50]
        
        results = []
        total_combinations = len(keynode_definitions) * len(activation_thresholds) * len(delays)
        current_combo = 0
        
        for keynode_followers in keynode_definitions:
            for act_thresh in activation_thresholds:
                for delay in delays:
                    current_combo += 1
                    print(f"      Fact check {current_combo}/{total_combinations}: "
                          f"key node={keynode_followers}followers, activation threshold={act_thresh}, delay={delay}")
                    
                    settings = {
                        'key_node_enabled': False,  # Do not activate keynode strategy
                        'key_node_followers': keynode_followers,  # Only used to define key nodes
                        'fact_check_enabled': True,
                        'fact_check_threshold': act_thresh,
                        'fact_check_delay': delay,
                        'early_warning_enabled': False
                    }
                    
                    strategy_name = f"factcheck_keydef{keynode_followers}_thresh{act_thresh}_delay{delay}"
                    combo_results = self.run_strategy_experiment(
                        strategy_name, settings, runs=1, steps=150
                    )
                    
                    # Add parameter markers
                    for result in combo_results:
                        result['strategy_type'] = 'factcheck'
                        result['keynode_definition'] = keynode_followers
                        result['activation_threshold'] = act_thresh
                        result['delay'] = delay
                    
                    results.extend(combo_results)
        
        return results
    
    def sensitivity_earlywarning_with_keynode_definitions(self, keynode_definitions):
        """Early warning strategy parameter sensitivity under different key node definitions"""
        
        # Early warning parameter space
        coverage_ratios = [0.1, 0.3, 0.5]
        activation_thresholds = [5, 10, 20]
        
        results = []
        total_combinations = len(keynode_definitions) * len(coverage_ratios) * len(activation_thresholds)
        current_combo = 0
        
        for keynode_followers in keynode_definitions:
            for coverage in coverage_ratios:
                for act_thresh in activation_thresholds:
                    current_combo += 1
                    print(f"      Early warning {current_combo}/{total_combinations}: "
                          f"key node={keynode_followers}followers, coverage={coverage}, activation threshold={act_thresh}")
                    
                    settings = {
                        'key_node_enabled': False,  # Do not activate keynode strategy
                        'key_node_followers': keynode_followers,  # Only used to define key nodes
                        'fact_check_enabled': False,
                        'early_warning_enabled': True,
                        'early_warning_threshold': act_thresh,
                        'early_warning_coverage': coverage
                    }
                    
                    strategy_name = f"earlywarning_keydef{keynode_followers}_cov{coverage}_thresh{act_thresh}"
                    combo_results = self.run_strategy_experiment(
                        strategy_name, settings, runs=1, steps=150
                    )
                    
                    # Add parameter markers
                    for result in combo_results:
                        result['strategy_type'] = 'earlywarning'
                        result['keynode_definition'] = keynode_followers
                        result['coverage_ratio'] = coverage
                        result['activation_threshold'] = act_thresh
                    
                    results.extend(combo_results)
        
        return results
    
    def experiment_3_optimal_combination_strategies(self):
        """Experiment 3: Parameter tuning experiment based on 1111 threshold"""
        
        print("  Parameter tuning based on 1111 threshold...")
        
        # Fixed: Consistent parameter range with experiment 2
        fact_check_configs = [
            {
                'name': 'factcheck_1111_thresh10_delay10',
                'key_node_enabled': False,
                'key_node_followers': 1111,
                'fact_check_enabled': True,
                'fact_check_threshold': 10,
                'fact_check_delay': 10,
                'early_warning_enabled': False
            },
            {
                'name': 'factcheck_1111_thresh30_delay30',
                'key_node_enabled': False,
                'key_node_followers': 1111,
                'fact_check_enabled': True,
                'fact_check_threshold': 30,
                'fact_check_delay': 30,
                'early_warning_enabled': False
            },
            {
                'name': 'factcheck_1111_thresh50_delay50',
                'key_node_enabled': False,
                'key_node_followers': 1111,
                'fact_check_enabled': True,
                'fact_check_threshold': 50,
                'fact_check_delay': 50,
                'early_warning_enabled': False
            }
        ]

        
        early_warning_configs = [
            {
                'name': 'earlywarning_1111_cov01_thresh5',
                'key_node_enabled': False,
                'key_node_followers': 1111,
                'fact_check_enabled': False,
                'early_warning_enabled': True,
                'early_warning_threshold': 5,
                'early_warning_coverage': 0.1
            },
            {
                'name': 'earlywarning_1111_cov03_thresh10',
                'key_node_enabled': False,
                'key_node_followers': 1111,
                'fact_check_enabled': False,
                'early_warning_enabled': True,
                'early_warning_threshold': 10,
                'early_warning_coverage': 0.3
            },
            {
                'name': 'earlywarning_1111_cov05_thresh20',
                'key_node_enabled': False,
                'key_node_followers': 1111,
                'fact_check_enabled': False,
                'early_warning_enabled': True,
                'early_warning_threshold': 20,
                'early_warning_coverage': 0.5
            }
        ]
        
        # Merge all single strategy configurations
        all_configs = fact_check_configs + early_warning_configs
        
        results = []
        for config in all_configs:
            strategy_name = config.pop('name')
            print(f"  Testing parameter combination: {strategy_name}")
            
            combo_results = self.run_strategy_experiment(
                strategy_name, config, runs=1, steps=200
            )
            results.extend(combo_results)
        
        # Save results
        combo_df = pd.DataFrame(results)
        combo_df.to_csv(self.output_dir / "experiment_3_parameter_tuning_1111.csv", index=False)
        
        # Analyze optimal parameters and test combination strategy
        optimal_combination_results = self.find_and_test_optimal_combination(combo_df)
        results.extend(optimal_combination_results)
        
        # Update complete results
        full_combo_df = pd.DataFrame(results)
        full_combo_df.to_csv(self.output_dir / "experiment_3_parameter_tuning_1111.csv", index=False)
        
        # Generate comparison chart
        self.plot_combination_results(full_combo_df)
        
        return results
    
    def find_and_test_optimal_combination(self, single_strategy_df):
        """Fixed: Update parameter parsing logic"""
        
        print("  Analyzing single strategy results, finding optimal parameters...")
        
        # Get final results
        final_results = single_strategy_df[single_strategy_df['step'] == single_strategy_df['step'].max()]
        
        # Find optimal fact check parameters
        factcheck_results = final_results[final_results['strategy'].str.contains('factcheck')]
        if len(factcheck_results) > 0:
            best_factcheck = factcheck_results.loc[factcheck_results['active_users'].idxmin()]
            print(f"    Optimal fact check strategy: {best_factcheck['strategy']}")
            
            # Fixed: Update parameter parsing logic
            if 'thresh10_delay10' in best_factcheck['strategy']:
                best_fc_thresh, best_fc_delay = 10, 10
            elif 'thresh30_delay30' in best_factcheck['strategy']:
                best_fc_thresh, best_fc_delay = 30, 30
            elif 'thresh50_delay50' in best_factcheck['strategy']:
                best_fc_thresh, best_fc_delay = 50, 50
            else:
                best_fc_thresh, best_fc_delay = 30, 30  # default value
        else:
            best_fc_thresh, best_fc_delay = 30, 30
            
        # Find optimal early warning parameters
        earlywarning_results = final_results[final_results['strategy'].str.contains('earlywarning')]
        if len(earlywarning_results) > 0:
            best_earlywarning = earlywarning_results.loc[earlywarning_results['active_users'].idxmin()]
            print(f"    Optimal early warning strategy: {best_earlywarning['strategy']}")
            
            # Parse parameters
            if 'cov01_thresh5' in best_earlywarning['strategy']:
                best_ew_cov, best_ew_thresh = 0.1, 5
            elif 'cov03_thresh10' in best_earlywarning['strategy']:
                best_ew_cov, best_ew_thresh = 0.3, 10
            else:  # cov05_thresh20
                best_ew_cov, best_ew_thresh = 0.5, 20
        else:
            best_ew_cov, best_ew_thresh = 0.3, 10  # default value
        
        # Test optimal combination strategy
        print(f"  Testing optimal combination strategy: FC(thresh={best_fc_thresh}, delay={best_fc_delay}) + EW(cov={best_ew_cov}, thresh={best_ew_thresh})")
        
        optimal_combination_config = {
            'key_node_enabled': False,
            'key_node_followers': 1111,
            'fact_check_enabled': True,
            'fact_check_threshold': best_fc_thresh,
            'fact_check_delay': best_fc_delay,
            'early_warning_enabled': True,
            'early_warning_threshold': best_ew_thresh,
            'early_warning_coverage': best_ew_cov
        }
        
        combination_results = self.run_strategy_experiment(
            f"combined_1111_optimal_fc{best_fc_thresh}d{best_fc_delay}_ew{best_ew_cov}t{best_ew_thresh}",
            optimal_combination_config, runs=1, steps=200
        )
        
        return combination_results
    
    def experiment_4_temporal_analysis(self):
        """Experiment 4: Temporal evolution analysis"""
        
        # Test the effect of starting intervention at different time points
        intervention_start_steps = [10, 30, 50, 100]
        
        results = []
        for start_step in intervention_start_steps:
            print(f"  Testing intervention start time: step {start_step}")
            
            # Use combination strategy for temporal analysis
            settings = {
                'key_node_enabled': False,
                'key_node_followers': 1111,  # Use median threshold
                'fact_check_enabled': True,
                'fact_check_threshold': 50,
                'fact_check_delay': 10,
                'early_warning_enabled': True,
                'early_warning_threshold': 10,
                'early_warning_coverage': 0.3,
                'intervention_start_step': start_step  # New parameter
            }
            
            temporal_results = self.run_strategy_experiment(
                f"temporal_start_{start_step}", settings, runs=1, steps=200
            )
            
            # Add intervention start time marker
            for result in temporal_results:
                result['intervention_start'] = start_step
            
            results.extend(temporal_results)
        
        # Save results
        temporal_df = pd.DataFrame(results)
        temporal_df.to_csv(self.output_dir / "experiment_4_temporal_analysis.csv", index=False)
        
        return results
    
    def plot_keynode_definition_impact(self, df):
        """Plot key node definition impact analysis chart"""
        
        final_df = df[df['step'] == df['step'].max()].copy()
        
        # Analyze fact check and early warning strategies separately
        strategy_types = final_df['strategy_type'].unique()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        for i, strategy_type in enumerate(strategy_types):
            strategy_data = final_df[final_df['strategy_type'] == strategy_type]
            
            # Impact of key node definition on active users
            ax1 = axes[i, 0]
            keynode_impact = strategy_data.groupby('keynode_definition')['active_users'].mean()
            ax1.plot(keynode_impact.index, keynode_impact.values, 'o-', linewidth=2, markersize=8)
            ax1.set_xlabel('Key Node Definition (Followers Threshold)')
            ax1.set_ylabel('Average Active Users')
            ax1.set_title(f'{strategy_type.capitalize()}: Key Node Definition Impact')
            ax1.grid(True, alpha=0.3)
            
            # Impact of key node definition on total retweets
            ax2 = axes[i, 1]
            retweet_impact = strategy_data.groupby('keynode_definition')['total_retweets'].mean()
            ax2.plot(retweet_impact.index, retweet_impact.values, 's-', linewidth=2, markersize=8)
            ax2.set_xlabel('Key Node Definition (Followers Threshold)')
            ax2.set_ylabel('Average Total Retweets')
            ax2.set_title(f'{strategy_type.capitalize()}: Retweet Impact')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "keynode_definition_impact.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate detailed parameter heatmaps
        self.plot_detailed_parameter_heatmaps(final_df)
    
    def plot_detailed_parameter_heatmaps(self, df):
        """Generate detailed parameter heatmaps"""
        
        strategy_types = df['strategy_type'].unique()
        
        for strategy_type in strategy_types:
            strategy_data = df[df['strategy_type'] == strategy_type]
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            if strategy_type == 'factcheck':
                # Fact check strategy heatmaps
                for i, keynode_def in enumerate([146, 1111]):  # Select two representative key node definitions
                    subset = strategy_data[strategy_data['keynode_definition'] == keynode_def]
                    
                    if len(subset) > 0:
                        pivot_data = subset.pivot_table(
                            index='delay',
                            columns='activation_threshold',
                            values='active_users',
                            aggfunc='mean'
                        )
                        
                        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=axes[i])
                        axes[i].set_title(f'Fact Check: Key Node {keynode_def} Followers')
                        axes[i].set_xlabel('Activation Threshold')
                        axes[i].set_ylabel('Delay')
            
            elif strategy_type == 'earlywarning':
                # Early warning strategy heatmaps
                for i, keynode_def in enumerate([146, 1111]):
                    subset = strategy_data[strategy_data['keynode_definition'] == keynode_def]
                    
                    if len(subset) > 0:
                        pivot_data = subset.pivot_table(
                            index='coverage_ratio',
                            columns='activation_threshold',
                            values='active_users',
                            aggfunc='mean'
                        )
                        
                        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=axes[i])
                        axes[i].set_title(f'Early Warning: Key Node {keynode_def} Followers')
                        axes[i].set_xlabel('Activation Threshold')
                        axes[i].set_ylabel('Coverage Ratio')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{strategy_type}_parameter_heatmaps.png",
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def run_strategy_experiment(self, strategy_name, settings, runs=1, steps=200):
        """Run single strategy experiment"""
        
        results = []
        
        for run_id in range(runs):
            try:
                # Create model
                model = MisinformationModel(self.preprocessed_data, settings)
                
                # Collect initial state
                initial_data = self.extract_model_data(model, 0)
                initial_data.update({
                    'strategy': strategy_name,
                    'run_id': run_id,
                    'step': 0
                })
                results.append(initial_data)
                
                # Run simulation
                for step in range(1, steps + 1):
                    model.step()
                    
                    # Collect data every 10 steps
                    if step % 10 == 0 or step == steps:
                        step_data = self.extract_model_data(model, step)
                        step_data.update({
                            'strategy': strategy_name,
                            'run_id': run_id,
                            'step': step
                        })
                        results.append(step_data)
                
                print(f"    Strategy {strategy_name} completed")
                
            except Exception as e:
                print(f"    Strategy {strategy_name} failed: {str(e)}")
                continue
        
        return results
    
    def extract_model_data(self, model, step):
        """Extract key data from model"""
        try:
            # Basic statistics
            active_users = len([u for u in model.user_agents.values() if u.is_active()])
            total_retweets = model.get_total_retweets()
            total_users = len(model.user_agents)
            
            # Countermeasure statistics
            activated_countermeasures = sum(len(cm.active_countermeasures) 
                                          for cm in model.countermeasure_agents)
            
            # Retraction statistics
            total_retractions = sum(len(users) for users in model.retracted_users.values())
            
            return {
                'active_users': active_users,
                'total_retweets': total_retweets,
                'total_users': total_users,
                'active_users_ratio': active_users / total_users if total_users > 0 else 0,
                'activated_countermeasures': activated_countermeasures,
                'total_retractions': total_retractions,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'active_users': 0,
                'total_retweets': 0,
                'total_users': 0,
                'active_users_ratio': 0,
                'activated_countermeasures': 0,
                'total_retractions': 0,
                'timestamp': datetime.now().isoformat()
            }
    
    def plot_baseline_comparison(self, df):
        """Plot baseline strategy comparison chart"""
        
        final_results = df[df['step'] == df['step'].max()].groupby('strategy').agg({
            'active_users': 'mean',
            'total_retweets': 'mean'
        }).reset_index()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        strategies = final_results['strategy'].tolist()
        active_users = final_results['active_users'].tolist()
        
        bars1 = ax1.bar(strategies, active_users, color=['red', 'blue', 'green'])
        ax1.set_title('Final Active Users by Strategy')
        ax1.set_ylabel('Active Users Count')
        ax1.tick_params(axis='x', rotation=45)
        
        for bar in bars1:
            height = bar.get_height()
            ax1.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        
        total_retweets = final_results['total_retweets'].tolist()
        bars2 = ax2.bar(strategies, total_retweets, color=['red', 'blue', 'green'])
        ax2.set_title('Final Total Retweets by Strategy')
        ax2.set_ylabel('Total Retweets Count')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "baseline_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_combination_results(self, df):
        """Plot combination strategy results - bar charts and line charts"""
        
        final_results = df[df['step'] == df['step'].max()].groupby('strategy').agg({
            'active_users': 'mean',
            'total_retweets': 'mean',
            'activated_countermeasures': 'mean'
        }).reset_index()
        
        # Create charts
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Active users bar chart comparison
        ax1 = axes[0, 0]
        x_pos = np.arange(len(final_results))
        bars1 = ax1.bar(x_pos, final_results['active_users'], alpha=0.8, color='skyblue')
        ax1.set_title('Active Users by Strategy (Lower is Better)')
        ax1.set_ylabel('Active Users Count')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(final_results['strategy'], rotation=45, ha='right')
        
        # Add value labels
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height + height*0.01,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        # 2. Total retweets bar chart comparison
        ax2 = axes[0, 1]
        bars2 = ax2.bar(x_pos, final_results['total_retweets'], alpha=0.8, color='lightcoral')
        ax2.set_title('Total Retweets by Strategy (Lower is Better)')
        ax2.set_ylabel('Total Retweets Count')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(final_results['strategy'], rotation=45, ha='right')
        
        # Add value labels
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height + height*0.01,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        # 3. Activated countermeasures bar chart
        ax3 = axes[1, 0]
        bars3 = ax3.bar(x_pos, final_results['activated_countermeasures'], alpha=0.8, color='lightgreen')
        ax3.set_title('Activated Countermeasures by Strategy (Higher is Better)')
        ax3.set_ylabel('Countermeasures Count')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(final_results['strategy'], rotation=45, ha='right')
        
        # Add value labels
        for i, bar in enumerate(bars3):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, height + height*0.01,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        # 4. Comprehensive effect line chart
        ax4 = axes[1, 1]
        
        # Group by different strategy types
        fact_check_strategies = [s for s in final_results['strategy'] if 'factcheck' in s]
        early_warning_strategies = [s for s in final_results['strategy'] if 'earlywarning' in s]
        combined_strategies = [s for s in final_results['strategy'] if 'combined' in s]
        
        # Plot fact check strategy line
        if fact_check_strategies:
            fc_data = final_results[final_results['strategy'].isin(fact_check_strategies)]
            ax4.plot(range(len(fc_data)), fc_data['active_users'], 'o-', 
                    label='Fact Check', linewidth=2, markersize=8)
        
        # Plot early warning strategy line
        if early_warning_strategies:
            ew_data = final_results[final_results['strategy'].isin(early_warning_strategies)]
            ax4.plot(range(len(ew_data)), ew_data['active_users'], 's-', 
                    label='Early Warning', linewidth=2, markersize=8)
        
        # Plot combined strategy line
        if combined_strategies:
            cb_data = final_results[final_results['strategy'].isin(combined_strategies)]
            ax4.plot(range(len(cb_data)), cb_data['active_users'], '^-', 
                    label='Combined', linewidth=2, markersize=8)
        
        ax4.set_title('Strategy Performance Comparison (Active Users)')
        ax4.set_ylabel('Active Users Count')
        ax4.set_xlabel('Strategy Variants')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "parameter_tuning_comparison.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate time series comparison chart
        self.plot_time_series_comparison_detailed(df)
    
    def plot_time_series_comparison_detailed(self, df):
        """Plot detailed time series comparison chart"""
        
        # Group by strategy type
        fact_check_strategies = [s for s in df['strategy'].unique() if 'factcheck' in s]
        early_warning_strategies = [s for s in df['strategy'].unique() if 'earlywarning' in s]
        combined_strategies = [s for s in df['strategy'].unique() if 'combined' in s]
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # Fact check strategy time series
        if fact_check_strategies:
            ax1 = axes[0]
            for strategy in fact_check_strategies:
                strategy_data = df[df['strategy'] == strategy]
                time_series = strategy_data.groupby('step')['active_users'].mean()
                ax1.plot(time_series.index, time_series.values, 
                        marker='o', label=strategy.replace('factcheck_1111_', ''), linewidth=2)
            
            ax1.set_title('Fact Check Strategies - Active Users Over Time')
            ax1.set_ylabel('Active Users')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Early warning strategy time series
        if early_warning_strategies:
            ax2 = axes[1]
            for strategy in early_warning_strategies:
                strategy_data = df[df['strategy'] == strategy]
                time_series = strategy_data.groupby('step')['active_users'].mean()
                ax2.plot(time_series.index, time_series.values, 
                        marker='s', label=strategy.replace('earlywarning_1111_', ''), linewidth=2)
            
            ax2.set_title('Early Warning Strategies - Active Users Over Time')
            ax2.set_ylabel('Active Users')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Combined strategy time series
        if combined_strategies:
            ax3 = axes[2]
            for strategy in combined_strategies:
                strategy_data = df[df['strategy'] == strategy]
                time_series = strategy_data.groupby('step')['active_users'].mean()
                ax3.plot(time_series.index, time_series.values, 
                        marker='^', label=strategy.replace('combined_1111_', ''), linewidth=2)
            
            ax3.set_title('Combined Strategy - Active Users Over Time')
            ax3.set_ylabel('Active Users')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        ax3.set_xlabel('Time Steps')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "parameter_tuning_time_series.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comprehensive_report(self):
        """Generate comprehensive experiment report"""
        
        report = {
            'experiment_summary': {
                'total_experiments': 4,
                'completion_time': datetime.now().isoformat(),
                'data_source': str(self.data_path),
                'focus': 'Key node definition impact on fact-check and early-warning strategies'
            },
            'key_findings': self.analyze_key_findings(),
            'keynode_definition_insights': self.analyze_keynode_definition_insights(),
            'recommendations': self.generate_recommendations()
        }
        
        # Save JSON report
        with open(self.output_dir / "comprehensive_experiment_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate Markdown report
        self.generate_enhanced_markdown_report(report)
        
        print("Comprehensive report generation completed")
    
    def analyze_key_findings(self):
        """Analyze key findings"""
        findings = []
        
        try:
            if (self.output_dir / "experiment_1_baseline_comparison.csv").exists():
                baseline_df = pd.read_csv(self.output_dir / "experiment_1_baseline_comparison.csv")
                final_baseline = baseline_df[baseline_df['step'] == baseline_df['step'].max()]
                strategy_performance = final_baseline.groupby('strategy')['active_users'].mean().sort_values()
                
                best_strategy = strategy_performance.index[0]
                findings.append(f"Most effective single strategy: {best_strategy}")
                findings.append(f"Strategy effectiveness ranking: {dict(strategy_performance)}")
            
        except Exception as e:
            findings.append(f"Baseline analysis error: {str(e)}")
        
        return findings
    
    def analyze_keynode_definition_insights(self):
        """Analyze key node definition insights"""
        insights = {}
        
        try:
            if (self.output_dir / "experiment_2_keynode_definition_impact.csv").exists():
                sensitivity_df = pd.read_csv(self.output_dir / "experiment_2_keynode_definition_impact.csv")
                final_sensitivity = sensitivity_df[sensitivity_df['step'] == sensitivity_df['step'].max()]
                
                # Analyze optimal key node definition for each strategy type
                for strategy_type in final_sensitivity['strategy_type'].unique():
                    type_data = final_sensitivity[final_sensitivity['strategy_type'] == strategy_type]
                    
                    # Find best performance under each key node definition
                    keynode_performance = type_data.groupby('keynode_definition')['active_users'].min()
                    best_keynode_def = keynode_performance.idxmin()
                    
                    insights[f'{strategy_type}_optimal_keynode_definition'] = int(best_keynode_def)
                    insights[f'{strategy_type}_keynode_performance'] = dict(keynode_performance)
        
        except Exception as e:
            insights['error'] = str(e)
        
        return insights
    



def main():
    """Main function - run redesigned automated experiments"""
    
    # Configure data path
    data_path = "/Users/oliviafeng/Desktop/uchi/agent_based_modeling/code/final_project/pheme-rumour-scheme-dataset/threads/en/charliehebdo"
    
    # Create experiment framework
    print("Initializing redesigned automated experiment framework...")
    print("Experiment focus: Impact of key node definitions on fact-checking and early warning strategies")
    
    experiment_framework = AutomatedExperimentFramework(
        data_path=data_path,
        output_dir="/Users/oliviafeng/Desktop/uchi/agent_based_modeling/code/final_final/automated_experiment_results3"
    )
    
    # Run all experiments
    try:
        experiment_framework.run_all_experiments()
        print("\nAll experiments completed successfully!")
        print("Please check the generated reports and visualization results")
        print("Focus on the impact analysis of key node definitions on strategy effectiveness")
    except Exception as e:
        print(f"\nError occurred during experiments: {str(e)}")
        print("Please check data path and model configuration")


if __name__ == "__main__":
    main()