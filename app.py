import solara
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
import traceback

from data_preprocessing import load_event_data, preprocess_event_data
from model import MisinformationModel

class SimulationState:
    """Simulation state management class"""
    def __init__(self):
        self.preprocessed_data = None
        self.baseline_model = None
        self.countermeasure_model = None
        self.baseline_data = []
        self.countermeasure_data = []
        self.current_step = 0
        self.countermeasure_settings = self._default_settings()
    
    def _default_settings(self):
        """Default countermeasure settings"""
        return {
            'key_node_enabled': False,
            'key_node_threshold': 0.01,
            'key_node_followers': 100,
            'fact_check_enabled': False,
            'fact_check_threshold': 50,
            'fact_check_delay': 30,
            'early_warning_enabled': False,
            'early_warning_threshold': 10,
            'early_warning_coverage': 0.3
        }
    
    def reset_models(self):
        """Reset model states"""
        self.baseline_model = None
        self.countermeasure_model = None
        self.baseline_data = []
        self.countermeasure_data = []
        self.current_step = 0
    
    def update_settings(self, **kwargs):
        """Update countermeasure settings"""
        self.countermeasure_settings.update(kwargs)

# Global state instance
sim_state = SimulationState()

@solara.component
def DataLoadingCard():
    """Data loading component"""
    load_progress, set_load_progress = solara.use_state(0)
    status_message, set_status_message = solara.use_state("Waiting to load data...")
    folder_path, set_folder_path = solara.use_state(
        "/Users/oliviafeng/Desktop/uchi/agent_based_modeling/code/final_project/pheme-rumour-scheme-dataset/threads/en/charliehebdo"
    )
    
    def load_data():
        set_status_message("Loading data...")
        set_load_progress(10)
        try:
            event_data = load_event_data(folder_path)
            set_load_progress(50)
            set_status_message(f"Successfully loaded {len(event_data)} misinformation events, preprocessing...")
            
            sim_state.preprocessed_data = preprocess_event_data(event_data)
            sim_state.reset_models()
            set_load_progress(100)
            
            users_count = len(sim_state.preprocessed_data['users'])
            tweets_count = len(sim_state.preprocessed_data['tweets_timeline'])
            historical_retweets = sim_state.preprocessed_data.get('historical_retweets', {})
            historical_count = sum(len(retweets) for retweets in historical_retweets.values())
            
            set_status_message(f"Data loading completed! Users: {users_count}, Tweets: {tweets_count}, Historical retweets: {historical_count}")
        
        except Exception as e:
            set_status_message(f"Error: {str(e)}")
            traceback.print_exc()
            set_load_progress(0)
    
    with solara.Card("Data Loading"):
        with solara.Row():
            solara.InputText(label="Data folder path", value=folder_path, on_value=set_folder_path)
            solara.Button(label="Load Data", on_click=load_data)
        
        solara.ProgressLinear(value=load_progress)
        solara.Info(status_message)

@solara.component
def CountermeasureSettingsCard():
    """Countermeasure settings component"""
    
    # State management
    initialization_status, set_initialization_status = solara.use_state("Waiting for initialization...")
    
    # Key node settings
    key_node_enabled, set_key_node_enabled = solara.use_state(
        sim_state.countermeasure_settings['key_node_enabled'])
    key_node_threshold, set_key_node_threshold = solara.use_state(
        sim_state.countermeasure_settings['key_node_threshold'])
    key_node_followers, set_key_node_followers = solara.use_state(
        sim_state.countermeasure_settings['key_node_followers'])
    
    # Fact checking settings
    fact_check_enabled, set_fact_check_enabled = solara.use_state(
        sim_state.countermeasure_settings['fact_check_enabled'])
    fact_check_threshold, set_fact_check_threshold = solara.use_state(
        sim_state.countermeasure_settings['fact_check_threshold'])
    fact_check_delay, set_fact_check_delay = solara.use_state(
        sim_state.countermeasure_settings['fact_check_delay'])
    
    # Early warning settings
    early_warning_enabled, set_early_warning_enabled = solara.use_state(
        sim_state.countermeasure_settings['early_warning_enabled'])
    early_warning_threshold, set_early_warning_threshold = solara.use_state(
        sim_state.countermeasure_settings['early_warning_threshold'])
    early_warning_coverage, set_early_warning_coverage = solara.use_state(
        sim_state.countermeasure_settings['early_warning_coverage'])
    
    def update_settings():
        """Update settings to global state"""
        try:
            sim_state.update_settings(
                key_node_enabled=key_node_enabled,
                key_node_threshold=key_node_threshold,
                key_node_followers=key_node_followers,
                fact_check_enabled=fact_check_enabled,
                fact_check_threshold=fact_check_threshold,
                fact_check_delay=fact_check_delay,
                early_warning_enabled=early_warning_enabled,
                early_warning_threshold=early_warning_threshold,
                early_warning_coverage=early_warning_coverage
            )
            print("Settings updated to global state")
        except Exception as e:
            print(f"Error updating settings: {e}")
            set_initialization_status(f"Failed to update settings: {str(e)}")
    
    def initialize_models():
        """Initialize dual models"""
        print("Starting model initialization...")
        set_initialization_status("Initializing models...")
        
        try:
            # Check if data is loaded
            if sim_state.preprocessed_data is None:
                set_initialization_status("❌ Please load data first")
                print("Data not loaded")
                return
            
            print("Data check passed, updating settings...")
            update_settings()
            
            print("Creating baseline model...")
            # Create baseline model (no countermeasures)
            baseline_settings = {}
            for key in sim_state.countermeasure_settings.keys():
                if key.endswith('_enabled'):
                    baseline_settings[key] = False
                else:
                    baseline_settings[key] = sim_state.countermeasure_settings[key]
            
            sim_state.baseline_model = MisinformationModel(
                sim_state.preprocessed_data, baseline_settings)
            print("Baseline model created successfully")
            
            print("Creating countermeasure model...")
            sim_state.countermeasure_model = MisinformationModel(
                sim_state.preprocessed_data, sim_state.countermeasure_settings)
            print("Countermeasure model created successfully")
            
            print("Collecting initial data...")
            try:
                baseline_data = get_model_data(sim_state.baseline_model)
                countermeasure_data = get_model_data(sim_state.countermeasure_model)
                
                sim_state.baseline_data = [baseline_data]
                sim_state.countermeasure_data = [countermeasure_data]
                sim_state.current_step = 0
                
                print("Initial data collection successful")
            except Exception as data_error:
                print(f"Data collection error: {data_error}")
                sim_state.baseline_data = [{"Active_Users": 0, "Total_Retweets": 0}]
                sim_state.countermeasure_data = [{"Active_Users": 0, "Total_Retweets": 0}]
                sim_state.current_step = 0
            
            # Generate success message
            enabled_measures = []
            if key_node_enabled:
                enabled_measures.append(f"Key Node(threshold:{key_node_threshold}, followers>{key_node_followers})")
            if fact_check_enabled:
                enabled_measures.append(f"Fact Check(threshold:{fact_check_threshold}, delay:{fact_check_delay})")
            if early_warning_enabled:
                enabled_measures.append(f"Early Warning(threshold:{early_warning_threshold}, coverage:{early_warning_coverage})")
            
            success_message = "✅ Dual model initialization successful!"
            if enabled_measures:
                success_message += f"\nEnabled countermeasures: {', '.join(enabled_measures)}"
            else:
                success_message += "\nNo countermeasures currently enabled"
            
            set_initialization_status(success_message)
            print("Model initialization completely successful")
            
        except Exception as e:
            error_message = f"❌ Model initialization failed: {str(e)}"
            set_initialization_status(error_message)
            print(f"Model initialization failed: {e}")
            import traceback
            traceback.print_exc()
    
    def reset_simulation():
        """Reset simulation"""
        try:
            sim_state.reset_models()
            set_initialization_status("🔄 Simulation reset")
            print("Simulation reset successful")
        except Exception as e:
            set_initialization_status(f"❌ Reset failed: {str(e)}")
            print(f"Reset failed: {e}")
    
    with solara.Card("Countermeasure Settings"):
        # Key node intervention
        solara.Markdown("**Key Node Intervention Settings**")
        with solara.Row():
            solara.Checkbox(label="Enable Key Node Intervention", value=key_node_enabled, on_value=set_key_node_enabled)
            solara.SliderFloat(label="Activation Threshold (ratio)", value=key_node_threshold, 
                             min=0.01, max=1.0, step=0.01, on_value=set_key_node_threshold)
            solara.SliderInt(label="Minimum Followers", value=key_node_followers, 
                           min=10, max=100000, step=100, on_value=set_key_node_followers)
        
        # Fact checking
        solara.Markdown("**Fact Checking Settings**")
        with solara.Row():
            solara.Checkbox(label="Enable Fact Checking", value=fact_check_enabled, on_value=set_fact_check_enabled)
            solara.SliderInt(label="Activation Threshold (retweets)", value=fact_check_threshold, 
                           min=1, max=500, step=10, on_value=set_fact_check_threshold)
            solara.SliderInt(label="Delay Time (steps)", value=fact_check_delay, 
                           min=1, max=100, step=5, on_value=set_fact_check_delay)
        
        # Early warning
        solara.Markdown("**Early Warning Settings**")
        with solara.Row():
            solara.Checkbox(label="Enable Early Warning", value=early_warning_enabled, on_value=set_early_warning_enabled)
            solara.SliderInt(label="Activation Threshold (retweets)", value=early_warning_threshold, 
                           min=1, max=100, step=5, on_value=set_early_warning_threshold)
            solara.SliderFloat(label="Coverage Ratio", value=early_warning_coverage, 
                             min=0.1, max=1.0, step=0.05, on_value=set_early_warning_coverage)
        
        # Control buttons
        with solara.Row():
            solara.Button(label="Initialize Dual Models", on_click=initialize_models, color="primary")
            solara.Button(label="Reset Simulation", on_click=reset_simulation, color="secondary")
        
        # Status display
        solara.Info(initialization_status)
        
        # Debug information
        if sim_state.preprocessed_data:
            users_count = len(sim_state.preprocessed_data['users'])
            tweets_count = len(sim_state.preprocessed_data['tweets_timeline'])
            solara.Text(f"Data loaded: {users_count} users, {tweets_count} tweets")
        else:
            solara.Text("⚠️ Data not loaded")

@solara.component 
def SimulationControlCard():
    """Simulation control component"""
    status_message, set_status_message = solara.use_state(f"Current step: {sim_state.current_step}")
    
    def run_single_step():
        """Run single step"""
        if not sim_state.baseline_model or not sim_state.countermeasure_model:
            set_status_message("Please initialize models first")
            return
        
        try:
            sim_state.baseline_model.step()
            sim_state.countermeasure_model.step()
            sim_state.current_step += 1
            
            baseline_data = get_model_data(sim_state.baseline_model)
            countermeasure_data = get_model_data(sim_state.countermeasure_model)
            
            sim_state.baseline_data.append(baseline_data)
            sim_state.countermeasure_data.append(countermeasure_data)
            
            set_status_message(f"Step {sim_state.current_step} - "
                             f"Baseline: {baseline_data['Active_Users']}active/{baseline_data['Total_Retweets']}retweets, "
                             f"Counter: {countermeasure_data['Active_Users']}active/{countermeasure_data['Total_Retweets']}retweets")
        except Exception as e:
            set_status_message(f"Simulation failed: {str(e)}")
    
    def run_multiple_steps(steps):
        """Run multiple steps"""
        if not sim_state.baseline_model or not sim_state.countermeasure_model:
            set_status_message("Please initialize models first")
            return
        
        try:
            for _ in range(steps):
                sim_state.baseline_model.step()
                sim_state.countermeasure_model.step()
                sim_state.current_step += 1
                
                sim_state.baseline_data.append(get_model_data(sim_state.baseline_model))
                sim_state.countermeasure_data.append(get_model_data(sim_state.countermeasure_model))
            
            final_baseline = sim_state.baseline_data[-1]
            final_countermeasure = sim_state.countermeasure_data[-1]
            set_status_message(f"Completed {steps} steps, current step: {sim_state.current_step} - "
                             f"Baseline: {final_baseline['Active_Users']}active/{final_baseline['Total_Retweets']}retweets, "
                             f"Counter: {final_countermeasure['Active_Users']}active/{final_countermeasure['Total_Retweets']}retweets")
        except Exception as e:
            set_status_message(f"Multi-step simulation failed: {str(e)}")
    
    with solara.Card("Simulation Control"):
        with solara.Row():
            solara.Button(label="Single Step", on_click=run_single_step)
            solara.Button(label="10 Steps", on_click=lambda: run_multiple_steps(10))
            solara.Button(label="50 Steps", on_click=lambda: run_multiple_steps(50))
            solara.Button(label="100 Steps", on_click=lambda: run_multiple_steps(100))
            solara.Button(label="200 Steps", on_click=lambda: run_multiple_steps(200))
        
        solara.Info(status_message)

@solara.component
def DebugDataCard():
    """Debug data status component"""
    with solara.Card("Data Debug Information"):
        solara.Markdown("**Current Data Status:**")
        
        # Baseline data status
        baseline_count = len(sim_state.baseline_data) if sim_state.baseline_data else 0
        countermeasure_count = len(sim_state.countermeasure_data) if sim_state.countermeasure_data else 0
        
        solara.Text(f"Baseline data count: {baseline_count}")
        solara.Text(f"Countermeasure data count: {countermeasure_count}")
        solara.Text(f"Current step: {sim_state.current_step}")
        
        # Show latest data samples
        if sim_state.baseline_data:
            latest_baseline = sim_state.baseline_data[-1]
            solara.Markdown(f"**Latest baseline data**: {latest_baseline}")
        
        if sim_state.countermeasure_data:
            latest_countermeasure = sim_state.countermeasure_data[-1]
            solara.Markdown(f"**Latest countermeasure data**: {latest_countermeasure}")
        
        # Model status
        baseline_model_status = "Initialized" if sim_state.baseline_model else "Not initialized"
        countermeasure_model_status = "Initialized" if sim_state.countermeasure_model else "Not initialized"
        
        solara.Text(f"Baseline model: {baseline_model_status}")
        solara.Text(f"Countermeasure model: {countermeasure_model_status}")

@solara.component
def StatisticsCard():
    """Statistical analysis component"""
    if not sim_state.baseline_data or not sim_state.countermeasure_data or len(sim_state.baseline_data) <= 1:
        with solara.Card("Usage Instructions"):
            solara.Markdown("""
            **Quick Parameter Tuning Guide**:
            1. Load data
            2. Adjust countermeasure parameters
            3. Initialize dual models
            4. Run simulation (recommended 50-100 steps for testing)
            5. View comparative statistical analysis
            
            **Parameter Tuning Suggestions**:
            - **Key Node Intervention**: Suitable for targeting high-influence users, lower threshold = more sensitive
            - **Fact Checking**: Suitable for intervention after large-scale spread, threshold determines when to trigger
            - **Early Warning**: Suitable for early prevention, coverage ratio determines impact scope
            
            **Optimization Goal**: Maximize misinformation spread reduction while maintaining reasonable costs
            """)
        return
    
    # Convert to DataFrame
    baseline_df = pd.DataFrame(sim_state.baseline_data)
    countermeasure_df = pd.DataFrame(sim_state.countermeasure_data)
    
    with solara.Card("Comparative Statistical Analysis"):
        # Active users comparison chart
        if 'Active_Users' in baseline_df.columns and 'Active_Users' in countermeasure_df.columns:
            fig_active = create_comparison_chart(
                baseline_df, countermeasure_df, 'Active_Users',
                'Active Users Comparison', 'Active Users', 'red', 'blue'
            )
            solara.FigureMatplotlib(fig_active)
        
        # Total retweets comparison chart
        if 'Total_Retweets' in baseline_df.columns and 'Total_Retweets' in countermeasure_df.columns:
            fig_retweets = create_comparison_chart(
                baseline_df, countermeasure_df, 'Total_Retweets',
                'Total Retweets Comparison', ' Total Retweets', 'orange', 'green'
            )
            solara.FigureMatplotlib(fig_retweets)
        
        # Detailed statistics table
        create_statistics_table(baseline_df, countermeasure_df)
        
        # Effectiveness summary
        create_effectiveness_summary(baseline_df, countermeasure_df)

def get_model_data(model):
    """Get current data from model (enhanced version)"""
    try:
        # Try to get from datacollector
        df = model.datacollector.get_model_vars_dataframe()
        if len(df) > 0:
            return df.iloc[-1].to_dict()
        else:
            # datacollector is empty, calculate manually
            return calculate_model_data_manually(model)
    except Exception as e:
        print(f"Failed to get data from datacollector: {e}")
        # Calculate data manually
        return calculate_model_data_manually(model)

def calculate_model_data_manually(model):
    """Calculate model data manually"""
    try:
        active_users = len([a for a in model.user_agents.values() if a.is_active()])
        total_retweets = model.get_total_retweets()
        
        misinfo_spread = {}
        countermeasure_coverage = {}
        for mid in model.misinfo_stats.keys():
            misinfo_spread[mid] = model.get_misinfo_spread(mid)
            countermeasure_coverage[mid] = model.get_countermeasure_coverage(mid)
        
        return {
            "Active_Users": active_users,
            "Total_Retweets": total_retweets,
            "Misinfo_Spread": misinfo_spread,
            "Countermeasure_Coverage": countermeasure_coverage
        }
    except Exception as e:
        print(f"Manual data calculation also failed: {e}")
        return {
            "Active_Users": 0,
            "Total_Retweets": 0,
            "Misinfo_Spread": {},
            "Countermeasure_Coverage": {}
        }

def create_comparison_chart(baseline_df, countermeasure_df, column, title, ylabel, 
                          baseline_color, countermeasure_color):
    """Create comparison chart"""
    fig = Figure(figsize=(12, 6))
    ax = fig.add_subplot()
    
    ax.plot(baseline_df.index, baseline_df[column], 
           marker='o', linewidth=2, label='Baseline Model (No Countermeasures)', color=baseline_color)
    ax.plot(countermeasure_df.index, countermeasure_df[column], 
           marker='s', linewidth=2, label='Countermeasure Model', color=countermeasure_color)
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('time steps', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add final value annotations
    baseline_final = baseline_df[column].iloc[-1]
    countermeasure_final = countermeasure_df[column].iloc[-1]
    
    ax.annotate(f'Baseline: {baseline_final}', 
               xy=(len(baseline_df)-1, baseline_final), 
               xytext=(10, 10), textcoords='offset points',
               bbox=dict(boxstyle='round,pad=0.3', facecolor=baseline_color, alpha=0.3))
    ax.annotate(f'Counter: {countermeasure_final}', 
               xy=(len(countermeasure_df)-1, countermeasure_final), 
               xytext=(10, -20), textcoords='offset points',
               bbox=dict(boxstyle='round,pad=0.3', facecolor=countermeasure_color, alpha=0.3))
    
    return fig

def create_statistics_table(baseline_df, countermeasure_df):
    """Create statistical comparison table"""
    stages = [0.25, 0.5, 0.75, 1.0]
    stage_names = ["25% Progress", "50% Progress", "75% Progress", "Final"]
    
    comparison_data = []
    for stage, name in zip(stages, stage_names):
        stage_index = max(1, int(len(baseline_df) * stage) - 1)
        baseline_stage = baseline_df.iloc[stage_index]
        countermeasure_stage = countermeasure_df.iloc[stage_index]
        
        active_reduction = baseline_stage['Active_Users'] - countermeasure_stage['Active_Users']
        retweet_reduction = baseline_stage['Total_Retweets'] - countermeasure_stage['Total_Retweets']
        
        active_reduction_pct = (active_reduction / max(baseline_stage['Active_Users'], 1)) * 100
        retweet_reduction_pct = (retweet_reduction / max(baseline_stage['Total_Retweets'], 1)) * 100
        
        comparison_data.append({
            "Stage": name,
            "Baseline Active": int(baseline_stage['Active_Users']),
            "Counter Active": int(countermeasure_stage['Active_Users']),
            "Active Reduction": f"{int(active_reduction)} ({active_reduction_pct:.1f}%)",
            "Baseline Retweets": int(baseline_stage['Total_Retweets']),
            "Counter Retweets": int(countermeasure_stage['Total_Retweets']),
            "Retweet Reduction": f"{int(retweet_reduction)} ({retweet_reduction_pct:.1f}%)"
        })
    
    solara.Markdown("**Stage-wise Effectiveness Comparison Analysis**")
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create table
    table_markdown = "| " + " | ".join(comparison_df.columns) + " |\n"
    table_markdown += "|" + "|".join(["---"] * len(comparison_df.columns)) + "|\n"
    for _, row in comparison_df.iterrows():
        table_markdown += "| " + " | ".join(str(row[col]) for col in comparison_df.columns) + " |\n"
    
    solara.Markdown(table_markdown)

def create_effectiveness_summary(baseline_df, countermeasure_df):
    """Create effectiveness summary"""
    baseline_latest = baseline_df.iloc[-1]
    countermeasure_latest = countermeasure_df.iloc[-1]
    
    final_active_reduction_pct = (
        (baseline_latest['Active_Users'] - countermeasure_latest['Active_Users']) / 
        max(baseline_latest['Active_Users'], 1) * 100
    )
    final_retweet_reduction_pct = (
        (baseline_latest['Total_Retweets'] - countermeasure_latest['Total_Retweets']) / 
        max(baseline_latest['Total_Retweets'], 1) * 100
    )
    
    # Get currently enabled countermeasures
    enabled_measures = []
    settings = sim_state.countermeasure_settings
    if settings.get('key_node_enabled'):
        enabled_measures.append(f"Key Node(threshold{settings['key_node_threshold']}, >{settings['key_node_followers']}followers)")
    if settings.get('fact_check_enabled'):
        enabled_measures.append(f"Fact Check(threshold{settings['fact_check_threshold']}, delay{settings['fact_check_delay']})")
    if settings.get('early_warning_enabled'):
        enabled_measures.append(f"Early Warning(threshold{settings['early_warning_threshold']}, coverage{settings['early_warning_coverage']})")
    
    effectiveness_icon = (
        "✅ Highly Effective!" if final_active_reduction_pct > 50 else 
        "⚠️ Moderately Effective" if final_active_reduction_pct > 25 else 
        "❌ Low Effectiveness"
    )
    
    solara.Markdown(f"""
    **Final Effectiveness Summary**:
    - Current Configuration: {', '.join(enabled_measures) if enabled_measures else 'No Countermeasures'}
    - Active Users Reduction: **{final_active_reduction_pct:.1f}%**
    - Retweets Reduction: **{final_retweet_reduction_pct:.1f}%**
    - Simulation Steps: {sim_state.current_step}
    
    {effectiveness_icon}
    """)

@solara.component
def Page():
    """Main page component"""
    solara.Title("Misinformation Propagation Simulator - Quick Comparison Statistics")
    
    with solara.Column():
        DataLoadingCard()
        CountermeasureSettingsCard()
        SimulationControlCard()
        DebugDataCard()
        StatisticsCard()