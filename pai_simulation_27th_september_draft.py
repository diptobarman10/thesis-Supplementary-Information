import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import warnings
import random
from model_selection import select_best_model

# Optional: For visualization
import matplotlib.pyplot as plt
import seaborn as sns  # Added for advanced plotting

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# -------------------------------
# Data Generation Functions
# -------------------------------

def generate_user_profiles(num_users):
    """Generates synthetic user profiles."""
    user_profiles = pd.DataFrame({
        'user_id': range(num_users),
        'CRT_score': np.random.randint(0, 8, num_users),
        'CMQ_score': np.random.randint(0, 8, num_users),
        'openness': np.random.rand(num_users),
        'conscientiousness': np.random.rand(num_users),
        'extraversion': np.random.rand(num_users),
        'agreeableness': np.random.rand(num_users),
        'neuroticism': np.random.rand(num_users),
        'political_ideology': np.random.choice(['liberal', 'moderate', 'conservative'], num_users),
        'age': np.random.randint(18, 65, num_users),
        'gender': np.random.choice(['male', 'female', 'other'], num_users),
        'education': np.random.choice(['high_school', 'bachelor', 'master', 'phd'], num_users),
        'past_interventions': np.zeros(num_users),  # Initialize past interventions
        'susceptibility_score': np.random.rand(num_users)  # Initial susceptibility score
    })
    return user_profiles

def generate_content_items(num_items):
    """Generates synthetic content items."""
    content_items = pd.DataFrame({
        'content_id': range(num_items),
        'is_misinformation': np.random.choice([0, 1], num_items, p=[0.7, 0.3]),
        'topic': np.random.choice(['health', 'politics', 'technology', 'sports'], num_items),
        'sentiment': np.random.choice(['positive', 'neutral', 'negative'], num_items),
        'complexity': np.random.rand(num_items),
        'readability_score': np.random.rand(num_items),
        'emotional_impact': np.random.rand(num_items),
        'credibility_indicator': np.random.rand(num_items)
    })
    return content_items

# -------------------------------
# Interaction Simulation
# -------------------------------

def simulate_interactions(user_profiles, content_items, time_steps):
    """Simulates user-content interactions over multiple time steps."""
    interactions_list = []
    for t in range(time_steps):
        # Merge user profiles and content items to simulate interactions
        interactions = pd.merge(user_profiles.assign(key=1), content_items.assign(key=1), on='key').drop('key', axis=1)
        # Create a synthetic target variable: 1 if user is susceptible, 0 otherwise
        interactions['susceptible'] = interactions.apply(lambda row: 
            int(
                row['is_misinformation'] == 1 and
                row['susceptibility_score'] > np.random.rand()
            ), axis=1)
        # Add time step
        interactions['time_step'] = t
        interactions_list.append(interactions)
    # Concatenate all interactions
    all_interactions = pd.concat(interactions_list, ignore_index=True)
    return all_interactions

# -------------------------------
# Feature Encoding
# -------------------------------

def encode_features(df):
    """Encodes categorical features using OneHotEncoder."""
    categorical_cols = ['political_ideology', 'gender', 'education', 'topic', 'sentiment']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
    df = pd.concat([df.drop(categorical_cols, axis=1).reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    return df

# -------------------------------
# Model Training
# -------------------------------

def train_model(interactions):
    """Selects and trains the best model to predict susceptibility."""
    best_model = select_best_model(interactions)
    print(f"Selected model: {type(best_model).__name__}")
    return best_model
# -------------------------------
# Intervention Selection Functions
# -------------------------------

# Define baseline interventions and their selection probabilities
baseline_interventions = [
    'Standard Warning',
    'Informational Message',
    'Neutral Feedback',
    'Engagement Prompt'
]

# Define selection probabilities (they should sum to 1)
baseline_intervention_probs = [0.4, 0.3, 0.2, 0.1]  # Example probabilities

def select_intervention_personalized(row):
    """Selects intervention based on user and content features."""
    # New condition for political ideology towards conservative
    if row['political_ideology'] == 'conservative':
        return 'Nudge Warning'  # Assign 'Nudge Warning' to conservative users
    # Existing personalized intervention logic
    if row['CMQ_score'] > 4 and row['topic'] == 'politics':
        return 'Prebunking (Context)'
    elif row['CRT_score'] < 4 and row['complexity'] > 0.5:
        return 'Boosting (Educational Video)'
    else:
        return 'Standard Warning'

def select_intervention_baseline(row):
    """Randomly selects a baseline intervention for each interaction."""
    return random.choices(baseline_interventions, weights=baseline_intervention_probs, k=1)[0]

# -------------------------------
# Intervention Effectiveness Functions
# -------------------------------

def intervention_effectiveness_personalized(row, parameters):
    """Determines the effectiveness of personalized interventions."""
    prebunking_effectiveness = parameters['prebunking_effectiveness']
    boosting_effectiveness = parameters['boosting_effectiveness']
    standard_warning_effectiveness = parameters['standard_warning_effectiveness']
    nudge_warning_effectiveness = parameters['nudge_warning_effectiveness']
    content_weight_complexity = parameters['content_weight_complexity']
    content_weight_emotional = parameters['content_weight_emotional']
    
    if row['susceptible'] == 1:
        base_effectiveness = 0.0
        if row['intervention'] == 'Prebunking (Context)':
            base_effectiveness = prebunking_effectiveness if row['CMQ_score'] > 4 else 0.5
        elif row['intervention'] == 'Boosting (Educational Video)':
            base_effectiveness = boosting_effectiveness if row['CRT_score'] < 4 else 0.5
        elif row['intervention'] == 'Standard Warning':
            base_effectiveness = standard_warning_effectiveness
        elif row['intervention'] == 'Nudge Warning':  # Handle 'Nudge Warning'
            base_effectiveness = nudge_warning_effectiveness  # Define base effectiveness for 'Nudge Warning'
        # Adjust effectiveness based on content complexity and emotional impact
        content_factor = (1 - row['complexity']) * content_weight_complexity + (1 - row['emotional_impact']) * content_weight_emotional
        effectiveness = base_effectiveness * content_factor
        return int(random.random() < effectiveness)
    else:
        return 0  # Not susceptible, so intervention not needed

def intervention_effectiveness_baseline(row, parameters):
    """Determines the effectiveness of baseline interventions."""
    baseline_effectiveness = parameters['baseline_effectiveness']
    content_weight_complexity = parameters['content_weight_complexity']
    content_weight_emotional = parameters['content_weight_emotional']
    
    if row['susceptible'] == 1:
        # Get the base effectiveness based on the intervention type
        base_effectiveness = baseline_effectiveness.get(row['intervention'], 0.5)  # Default to 0.5 if not found
        
        # Adjust effectiveness based on content complexity and emotional impact
        content_factor = (1 - row['complexity']) * content_weight_complexity + (1 - row['emotional_impact']) * content_weight_emotional
        effectiveness = base_effectiveness * content_factor
        
        # Determine if the intervention is effective
        return int(random.random() < effectiveness)
    else:
        return 0  # Not susceptible, so intervention not needed

# -------------------------------
# Intervention Simulation Functions
# -------------------------------

def simulate_interventions_over_time_personalized(interactions, model, user_profiles, parameters):
    """Simulates personalized interventions and updates user profiles."""
    interactions_encoded = encode_features(interactions)
    X = interactions_encoded.drop(['user_id', 'content_id', 'susceptible', 'is_misinformation', 'time_step'], axis=1)
    interactions['predicted_susceptibility'] = model.predict_proba(X)[:,1]
    # Select interventions (Personalized)
    interactions['intervention'] = interactions.apply(select_intervention_personalized, axis=1)
    
    # Simulate user response
    interactions['effective'] = interactions.apply(lambda row: intervention_effectiveness_personalized(row, parameters), axis=1)
    
    # Update user profiles based on effective interventions
    effective_interventions = interactions[interactions['effective'] == 1]
    users_with_effective_interventions = effective_interventions['user_id'].unique()
    user_profiles.loc[user_profiles['user_id'].isin(users_with_effective_interventions), 'past_interventions'] += 1
    
    # Update user susceptibility based on past interventions
    susceptibility_decay = parameters['susceptibility_decay']
    users_with_effective_interventions = effective_interventions['user_id'].unique()
    user_profiles.loc[user_profiles['user_id'].isin(users_with_effective_interventions), 'susceptibility_score'] *= (1 - susceptibility_decay)
    user_profiles['susceptibility_score'] = user_profiles['susceptibility_score'].clip(0, 1)
    
    # Calculate effectiveness
    total_susceptible = interactions['susceptible'].sum()
    total_effective = interactions['effective'].sum()
    effectiveness_percentage = (total_effective / total_susceptible * 100) if total_susceptible > 0 else 0
    print(f"Personalized Intervention Effectiveness at time step {interactions['time_step'].iloc[0]+1}: {total_effective}/{total_susceptible} ({effectiveness_percentage:.2f}%)")
    
    return interactions, user_profiles, total_effective, total_susceptible

def simulate_interventions_over_time_baseline(interactions, model, user_profiles, parameters):
    """Simulates baseline interventions and updates user profiles."""
    interactions_encoded = encode_features(interactions)
    X = interactions_encoded.drop(['user_id', 'content_id', 'susceptible', 'is_misinformation', 'time_step'], axis=1)
    interactions['predicted_susceptibility'] = model.predict_proba(X)[:,1]
    # Select interventions (Baseline)
    interactions['intervention'] = interactions.apply(select_intervention_baseline, axis=1)
    
    # Simulate user response
    interactions['effective'] = interactions.apply(lambda row: intervention_effectiveness_baseline(row, parameters), axis=1)
    
    # Update user profiles based on effective interventions
    effective_interventions = interactions[interactions['effective'] == 1]
    users_with_effective_interventions = effective_interventions['user_id'].unique()
    user_profiles.loc[user_profiles['user_id'].isin(users_with_effective_interventions), 'past_interventions'] += 1
    
    # Update susceptibility score for users with effective interventions
    susceptibility_decay = parameters['susceptibility_decay']
    users_with_effective_interventions = effective_interventions['user_id'].unique()
    user_profiles.loc[user_profiles['user_id'].isin(users_with_effective_interventions), 'susceptibility_score'] *= (1 - susceptibility_decay)
    user_profiles['susceptibility_score'] = user_profiles['susceptibility_score'].clip(0, 1)
    
    # Calculate effectiveness
    total_susceptible = interactions['susceptible'].sum()
    total_effective = interactions['effective'].sum()
    effectiveness_percentage = (total_effective / total_susceptible * 100) if total_susceptible > 0 else 0
    print(f"Baseline Intervention Effectiveness at time step {interactions['time_step'].iloc[0]+1}: {total_effective}/{total_susceptible} ({effectiveness_percentage:.2f}%)")
    
    return interactions, user_profiles, total_effective, total_susceptible

# -------------------------------
# Main Comparison Function
# -------------------------------

def main_comparison(parameters):
    # Unpack parameters
    num_users = parameters['num_users']
    num_content = parameters['num_content']
    time_steps = parameters['time_steps']
    susceptibility_decay = parameters['susceptibility_decay']
    prebunking_effectiveness = parameters['prebunking_effectiveness']
    boosting_effectiveness = parameters['boosting_effectiveness']
    standard_warning_effectiveness = parameters['standard_warning_effectiveness']
    nudge_warning_effectiveness = parameters['nudge_warning_effectiveness']
    content_weight_complexity = parameters['content_weight_complexity']
    content_weight_emotional = parameters['content_weight_emotional']
    
    num_users = parameters['num_users']
    num_content = parameters['num_content']
    time_steps = parameters['time_steps']
    # Add other parameters as needed

    # Generate synthetic user profiles for both systems
    print("\nGenerating synthetic user profiles for Personalized and Baseline systems...")
    user_profiles_personalized = generate_user_profiles(num_users)
    user_profiles_baseline = user_profiles_personalized.copy(deep=True)  # Use the same profiles for fair comparison

    # Generate synthetic content items (shared by both systems)
    print("Generating synthetic content items...")
    content_items = generate_content_items(num_content)

    # Simulate interactions for both systems
    print("\nSimulating interactions for Personalized system...")
    interactions_personalized = simulate_interactions(user_profiles_personalized, content_items, time_steps)

    print("\nSimulating interactions for Baseline system...")
    interactions_baseline = interactions_personalized.copy(deep=True)  # Use the same interactions for fair comparison

    # Train predictive models for both systems
    print("\nTraining predictive model for Personalized system...")
    model_personalized = train_model(interactions_personalized)

    # For fair comparison, use the same model
    model_baseline = model_personalized

    # Initialize lists to store effectiveness metrics
    effectiveness_personalized = []
    total_susceptible_personalized = []

    effectiveness_baseline = []
    total_susceptible_baseline = []

    # Initialize lists to collect all interactions after interventions
    all_interactions_personalized = []
    all_interactions_baseline = []
    
    avg_susceptibility_personalized = []
    avg_susceptibility_baseline = []

    print("\nSimulating interventions over time for both systems...")
    for t in range(time_steps):
        print(f"\n--- Time Step {t+1} ---")

        # Personalized Interventions
        interactions_t_pers = interactions_personalized[interactions_personalized['time_step'] == t]
        interactions_t_pers, user_profiles_personalized, eff_pers, tot_sus_pers = simulate_interventions_over_time_personalized(
            interactions_t_pers, model_personalized, user_profiles_personalized, parameters
        )
        effectiveness_personalized.append(eff_pers)
        total_susceptible_personalized.append(tot_sus_pers)
        all_interactions_personalized.append(interactions_t_pers)
        
        avg_sus_pers = user_profiles_personalized['susceptibility_score'].mean()
        avg_susceptibility_personalized.append(avg_sus_pers)

        # Baseline Interventions
        interactions_t_base = interactions_baseline[interactions_baseline['time_step'] == t]
        interactions_t_base, user_profiles_baseline, eff_base, tot_sus_base = simulate_interventions_over_time_baseline(
            interactions_t_base, model_baseline, user_profiles_baseline, parameters
        )
        effectiveness_baseline.append(eff_base)
        total_susceptible_baseline.append(tot_sus_base)
        all_interactions_baseline.append(interactions_t_base)
        avg_sus_base = user_profiles_baseline['susceptibility_score'].mean()
        avg_susceptibility_baseline.append(avg_sus_base)

    # Concatenate all interactions
    interactions_personalized_all = pd.concat(all_interactions_personalized, ignore_index=True)
    interactions_baseline_all = pd.concat(all_interactions_baseline, ignore_index=True)

    # Calculate average effectiveness rates
    total_eff_pers = sum(effectiveness_personalized)
    total_sus_pers = sum(total_susceptible_personalized)
    avg_effectiveness_personalized = (total_eff_pers / total_sus_pers) * 100 if total_sus_pers > 0 else 0

    total_eff_base = sum(effectiveness_baseline)
    total_sus_base = sum(total_susceptible_baseline)
    avg_effectiveness_baseline = (total_eff_base / total_sus_base) * 100 if total_sus_base > 0 else 0

    avg_difference = avg_effectiveness_personalized - avg_effectiveness_baseline

    print(f"\nAverage Effectiveness Rate (Personalized System): {avg_effectiveness_personalized:.2f}%")
    print(f"Average Effectiveness Rate (Baseline System): {avg_effectiveness_baseline:.2f}%")
    print(f"Average Difference: {avg_difference:.2f}%")

    # Calculate cumulative effectiveness over time
    cumulative_effectiveness_personalized = np.cumsum(effectiveness_personalized)
    cumulative_effectiveness_baseline = np.cumsum(effectiveness_baseline)

    # Return cumulative effectiveness data along with averages
    return (avg_effectiveness_personalized, avg_effectiveness_baseline, avg_difference,
            cumulative_effectiveness_personalized, cumulative_effectiveness_baseline,
            avg_susceptibility_personalized, avg_susceptibility_baseline)

# -------------------------------
# Execute the Comparison
# -------------------------------

if __name__ == "__main__":
    import itertools

    # Define parameter ranges
    num_users_list = [100]  # Adjust as needed
    susceptibility_decay_list = [0, 0.1]
    prebunking_effectiveness_list = [0.7]
    num_content_list = [100]
    content_weights_list = [
        (0.5, 0.5),
        (0.7, 0.3),
        (0.3, 0.7)
    ]  # List of tuples (content_weight_complexity, content_weight_emotional)

    # Create a list of all combinations
    parameter_grid = list(itertools.product(
        num_users_list,
        susceptibility_decay_list,
        prebunking_effectiveness_list,
        num_content_list,
        content_weights_list
    ))

    # Initialize a list to collect results
    results_list = []
    cumulative_data = []

    for num_users, susceptibility_decay, prebunking_effectiveness, num_content, (content_weight_complexity, content_weight_emotional) in parameter_grid:
        print(f"\nRunning simulation with num_users={num_users}, susceptibility_decay={susceptibility_decay}, prebunking_effectiveness={prebunking_effectiveness}, num_content={num_content}, content_weights=({content_weight_complexity}, {content_weight_emotional})")
        parameters = {
            'num_users': num_users,
            'num_content': num_content,
            'time_steps': 10,  # Adjust as needed
            'susceptibility_decay': susceptibility_decay,
            'prebunking_effectiveness': prebunking_effectiveness,
            'boosting_effectiveness': 0.7,  # You can also vary this
            'standard_warning_effectiveness': 0.5,
            'nudge_warning_effectiveness': 0.6,
            'content_weight_complexity': content_weight_complexity,
            'content_weight_emotional': content_weight_emotional,
            'baseline_effectiveness': {
                'Standard Warning': 0.5,
                'Informational Message': 0.4,
                'Neutral Feedback': 0.3,
                'Engagement Prompt': 0.6
            },
        }

        (avg_effectiveness_personalized, avg_effectiveness_baseline, avg_difference,
         cumulative_effectiveness_personalized, cumulative_effectiveness_baseline,
         avg_susceptibility_personalized, avg_susceptibility_baseline) = main_comparison(parameters)

        results_list.append({
            'num_users': num_users,
            'susceptibility_decay': susceptibility_decay,
            'prebunking_effectiveness': prebunking_effectiveness,
            'num_content': num_content,
            'content_weight_complexity': content_weight_complexity,
            'content_weight_emotional': content_weight_emotional,
            'avg_effectiveness_personalized': avg_effectiveness_personalized,
            'avg_effectiveness_baseline': avg_effectiveness_baseline,
            'avg_difference': avg_difference
        })

        # Collect cumulative effectiveness data
        cumulative_data.append({
            'parameters': parameters,
            'cumulative_effectiveness_personalized': cumulative_effectiveness_personalized,
            'cumulative_effectiveness_baseline': cumulative_effectiveness_baseline,
            'avg_susceptibility_personalized': avg_susceptibility_personalized,
            'avg_susceptibility_baseline': avg_susceptibility_baseline
        })

    # Create the results DataFrame
    results = pd.DataFrame(results_list)

    # Display the results
    print("\nSimulation Results:")
    print(results.to_string(index=False))

    # Convert columns to appropriate data types
    results['avg_effectiveness_personalized'] = results['avg_effectiveness_personalized'].astype(float)
    results['avg_effectiveness_baseline'] = results['avg_effectiveness_baseline'].astype(float)
    results['avg_difference'] = results['avg_difference'].astype(float)
    results['susceptibility_decay'] = results['susceptibility_decay'].astype(float)
    results['prebunking_effectiveness'] = results['prebunking_effectiveness'].astype(float)
    results['num_users'] = results['num_users'].astype(int)
    results['num_content'] = results['num_content'].astype(int)
    results['content_weight_complexity'] = results['content_weight_complexity'].astype(float)
    results['content_weight_emotional'] = results['content_weight_emotional'].astype(float)

    # -------------------------------
    # Plot Cumulative Effective Interventions Over Time After All Simulations
    # -------------------------------

    # Visualization: Comparison between Personalized and Baseline Systems



    # Plotting average difference between Personalized and Baseline Systems
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=results, x='prebunking_effectiveness', y='avg_difference', hue='susceptibility_decay', style='num_users', markers=True, dashes=False)
    plt.title('Average Difference in Effectiveness (Personalized - Baseline)')
    plt.xlabel('Prebunking Effectiveness')
    plt.ylabel('Average Difference in Effectiveness (%)')
    plt.legend(title='Susceptibility Decay')
    plt.grid(True)
    plt.show()

    # Additional Visualization for Content Factor Weights
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=results, x='content_weight_complexity', y='avg_difference', hue='susceptibility_decay', style='num_users', markers=True, dashes=False)
    plt.title('Effect of Content Weight (Complexity) on Effectiveness Difference')
    plt.xlabel('Content Weight Complexity')
    plt.ylabel('Average Difference in Effectiveness (%)')
    plt.legend(title='Susceptibility Decay')
    plt.grid(True)
    plt.show()

    # Visualization comparing Personalized and Baseline systems with respect to Content Factor Weights
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=results, x='content_weight_complexity', y='avg_effectiveness_personalized', hue='susceptibility_decay', style='num_users', markers=True, dashes=False)
    sns.lineplot(data=results, x='content_weight_complexity', y='avg_effectiveness_baseline', hue='susceptibility_decay', style='num_users', markers=True, dashes=False, legend=False)
    plt.title('Average Effectiveness Rates vs Content Weight Complexity')
    plt.xlabel('Content Weight Complexity')
    plt.ylabel('Average Effectiveness (%)')
    plt.legend(title='System', labels=['Personalized', 'Baseline'])
    plt.grid(True)
    plt.show()
    
    # -------------------------------
    # Plot Average Susceptibility Over Time
    # -------------------------------

    # Select simulations with specific parameters for plotting
    selected_cumulative_data = [entry for entry in cumulative_data if
                            entry['parameters']['num_users'] == 100 and
                            entry['parameters']['content_weight_complexity'] == 0.5 and
                            entry['parameters']['susceptibility_decay'] == 0.1 and
                            entry['parameters']['prebunking_effectiveness'] == 0.7]

    for entry in selected_cumulative_data:
        steps = range(1, entry['parameters']['time_steps'] + 1)
        plt.figure(figsize=(12, 6))
        plt.plot(steps, entry['avg_susceptibility_personalized'], label='Personalized System', linestyle='-', marker='o')
        plt.plot(steps, entry['avg_susceptibility_baseline'], label='Baseline System', linestyle='--', marker='s')
        plt.xlabel('Time Step')
        plt.ylabel('Average Susceptibility Score')
        plt.title('Average User Susceptibility Over Time\n' +
                f"Parameters: num_users={entry['parameters']['num_users']}, " +
                f"prebunking_effectiveness={entry['parameters']['prebunking_effectiveness']}, " +
                f"susceptibility_decay={entry['parameters']['susceptibility_decay']}")
        plt.legend()
        plt.grid(True)
        plt.show()

    # After all simulations are complete and results are collected

    # Plot Effectiveness Over Time
    plt.figure(figsize=(12, 6))

    for entry in cumulative_data:
        time_steps = range(1, len(entry['cumulative_effectiveness_personalized']) + 1)
        
        # Calculate effectiveness rate at each time step
        effectiveness_personalized = [eff / (step * entry['parameters']['num_users']) * 100 
                                      for step, eff in zip(time_steps, entry['cumulative_effectiveness_personalized'])]
        effectiveness_baseline = [eff / (step * entry['parameters']['num_users']) * 100 
                                  for step, eff in zip(time_steps, entry['cumulative_effectiveness_baseline'])]
        
        # Create legend labels with both decay and content weight
        decay = entry['parameters']['susceptibility_decay']
        content_weight = (entry['parameters']['content_weight_complexity'], entry['parameters']['content_weight_emotional'])
        
        label_personalized = f"Personalized (Decay: {decay}, Weight: {content_weight})"
        label_baseline = f"Baseline (Decay: {decay}, Weight: {content_weight})"
        
        # Plot lines for this simulation
        plt.plot(time_steps, effectiveness_personalized, label=label_personalized)
        plt.plot(time_steps, effectiveness_baseline, label=label_baseline, linestyle='--')

    plt.title('Effectiveness Rates Over Time: Personalized vs Baseline Systems')
    plt.xlabel('Time Steps')
    plt.ylabel('Effectiveness Rate (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

