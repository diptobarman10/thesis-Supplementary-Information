import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import warnings
import random

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
        print(f"Simulating interactions for time step {t+1}...")
        # Update user susceptibility based on past interventions
        user_profiles['susceptibility_score'] = user_profiles['susceptibility_score'] * (1 - 0.1 * user_profiles['past_interventions'])
        # Ensure susceptibility_score stays between 0 and 1
        user_profiles['susceptibility_score'] = user_profiles['susceptibility_score'].clip(0, 1)
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
    """Trains an XGBoost classifier to predict susceptibility."""
    # Encode features
    interactions_encoded = encode_features(interactions)
    # Define features and target
    X = interactions_encoded.drop(['user_id', 'content_id', 'susceptible', 'is_misinformation', 'time_step'], axis=1)
    y = interactions_encoded['susceptible']
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train model
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    # Predict
    y_pred = model.predict(X_test)
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    return model

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

# Define base effectiveness for each baseline intervention
baseline_effectiveness = {
    'Standard Warning': 0.5,          # 50% base effectiveness
    'Informational Message': 0.4,     # 40% base effectiveness
    'Neutral Feedback': 0.3,          # 30% base effectiveness
    'Engagement Prompt': 0.6          # 60% base effectiveness
}

def intervention_effectiveness_personalized(row):
    """Determines the effectiveness of personalized interventions."""
    if row['susceptible'] == 1:
        base_effectiveness = 0.0
        if row['intervention'] == 'Prebunking (Context)':
            base_effectiveness = 0.8 if row['CMQ_score'] > 4 else 0.5
        elif row['intervention'] == 'Boosting (Educational Video)':
            base_effectiveness = 0.7 if row['CRT_score'] < 4 else 0.5
        elif row['intervention'] == 'Standard Warning':
            base_effectiveness = 0.5
        elif row['intervention'] == 'Nudge Warning':  # Handle 'Nudge Warning'
            base_effectiveness = 0.6  # Define base effectiveness for 'Nudge Warning'
        # Adjust effectiveness based on content complexity and emotional impact
        content_factor = (1 - row['complexity']) * 0.5 + (1 - row['emotional_impact']) * 0.5
        effectiveness = base_effectiveness * content_factor
        return int(random.random() < effectiveness)
    else:
        return 0  # Not susceptible, so intervention not needed

def intervention_effectiveness_baseline(row):
    """Determines the effectiveness of baseline interventions."""
    if row['susceptible'] == 1:
        # Get the base effectiveness based on the intervention type
        base_effectiveness = baseline_effectiveness.get(row['intervention'], 0.5)  # Default to 0.5 if not found
        
        # Adjust effectiveness based on content complexity and emotional impact
        content_factor = (1 - row['complexity']) * 0.5 + (1 - row['emotional_impact']) * 0.5
        effectiveness = base_effectiveness * content_factor
        
        # Determine if the intervention is effective
        return int(random.random() < effectiveness)
    else:
        return 0  # Not susceptible, so intervention not needed

# -------------------------------
# Intervention Simulation Functions
# -------------------------------

def simulate_interventions_over_time_personalized(interactions, model, user_profiles):
    """Simulates personalized interventions and updates user profiles."""
    interactions_encoded = encode_features(interactions)
    X = interactions_encoded.drop(['user_id', 'content_id', 'susceptible', 'is_misinformation', 'time_step'], axis=1)
    interactions['predicted_susceptibility'] = model.predict_proba(X)[:,1]
    # Select interventions (Personalized)
    interactions['intervention'] = interactions.apply(select_intervention_personalized, axis=1)
    
    # Simulate user response
    interactions['effective'] = interactions.apply(intervention_effectiveness_personalized, axis=1)
    
    # Update user profiles based on effective interventions
    effective_interventions = interactions[interactions['effective'] == 1]
    users_with_effective_interventions = effective_interventions['user_id'].unique()
    user_profiles.loc[user_profiles['user_id'].isin(users_with_effective_interventions), 'past_interventions'] += 1
    
    # Calculate effectiveness
    total_susceptible = interactions['susceptible'].sum()
    total_effective = interactions['effective'].sum()
    effectiveness_percentage = (total_effective / total_susceptible * 100) if total_susceptible > 0 else 0
    print(f"Personalized Intervention Effectiveness at time step {interactions['time_step'].iloc[0]+1}: {total_effective}/{total_susceptible} ({effectiveness_percentage:.2f}%)")
    
    return interactions, user_profiles, total_effective, total_susceptible

def simulate_interventions_over_time_baseline(interactions, model, user_profiles):
    """Simulates baseline interventions and updates user profiles."""
    interactions_encoded = encode_features(interactions)
    X = interactions_encoded.drop(['user_id', 'content_id', 'susceptible', 'is_misinformation', 'time_step'], axis=1)
    interactions['predicted_susceptibility'] = model.predict_proba(X)[:,1]
    # Select interventions (Baseline)
    interactions['intervention'] = interactions.apply(select_intervention_baseline, axis=1)
    
    # Simulate user response
    interactions['effective'] = interactions.apply(intervention_effectiveness_baseline, axis=1)
    
    # Update user profiles based on effective interventions
    effective_interventions = interactions[interactions['effective'] == 1]
    users_with_effective_interventions = effective_interventions['user_id'].unique()
    user_profiles.loc[user_profiles['user_id'].isin(users_with_effective_interventions), 'past_interventions'] += 1
    
    # Calculate effectiveness
    total_susceptible = interactions['susceptible'].sum()
    total_effective = interactions['effective'].sum()
    effectiveness_percentage = (total_effective / total_susceptible * 100) if total_susceptible > 0 else 0
    print(f"Baseline Intervention Effectiveness at time step {interactions['time_step'].iloc[0]+1}: {total_effective}/{total_susceptible} ({effectiveness_percentage:.2f}%)")
    
    return interactions, user_profiles, total_effective, total_susceptible

# -------------------------------
# Main Comparison Function
# -------------------------------

def main_comparison():
    # Parameters
    num_users = 100  # Number of users
    num_content = 200  # Number of content items
    time_steps = 100  # Number of time steps to simulate

    # Generate synthetic user profiles for both systems
    print("Generating synthetic user profiles for Personalized and Baseline systems...")
    user_profiles_personalized = generate_user_profiles(num_users)
    user_profiles_baseline = generate_user_profiles(num_users)  # Separate profiles for baseline

    # Generate synthetic content items (shared by both systems)
    print("Generating synthetic content items...")
    content_items = generate_content_items(num_content)

    # Simulate interactions for both systems
    print("\nSimulating interactions for Personalized system...")
    interactions_personalized = simulate_interactions(user_profiles_personalized, content_items, time_steps)
    
    print("\nSimulating interactions for Baseline system...")
    interactions_baseline = simulate_interactions(user_profiles_baseline, content_items, time_steps)

    # Train predictive models for both systems
    print("\nTraining predictive model for Personalized system...")
    model_personalized = train_model(interactions_personalized)
    
    print("\nTraining predictive model for Baseline system...")
    model_baseline = train_model(interactions_baseline)

    # Initialize lists to store effectiveness metrics
    effectiveness_personalized = []
    total_susceptible_personalized = []
    
    effectiveness_baseline = []
    total_susceptible_baseline = []
    
    # Lists to store average susceptibility scores
    avg_susceptibility_personalized = []
    avg_susceptibility_baseline = []
    
    # Lists for group comparisons (e.g., high vs. low CRT scores)
    avg_sus_high_CRT_over_time = []
    avg_sus_low_CRT_over_time = []
    
    # Initialize lists to collect all interactions after interventions
    all_interactions_personalized = []
    all_interactions_baseline = []
    
    # Select a sample of users to track
    sample_user_ids = user_profiles_personalized['user_id'].sample(5, random_state=42).tolist()
    # Initialize a dictionary to store susceptibility over time
    user_susceptibility_over_time = {user_id: [] for user_id in sample_user_ids}
    
    print("\nSimulating interventions over time for both systems...")
    for t in range(time_steps):
        print(f"\n--- Time Step {t+1} ---")
        
        # Personalized Interventions
        interactions_t_pers = interactions_personalized[interactions_personalized['time_step'] == t]
        interactions_t_pers, user_profiles_personalized, eff_pers, tot_sus_pers = simulate_interventions_over_time_personalized(interactions_t_pers, model_personalized, user_profiles_personalized)
        effectiveness_personalized.append(eff_pers)
        total_susceptible_personalized.append(tot_sus_pers)
        all_interactions_personalized.append(interactions_t_pers)
        
        # Baseline Interventions
        interactions_t_base = interactions_baseline[interactions_baseline['time_step'] == t]
        interactions_t_base, user_profiles_baseline, eff_base, tot_sus_base = simulate_interventions_over_time_baseline(interactions_t_base, model_baseline, user_profiles_baseline)
        effectiveness_baseline.append(eff_base)
        total_susceptible_baseline.append(tot_sus_base)
        all_interactions_baseline.append(interactions_t_base)
        
        # Calculate average susceptibility
        avg_sus_pers = user_profiles_personalized['susceptibility_score'].mean()
        avg_sus_base = user_profiles_baseline['susceptibility_score'].mean()
        
        avg_susceptibility_personalized.append(avg_sus_pers)
        avg_susceptibility_baseline.append(avg_sus_base)
        
        # Group comparisons based on CRT score
        high_CRT_users = user_profiles_personalized[user_profiles_personalized['CRT_score'] >= 4]
        low_CRT_users = user_profiles_personalized[user_profiles_personalized['CRT_score'] < 4]
        
        avg_sus_high_CRT_over_time.append(high_CRT_users['susceptibility_score'].mean())
        avg_sus_low_CRT_over_time.append(low_CRT_users['susceptibility_score'].mean())
        
        # Track individual user susceptibility trajectories
        for user_id in sample_user_ids:
            sus_score = user_profiles_personalized.loc[user_profiles_personalized['user_id'] == user_id, 'susceptibility_score'].values[0]
            user_susceptibility_over_time[user_id].append(sus_score)

    # Concatenate all interactions
    interactions_personalized_all = pd.concat(all_interactions_personalized, ignore_index=True)
    interactions_baseline_all = pd.concat(all_interactions_baseline, ignore_index=True)
    
    # Summary of Results
    print("\n--- Summary of Intervention Effectiveness ---")
    print("{:<10} {:<25} {:<25}".format('Time Step', 'Personalized Effective', 'Baseline Effective'))
    for t in range(time_steps):
        print("{:<10} {:<25} {:<25}".format(
            t+1,
            effectiveness_personalized[t],
            effectiveness_baseline[t]
        ))
    
    # Visualization

    # 1. Plot Effective Interventions Over Time
    steps = range(1, time_steps + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(steps, effectiveness_personalized, label='Personalized System', marker='o')
    plt.plot(steps, effectiveness_baseline, label='Baseline System', marker='s')
    plt.xlabel('Time Step')
    plt.ylabel('Number of Effective Interventions')
    plt.title('Comparison of Personalized vs. Baseline Intervention Effectiveness')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    # 3. Distribution of Susceptibility Scores at Final Time Step
    plt.figure(figsize=(12, 6))
    sns.histplot(user_profiles_personalized['susceptibility_score'], bins=20, kde=True, label='Personalized', color='blue')
    sns.histplot(user_profiles_baseline['susceptibility_score'], bins=20, kde=True, label='Baseline', color='orange', alpha=0.7)
    plt.xlabel('Susceptibility Score')
    plt.ylabel('Number of Users')
    plt.title('Distribution of User Susceptibility Scores at Final Time Step')
    plt.legend()
    plt.show()
    
    
    
    # 6. Cumulative Effective Interventions Over Time
    cumulative_effectiveness_personalized = np.cumsum(effectiveness_personalized)
    cumulative_effectiveness_baseline = np.cumsum(effectiveness_baseline)
    
    plt.figure(figsize=(12, 6))
    plt.plot(steps, cumulative_effectiveness_personalized, label='Personalized System', marker='o')
    plt.plot(steps, cumulative_effectiveness_baseline, label='Baseline System', marker='s')
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Effective Interventions')
    plt.title('Cumulative Effective Interventions Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
    # 9. Intervention Effectiveness by Content Topic (Personalized System)
    topic_effectiveness = interactions_personalized_all.groupby('topic')['effective'].sum()
    topic_total = interactions_personalized_all.groupby('topic')['susceptible'].sum()
    topic_effectiveness_rate = (topic_effectiveness / topic_total) * 100
    
    # Plotting
    plt.figure(figsize=(10, 6))
    topic_effectiveness_rate.plot(kind='bar')
    plt.xlabel('Content Topic')
    plt.ylabel('Intervention Effectiveness (%)')
    plt.title('Intervention Effectiveness by Content Topic (Personalized System)')
    plt.show()
    
    # 10. Effectiveness of Different Interventions in Personalized System
    intervention_effectiveness = interactions_personalized_all.groupby('intervention')['effective'].sum()
    intervention_total = interactions_personalized_all.groupby('intervention')['susceptible'].sum()
    intervention_effectiveness_rate = (intervention_effectiveness / intervention_total) * 100
    
    # Plotting
    plt.figure(figsize=(10, 6))
    intervention_effectiveness_rate.plot(kind='bar')
    plt.xlabel('Intervention Type')
    plt.ylabel('Effectiveness (%)')
    plt.title('Effectiveness of Different Interventions in Personalized System')
    plt.show()
    
# -------------------------------
# Execute the Comparison
# -------------------------------

if __name__ == "__main__":
    main_comparison()