import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import warnings
import random
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu

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
    # Add a column to track learning effect
    user_profiles['learning_effect'] = np.zeros(num_users)  # Initialize learning effect
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
# Feature Encoding
# -------------------------------

def encode_features(df):
    """Encodes categorical features using OneHotEncoder."""
    categorical_cols = ['political_ideology', 'gender', 'education', 'topic', 'sentiment']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
    # Preserve user_id and learning_effect
    preserved_cols = ['user_id', 'content_id', 'learning_effect'] if 'learning_effect' in df.columns else ['user_id', 'content_id']
    # Drop only the categorical columns (not preserved columns)
    df_numeric = df.drop(categorical_cols, axis=1)
    # Ensure all columns in df_numeric are numeric
    for col in df_numeric.columns:
        if df_numeric[col].dtype == 'object':
            df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
    df_encoded = pd.concat([df[preserved_cols], df_numeric, encoded_df], axis=1)
    return df_encoded

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
    # Convert all data to numpy arrays
    X_train_array = X_train.to_numpy()
    X_test_array = X_test.to_numpy()
    y_train_array = y_train.to_numpy()
    y_test_array = y_test.to_numpy()
    # Train model
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train_array, y_train_array)
    # Predict
    y_pred = model.predict(X_test_array)
    # Evaluate
    accuracy = accuracy_score(y_test_array, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    return model, list(X.columns)  # Return the model and the feature names used for training

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
baseline_intervention_probs = [0.25, 0.25, 0.25, 0.25]  # Example probabilities

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
    'Informational Message': 0.5,     # 50% base effectiveness
    'Neutral Feedback': 0.5,          # 50% base effectiveness
    'Engagement Prompt': 0.5          # 50% base effectiveness
}

def intervention_effectiveness_personalized(row):
    """Determines the effectiveness of personalized interventions."""
    if row['susceptible'] == 1:
        base_effectiveness = 0.0
        if row['intervention'] == 'Prebunking (Context)':
            base_effectiveness = 0.8 if row['CMQ_score'] > 4 and row['topic'] == 'politics' else 0.5
        elif row['intervention'] == 'Boosting (Educational Video)':
            base_effectiveness = 0.8 if row['CRT_score'] < 4 else 0.5
        elif row['intervention'] == 'Standard Warning':
            base_effectiveness = 0.5
        elif row['intervention'] == 'Nudge Warning':  # Handle 'Nudge Warning'
            base_effectiveness = 0.8  # Define base effectiveness for 'Nudge Warning'
        # Adjust effectiveness based on content complexity and emotional impact
        content_factor = (1 - row['complexity']) * 0.5 + (1 - row['emotional_impact']) * 0.5
        # Incorporate learning effect (positive or negative)
        learning_effect = row['learning_effect']
        effectiveness = base_effectiveness * content_factor * (1 + learning_effect)
        # Ensure effectiveness is between 0 and 1
        effectiveness = min(max(effectiveness, 0), 1)
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

        # Incorporate learning effect (positive or negative)
        learning_effect = row['learning_effect']
        effectiveness = base_effectiveness * content_factor * (1 + learning_effect)
        # Ensure effectiveness is between 0 and 1
        effectiveness = min(max(effectiveness, 0), 1)
        # Determine if the intervention is effective
        return int(random.random() < effectiveness)
    else:
        return 0  # Not susceptible, so intervention not needed

# -------------------------------
# Interaction Simulation Functions
# -------------------------------

def simulate_interactions_for_time_step(user_profiles, content_items, time_step):
    """Simulates user-content interactions for a single time step."""
    # Merge user profiles and content items to simulate interactions
    interactions = pd.merge(user_profiles.assign(key=1), content_items.assign(key=1), on='key').drop('key', axis=1)
    # Ensure 'user_id' is integer
    interactions['user_id'] = interactions['user_id'].astype(int)
    # Create a synthetic target variable: 1 if user is susceptible, 0 otherwise
    interactions['susceptible'] = interactions.apply(lambda row: 
        int(
            row['is_misinformation'] == 1 and
            row['susceptibility_score'] > np.random.rand()
        ), axis=1)
    # Add time step
    interactions['time_step'] = time_step
    return interactions

def simulate_interventions_over_time_personalized(interactions, model, user_profiles, feature_names):
    """Simulates personalized interventions and updates user profiles."""
    # Add learning effect to interactions before encoding
    interactions = interactions.merge(user_profiles[['user_id', 'learning_effect']], on='user_id', how='left')
    # Encode features
    interactions_encoded = encode_features(interactions)
    # Ensure the encoded interactions have the same features as the training data
    for feature in feature_names:
        if feature not in interactions_encoded.columns:
            interactions_encoded[feature] = 0  # Add missing columns with default value 0
    # Select only the features used in training
    X = interactions_encoded[feature_names]
    # Predict susceptibility
    interactions['predicted_susceptibility'] = model.predict_proba(X)[:,1]
    # Select interventions (Personalized)
    interactions['intervention'] = interactions.apply(select_intervention_personalized, axis=1)
    # Ensure 'learning_effect' is present
    if 'learning_effect' not in interactions.columns:
        print("Warning: 'learning_effect' not found in interactions. Adding it with default value 0.")
        interactions['learning_effect'] = 0
    # Simulate user response
    interactions['effective'] = interactions.apply(intervention_effectiveness_personalized, axis=1)
    # Update user profiles based on effective interventions
    effective_interventions = interactions[interactions['effective'] == 1]
    ineffective_interventions = interactions[(interactions['susceptible'] == 1) & (interactions['effective'] == 0)]
    users_with_effective_interventions = effective_interventions['user_id'].unique()
    users_with_ineffective_interventions = ineffective_interventions['user_id'].unique()
    # Update past interventions
    user_profiles.loc[user_profiles['user_id'].isin(users_with_effective_interventions), 'past_interventions'] += 1
    # Update learning effect
    # Positive learning effect for effective interventions, negative for ineffective
    user_profiles.loc[user_profiles['user_id'].isin(users_with_effective_interventions), 'learning_effect'] += 0.05  # Positive learning
    user_profiles.loc[user_profiles['user_id'].isin(users_with_ineffective_interventions), 'learning_effect'] -= 0.03  # Negative learning
    # Clip learning effect to a reasonable range, e.g., [-0.5, 0.5]
    user_profiles['learning_effect'] = user_profiles['learning_effect'].clip(-0.5, 0.5)
    # Calculate effectiveness
    total_susceptible = interactions['susceptible'].sum()
    total_effective = interactions['effective'].sum()
    effectiveness_percentage = (total_effective / total_susceptible * 100) if total_susceptible > 0 else 0
    print(f"Personalized Intervention Effectiveness at time step {interactions['time_step'].iloc[0]+1}: {total_effective}/{total_susceptible} ({effectiveness_percentage:.2f}%)")
    return interactions, user_profiles, total_effective, total_susceptible

def simulate_interventions_over_time_baseline(interactions, model, user_profiles, feature_names):
    """Simulates baseline interventions and updates user profiles."""
    # Add learning effect to interactions before encoding
    interactions = interactions.merge(user_profiles[['user_id', 'learning_effect']], on='user_id', how='left')
    # Encode features
    interactions_encoded = encode_features(interactions)
    # Ensure the encoded interactions have the same features as the training data
    for feature in feature_names:
        if feature not in interactions_encoded.columns:
            interactions_encoded[feature] = 0  # Add missing columns with default value 0
    # Select only the features used in training
    X = interactions_encoded[feature_names]
    # Predict susceptibility
    interactions['predicted_susceptibility'] = model.predict_proba(X)[:,1]
    # Select interventions (Baseline)
    interactions['intervention'] = interactions.apply(select_intervention_baseline, axis=1)
    # Ensure 'learning_effect' is present
    if 'learning_effect' not in interactions.columns:
        print("Warning: 'learning_effect' not found in interactions. Adding it with default value 0.")
        interactions['learning_effect'] = 0
    # Simulate user response
    interactions['effective'] = interactions.apply(intervention_effectiveness_baseline, axis=1)
    # Update user profiles based on effective interventions
    effective_interventions = interactions[interactions['effective'] == 1]
    ineffective_interventions = interactions[(interactions['susceptible'] == 1) & (interactions['effective'] == 0)]
    users_with_effective_interventions = effective_interventions['user_id'].unique()
    users_with_ineffective_interventions = ineffective_interventions['user_id'].unique()
    # Update past interventions
    user_profiles.loc[user_profiles['user_id'].isin(users_with_effective_interventions), 'past_interventions'] += 1
    # Update learning effect
    # Positive learning effect for effective interventions, negative for ineffective
    user_profiles.loc[user_profiles['user_id'].isin(users_with_effective_interventions), 'learning_effect'] += 0.02  # Smaller positive learning
    user_profiles.loc[user_profiles['user_id'].isin(users_with_ineffective_interventions), 'learning_effect'] -= 0.05  # Larger negative learning
    # Clip learning effect to a reasonable range, e.g., [-0.5, 0.5]
    user_profiles['learning_effect'] = user_profiles['learning_effect'].clip(-0.5, 0.5)
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
    num_users = 500  # Number of users
    num_content = 500  # Number of content items
    time_steps = 100  # Number of time steps to simulate

    # Generate synthetic user profiles for both systems
    print("Generating synthetic user profiles for Personalized and Baseline systems...")
    user_profiles_personalized = generate_user_profiles(num_users)
    user_profiles_baseline = user_profiles_personalized.copy(deep=True)  # Separate profiles for baseline

    # Generate synthetic content items (shared by both systems)
    print("Generating synthetic content items...")
    content_items = generate_content_items(num_content)

    # Initialize lists to store effectiveness metrics
    effectiveness_personalized = []
    total_susceptible_personalized = []

    effectiveness_baseline = []
    total_susceptible_baseline = []

    # Lists to store average susceptibility scores
    avg_susceptibility_personalized = []
    avg_susceptibility_baseline = []

    # Initialize lists to collect all interactions after interventions
    all_interactions_personalized = []
    all_interactions_baseline = []

    effectiveness_rate_personalized = []
    effectiveness_rate_baseline = []

    # Select a sample of users to track
    sample_user_ids = user_profiles_personalized['user_id'].sample(5, random_state=42).tolist()
    # Initialize a dictionary to store susceptibility over time
    user_susceptibility_over_time = {user_id: [] for user_id in sample_user_ids}

    # Initialize model (trained on initial data)
    # Simulate initial interactions to train the model
    interactions_initial = simulate_interactions_for_time_step(user_profiles_personalized, content_items, 0)
    # Add learning_effect column to initial interactions
    interactions_initial['learning_effect'] = 0
    print("\nTraining predictive model for Personalized system...")
    model_personalized, feature_names = train_model(interactions_initial)

    print("\nSimulating interventions over time for both systems...")
    for t in range(time_steps):
        print(f"\n--- Time Step {t+1} ---")

        # Update user susceptibility based on past interventions and learning effects
        # Apply a more realistic decay function
        # For example, susceptibility could decrease exponentially with diminishing returns
        user_profiles_personalized['susceptibility_score'] *= np.exp(-0.1 * user_profiles_personalized['past_interventions'])
        user_profiles_personalized['susceptibility_score'] = user_profiles_personalized['susceptibility_score'].clip(0.1, 1)  # Avoid zero susceptibility

        user_profiles_baseline['susceptibility_score'] *= np.exp(-0.1 * user_profiles_baseline['past_interventions'])
        user_profiles_baseline['susceptibility_score'] = user_profiles_baseline['susceptibility_score'].clip(0.1, 1)

        # Personalized Interventions
        interactions_t_pers = simulate_interactions_for_time_step(user_profiles_personalized, content_items, t)
        interactions_t_pers, user_profiles_personalized, eff_pers, tot_sus_pers = simulate_interventions_over_time_personalized(
            interactions_t_pers, model_personalized, user_profiles_personalized, feature_names)
        effectiveness_personalized.append(eff_pers)
        total_susceptible_personalized.append(tot_sus_pers)
        all_interactions_personalized.append(interactions_t_pers)

        # Calculate effectiveness rate for personalized system
        rate_pers = (eff_pers / tot_sus_pers) if tot_sus_pers > 0 else 0
        effectiveness_rate_personalized.append(rate_pers)

        # Baseline Interventions
        interactions_t_base = simulate_interactions_for_time_step(user_profiles_baseline, content_items, t)
        interactions_t_base, user_profiles_baseline, eff_base, tot_sus_base = simulate_interventions_over_time_baseline(
            interactions_t_base, model_personalized, user_profiles_baseline, feature_names)
        effectiveness_baseline.append(eff_base)
        total_susceptible_baseline.append(tot_sus_base)
        all_interactions_baseline.append(interactions_t_base)

        rate_base = (eff_base / tot_sus_base) if tot_sus_base > 0 else 0
        effectiveness_rate_baseline.append(rate_base)

        # Calculate average susceptibility
        avg_sus_pers = user_profiles_personalized['susceptibility_score'].mean()
        avg_sus_base = user_profiles_baseline['susceptibility_score'].mean()

        avg_susceptibility_personalized.append(avg_sus_pers)
        avg_susceptibility_baseline.append(avg_sus_base)

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

    # Convert lists to numpy arrays
    eff_rate_pers = np.array(effectiveness_rate_personalized)
    eff_rate_base = np.array(effectiveness_rate_baseline)

    # Perform independent t-test
    t_stat, p_value = ttest_ind(eff_rate_pers, eff_rate_base, equal_var=False)

    print(f"\nStatistical Comparison of Effectiveness Rates (t-test):")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4f}")

    alpha = 0.05  # Significance level
    if p_value < alpha:
        print("The difference in effectiveness rates is statistically significant.")
    else:
        print("The difference in effectiveness rates is not statistically significant.")

    # Optional: Perform Mann-Whitney U test
    u_stat, p_value_mwu = mannwhitneyu(eff_rate_pers, eff_rate_base, alternative='two-sided')

    print(f"\nStatistical Comparison of Effectiveness Rates (Mann-Whitney U Test):")
    print(f"U-statistic: {u_stat:.4f}")
    print(f"P-value: {p_value_mwu:.4f}")

    if p_value_mwu < alpha:
        print("The difference in effectiveness rates is statistically significant (Mann-Whitney U Test).")
    else:
        print("The difference in effectiveness rates is not statistically significant (Mann-Whitney U Test).")

    # Calculate average effectiveness rates
    avg_effectiveness_personalized = (sum(effectiveness_personalized) / sum(total_susceptible_personalized)) * 100
    avg_effectiveness_baseline = (sum(effectiveness_baseline) / sum(total_susceptible_baseline)) * 100

    print(f"\nAverage Effectiveness Rate (Personalized System): {avg_effectiveness_personalized:.2f}%")
    print(f"Average Effectiveness Rate (Baseline System): {avg_effectiveness_baseline:.2f}%")

    # Visualization

    steps = range(1, time_steps + 1)

    # 1. Plot Effective Interventions Over Time
    plt.figure(figsize=(12, 6))
    plt.plot(steps, effectiveness_personalized, label='Personalized System', marker='o')
    plt.plot(steps, effectiveness_baseline, label='Baseline System', marker='s')
    plt.xlabel('Time Step')
    plt.ylabel('Number of Effective Interventions')
    plt.title('Comparison of Effective Interventions Over Time Between Personalized and Baseline Systems')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 2. Average Susceptibility Over Time
    plt.figure(figsize=(12, 6))
    plt.plot(steps, avg_susceptibility_personalized, label='Personalized System', marker='o')
    plt.plot(steps, avg_susceptibility_baseline, label='Baseline System', marker='s')
    plt.xlabel('Time Step')
    plt.ylabel('Average Susceptibility Score')
    plt.title('Average Susceptibility Score Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 3. Cumulative Effective Interventions Over Time
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

    # 4. User Susceptibility Trajectories
    plt.figure(figsize=(12, 6))
    for user_id in sample_user_ids:
        plt.plot(steps, user_susceptibility_over_time[user_id], label=f'User {user_id}')
    plt.xlabel('Time Step')
    plt.ylabel('Susceptibility Score')
    plt.title('Susceptibility Score Trajectories of Sample Users (Personalized System)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 5. Distribution of Susceptibility Scores at Final Time Step
    plt.figure(figsize=(12, 6))
    sns.histplot(user_profiles_personalized['susceptibility_score'], bins=20, kde=True, label='Personalized', color='blue')
    sns.histplot(user_profiles_baseline['susceptibility_score'], bins=20, kde=True, label='Baseline', color='orange', alpha=0.7)
    plt.xlabel('Susceptibility Score')
    plt.ylabel('Number of Users')
    plt.title('Distribution of User Susceptibility Scores at Final Time Step')
    plt.legend()
    plt.show()

    # 6. Intervention Effectiveness by Content Topic (Personalized System)
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

    # Baseline System
    topic_effectiveness_base = interactions_baseline_all.groupby('topic')['effective'].sum()
    topic_total_base = interactions_baseline_all.groupby('topic')['susceptible'].sum()
    topic_effectiveness_rate_base = (topic_effectiveness_base / topic_total_base) * 100

    # Plotting
    plt.figure(figsize=(10, 6))
    topic_effectiveness_rate_base.plot(kind='bar', color='orange')
    plt.xlabel('Content Topic')
    plt.ylabel('Intervention Effectiveness (%)')
    plt.title('Intervention Effectiveness by Content Topic (Baseline System)')
    plt.show()

    # 7. Effectiveness of Different Interventions in Personalized System
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

    # Baseline System
    intervention_effectiveness_base = interactions_baseline_all.groupby('intervention')['effective'].sum()
    intervention_total_base = interactions_baseline_all.groupby('intervention')['susceptible'].sum()
    intervention_effectiveness_rate_base = (intervention_effectiveness_base / intervention_total_base) * 100

    # Plotting
    plt.figure(figsize=(10, 6))
    intervention_effectiveness_rate_base.plot(kind='bar', color='orange')
    plt.xlabel('Intervention Type')
    plt.ylabel('Effectiveness (%)')
    plt.title('Effectiveness of Different Interventions in Baseline System')
    plt.show()

# -------------------------------
# Execute the Comparison
# -------------------------------

if __name__ == "__main__":
    main_comparison()