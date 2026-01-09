import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# 1. Load  datasets
stats_df = pd.read_csv('player_stats_2024_2025_season.csv')
values_df = pd.read_csv('premier_league_players.csv')
club_stats = pd.read_csv('2024_season_club_stats.csv')

# 2. Clean the Value data (Deduplicate)

values_df = values_df.drop_duplicates(subset=['Player Name'], keep='first')

# This line automatically finds the 'club' column regardless of exact spelling/case
club_col = [c for c in club_stats.columns if 'club' in c.lower()][0]
# Cleaning up team data to user later on
club_stats['Team_Quality'] = club_stats['Goals'] - club_stats['Goals Conceded']
# Setup columns so they don't interfere with player stats
club_subset = club_stats[[club_col, 'Goals Conceded', 'Passes', 'Team_Quality']].copy()
club_subset.columns = ['club_join_name', 'Team_Goals_Conceded', 'Team_Total_Passes', 'Team_Quality_Score']


# Map each club name to be the same so that all teams are included
team_mapping = {
    'AFC Bournemouth': 'Bournemouth',
    'Chelsea FC': 'Chelsea',
    'Brighton & Hove Albion': 'Brighton and Hove Albion',
    'Brentford FC': 'Brentford',
    'Fulham FC': 'Fulham',
    'Everton FC': 'Everton',
    'Liverpool FC': 'Liverpool',
    'Arsenal FC': 'Arsenal',
    'Burnley FC': 'Burnley',
}

# converts market values from strings to ints
def convert_value(val):
    if pd.isna(val) or val == '':
        return 0.0

    # Remove Euro symbol and any commas
    val = val.replace('€', '').replace(',', '')

    if 'm' in val:
        # Convert '180.00m' -> 180.0
        return float(val.replace('m', ''))
    elif 'k' in val:
        # Convert '600k' -> 0.6 (Standardizing everything to Millions)
        return float(val.replace('k', '')) / 1000

    try:
        return float(val)
    except ValueError:
        return 0.0
# 3. Apply the function to the 'Market Value' column
values_df['Market Value'] = values_df['Market Value'].apply(convert_value)
merged_df = pd.merge(
    stats_df,
    values_df[['Player Name', 'Position', 'Club', 'Market Value', 'Age']],
    left_on='player_name',
    right_on='Player Name',
    how='inner'
)
# Fix club name so that all club stats are included
merged_df['Club Cleaned'] = merged_df['Club'].replace(team_mapping)
# Add club data to merged player data
merged_df = pd.merge(
    merged_df,
    club_subset,
    left_on='Club Cleaned',
    right_on='club_join_name',
    how='left'
)


# 4. Remove the extra name column created by the match
# (Since 'player_name' and 'Player Name' are now identical)
merged_df = merged_df.drop(columns=['Player Name'])


# 5. Eliminate the "Just Transferred" / Low-Minute players
merged_df = merged_df[merged_df['Minutes Played'] >= 700]
# Have the option to remove high value players but can skew data
#merged_df = merged_df[merged_df['Market Value'] <= 120]


# 6. Final verification
print(f"Data Cleaned! Total players for analysis: {len(merged_df)}")
print(merged_df.head())

# 7 Pre-processing Data
# Get an understanding on variance of data
for column in merged_df:
    unique_vals = merged_df[column].unique()
    nr_values = len(unique_vals)
    if nr_values <= 12:
        print("The number of values for feature {} :{} -- {}".format(column, nr_values, unique_vals))
    else:
        print("The number of values for feature {} :{}".format(column, nr_values))

# Grouping players based on position to make analysis more accurate
position_groups = {
    'Attacking Midfield': 'Midfielder',
    'Central Midfield': 'Midfielder',
    'Defensive Midfield': 'Midfielder',
    'Left Midfield': 'Midfielder',
    'Centre-Back': 'Defender',
    'Left-Back': 'Defender',
    'Right-Back': 'Defender',
    'Second Striker': 'Forward',
    'Right Winger': 'Forward',
    'Left Winger': 'Forward',
    'Centre-Forward': 'Forward'
}



# Apply it to dataset
merged_df['Group'] = merged_df['Position'].map(position_groups)

forwards_df = merged_df[merged_df['Group'] == 'Forward'].copy()
# Need to distinguish between striker and wingers
forwards_df['is_striker'] = forwards_df['Position'].isin(['Centre-Forward', 'Second Striker']).astype(int)
mids_df = merged_df[merged_df['Group'] == 'Midfielder'].copy()
def_df = merged_df[merged_df['Group'] == 'Defender'].copy()


# Convert stats into per 90 to get better correlation data
stats_to_convert = ['Goals', 'Assists', 'XG', 'XA', 'Touches in the Opposition Box', 'Shots On Target Inside the Box',
                    'dribble_attempts', 'dribble_accuracy', 'Shots On Target Outside the Box', 'pass_attempts',
                    'pass_accuracy', 'long_pass_accuracy', 'Interceptions', 'Duels Won', 'Blocks', 'Total Tackles',
                    'Aerial Duels Won', 'Total Tackles']

# Function that converts stats into per 90 minutes
def convert_to_per_90(df, stats_list):
    for stat in stats_list:
        df[f'{stat}_per_90'] = (df[stat] / df['Minutes Played']) * 90
    return df

# Apply it to all position groups
forwards_df = convert_to_per_90(forwards_df, stats_to_convert)
mids_df = convert_to_per_90(mids_df, stats_to_convert)
def_df = convert_to_per_90(def_df, stats_to_convert)

# 8 Generating Heatmaps for each position group
# Want to find variables that have direct correlation with market value

forwards_cols = ['Market Value', 'Age', 'Minutes Played', 'Team_Total_Passes', 'Team_Quality_Score', 'XG_per_90', 'Goals_per_90',
                 'XA_per_90', 'Assists_per_90', 'Touches in the Opposition Box_per_90', 'dribble_accuracy',
                 'dribble_attempts_per_90', 'Shots On Target Inside the Box_per_90', 'cross_accuracy']

mids_cols = ['Market Value','Age','Minutes Played', 'Assists', 'XA_per_90','pass_attempts', 'pass_accuracy_per_90',
             'long_pass_accuracy_per_90', 'Interceptions_per_90', 'Team_Quality_Score', 'Team_Total_Passes',
             'dribble_attempts_per_90', 'dribble_accuracy_per_90']

def_cols = ['Market Value','Age', 'Minutes Played', 'Duels Won_per_90', 'Aerial Duels Won_per_90',
            'Interceptions_per_90', 'Blocks_per_90','Total Tackles','pass_accuracy_per_90',
            'long_pass_accuracy_per_90', 'Team_Goals_Conceded', 'Team_Quality_Score', 'Team_Total_Passes' ]

def generate_heatmap(group_df, columns, title, filename):
    # Get correlation matrix for the specified columns
    corr = group_df[columns].corr()

    # Set up visual
    plt.figure(figsize=(12, 10))

    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f",
                linewidths=0.5, annot_kws={'size': 9}, cbar_kws={'shrink': .8})

    plt.title(f'Correlation Heatmap: {title}', fontsize=16)

    # Rotate labels so they don't overlap
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig('my_heatmap.png')


# Run the heatmaps
# Call for Forwards
generate_heatmap(forwards_df, forwards_cols, "Forward Correlations", "forward_heatmap.png")

# Call for Midfielders
generate_heatmap(mids_df, mids_cols, "Midfielder Correlations", "midfielder_heatmap.png")

# Call for Defenders
generate_heatmap(def_df, def_cols, "Defender Correlations", "defender_heatmap.png")

# 9 Using a decision tree to calculat most important features
# Gathering what data values could be important for strikers and wingers
important_stats = ['Minutes Played', 'Age', 'Goals_per_90', 'XG_per_90', 'Touches in the Opposition Box_per_90',
                   'XA_per_90', 'Team_Total_Passes', 'XA', 'Assists_per_90', 'dribble_attempts', 'pass_attempts',
                   'long_pass_accuracy', 'Duels Won', 'pass_accuracy', 'Blocks', 'Team_Goals_Conceded',
                   'Team_Quality_Score', 'dribble_accuracy']
# Creating variables
X = forwards_df[important_stats]
Y = forwards_df['Market Value']

# Running the decision tree model
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
rf.fit(X, Y)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# 4. Print the Ranking
print("Feature Ranking according to Random Forest:")
for i in range(len(important_stats)):
    print(f"{i+1}. {important_stats[indices[i]]} ({importances[indices[i]]:.3f})")

# Doing the same thing but for midfielders and defenders
critical_stats = ['Age', 'XA', 'dribble_attempts', 'pass_attempts', 'long_pass_accuracy', 'Duels Won', 'Minutes Played',
                  'pass_accuracy', 'Team_Goals_Conceded', 'Team_Quality_Score', 'Interceptions', 'Blocks',
                  'Team_Total_Passes', 'Total Tackles']

# Creating variables
G = mids_df[critical_stats]
Z = mids_df['Market Value']

# Running the decision tree model
dn = RandomForestRegressor(n_estimators = 100, random_state = 42)
dn.fit(G, Z)

importancess = dn.feature_importances_
indicess = np.argsort(importancess)[::-1]

# 4. Print the Ranking
print("Feature Ranking according to Random Forest:")
for i in range(len(critical_stats)):
    print(f"{i+1}. {critical_stats[indicess[i]]} ({importancess[indicess[i]]:.3f})")


# 10 Training model based off data and weighted variables we have
# Create function so we don't have to do models 4 times

def analyze_position_value(df, features, position_label):
    print(f"\n{'=' * 20} {position_label.upper()} ANALYSIS {'=' * 20}")

    # 1. Prepare Data & Log Transform
    X = df[features]
    y_log = np.log1p(df['Market Value'])

    # Split data
    X_train, X_test, y_log_train, y_log_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_all_scaled = scaler.transform(X)


    # 2. Train Model
    model = Ridge(alpha=12.0)
    model.fit(X_train_scaled, y_log_train)

    # 3. Metrics (on Log Scale)
    train_r2 = model.score(X_train_scaled, y_log_train)
    test_r2 = model.score(X_test_scaled, y_log_test)

    # Get errors
    test_preds_log = model.predict(X_test_scaled)
    test_preds_real = np.expm1(test_preds_log)
    test_actual_real = np.expm1(y_log_test)

    mae = mean_absolute_error(test_actual_real, test_preds_real)
    mse = mean_squared_error(test_actual_real, test_preds_real)

    # 4. Predictions & Reverse Log (Back to £M)
    all_preds_log = model.predict(X_all_scaled)
    df.loc[:, 'Predicted_Value'] = np.expm1(all_preds_log)
    df.loc[:, 'Difference'] = df['Predicted_Value'] - df['Market Value']


    # 6. Print the "Report Card"
    print(f"Train R2 Score: {train_r2:.4f}")
    print(f"Test R2 Score:  {test_r2:.4f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Avg Error (MAE): £{mae:.2f}M")

    # 7. Identify Outliers
    top_undervalued = df.sort_values(by='Difference', ascending=False).head(5)

    print(f"\nTop 5 {position_label} 'Bargains' (Should be worth MORE):")
    print(top_undervalued[['player_name', 'Market Value', 'Predicted_Value', 'Difference']].round(2))

    return df

# Run the model on each position Group
# Start with midfielders
mid_features = [ 'Team_Goals_Conceded', 'Minutes Played',   'XA', 'dribble_attempts', 'Age',
                 'Blocks', 'pass_accuracy']
mid_model = analyze_position_value(mids_df, mid_features, "Midfielder")

# Defenders
# First remove defenders with Nan values
defenders_clean = def_df.dropna(subset=['Team_Quality_Score', 'Team_Total_Passes']).copy()
def_features = ['Age', 'pass_accuracy',   'pass_attempts', 'Interceptions', 'Blocks', 'Team_Total_Passes',
                'Team_Quality_Score']
def_model = analyze_position_value(defenders_clean, def_features, "Defender")

# Forwards
forwards_features = [ 'XG_per_90', 'XA_per_90',  'Assists_per_90', 'Goals_per_90', 'Team_Quality_Score',
                      'Touches in the Opposition Box_per_90', 'dribble_attempts', 'is_striker', 'Age']
forward_model = analyze_position_value(forwards_df, forwards_features, "Forward")



# 11 FINAL PART Put it all together
# Create Graph that highlights best players to purchase

# Combine all position groups into one
all_results = pd.concat([mid_model, def_model, forward_model])

# Clean up data to only variables we care about
final_data = all_results[['player_name', 'Position', 'Market Value', 'Predicted_Value', 'Difference']].copy()
final_data = final_data[final_data['Market Value'] <= 80]

# Create graph
fig_log = px.scatter(
    final_data,
    x="Market Value",
    y="Predicted_Value",
    color="Position",
    hover_name="player_name",
    log_x=True,
    log_y=True,
    hover_data={
        "Market Value": ":.2f",
        "Predicted_Value": ":.2f",
        "Difference": ":.2f"
    },
    title="League-Wide Scouting Map (Logarithmic Scale)",
    labels={
        "Market Value": "Actual Market Value (£M) [Log Scale]",
        "Predicted_Value": "Model Prediction (£M) [Log Scale]"
    },
    template="plotly_dark" # Using a dark theme can make the dots pop more
)



# Add the 'Perfect Prediction' line
# On a log-log plot, the y=x line is still a perfect 45-degree diagonal
max_val = final_data['Market Value'].max()
min_val = final_data['Market Value'].min()

# To account for Error we will create a buffer along the perfect value line
# We will consider players that fall in this buffer fairly valued
# Lower Boundary
# 1. Add the UPPER buffer line (+2M)
fig_log.add_shape(
    type="line", line=dict(color="pink", width=1, dash="dot"),
    x0=min_val, x1=max_val,
    y0=min_val + 0.2 , y1=max_val + 0.2
)

# 2. Add the LOWER buffer line (-2M)
fig_log.add_shape(
    type="line", line=dict(color="pink", width=1, dash="dot"),
    x0=min_val, x1=max_val ,
    y0=min_val - 0.2, y1=max_val - 0.2
)

fig_log.add_shape(
    type="line", line=dict(dash="dash", color="yellow", width=2),
    x0=min_val, x1=max_val, y0=min_val, y1=max_val
)


fig_log.show()
fig_log.write_html("index.html")

# Sort by the highest positive difference (Model says worth MORE than market)
top_10_bargains = final_data.sort_values(by='Difference', ascending=False).head(10)
top_10_bargains = top_10_bargains.round(2)
print("\n--- TOP 10 MARKET BARGAINS (The Scouting List) ---")
print(top_10_bargains[['player_name', 'Position', 'Market Value', 'Predicted_Value', 'Difference']]
      .to_string(index=False))