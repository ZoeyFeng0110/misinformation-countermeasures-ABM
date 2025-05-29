import os
import json
import pandas as pd
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt

def parse_tweet_time(time_str):
    """Parse Twitter time format"""
    try:
        return datetime.strptime(time_str, '%a %b %d %H:%M:%S +0000 %Y')
    except ValueError:
        print(f"Unable to parse time format: {time_str}")
        return datetime.now()

def load_network_data(network_file_path):
    """Load who-follows-whom network data"""
    network_edges = []
    try:
        with open(network_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split(',')
                    if len(parts) >= 2:
                        follower = parts[0].strip()
                        followed = parts[1].strip()
                        network_edges.append((follower, followed))
    except Exception as e:
        print(f"Error loading network data: {e}")
    return network_edges

def load_event_data(event_folder_path):
    """Load event folder data"""
    if not os.path.exists(event_folder_path):
        raise FileNotFoundError(f"Event folder does not exist: {event_folder_path}")
    
    event_data = {}
    
    for misinfo_id in os.listdir(event_folder_path):
        misinfo_path = os.path.join(event_folder_path, misinfo_id)
        
        if not os.path.isdir(misinfo_path) or misinfo_id.startswith('.'):
            continue
        
        # Define file paths
        source_tweets_folder = os.path.join(misinfo_path, 'source-tweets')
        retweets_file = os.path.join(misinfo_path, 'retweets.json')
        network_file = os.path.join(misinfo_path, 'who-follows-whom.dat')
        
        # Load source tweet
        source_tweet = load_source_tweet(source_tweets_folder)
        
        # Load retweets
        retweets = load_retweets(retweets_file)
        
        # Load network data (if exists)
        network_data = None
        if os.path.exists(network_file):
            network_data = load_network_data(network_file)
        
        # Process time format
        if source_tweet:
            process_tweet_time(source_tweet)
            for retweet in retweets:
                process_tweet_time(retweet)
            
            event_data[misinfo_id] = {
                'source_tweet': source_tweet,
                'retweets': retweets,
                'network': network_data
            }
    
    return event_data

def load_source_tweet(source_tweets_folder):
    """Load source tweet"""
    if not os.path.exists(source_tweets_folder):
        return None
    
    for file_name in os.listdir(source_tweets_folder):
        if file_name.endswith('.json'):
            source_file_path = os.path.join(source_tweets_folder, file_name)
            with open(source_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    return None

def load_retweets(retweets_file, max_retweets=100):
    """Load retweet data"""
    retweets = []
    if not os.path.exists(retweets_file):
        return retweets
    
    with open(retweets_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    tweet = json.loads(line)
                    retweets.append(tweet)
                except json.JSONDecodeError as e:
                    print(f"Warning: Unable to parse JSON line: {line[:50]}... Error: {e}")
    
    # Limit retweet count
    if len(retweets) > max_retweets:
        retweets = retweets[:max_retweets]
    
    return retweets

def process_tweet_time(tweet):
    """Process tweet time"""
    if 'created_at' in tweet:
        tweet['created_at_dt'] = parse_tweet_time(tweet['created_at'])

def extract_user_data(tweet, user_source='user'):
    """Extract user data from tweet"""
    user_data = None
    user_id = None
    
    # Case 1: Standard user field
    if user_source in tweet and isinstance(tweet[user_source], dict):
        user_dict = tweet[user_source]
        if 'user_id' in user_dict:
            user_data = user_dict
            user_id = user_dict['user_id']
        else:
            # Try other ID fields
            for id_field in ['id', 'id_str', 'userId']:
                if id_field in user_dict:
                    user_data = user_dict.copy()
                    user_id = str(user_dict[id_field])
                    user_data['user_id'] = user_id
                    break
    
    # Case 2: User data at root level
    if not user_data:
        root_level_fields = ['user_id', 'userId', 'id', 'id_str', 'screen_name', 
                           'name', 'verified', 'followers_count', 'location', 'description']
        if any(field in tweet for field in root_level_fields):
            user_data = {}
            for field in root_level_fields:
                if field in tweet:
                    user_data[field] = tweet[field]
            
            # Ensure user_id field exists
            for id_field in ['user_id', 'userId', 'id', 'id_str']:
                if id_field in tweet:
                    user_id = str(tweet[id_field])
                    user_data['user_id'] = user_id
                    break
    
    return user_data, user_id

def preprocess_event_data(event_data):
    """Preprocess event data for model preparation"""
    users_dict = {}
    all_tweets = []
    historical_retweets = {}
    
    print("Starting event data processing...")
    
    for misinfo_id, misinfo_data in event_data.items():
        historical_retweets[misinfo_id] = []
        
        # Process source tweet
        source_tweet = misinfo_data['source_tweet']
        source_user_data, source_user_id = extract_user_data(source_tweet)
        
        if source_user_data and source_user_id:
            users_dict[source_user_id] = source_user_data
        
        # Prepare source tweet data
        source_tweet_processed = prepare_tweet_for_timeline(source_tweet, misinfo_id, True)
        all_tweets.append(source_tweet_processed)
        
        # Process retweets
        for retweet in misinfo_data['retweets']:
            user_data, user_id = extract_user_data(retweet)
            
            if user_data and user_id:
                # Add user (if not exists)
                if user_id not in users_dict:
                    users_dict[user_id] = user_data
                
                # Prepare retweet data
                retweet_processed = prepare_tweet_for_timeline(retweet, misinfo_id, False)
                all_tweets.append(retweet_processed)
                
                # Record historical retweet relationships
                retweet_source_id = determine_retweet_source(retweet, source_user_id)
                retweet_time = retweet.get('created_at_dt')
                
                historical_retweets[misinfo_id].append({
                    'user_id': user_id,
                    'source_id': retweet_source_id,
                    'time': retweet_time,
                    'tweet_id': retweet.get('id_str', retweet.get('id', f'retweet_{user_id}_{misinfo_id}'))
                })
    
    # Create timeline DataFrame
    tweets_df = create_tweets_timeline(all_tweets)
    
    # Build network edges
    network_edges = []
    for misinfo_data in event_data.values():
        if misinfo_data.get('network'):
            network_edges.extend(misinfo_data['network'])
    
    # Output statistics
    print_preprocessing_stats(users_dict, tweets_df, historical_retweets, network_edges)
    
    return {
        'users': list(users_dict.values()),
        'tweets_timeline': tweets_df,
        'earliest_time': tweets_df['created_at_dt'].min() if len(tweets_df) > 0 else datetime.now(),
        'network_edges': network_edges,
        'historical_retweets': historical_retweets
    }

def prepare_tweet_for_timeline(tweet, misinfo_id, is_source):
    """Prepare tweet data for timeline"""
    processed_tweet = {
        'misinfo_id': misinfo_id,
        'is_source': is_source,
        'tweet_id': tweet.get('id_str', tweet.get('id', f'tweet_{misinfo_id}')),
        'created_at_dt': tweet.get('created_at_dt', datetime.now())
    }
    
    # Add user ID (if exists)
    if 'user' in tweet and isinstance(tweet['user'], dict):
        user_dict = tweet['user']
        for id_field in ['user_id', 'id', 'id_str']:
            if id_field in user_dict:
                processed_tweet['user_id'] = str(user_dict[id_field])
                break
    else:
        # Try to get from root level
        for id_field in ['user_id', 'id', 'id_str']:
            if id_field in tweet:
                processed_tweet['user_id'] = str(tweet[id_field])
                break
    
    return processed_tweet

def determine_retweet_source(retweet, default_source_id):
    """Determine retweet source"""
    # Check if it's a retweet of someone else's retweet
    if 'retweeted_status' in retweet:
        retweeted_status = retweet['retweeted_status']
        if 'user' in retweeted_status:
            original_user = retweeted_status['user']
            for id_field in ['user_id', 'id', 'id_str']:
                if id_field in original_user:
                    return str(original_user[id_field])
    
    return default_source_id

def create_tweets_timeline(all_tweets):
    """Create tweet timeline DataFrame"""
    if not all_tweets:
        return pd.DataFrame()
    
    tweets_df = pd.DataFrame(all_tweets)
    
    if 'created_at_dt' in tweets_df.columns:
        tweets_df = tweets_df.sort_values(by='created_at_dt')
        earliest_time = tweets_df['created_at_dt'].min()
        tweets_df['time_since_start'] = tweets_df['created_at_dt'].apply(
            lambda x: (x - earliest_time).total_seconds()
        )
    else:
        # If no time data, use index
        tweets_df['time_since_start'] = tweets_df.index * 60  # Assume one tweet per minute
    
    return tweets_df

def print_preprocessing_stats(users_dict, tweets_df, historical_retweets, network_edges):
    """Print preprocessing statistics"""
    total_users = len(users_dict)
    total_tweets = len(tweets_df)
    total_historical_retweets = sum(len(retweets) for retweets in historical_retweets.values())
    total_network_edges = len(network_edges)
    
    print(f"\nPreprocessing completed!")
    print(f"Number of users: {total_users}")
    print(f"Number of tweets: {total_tweets}")
    print(f"Historical retweet relationships: {total_historical_retweets}")
    print(f"Number of network edges: {total_network_edges}")
    
    if total_tweets > 0:
        source_tweets = len(tweets_df[tweets_df['is_source'] == True])
        retweets = total_tweets - source_tweets
        print(f"  - Source tweets: {source_tweets}")
        print(f"  - Retweets: {retweets}")

# Visualization functions
def visualize_retweet_network(tweets_df, max_nodes=100):
    """Visualize retweet network"""
    G = nx.DiGraph()
    
    # Only process first max_nodes retweets
    retweets = tweets_df[tweets_df['is_source'] == False].head(max_nodes)
    
    for _, row in retweets.iterrows():
        source = row.get('source_id', 'unknown')
        target = row.get('user_id', 'unknown')
        if source != 'unknown' and target != 'unknown':
            G.add_edge(source, target)

    if len(G.nodes()) == 0:
        print("Insufficient retweet data to create network graph")
        return

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=500, 
            node_color='skyblue', edge_color='gray', arrows=True)
    plt.title("Retweet Propagation Network")
    plt.tight_layout()

def plot_tweet_activity_over_time(tweets_df, interval_minutes=10):
    """Plot tweet activity time distribution"""
    if 'time_since_start' not in tweets_df.columns:
        print("No time data available, cannot plot time distribution")
        return
    
    df = tweets_df.copy()
    df['time_bin'] = (df['time_since_start'] // (interval_minutes * 60)).astype(int)
    counts = df.groupby('time_bin').size()

    plt.figure(figsize=(10, 4))
    counts.plot(kind='bar', color='orange')
    plt.xlabel(f'Time Interval (every {interval_minutes} minutes)')
    plt.ylabel('Number of Tweets')
    plt.title('Tweet Activity Time Distribution')
    plt.tight_layout()

def main():
    """Main function example"""
    event_folder_path = "/Users/oliviafeng/Desktop/uchi/agent_based_modeling/code/final_project/pheme-rumour-scheme-dataset/threads/en/charliehebdo"  # Please replace with actual path
    
    try:
        print(f"Loading event data: {event_folder_path}")
        event_data = load_event_data(event_folder_path)
        print(f"Successfully loaded {len(event_data)} misinformation events")
        
        print("Starting data preprocessing...")
        preprocessed_data = preprocess_event_data(event_data)
        
        print("Generating visualizations...")
        visualize_retweet_network(preprocessed_data['tweets_timeline'])
        plt.show()
        
        plot_tweet_activity_over_time(preprocessed_data['tweets_timeline'])
        plt.show()
        
        return preprocessed_data
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    preprocessed_data = main()