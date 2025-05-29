import mesa
import networkx as nx
import pandas as pd
from datetime import datetime, timedelta
from mesa.space import NetworkGrid
from agent import UserAgent, CountermeasureAgent

class MisinformationModel(mesa.Model):
    
    def __init__(self, preprocessed_data, countermeasure_settings=None):
        super().__init__()
        self.running = True
        
        # data initialization
        self.preprocessed_data = preprocessed_data
        self.users_data = preprocessed_data['users']
        self.tweets_timeline = self._ensure_dataframe(preprocessed_data['tweets_timeline'])
        self.earliest_time = preprocessed_data['earliest_time']
        self.network_edges = preprocessed_data.get('network_edges', [])
        self.countermeasure_settings = countermeasure_settings or {}
        
        # Time-related parameters
        self.time_unit = 1000  # Each time step represents seconds
        self.current_real_time = self.earliest_time
        self.step_count = 0
        
        
        self.custom_agents = []  
        self.user_agents = {} 
        self.countermeasure_agents = []
        self.misinfo_stats = {} 
        self.scheduled_tweets = []  
        
        # Track users who have retracted
        self.retracted_users = {}  # {misinfo_id: set(user_ids)}
        
        # Network and retweet tracking
        self.retweet_graph = nx.DiGraph()
        self.historical_retweets = self._process_historical_retweets()
        
        
        self._setup_network()
        self._create_agents()
        self._schedule_tweets()
        self._setup_datacollector()
        
        # Collect initial data
        self.datacollector.collect(self)
        
        print(f"Model initialization complete: {len(self.user_agents)} user agents, {len(self.countermeasure_agents)} countermeasure agents")

    def _ensure_dataframe(self, data):

        return pd.DataFrame(data) if isinstance(data, list) else data

    def _process_historical_retweets(self):
        """Process historical retweet relationships, organized by time step"""
        historical_retweets = {}
        historical_mapping = self.preprocessed_data.get('historical_retweets', {})
        
        for misinfo_id, retweet_list in historical_mapping.items():
            for retweet_info in retweet_list:
                # Compute time step
                if retweet_info.get('time'):
                    time_since_start = (retweet_info['time'] - self.earliest_time).total_seconds()
                    time_step = int(time_since_start / self.time_unit)
                else:
                    time_step = 1
                
                if time_step not in historical_retweets:
                    historical_retweets[time_step] = []
                
                historical_retweets[time_step].append({
                    'user_id': retweet_info['user_id'],
                    'misinfo_id': misinfo_id,
                    'source_id': retweet_info['source_id'],
                    'tweet_id': retweet_info.get('tweet_id', f'historical_{retweet_info["user_id"]}_{misinfo_id}')
                })
        
        print(f"Historical retweet processing complete, covering {len(historical_retweets)} time steps")
        return historical_retweets

    def add_agent(self, agent):
        """Unified agent addition method"""
        if agent not in self.custom_agents:
            self.custom_agents.append(agent)

    def remove_agent(self, agent):
        """Unified agent removal method"""
        if agent in self.custom_agents:
            self.custom_agents.remove(agent)

    def mark_user_retracted(self, user_id, misinfo_id):
        """ Mark that a user has retracted a specific misinformation"""
        if misinfo_id not in self.retracted_users:
            self.retracted_users[misinfo_id] = set()
        self.retracted_users[misinfo_id].add(user_id)
        print(f"User {user_id} marked as retracted misinformation {misinfo_id}")

    def _setup_network(self):
        self.G = nx.DiGraph()
        
        # Add nodes
        for user_data in self.users_data:
            user_id = user_data.get('user_id')
            if user_id:
                self.G.add_node(user_id)

        # Add edges
        if self.network_edges:
            edge_count = 0
            for follower, followed in self.network_edges:
                if follower in self.G.nodes and followed in self.G.nodes:
                    self.G.add_edge(followed, follower)
                    edge_count += 1
            print(f"Added {edge_count} network edges from data")
        else:
            self._create_synthetic_network()

        self.grid = NetworkGrid(self.G)

    def _create_synthetic_network(self):
        """Create synthetic network (when no real network data is available)"""
        edge_count = 0
        for source_user in self.users_data:
            source_id = source_user.get('user_id')
            if not source_id:
                continue
                
            influence = source_user.get('followers_count', 0)
            if influence > 0:
                follow_prob = min(0.1, influence / 1000000)
                for target_user in self.users_data:
                    target_id = target_user.get('user_id')
                    if (target_id and target_id != source_id and 
                        self.random.random() < follow_prob):
                        self.G.add_edge(source_id, target_id)
                        edge_count += 1
        print(f"Created synthetic network with {edge_count} edges")

    def _create_agents(self):
        """Create all agents"""
        # Create user agents
        for user_data in self.users_data:
            user_id = user_data.get('user_id')
            if not user_id:
                continue

            # Ensure necessary fields exist
            user_data.setdefault('followers_count', 0)
            
            agent = UserAgent(user_id, self, user_data)
            self.user_agents[user_id] = agent
            self.grid.place_agent(agent, user_id)

        # Create countermeasure agents
        self._create_countermeasure_agents()
        
        print(f"Created {len(self.user_agents)} user agents")

    def _create_countermeasure_agents(self):
        """Create countermeasure agents based on settings"""
        countermeasure_configs = {
            'key_node': {
                'enabled_key': 'key_node_enabled',
                'settings': {
                    'activation_threshold': 'key_node_threshold',
                    'target_threshold': 'key_node_followers'
                }
            },
            'fact_check': {
                'enabled_key': 'fact_check_enabled',
                'settings': {
                    'activation_threshold': 'fact_check_threshold',
                    'delay': 'fact_check_delay'
                }
            },
            'early_warning': {
                'enabled_key': 'early_warning_enabled',
                'settings': {
                    'activation_threshold': 'early_warning_threshold',
                    'coverage_ratio': 'early_warning_coverage'
                }
            }
        }
        
        for cm_type, config in countermeasure_configs.items():
            if self.countermeasure_settings.get(config['enabled_key']):
                # Extract settings
                settings = {}
                for param_name, setting_key in config['settings'].items():
                    if setting_key in self.countermeasure_settings:
                        settings[param_name] = self.countermeasure_settings[setting_key]
                
                # Create agent
                cm_agent = CountermeasureAgent(f"cm_{cm_type}", self, cm_type, settings)
                self.countermeasure_agents.append(cm_agent)

    def _schedule_tweets(self):
        """Schedule tweet publishing times"""
        if 'time_since_start' not in self.tweets_timeline.columns:
            print("Warning: No time data available, cannot schedule tweets")
            return

        # Only schedule source tweets
        source_tweets = self.tweets_timeline[self.tweets_timeline['is_source'] == True]
        for _, tweet in source_tweets.iterrows():
            time_step = int(tweet['time_since_start'] / self.time_unit)
            self.scheduled_tweets.append((time_step, tweet.to_dict()))

        print(f"Scheduled {len(source_tweets)} source tweets")

    def _setup_datacollector(self):
        """Setup data collector"""
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Active_Users": lambda m: len([u for u in m.user_agents.values() 
                                             if u.is_active()]),
                "Total_Retweets": self.get_total_retweets,
                "Misinfo_Spread": lambda m: {mid: self.get_misinfo_spread(mid) 
                                           for mid in m.misinfo_stats.keys()},
                "Countermeasure_Coverage": lambda m: {mid: self.get_countermeasure_coverage(mid) 
                                                    for mid in m.misinfo_stats.keys()},
                "Retracted_Users": lambda m: {mid: len(users) for mid, users in m.retracted_users.items()},
            }
        )

    def get_total_retweets(self):
        """Get total number of retweets"""
        return sum(stats['total_spread'] for stats in self.misinfo_stats.values())

    def get_misinfo_spread(self, misinfo_id):
        """Get spread count for specific misinformation"""
        return len(self.misinfo_stats.get(misinfo_id, {}).get('active_users', set()))

    def get_countermeasure_coverage(self, misinfo_id):
        """Get countermeasure coverage count for specific misinformation"""
        return len(self.misinfo_stats.get(misinfo_id, {}).get('countermeasure_coverage', set()))

    def user_retweet(self, user_agent, tweet_data):
        """Handle user retweet event (improved version)"""
        misinfo_id = tweet_data['misinfo_id']
        source_id = tweet_data.get('source_id', 'unknown')
        retweeter_id = user_agent.unique_id
        
        # Improvement: Check if propagation source has retracted
        if (misinfo_id in self.retracted_users and 
            source_id in self.retracted_users[misinfo_id]):
            print(f"🚫 Blocked propagation: User {retweeter_id} cannot get misinformation {misinfo_id} from retracted user {source_id}")
            return  # Block propagation!
        
        # Record retweet relationship
        self.retweet_graph.add_edge(source_id, retweeter_id, 
                                   time_step=self.step_count,
                                   misinfo_id=misinfo_id)
        
        # Update statistics
        if misinfo_id not in self.misinfo_stats:
            self.misinfo_stats[misinfo_id] = {
                'total_spread': 0,
                'active_users': set(),
                'countermeasure_coverage': set()
            }
        
        self.misinfo_stats[misinfo_id]['total_spread'] += 1
        self.misinfo_stats[misinfo_id]['active_users'].add(retweeter_id)
        
        # Propagate to followers (but check if retracted)
        tweet_data_copy = tweet_data.copy()
        tweet_data_copy['source_id'] = retweeter_id
        
        neighbors = list(self.grid.get_neighbors(user_agent.pos))
        successful_propagations = 0
        blocked_propagations = 0
        
        for follower_id in neighbors:
            follower = self.user_agents.get(follower_id)
            if follower:
                # Check if current user has retracted
                if (misinfo_id in self.retracted_users and 
                    retweeter_id in self.retracted_users[misinfo_id]):
                    print(f"🚫 Blocked propagation: User {retweeter_id} has retracted, not propagating to {follower_id}")
                    blocked_propagations += 1
                    continue
                
                follower.receive_tweet(tweet_data_copy)
                successful_propagations += 1
        
        if blocked_propagations > 0:
            print(f"📊 User {retweeter_id}: Successfully propagated {successful_propagations}, blocked {blocked_propagations}")
        
        # Notify countermeasure agents
        spread_count = self.misinfo_stats[misinfo_id]['total_spread']
        for cm_agent in self.countermeasure_agents:
            cm_agent.monitor_misinfo(misinfo_id, spread_count)

    def deploy_countermeasure(self, cm_agent, misinfo_id):
        """Deploy countermeasure"""
        countermeasure_tweet = {
            'misinfo_id': misinfo_id,
            'is_countermeasure': True,
            'countermeasure_type': cm_agent.type
        }
        
        target_users = self._get_countermeasure_targets(cm_agent, misinfo_id)
        
        deployment_count = 0
        for user_id in target_users:
            user_agent = self.user_agents.get(user_id)
            if user_agent:
                user_agent.receive_tweet(countermeasure_tweet, is_countermeasure=True)
                if misinfo_id in self.misinfo_stats:
                    self.misinfo_stats[misinfo_id]['countermeasure_coverage'].add(user_id)
                deployment_count += 1
        
        print(f"🛡️  {cm_agent.type} countermeasure deployed to {deployment_count} users")

    def _get_countermeasure_targets(self, cm_agent, misinfo_id):
        """Get target users for countermeasures"""
        if cm_agent.type == 'key_node':
            # Target high-influence users
            return [user_id for user_id, user_agent in self.user_agents.items()
                    if user_agent.followers_count >= cm_agent.target_threshold]
        
        elif cm_agent.type == 'fact_check':
            # Target active spreading users
            if misinfo_id in self.misinfo_stats:
                return list(self.misinfo_stats[misinfo_id]['active_users'])
            return []
        
        elif cm_agent.type == 'early_warning':
            # Random user sample
            all_users = list(self.user_agents.keys())
            target_count = int(len(all_users) * cm_agent.coverage_ratio)
            return self.random.sample(all_users, min(target_count, len(all_users)))
        
        return []

    def record_retraction(self, user_agent, misinfo_id):
        """Record user retraction of retweet"""
        if misinfo_id in self.misinfo_stats:
            self.misinfo_stats[misinfo_id]['active_users'].discard(user_agent.unique_id)
            self.misinfo_stats[misinfo_id]['total_spread'] = max(
                0, self.misinfo_stats[misinfo_id]['total_spread'] - 1)

    def step(self):
        """Execute one time step"""
        self.step_count += 1
        
        # Process scheduled tweets
        self._process_scheduled_tweets()
        
        # Process historical retweets
        self._process_historical_retweets_step()
        
        # Update time
        if isinstance(self.current_real_time, datetime):
            self.current_real_time += timedelta(seconds=self.time_unit)
        else:
            self.current_real_time += self.time_unit
        
        # Execute all agents' steps
        for agent in list(self.custom_agents):  # Create copy to avoid modification during iteration
            agent.step()
        
        # Collect data
        self.datacollector.collect(self)

    def _process_scheduled_tweets(self):
        """Process scheduled tweets for current time step"""
        current_tweets = [tweet for step, tweet in self.scheduled_tweets 
                         if step == self.step_count]
        
        for tweet in current_tweets:
            misinfo_id = tweet.get('misinfo_id')
            source_user_id = tweet.get('user_id')
            
            if source_user_id and source_user_id in self.user_agents:
                source_user = self.user_agents[source_user_id]
                source_user.received_misinfo.add(misinfo_id)
                
                # Propagate to followers
                neighbors = list(self.grid.get_neighbors(source_user.pos))
                for follower_id in neighbors:
                    follower = self.user_agents.get(follower_id)
                    if follower:
                        tweet_with_source = tweet.copy()
                        tweet_with_source['source_id'] = source_user_id
                        follower.receive_tweet(tweet_with_source)

    def _process_historical_retweets_step(self):
        """Process historical retweets for current time step (improved version)"""
        if self.step_count not in self.historical_retweets:
            return
            
        total_attempts = 0
        blocked_attempts = 0
        successful_retweets = 0
        
        for retweet_data in list(self.historical_retweets[self.step_count]):
            user_id = retweet_data['user_id']
            misinfo_id = retweet_data['misinfo_id']
            source_id = retweet_data['source_id']
            total_attempts += 1
            
            if user_id not in self.user_agents:
                continue
            
            # Improvement: Check if historical retweet source has retracted
            if (misinfo_id in self.retracted_users and 
                source_id in self.retracted_users[misinfo_id]):
                print(f"🚫 Blocked historical retweet: {user_id} cannot get information from retracted user {source_id}")
                blocked_attempts += 1
                continue
                
            user_agent = self.user_agents[user_id]
            
            # Ensure user received the information
            if misinfo_id not in user_agent.received_misinfo:
                user_agent.received_misinfo.add(misinfo_id)
            
            # Decide whether to retweet (considering countermeasure effects)
            should_retweet = True
            if misinfo_id in user_agent.received_countermeasures:
                rt_prob = user_agent.retweet_probability
                rt_prob *= (1 - user_agent.susceptibility_to_countermeasures)
                should_retweet = user_agent._random_generator.random() < rt_prob
            
            if should_retweet and misinfo_id not in user_agent.retweeted_misinfo:
                user_agent.retweeted_misinfo.add(misinfo_id)
                tweet_data = {
                    'misinfo_id': misinfo_id,
                    'source_id': source_id,
                    'tweet_id': retweet_data.get('tweet_id')
                }
                self.user_retweet(user_agent, tweet_data)
                successful_retweets += 1
        
        if total_attempts > 0:
            print(f"📊 Historical retweet statistics: {total_attempts} attempts, {successful_retweets} successful, {blocked_attempts} blocked")

    def get_retraction_stats(self):
        """Get retraction statistics"""
        stats = {}
        for misinfo_id, retracted_users in self.retracted_users.items():
            high_influence_retractions = 0
            total_retractions = len(retracted_users)
            
            for user_id in retracted_users:
                if user_id in self.user_agents:
                    user_agent = self.user_agents[user_id]
                    if user_agent.followers_count >= 1000:  # High-influence users
                        high_influence_retractions += 1
            
            stats[misinfo_id] = {
                'total_retractions': total_retractions,
                'high_influence_retractions': high_influence_retractions,
                'retraction_ratio': high_influence_retractions / max(total_retractions, 1)
            }
        
        return stats

    def print_simulation_summary(self):
        """Print simulation summary"""
        print("\n" + "="*50)
        print("🎯 Simulation Summary")
        print("="*50)
        
        # Basic statistics
        total_users = len(self.user_agents)
        active_users = len([u for u in self.user_agents.values() if u.is_active()])
        total_retweets = self.get_total_retweets()
        
        print(f"📊 Basic Statistics:")
        print(f"   Total users: {total_users}")
        print(f"   Active users: {active_users} ({active_users/total_users*100:.1f}%)")
        print(f"   Total retweets: {total_retweets}")
        
        # Retraction statistics
        retraction_stats = self.get_retraction_stats()
        if retraction_stats:
            print(f"\n🔄 Retraction Statistics:")
            for misinfo_id, stats in retraction_stats.items():
                print(f"   Misinformation {misinfo_id}:")
                print(f"     Total retractions: {stats['total_retractions']}")
                print(f"     High-influence retractions: {stats['high_influence_retractions']}")
                print(f"     High-influence ratio: {stats['retraction_ratio']*100:.1f}%")
        
        # Countermeasure statistics
        print(f"\n🛡️  Countermeasure Statistics:")
        for cm_agent in self.countermeasure_agents:
            active_count = len(cm_agent.active_countermeasures)
            pending_count = len(cm_agent.pending_countermeasures)
            print(f"   {cm_agent.type}: Activated {active_count} times, Pending {pending_count} times")
        
        print("="*50)