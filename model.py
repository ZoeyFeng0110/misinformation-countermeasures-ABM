import pandas as pd
import networkx as nx
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import NetworkGrid
from agent import UserAgent, CountermeasureAgent

class MisinformationModel(Model):
    def __init__(self, preprocessed_data, countermeasure_settings=None):
        super().__init__()
        self.preprocessed_data = preprocessed_data
        self.users_data = preprocessed_data['users']
        self.tweets_timeline = preprocessed_data['tweets_timeline']
        if isinstance(self.tweets_timeline, list):
            self.tweets_timeline = pd.DataFrame(self.tweets_timeline)
        self.earliest_time = preprocessed_data['earliest_time']
        self.network_edges = preprocessed_data.get('network_edges', [])
        self.countermeasure_settings = countermeasure_settings or {}
        self.time_unit = 60  # 1 step = 60 seconds
        self.current_real_time = self.earliest_time

        self.misinfo_stats = {}  # Used to store diffusion stats per misinformation
        self.scheduled_tweets = []  # Tweets to be dispatched over time

        # Set up data collection metrics
        self.datacollector = DataCollector(
            model_reporters={
                "Active_Users": lambda m: len([u for u in m.agents if isinstance(u, UserAgent) and len(u.retweeted_misinfo) > 0]),
                "Total_Retweets": self.get_total_retweets,
                "Misinfo_Spread": lambda m: {mid: self.get_misinfo_spread(mid) for mid in self.misinfo_stats},
                "Countermeasure_Coverage": lambda m: {mid: self.get_countermeasure_coverage(mid) for mid in self.misinfo_stats},
            }
        )

        self._setup_network()
        self._create_agents()
        self._schedule_tweets()
        self.datacollector.collect(self)

    def _setup_network(self):
        # Build graph and add user nodes
        self.G = nx.DiGraph()
        for user_data in self.users_data:
            user_id = user_data.get('user_id')
            if user_id:
                self.G.add_node(user_id)

        # Add edges (follower -> followed)
        if self.network_edges:
            for follower, followed in self.network_edges:
                if follower in self.G and followed in self.G:
                    self.G.add_edge(followed, follower)
        else:
            self._create_synthetic_network()

        # Wrap graph into Mesa grid
        self.grid = NetworkGrid(self.G)

    def _create_synthetic_network(self):
        # Create synthetic links based on influence
        for source_user in self.users_data:
            source_id = source_user.get('user_id')
            if not source_id:
                continue
            influence = int(source_user.get('followers_count', 0))
            if influence > 1000:
                follow_prob = min(0.1, influence / 1000000)
                for target_user in self.users_data:
                    target_id = target_user.get('user_id')
                    if target_id and target_id != source_id and self.random.random() < follow_prob:
                        self.G.add_edge(source_id, target_id)

    def _create_agents(self):
        # Instantiate and place user agents
        for user_data in self.users_data:
            user_id = user_data.get('user_id')
            if not user_id:
                continue
            agent = UserAgent(user_id, self, user_data)
            self.add(agent)
            self.grid.place_agent(agent, user_id)

        # Add countermeasure agents based on settings
        if self.countermeasure_settings.get('key_node_enabled'):
            cm_agent = CountermeasureAgent("cm_key_node", self, 'key_node', {
                'activation_threshold': self.countermeasure_settings.get('key_node_threshold', 0.1),
                'target_threshold': self.countermeasure_settings.get('key_node_followers', 100000)
            })
            self.add(cm_agent)

        if self.countermeasure_settings.get('fact_check_enabled'):
            cm_agent = CountermeasureAgent("cm_fact_check", self, 'fact_check', {
                'activation_threshold': self.countermeasure_settings.get('fact_check_threshold', 50),
                'delay': self.countermeasure_settings.get('fact_check_delay', 30)
            })
            self.add_agent(cm_agent)

        if self.countermeasure_settings.get('early_warning_enabled'):
            cm_agent = CountermeasureAgent("cm_early_warning", self, 'early_warning', {
                'activation_threshold': self.countermeasure_settings.get('early_warning_threshold', 10),
                'coverage_ratio': self.countermeasure_settings.get('early_warning_coverage', 0.3)
            })
            self.add_agent(cm_agent)

    def _schedule_tweets(self):
        # Organize tweets into step-wise schedule
        if 'time_since_start' not in self.tweets_timeline.columns:
            print("Warning: Missing timestamp data.")
            return
        sorted_tweets = self.tweets_timeline.sort_values('time_since_start').to_dict('records')
        for tweet in sorted_tweets:
            time_step = int(tweet['time_since_start'] / self.time_unit)
            self.scheduled_tweets.append((time_step, tweet))

    def step(self):
        # Execute one simulation tick
        current_step = self.num_steps
        for step, tweet in self.scheduled_tweets[:]:
            if step <= current_step:
                self.process_tweet(tweet)
                self.scheduled_tweets.remove((step, tweet))
        for agent in self.agents:
            agent.step()
        self.datacollector.collect(self)

    def process_tweet(self, tweet_data):
        pass  # to be implemented

    def get_total_retweets(self):
        return sum(stats['retweet_count'] for stats in self.misinfo_stats.values())

    def get_misinfo_spread(self, misinfo_id):
        if misinfo_id in self.misinfo_stats:
            return len(self.misinfo_stats[misinfo_id]['affected_users'])
        return 0

    def get_countermeasure_coverage(self, misinfo_id):
        if misinfo_id in self.misinfo_stats:
            return len(self.misinfo_stats[misinfo_id]['countermeasure_received_users'])
        return 0
