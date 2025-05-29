import numpy as np
from mesa import Agent

class BaseAgent(Agent):
    """Base agent class ensuring Mesa network compatibility"""
    def __init__(self, unique_id, model):
        self.unique_id = unique_id
        self.model = model
        self.pos = None
        self._random_generator = model.random
        
        # Use model's unified agent management (avoid Mesa 3.0+ agents attribute conflict)
        if hasattr(model, 'add_agent'):
            model.add_agent(self)

    def remove(self):
        """Remove agent from model"""
        if hasattr(self.model, 'remove_agent'):
            self.model.remove_agent(self)
        if hasattr(self.model, 'grid') and hasattr(self.model.grid, 'remove_agent'):
            self.model.grid.remove_agent(self)

class UserAgent(BaseAgent):
    """Social media user agent (improved version)"""
    def __init__(self, unique_id, model, user_data):
        super().__init__(unique_id, model)
        
        # Basic user information
        self.user_id = user_data.get('user_id', str(unique_id))
        self.screen_name = user_data.get('screen_name', '')
        self.name = user_data.get('name', '')
        self.verified = user_data.get('verified', False)
        self.followers_count = user_data.get('followers_count', 0)
        self.location = user_data.get('location', '')
        self.description = user_data.get('description', '')

        # Status tracking
        self.received_misinfo = set()
        self.retweeted_misinfo = set()
        self.received_countermeasures = set()
        self.retweet_sources = {}  # {misinfo_id: source_id}

        # Behavioral probabilities
        self.retweet_probability = self._calculate_retweet_probability()
        self.susceptibility_to_countermeasures = self._calculate_susceptibility()

    def _calculate_retweet_probability(self):
        """Calculate base probability of retweeting misinformation"""
        base_prob = 0.6
        if self.verified:
            base_prob *= 0.8
        if self.followers_count > 0:
            follower_factor = 1.0 - min(0.5, np.log(1 + self.followers_count) / 30)
            base_prob *= follower_factor
        return base_prob

    def _calculate_susceptibility(self):
        """Calculate susceptibility to countermeasures"""
        return 0.7 if self.verified else 0.3 + (0.4 * self._random_generator.random())

    def receive_tweet(self, tweet_data, is_countermeasure=False):
        """Process received tweet (improved version)"""
        misinfo_id = tweet_data['misinfo_id']
        
        if is_countermeasure:
            self.received_countermeasures.add(misinfo_id)
            # If already retweeted, consider retraction
            if misinfo_id in self.retweeted_misinfo:
                # Improvement: Significantly increase retraction probability for key node users
                base_retraction_prob = self.susceptibility_to_countermeasures
                
                # Get key node intervention target threshold from model
                key_node_threshold = None
                for cm_agent in self.model.countermeasure_agents:
                    if cm_agent.type == 'key_node':
                        key_node_threshold = cm_agent.target_threshold
                        break
                
                # If high-influence user (key node), higher retraction probability
                if key_node_threshold and self.followers_count >= key_node_threshold:
                    base_retraction_prob *= 3.0  # 3x retraction probability
                    print(f"High-influence user {self.user_id} (followers {self.followers_count}) retraction probability increased to {base_retraction_prob:.2f}")
                
                # Key node intervention has stronger effect
                if tweet_data.get('countermeasure_type') == 'key_node':
                    base_retraction_prob *= 2.0  # Multiply by 2x again
                
                # Limit maximum retraction probability to 95%
                final_retraction_prob = min(0.95, base_retraction_prob)
                
                if self._random_generator.random() < final_retraction_prob:
                    self.retweeted_misinfo.remove(misinfo_id)
                    self.model.record_retraction(self, misinfo_id)
                    self.retweet_sources.pop(misinfo_id, None)
                    
                    # Critical: Mark this user as retracted, blocking subsequent propagation
                    self.model.mark_user_retracted(self.user_id, misinfo_id)
                    print(f"High-influence user {self.user_id} retracted misinformation {misinfo_id}")
        else:
            self.received_misinfo.add(misinfo_id)
            # Record retweet source
            source_id = tweet_data.get('original_source') or tweet_data.get('source_id', 'unknown')
            self.retweet_sources[misinfo_id] = source_id
            self._consider_retweeting(tweet_data)

    def _consider_retweeting(self, tweet_data):
        """Decide whether to retweet misinformation"""
        misinfo_id = tweet_data['misinfo_id']
        if misinfo_id in self.retweeted_misinfo:
            return

        # Calculate retweet probability
        rt_prob = self.retweet_probability
        if misinfo_id in self.received_countermeasures:
            rt_prob *= (1 - self.susceptibility_to_countermeasures)

        if self._random_generator.random() < rt_prob:
            self.retweeted_misinfo.add(misinfo_id)
            self.model.user_retweet(self, tweet_data)

    def step(self):
        """User agent has no autonomous behavior"""
        pass

    def get_neighbors(self):
        """Get neighbors (followers)"""
        return list(self.model.grid.get_neighbors(self.pos))

    def is_active(self):
        """Determine if user is actively spreading misinformation"""
        return len(self.retweeted_misinfo) > 0

class CountermeasureAgent(BaseAgent):
    """Platform-level countermeasure agent (improved version)"""
    
    # Unified default configuration
    DEFAULT_SETTINGS = {
        'key_node': {
            'activation_threshold': 0.05,   # Improvement: Lower to 5% for more sensitivity
            'target_threshold': 1111       # Improvement: Increase to 10k followers for more precision
        },
        'fact_check': {
            'activation_threshold': 50,
            'delay': 30
        },
        'early_warning': {
            'activation_threshold': 10,
            'coverage_ratio': 0.3
        }
    }

    def __init__(self, unique_id, model, countermeasure_type, settings=None):
        super().__init__(unique_id, model)
        self.type = countermeasure_type
        
        # Merge default settings with passed settings
        default_settings = self.DEFAULT_SETTINGS.get(countermeasure_type, {})
        final_settings = {**default_settings, **(settings or {})}
        
        # Set parameters
        for key, value in final_settings.items():
            setattr(self, key, value)

        # Status tracking
        self.active_countermeasures = {}   # {misinfo_id: step when active}
        self.pending_countermeasures = {}  # {misinfo_id: step to activate}

    def activate_countermeasure(self, misinfo_id, current_step):
        """Activate countermeasure"""
        if self.type == 'fact_check':
            # Fact checking has delay
            self.pending_countermeasures[misinfo_id] = current_step + self.delay
        else:
            # Other types activate immediately
            self.active_countermeasures[misinfo_id] = current_step
            self.model.deploy_countermeasure(self, misinfo_id)

    def step(self):
        """Check pending countermeasures"""
        current_step = self.model.step_count
        to_activate = []
        
        for misinfo_id, activation_step in self.pending_countermeasures.items():
            if current_step >= activation_step:
                to_activate.append(misinfo_id)
                self.active_countermeasures[misinfo_id] = current_step
        
        for misinfo_id in to_activate:
            del self.pending_countermeasures[misinfo_id]
            self.model.deploy_countermeasure(self, misinfo_id)

    def monitor_misinfo(self, misinfo_id, spread_count):
        """Monitor misinformation spread (improved version - with complete debugging)"""
        if misinfo_id in self.active_countermeasures or misinfo_id in self.pending_countermeasures:
            return
        
        current_step = self.model.step_count
        should_activate = False
        
        # Add basic debugging information
        print(f"🔍 [Step {current_step}] Monitoring {self.type} countermeasure - Misinformation {misinfo_id}, spread count {spread_count}")
        
        if self.type == 'key_node':
            # Basic statistics
            user_agents = list(self.model.user_agents.values())
            total_users = len(user_agents)
            print(f"📊 Total user agents: {total_users}")
            
            # Followers distribution statistics
            if total_users > 0:
                followers_list = [agent.followers_count for agent in user_agents]
                max_followers = max(followers_list)
                avg_followers = sum(followers_list) / len(followers_list)
                print(f"📈 Followers distribution: Max {max_followers}, Average {avg_followers:.1f}")
            
            # High-influence users statistics
            high_influence_users = [agent for agent in user_agents
                                if agent.followers_count >= self.target_threshold]
            total_influential = len(high_influence_users)
            print(f"👑 High-influence users: {total_influential} users (followers >= {self.target_threshold})")
            
            if total_influential == 0:
                print(f"❌ No qualified high-influence users! Cannot trigger key node intervention")
                print(f"   Suggestion: Lower target_threshold from {self.target_threshold} to a lower value")
                return
            
            # Show high-influence user details (top 5)
            if total_influential > 0:
                top_users = sorted(high_influence_users, key=lambda x: x.followers_count, reverse=True)[:5]
                user_info = [(u.user_id, u.followers_count) for u in top_users]
                print(f"   Top 5: {user_info}")
            
            # Detailed retweet status analysis
            influential_spreaders = [agent for agent in high_influence_users
                                if misinfo_id in agent.retweeted_misinfo]
            received_count = sum(1 for agent in high_influence_users 
                            if misinfo_id in agent.received_misinfo)
            
            print(f"📢 High-influence user status:")
            print(f"   Received misinformation: {received_count}/{total_influential}")
            print(f"   Retweeted misinformation: {len(influential_spreaders)}/{total_influential}")
            
            if len(influential_spreaders) > 0:
                spreader_details = [(agent.user_id, agent.followers_count) for agent in influential_spreaders]
                print(f"   Spreader details: {spreader_details}")
            
            # Calculate activation condition
            if total_influential > 0:
                spreader_ratio = len(influential_spreaders) / total_influential
                print(f"📊 Retweet ratio: {spreader_ratio:.4f} ({spreader_ratio*100:.2f}%)")
                print(f"🎯 Activation threshold: {self.activation_threshold:.4f} ({self.activation_threshold*100:.2f}%)")
                
                should_activate = spreader_ratio >= self.activation_threshold
                
                if should_activate:
                    print(f"🚨 Key node intervention triggered!")
                    print(f"   Condition: {len(influential_spreaders)}/{total_influential} = {spreader_ratio:.2%} >= {self.activation_threshold:.2%}")
                    print(f"   High-influence spreaders: {[agent.user_id for agent in influential_spreaders]}")
                else:
                    print(f"❌ Activation threshold not met")
                    needed_spreaders = int(total_influential * self.activation_threshold) + 1
                    print(f"   Need at least {needed_spreaders} high-influence users to retweet to trigger")
            
        elif self.type == 'fact_check':
            print(f"📋 Fact check monitoring: Spread count {spread_count}, threshold {self.activation_threshold}")
            should_activate = spread_count >= self.activation_threshold
            if should_activate:
                print(f"🚨 Fact check triggered! Spread count {spread_count} >= {self.activation_threshold}")
            else:
                print(f"❌ Fact check not triggered, need {self.activation_threshold} spreads")
                
        elif self.type == 'early_warning':
            print(f"⚠️ Early warning monitoring: Spread count {spread_count}, threshold {self.activation_threshold}")
            should_activate = spread_count >= self.activation_threshold
            if should_activate:
                print(f"🚨 Early warning triggered! Spread count {spread_count} >= {self.activation_threshold}")
            else:
                print(f"❌ Early warning not triggered, need {self.activation_threshold} spreads")
        
        if should_activate:
            print(f"✅ Activating {self.type} countermeasure")
            self.activate_countermeasure(misinfo_id, current_step)
        else:
            print(f"⏸️  {self.type} countermeasure not activated yet")
        
        print("-" * 50)  # Separator line