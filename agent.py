# In this project, I defined a custom BaseAgent class instead of directly using mesa.Agent. This is because my model uses a network-based structure (NetworkGrid) where agents don’t move, and most of the behavior is event-driven (triggered by tweet events), not by agents taking autonomous steps.
# By using a lightweight BaseAgent, I have more control over how agents behave, and I avoid potential conflicts or complexity that can come from Mesa’s built-in scheduling or grid movement features. This approach also makes the model simpler and easier to manage, since I don’t need all the features of the full mesa.Agent class.

import numpy as np

# Base agent class to ensure compatibility with Mesa's network-based grid
class BaseAgent:
    def __init__(self, unique_id, model):
        self.unique_id = unique_id  # Agent's unique identifier
        self.model = model          # Reference to the model
        self.random = model.random  # Random generator shared with the model
        self.pos = None             # Required by NetworkGrid to track position

# Agent representing a social media user
class UserAgent(BaseAgent):
    def __init__(self, unique_id, model, user_data):
        super().__init__(unique_id, model)
        # Basic user metadata
        self.user_id = user_data.get('user_id', str(unique_id))
        self.screen_name = user_data.get('screen_name', '')
        self.name = user_data.get('name', '')
        self.verified = user_data.get('verified', False)
        self.followers_count = user_data.get('followers_count', 0)
        self.location = user_data.get('location', '')
        self.description = user_data.get('description', '')

        # State tracking
        self.received_misinfo = set()            # Misinformation received
        self.retweeted_misinfo = set()           # Misinformation retweeted
        self.received_countermeasures = set()    # Countermeasures received

        # Behavior probabilities
        self.retweet_probability = self._calculate_retweet_probability()
        self.susceptibility_to_countermeasures = self._calculate_susceptibility()

    def _calculate_retweet_probability(self):
        """Determine base probability of retweeting misinformation"""
        base_prob = 0.1
        if self.verified:
            base_prob *= 0.8  # Verified users are more cautious
        follower_factor = 1.0
        if self.followers_count > 0:
            follower_factor = 1.0 - min(0.5, np.log(1 + self.followers_count) / 30)
        return base_prob * follower_factor

    def _calculate_susceptibility(self):
        """Determine how much the agent is influenced by countermeasures"""
        return 0.7 if self.verified else 0.3 + (0.4 * self.random.random())

    def receive_tweet(self, tweet_data, is_countermeasure=False):
        """Process incoming tweet—either misinformation or countermeasure"""
        misinfo_id = tweet_data['misinfo_id']
        if is_countermeasure:
            self.received_countermeasures.add(misinfo_id)
            # If already retweeted, might retract based on susceptibility
            if misinfo_id in self.retweeted_misinfo and self.random.random() < self.susceptibility_to_countermeasures * 0.3:
                self.retweeted_misinfo.remove(misinfo_id)
                self.model.record_retraction(self, misinfo_id)
        else:
            self.received_misinfo.add(misinfo_id)
            self._consider_retweeting(tweet_data)

    def _consider_retweeting(self, tweet_data):
        """Decide whether to retweet a misinformation tweet"""
        misinfo_id = tweet_data['misinfo_id']
        if misinfo_id in self.retweeted_misinfo:
            return  # Already retweeted

        rt_prob = self.retweet_probability
        if misinfo_id in self.received_countermeasures:
            rt_prob *= (1 - self.susceptibility_to_countermeasures)  # Reduce prob

        if self.random.random() < rt_prob:
            self.retweeted_misinfo.add(misinfo_id)
            self.model.user_retweet(self, tweet_data)

    def step(self):
        """No autonomous behavior; acts only on received messages"""
        pass

# Agent representing a platform-level countermeasure
class CountermeasureAgent(BaseAgent):
    def __init__(self, unique_id, model, countermeasure_type, settings=None):
        super().__init__(unique_id, model)
        self.type = countermeasure_type  # 'key_node', 'fact_check', 'early_warning'
        settings = settings or {}

        # Parameter initialization based on type
        if countermeasure_type == 'key_node':
            self.activation_threshold = settings.get('activation_threshold', 0.1)
            self.target_threshold = settings.get('target_threshold', 100000)
        elif countermeasure_type == 'fact_check':
            self.activation_threshold = settings.get('activation_threshold', 50)
            self.delay = settings.get('delay', 30)
        elif countermeasure_type == 'early_warning':
            self.activation_threshold = settings.get('activation_threshold', 10)
            self.coverage_ratio = settings.get('coverage_ratio', 0.3)

        # State tracking
        self.active_countermeasures = {}   # {misinfo_id: step when active}
        self.pending_countermeasures = {}  # {misinfo_id: step to activate}

    def activate_countermeasure(self, misinfo_id, current_step):
        """Trigger a countermeasure activation for a specific misinformation"""
        if self.type == 'fact_check':
            # Fact-check delayed activation
            self.pending_countermeasures[misinfo_id] = current_step + self.delay
        else:
            # Immediate activation for other types
            self.active_countermeasures[misinfo_id] = current_step
            self.model.deploy_countermeasure(self, misinfo_id)

    def step(self):
        """Each step checks if any pending countermeasures should be activated"""
        current_step = getattr(self.model, "step_count", 0)
        to_activate = []
        for misinfo_id, activation_step in self.pending_countermeasures.items():
            if current_step >= activation_step:
                to_activate.append(misinfo_id)
                self.active_countermeasures[misinfo_id] = current_step
        for misinfo_id in to_activate:
            del self.pending_countermeasures[misinfo_id]
            self.model.deploy_countermeasure(self, misinfo_id)

# import mesa
# import numpy as np
# from mesa import Agent


# class UserAgent(Agent):
#     """代表Twitter用户的Agent"""
    
#     def __init__(self, unique_id, model, user_data):
#         Agent.__init__(self, unique_id, model)
#         # 基础用户属性
#         self.user_id = user_data.get('user_id', str(unique_id))
#         self.screen_name = user_data.get('screen_name', '')
#         self.name = user_data.get('name', '')
#         self.verified = user_data.get('verified', False)
#         self.followers_count = user_data.get('followers_count', 0)
#         self.location = user_data.get('location', '')
#         self.description = user_data.get('description', '')
        
#         # 状态属性
#         self.received_misinfo = set()  # 存储收到的misinformation ID
#         self.retweeted_misinfo = set()  # 存储已转发的misinformation ID
#         self.received_countermeasures = set()  # 存储收到的对抗措施ID
        
#         # 行为参数 (可以基于用户属性进行调整)
#         self.retweet_probability = self._calculate_retweet_probability()
#         self.susceptibility_to_countermeasures = self._calculate_susceptibility()
    
#     def _calculate_retweet_probability(self):
#         """计算该用户转发misinformation的基础概率"""
#         # 基础概率
#         base_prob = 0.1
        
#         # 认证用户可能更谨慎
#         if self.verified:
#             base_prob *= 0.8
        
#         # 关注者多的用户可能更谨慎
#         follower_factor = 1.0
#         if self.followers_count > 0:
#             follower_factor = 1.0 - min(0.5, np.log(1 + self.followers_count) / 30)
        
#         return base_prob * follower_factor
    
#     def _calculate_susceptibility(self):
#         """计算用户对对抗措施的敏感度"""
#         # 认证用户可能更相信对抗措施
#         if self.verified:
#             return 0.7
#         else:
#             return 0.3 + (0.4 * self.random.random())  # 0.3-0.7之间随机
    
#     def receive_tweet(self, tweet_data, is_countermeasure=False):
#         """接收推文 (可能是misinformation或对抗措施)"""
#         misinfo_id = tweet_data['misinfo_id']
        
#         if is_countermeasure:
#             self.received_countermeasures.add(misinfo_id)
#             # 已经转发过，但收到了对抗措施，可能会撤回转发 (简化模型)
#             if misinfo_id in self.retweeted_misinfo and self.random.random() < self.susceptibility_to_countermeasures * 0.3:
#                 self.retweeted_misinfo.remove(misinfo_id)
#                 self.model.record_retraction(self, misinfo_id)
#         else:
#             # 接收到misinformation
#             self.received_misinfo.add(misinfo_id)
            
#             # 考虑是否转发
#             self._consider_retweeting(tweet_data)
    
#     def _consider_retweeting(self, tweet_data):
#         """决定是否转发收到的信息"""
#         misinfo_id = tweet_data['misinfo_id']
        
#         # 已经转发过的不再转发
#         if misinfo_id in self.retweeted_misinfo:
#             return
        
#         # 基础转发概率
#         rt_prob = self.retweet_probability
        
#         # 如果收到了对抗措施，降低转发概率
#         if misinfo_id in self.received_countermeasures:
#             rt_prob *= (1 - self.susceptibility_to_countermeasures)
        
#         # 决定是否转发
#         if self.random.random() < rt_prob:
#             self.retweeted_misinfo.add(misinfo_id)
            
#             # 通知模型该用户转发了信息
#             self.model.user_retweet(self, tweet_data)
    
#     def step(self):
#         """Agent的步进函数 - 在模型中的每个时间步执行"""
#         # 大部分行为由事件驱动，这里可以留空或添加主动行为
#         pass


# class CountermeasureAgent(Agent):
#     """实施对抗措施的Agent"""
    
#     def __init__(self, unique_id, model, countermeasure_type, settings=None):
#         Agent.__init__(self, unique_id, model)
#         self.type = countermeasure_type  # 'key_node', 'fact_check', 'early_warning'
#         settings = settings or {}
        
#         # 对抗措施的参数
#         if countermeasure_type == 'key_node':
#             self.activation_threshold = settings.get('activation_threshold', 0.1)  # 默认10%的高影响力用户转发后激活
#             self.target_threshold = settings.get('target_threshold', 100000)  # 默认"高影响力"用户的粉丝数阈值
#         elif countermeasure_type == 'fact_check':
#             self.activation_threshold = settings.get('activation_threshold', 50)  # 默认当转发数达到50时激活
#             self.delay = settings.get('delay', 30)  # 默认延迟30个时间步
#         elif countermeasure_type == 'early_warning':
#             self.activation_threshold = settings.get('activation_threshold', 10)  # 默认10次转发就激活
#             self.coverage_ratio = settings.get('coverage_ratio', 0.3)  # 默认覆盖30%的用户
        
#         # 状态
#         self.active_countermeasures = {}  # {misinfo_id: activation_step}
#         self.pending_countermeasures = {}  # {misinfo_id: activation_step}
    
#     def activate_countermeasure(self, misinfo_id, current_step):
#         """激活对某个misinformation的对抗措施"""
#         if self.type == 'fact_check':
#             # 事实核查有延迟
#             self.pending_countermeasures[misinfo_id] = current_step + self.delay
#         else:
#             # 其他对抗措施立即生效
#             self.active_countermeasures[misinfo_id] = current_step
#             self.model.deploy_countermeasure(self, misinfo_id)
    
#     def step(self):
#         """每个时间步检查是否有需要部署的对抗措施"""
#         current_step = self.model.schedule.steps
        
#         # 检查是否有延迟到期的对抗措施
#         to_activate = []
#         for misinfo_id, activation_step in self.pending_countermeasures.items():
#             if current_step >= activation_step:
#                 to_activate.append(misinfo_id)
#                 self.active_countermeasures[misinfo_id] = current_step
        
#         # 部署到期的对抗措施
#         for misinfo_id in to_activate:
#             del self.pending_countermeasures[misinfo_id]
#             self.model.deploy_countermeasure(self, misinfo_id)



