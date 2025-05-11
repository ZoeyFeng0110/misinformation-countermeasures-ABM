import os
import json
import pandas as pd
from datetime import datetime

def parse_tweet_time(time_str):
    """解析Twitter时间格式"""
    try:
        # Twitter时间格式如: "Wed Jan 07 11:07:51 +0000 2015"
        return datetime.strptime(time_str, '%a %b %d %H:%M:%S +0000 %Y')
    except ValueError:
        # 如果格式不匹配，尝试其他格式或返回默认时间
        print(f"无法解析时间格式: {time_str}")
        return datetime.now()  # 返回当前时间作为默认值

def load_network_data(network_file_path):
    """加载who-follows-whom网络数据"""
    # 这个函数处理具体的网络数据格式，返回适合构建网络的数据结构
    # 假设文件是每行一个关注关系，格式为: follower_id,followed_id
    network_edges = []
    
    try:
        with open(network_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # 跳过注释行
                    parts = line.split(',')
                    if len(parts) >= 2:
                        follower = parts[0].strip()
                        followed = parts[1].strip()
                        network_edges.append((follower, followed))
    except Exception as e:
        print(f"加载网络数据时出错: {e}")
    
    return network_edges

def load_event_data(event_folder_path):
    """加载事件文件夹的数据，适配实际的文件结构"""
    event_data = {}
    
    # 检查事件文件夹是否存在
    if not os.path.exists(event_folder_path):
        raise FileNotFoundError(f"事件文件夹不存在: {event_folder_path}")
    
    # 遍历事件文件夹下的所有misinformation文件夹（即ID文件夹）
    for misinfo_id in os.listdir(event_folder_path):
        misinfo_path = os.path.join(event_folder_path, misinfo_id)
        
        # 跳过非文件夹和隐藏文件夹
        if not os.path.isdir(misinfo_path) or misinfo_id.startswith('.'):
            continue
        
        # 定义各文件路径
        source_tweets_folder = os.path.join(misinfo_path, 'source-tweets')
        retweets_json_file = os.path.join(misinfo_path, 'retweets.json')
        network_file = os.path.join(misinfo_path, 'who-follows-whom.dat')
        
        # 加载source tweet
        source_tweet = None
        if os.path.exists(source_tweets_folder):
            # 查找source-tweets文件夹中的json文件
            for file_name in os.listdir(source_tweets_folder):
                if file_name.endswith('.json'):
                    source_file_path = os.path.join(source_tweets_folder, file_name)
                    with open(source_file_path, 'r', encoding='utf-8') as f:
                        source_tweet = json.load(f)
                    break
        
        # 加载retweets - 修改为按行解析JSONL格式
        retweets = []
        if os.path.exists(retweets_json_file):
            with open(retweets_json_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:  # 跳过空行
                        try:
                            tweet = json.loads(line)
                            retweets.append(tweet)
                        except json.JSONDecodeError as e:
                            print(f"警告：无法解析JSON行: {line[:50]}... 错误: {e}")
        
        # 加载网络结构 (如果存在)
        network_data = None
        if os.path.exists(network_file):
            network_data = load_network_data(network_file)
        
        # 处理时间格式并收集数据
        if source_tweet:
            # 转换source tweet的时间格式
            if 'created_at' in source_tweet:
                source_tweet['created_at_dt'] = parse_tweet_time(source_tweet['created_at'])
            
            # 转换所有retweet的时间格式
            for retweet in retweets:
                if 'created_at' in retweet:
                    retweet['created_at_dt'] = parse_tweet_time(retweet['created_at'])
            
            # 存储该misinformation的数据
            event_data[misinfo_id] = {
                'source_tweet': source_tweet,
                'retweets': retweets,
                'network': network_data
            }
    
    return event_data

def inspect_retweet_user_structure(event_data):
    """检查retweet中的用户数据结构"""
    print("\n检查retweet用户结构...")
    
    sample_count = 0
    for misinfo_id, misinfo_data in event_data.items():
        if misinfo_data['retweets'] and sample_count < 5:  # 只检查5个样本
            retweet = misinfo_data['retweets'][0]
            print(f"\nMisinfo ID {misinfo_id} 的第一条retweet:")
            if 'user' in retweet:
                print(f"用户数据键: {retweet['user'].keys()}")
                print(f"用户数据样例: {retweet['user']}")
            else:
                print("没有'user'字段!")
                print(f"可用字段: {retweet.keys()}")
            sample_count += 1
    
    # 检查是否每个misinformation都有retweets
    misinfo_with_retweets = sum(1 for data in event_data.values() if data['retweets'])
    print(f"\n含有retweets的misinformation数: {misinfo_with_retweets}/{len(event_data)}")

def preprocess_event_data(event_data):
    """预处理事件数据，准备用于Mesa模型"""
    # 创建用户字典，避免重复
    users_dict = {}
    
    # 添加调试计数器
    total_tweets = 0
    total_users_with_id = 0
    
    for misinfo_id, misinfo_data in event_data.items():
        # 处理source tweet
        source_tweet = misinfo_data['source_tweet']
        total_tweets += 1
        
        if 'user' in source_tweet:
            source_user = source_tweet['user']
            
            # 添加用户并打印详细日志
            if 'user_id' in source_user:
                total_users_with_id += 1
                user_id = source_user['user_id']
                if user_id not in users_dict:
                    users_dict[user_id] = source_user
                    print(f"添加新用户: {user_id}, 当前用户总数: {len(users_dict)}")
                else:
                    print(f"用户已存在: {user_id}")
            else:
                print(f"警告: source tweet中没有user_id字段! 用户数据: {source_user}")
        
        # 添加source tweet到时间序列
        source_tweet['misinfo_id'] = misinfo_id
        source_tweet['is_source'] = True
        
        # 处理retweets
        for retweet in misinfo_data['retweets']:
            total_tweets += 1
            
            # 处理不同的用户数据结构情况
            user_data = None
            user_id = None
            
            # 情况1: 标准结构，user字段包含user_id
            if 'user' in retweet and isinstance(retweet['user'], dict) and 'user_id' in retweet['user']:
                user_data = retweet['user']
                user_id = user_data['user_id']
            
            # 情况2: user字段存在但结构不同
            elif 'user' in retweet and isinstance(retweet['user'], dict):
                user_data = retweet['user']
                # 尝试其他可能的ID字段
                for id_field in ['id', 'id_str', 'userId', 'user_id_str']:
                    if id_field in user_data:
                        user_id = str(user_data[id_field])  # 确保ID是字符串
                        user_data['user_id'] = user_id  # 添加标准字段
                        break
            
            # 情况3: 用户数据直接在retweet根级别
            elif any(field in retweet for field in ['user_id', 'userId', 'id', 'id_str']):
                user_data = {}
                for field in retweet.keys():
                    if field in ['user_id', 'userId', 'id', 'id_str', 'screen_name', 'name', 
                                'verified', 'followers_count', 'location', 'description']:
                        user_data[field] = retweet[field]
                
                # 确保有user_id字段
                for id_field in ['user_id', 'userId', 'id', 'id_str']:
                    if id_field in retweet:
                        user_id = str(retweet[id_field])
                        user_data['user_id'] = user_id
                        break
            
            # 如果成功提取了用户数据和ID
            if user_data and user_id:
                total_users_with_id += 1
                if user_id not in users_dict:
                    users_dict[user_id] = user_data
                    if len(users_dict) % 100 == 0:  # 每100个新用户打印一次
                        print(f"添加新用户: {user_id}, 当前用户总数: {len(users_dict)}")
                # 不打印已存在用户，避免日志过多
            else:
                # 尝试从用户字段中提取更多信息
                if 'user' in retweet:
                    print(f"警告: 无法从retweet的user字段提取用户ID! user keys: {retweet['user'].keys() if isinstance(retweet['user'], dict) else 'not a dict'}")
                else:
                    print(f"警告: retweet中没有user字段! retweet keys: {retweet.keys()}")
    
    # 打印最终统计数据
    print(f"\n用户提取统计:")
    print(f"总推文数: {total_tweets}")
    print(f"包含用户ID的推文数: {total_users_with_id}")
    print(f"唯一用户数: {len(users_dict)}")
    
    # 创建推文时间序列
    all_tweets = []
    tweets_without_user = 0
    
    for misinfo_id, misinfo_data in event_data.items():
        # 处理source tweet
        source_tweet = misinfo_data['source_tweet']
        source_tweet['misinfo_id'] = misinfo_id
        source_tweet['is_source'] = True
        
        # 确保source tweet有关联用户
        if 'user' in source_tweet and 'user_id' in source_tweet['user']:
            source_tweet['user_id'] = source_tweet['user']['user_id']
        else:
            tweets_without_user += 1
        
        all_tweets.append(source_tweet)
        
        # 处理retweets
        for retweet in misinfo_data['retweets']:
            retweet['misinfo_id'] = misinfo_id
            retweet['is_source'] = False
            retweet['source_id'] = source_tweet.get('tweet_id', 'unknown')
            
            # 确保retweet有关联用户
            user_id = None
            
            # 尝试从user字段提取
            if 'user' in retweet and isinstance(retweet['user'], dict):
                user_dict = retweet['user']
                # 尝试多个可能的ID字段
                for id_field in ['user_id', 'id', 'id_str', 'userId']:
                    if id_field in user_dict:
                        user_id = str(user_dict[id_field])
                        retweet['user_id'] = user_id
                        break
            
            # 如果user字段没有ID，尝试从retweet根级别提取
            if not user_id:
                for id_field in ['user_id', 'id', 'id_str', 'userId']:
                    if id_field in retweet:
                        user_id = str(retweet[id_field])
                        retweet['user_id'] = user_id
                        break
            
            if not user_id:
                tweets_without_user += 1
            
            all_tweets.append(retweet)
    
    print(f"没有关联用户的推文数: {tweets_without_user} / {len(all_tweets)}")
    
    # 按时间排序所有推文
    all_tweets_df = pd.DataFrame(all_tweets)
    if 'created_at_dt' in all_tweets_df.columns:
        all_tweets_df = all_tweets_df.sort_values(by='created_at_dt')
        
        # 计算推文时间间隔和最早时间
        earliest_time = all_tweets_df['created_at_dt'].min()
        all_tweets_df['time_since_start'] = (all_tweets_df['created_at_dt'] - earliest_time).dt.total_seconds()
    else:
        # 如果没有时间数据，使用索引作为时间
        earliest_time = datetime.now()
        all_tweets_df['time_since_start'] = all_tweets_df.index * 60  # 假设每条推文间隔60秒
    
    # 构建网络数据
    network_edges = []
    for misinfo_data in event_data.values():
        if 'network' in misinfo_data and misinfo_data['network']:
            network_edges.extend(misinfo_data['network'])
    
    return {
        'users': list(users_dict.values()),
        'tweets_timeline': all_tweets_df,
        'earliest_time': earliest_time,
        'network_edges': network_edges
    }

def main():
    """主函数"""
    # 设置事件文件夹路径
    event_folder_path = "/Users/oliviafeng/Desktop/uchi/agent_based_modeling/code/final_project/pheme-rumour-scheme-dataset/threads/en/charliehebdo"  # 请替换为您的实际路径
    
    try:
        # 1. 加载事件数据
        print(f"正在加载事件数据: {event_folder_path}")
        event_data = load_event_data(event_folder_path)
        print(f"成功加载 {len(event_data)} 个misinformation数据")
        
        # 2. 检查retweet用户结构
        inspect_retweet_user_structure(event_data)
        
        # 3. 预处理数据
        print("\n开始数据预处理...")
        preprocessed_data = preprocess_event_data(event_data)
        
        # 4. 显示处理结果
        print(f"\n预处理完成!")
        print(f"加载了 {len(event_data)} 个misinformation数据")
        print(f"共有 {len(preprocessed_data['users'])} 个用户")
        print(f"推文时间线包含 {len(preprocessed_data['tweets_timeline'])} 条推文")
        print(f"最早的推文时间: {preprocessed_data['earliest_time']}")
        
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()