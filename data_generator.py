import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker()

def generate_data(
    num_influencers=100,
    min_posts_per_influencer=5,
    max_posts_per_influencer=20,
    min_tracking_events_per_influencer=10,
    max_tracking_events_per_influencer=100,
    start_date=datetime(2024, 7, 1),
    end_date=datetime(2025, 7, 26)
):
    """
    Generates simulated data for influencers, posts, tracking_data, and payouts.
    """

    print("Generating Influencers data...")
    # 1. influencers DataFrame
    influencers_data = []
    influencer_categories = ['Fitness', 'Nutrition', 'Lifestyle', 'Beauty', 'Gaming', 'Tech']
    # Updated: Added more platforms for simulation
    platforms = ['Instagram', 'YouTube', 'Twitter', 'TikTok', 'Facebook', 'Pinterest', 'LinkedIn', 'Snapchat', 'Twitch', 'Reddit']

    for i in range(num_influencers):
        followers = random.randint(10_000, 1_000_000)
        influencers_data.append({
            'ID': i + 1,
            'name': fake.name(),
            'category': random.choice(influencer_categories),
            'gender': random.choice(['Male', 'Female', 'Other']),
            'follower_count': followers,
            'platform': random.choice(platforms) # Primary platform for the influencer
        })
    df_influencers = pd.DataFrame(influencers_data)
    print(f"Generated {len(df_influencers)} influencers.")

    print("Generating Posts data...")
    # 2. posts DataFrame
    posts_data = []
    for index, influencer in df_influencers.iterrows():
        num_posts = random.randint(min_posts_per_influencer, max_posts_per_influencer)
        for _ in range(num_posts):
            post_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
            reach = int(influencer['follower_count'] * random.uniform(0.3, 0.8)) # 30-80% of followers
            likes = int(reach * random.uniform(0.01, 0.08)) # 1-8% of reach
            comments = int(reach * random.uniform(0.001, 0.01)) # 0.1-1% of reach

            posts_data.append({
                'influencer_id': influencer['ID'],
                'platform': random.choice(platforms), # Post could be on any platform
                'date': post_date.strftime('%Y-%m-%d'),
                'URL': fake.url(),
                'caption': fake.sentence(),
                'reach': reach,
                'likes': likes,
                'comments': comments
            })
    df_posts = pd.DataFrame(posts_data)
    print(f"Generated {len(df_posts)} posts.")


    print("Generating Tracking Data...")
    # 3. tracking_data DataFrame
    tracking_data = []
    products = ['Whey Protein', 'Multivitamin', 'Creatine', 'BCAA', 'Fish Oil', 'Kids Nutrition', 'Hair & Skin Vitamins']
    campaigns = ['Summer Sale', 'New Product Launch', 'Winter Fitness', 'HealthFest', 'Everyday Wellness']
    user_ids = [fake.uuid4() for _ in range(num_influencers * max_tracking_events_per_influencer * 2)] # More user IDs than events

    for index, influencer in df_influencers.iterrows():
        num_events = random.randint(min_tracking_events_per_influencer, max_tracking_events_per_influencer)
        for _ in range(num_events):
            event_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
            orders = random.randint(1, 5) # 1 to 5 items per transaction
            revenue_per_order = random.uniform(200, 2000) # Rs. 200 to 2000 per order
            total_revenue = orders * revenue_per_order

            tracking_data.append({
                'source': 'Influencer Campaign',
                'campaign': random.choice(campaigns),
                'influencer_id': influencer['ID'],
                'user_id': random.choice(user_ids),
                'product': random.choice(products),
                'date': event_date.strftime('%Y-%m-%d'),
                'orders': orders,
                'revenue': total_revenue
            })
    df_tracking_data = pd.DataFrame(tracking_data)
    print(f"Generated {len(df_tracking_data)} tracking events.")

    print("Generating Payouts Data...")
    # 4. payouts DataFrame
    payouts_data = []
    for influencer_id in df_influencers['ID'].unique():
        basis = random.choice(['post', 'order'])
        rate = 0
        orders_attributed = 0 # Will be calculated if basis is 'order'

        if basis == 'post':
            rate = random.randint(500, 15000) # Rs. 500 to 15000 per post
            total_payout = rate * df_posts[df_posts['influencer_id'] == influencer_id].shape[0] # Payout per post * number of posts
            if df_posts[df_posts['influencer_id'] == influencer_id].empty: # If an influencer has no posts, make payout 0
                total_payout = 0
        else: # basis == 'order'
            rate = random.uniform(20, 150) # Rs. 20 to 150 per order
            # Sum orders from tracking data for this influencer
            orders_attributed = df_tracking_data[df_tracking_data['influencer_id'] == influencer_id]['orders'].sum()
            total_payout = rate * orders_attributed

        payouts_data.append({
            'influencer_id': influencer_id,
            'basis': basis,
            'rate': round(rate, 2),
            'orders': int(orders_attributed), # Store total orders for 'order' basis
            'total_payout': round(total_payout, 2)
        })
    df_payouts = pd.DataFrame(payouts_data)
    print(f"Generated {len(df_payouts)} payout records.")


    print("Saving data to CSVs...")
    df_influencers.to_csv('influencers.csv', index=False)
    df_posts.to_csv('posts.csv', index=False)
    df_tracking_data.to_csv('tracking_data.csv', index=False)
    df_payouts.to_csv('payouts.csv', index=False)
    print("All data saved successfully!")

if __name__ == "__main__":
    generate_data()
