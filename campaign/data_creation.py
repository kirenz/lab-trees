import pandas as pd
import numpy as np

np.random.seed(42)


def generate_data(num_samples=1000):
    age = np.random.randint(18, 65, size=num_samples)
    income = np.random.randint(20000, 150000, size=num_samples)
    membership_days = np.random.randint(100, 3000, size=num_samples)
    campaign_engagement = np.random.randint(0, 11, size=num_samples)

    # Generate a probability score based on age, income, and campaign_engagement
    score = (age / 100) + (income / 150000) + (campaign_engagement / 10)

    # Add some noise to the score
    noise = np.random.normal(0, 0.1, size=num_samples)
    score += noise

    # Set the target variable based on the probability score
    target = [1 if s >= 1.5 else 0 for s in score]

    city = np.random.choice(['Berlin', 'Stuttgart'], size=num_samples)

    df = pd.DataFrame({
        'age': age,
        'city': city,
        'income': income,
        'membership_days': membership_days,
        'campaign_engagement': campaign_engagement,
        'target': target
    })

    return df


df = generate_data()

df.to_csv("data.csv", index=False)
