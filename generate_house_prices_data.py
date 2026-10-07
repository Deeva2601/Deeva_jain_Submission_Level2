"""
Generate realistic, comprehensive Residential Housing Sales dataset for Level 2 Task 1: House Price Prediction with Linear Regression.
"""
import numpy as np
import pandas as pd

def generate_housing_dataset(n_samples=2800, random_state=42):
    np.random.seed(random_state)
    
    neighborhoods = {
        'Downtown Metro': {'base_price': 180000, 'sqft_rate': 240, 'weight': 0.18},
        'Green Hills': {'base_price': 220000, 'sqft_rate': 210, 'weight': 0.22},
        'Suburban West': {'base_price': 140000, 'sqft_rate': 160, 'weight': 0.25},
        'Riverside Park': {'base_price': 190000, 'sqft_rate': 195, 'weight': 0.15},
        'Metro Uptown': {'base_price': 165000, 'sqft_rate': 180, 'weight': 0.12},
        'East Bay Outskirts': {'base_price': 95000, 'sqft_rate': 130, 'weight': 0.08}
    }
    
    n_list = list(neighborhoods.keys())
    weights = [neighborhoods[n]['weight'] for n in n_list]
    
    records = []
    
    for i in range(1, n_samples + 1):
        prop_id = f"PROP-{10000 + i}"
        loc = np.random.choice(n_list, p=weights)
        loc_data = neighborhoods[loc]
        
        # Area in sqft (normally distributed between 750 and 5200)
        area_sqft = int(np.clip(np.random.normal(2150, 680), 750, 5500))
        
        # Bedrooms dependent on area
        if area_sqft < 1200:
            bedrooms = int(np.random.choice([1, 2], p=[0.6, 0.4]))
        elif area_sqft < 2000:
            bedrooms = int(np.random.choice([2, 3], p=[0.4, 0.6]))
        elif area_sqft < 3200:
            bedrooms = int(np.random.choice([3, 4], p=[0.55, 0.45]))
        else:
            bedrooms = int(np.random.choice([4, 5, 6], p=[0.5, 0.35, 0.15]))
            
        # Bathrooms
        bathrooms = max(1.0, round(bedrooms * 0.75 + np.random.choice([0, 0.5, 1.0], p=[0.5, 0.35, 0.15]), 1))
        
        # Stories
        stories = int(np.random.choice([1, 2, 3], p=[0.45, 0.45, 0.10]))
        
        # House Age (0 to 55 years)
        house_age = int(np.clip(np.random.exponential(15), 0, 55))
        
        # Garage capacity
        garage = int(np.random.choice([0, 1, 2, 3], p=[0.12, 0.30, 0.45, 0.13]))
        
        # Pool & AC
        has_pool = int(np.random.choice([0, 1], p=[0.78, 0.22]))
        has_central_ac = int(np.random.choice([0, 1], p=[0.10, 0.90]))
        
        # Distance to City Center
        dist_city = round(float(np.clip(np.random.normal(12.5, 6.0), 1.0, 35.0)), 1)
        
        # Overall condition score (1 to 10)
        condition_score = int(np.clip(np.random.normal(7.2, 1.6), 2, 10))
        
        # Property Price Formula with realistic noise
        price = (
            loc_data['base_price']
            + (area_sqft * loc_data['sqft_rate'])
            + (bedrooms * 18000)
            + (bathrooms * 24000)
            + (stories * 12000)
            - (house_age * 1650)
            + (garage * 19500)
            + (has_pool * 38000)
            + (has_central_ac * 15000)
            - (dist_city * 2200)
            + (condition_score * 8500)
            + np.random.normal(0, 22000) # Unobserved variation
        )
        
        price = max(110000.0, round(price, 2))
        
        records.append({
            'Property_ID': prop_id,
            'Area_SqFt': area_sqft,
            'Bedrooms': bedrooms,
            'Bathrooms': bathrooms,
            'Stories': stories,
            'Location_Neighborhood': loc,
            'House_Age_Years': house_age,
            'Garage_Capacity': garage,
            'Has_Pool': has_pool,
            'Has_Central_AC': has_central_ac,
            'Distance_to_CityCenter_Miles': dist_city,
            'Overall_Condition_Score': condition_score,
            'Price': price
        })
        
    df = pd.DataFrame(records)
    return df

if __name__ == '__main__':
    df = generate_housing_dataset(n_samples=2800)
    output_path = 'c:/Users/jainp/Downloads/Data_Analytics_Oasis_internship_Level2/house_prices_dataset.csv'
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df):,} housing records with {df.shape[1]} columns in {output_path}")
    print(df.head())
