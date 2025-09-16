🏪 Predictive Restaurant Recommender

📌 Overview

This project aims to build a Predictive Restaurant Recommendation Engine using customer, location, order, and vendor data.
The goal is to recommend the most relevant restaurants to users based on their past behavior, preferences, and location proximity.

📂 Datasets Used

train_customers — Basic customer information

train_locations — Location details for each customer

train_orders — Order history and transaction details

vendors — Information about available restaurants/vendors

🧹 Data Cleaning & Preprocessing

✅ train_customers

Removed duplicate customer_id entries

Dropped high-missing columns (gender, dob) due to low predictive value

Retained only useful identifiers

✅ train_locations

Verified uniqueness of customer_id + location_number pairs

Dropped rows with missing latitude/longitude

Handled missing item_count (filled with median)

Converted is_favorite from Yes/No to 1/0

✅ train_orders

Removed duplicate order_id entries

Dropped irrelevant columns like status, verified, created_at, and other timestamps

Cleaned and normalized key transaction features

✅ vendors

Planned to drop uninformative columns (authentication_id, created_at, detailed opening times)

Retain essential columns:

latitude, longitude for distance calculation

vendor_category_en and vendor_tag_name for cuisine preferences

Process vendor_tag_name into multiple binary tag features (e.g. is_pizza, is_sushi)

⚙️ Feature Engineering

Created has_vendor_rating flag to capture missing rating info

Converted categorical flags (is_favorite, is_rated) to numeric

Plan to calculate geodesic distance between customer and vendor using coordinates

Plan to create a user–vendor interaction matrix from order history

🔗 Dataset Merging Plan

Merge on these keys:

train_orders + train_locations → by customer_id, location_number

vendors → by vendor_id

train_customers → by customer_id

Final merged dataset will have customer info + location + vendor details + order behavior for each record

🧠 Modeling Approach

Explore:

Collaborative Filtering (user-item interaction based)

Content-Based Filtering (restaurant features based)

Hybrid Models (LightGBM/XGBoost using combined features)

Key features:

Customer spending behavior

Vendor category and rating

Interaction frequency

Distance between customer and vendor

📊 Tools and Technologies

Python, Pandas, NumPy

Scikit-learn, LightGBM / XGBoost

Geopy (for distance calculation)

Matplotlib / Seaborn (for EDA)

🚀 Future Work

Implement model evaluation using Precision@K / Recall@K

Add real-time recommendation capability

📁 Project Structure

Restaurant-Recommendation 

1.Train Data 

     1. vendors
     2. orders
     3. train_customers
     4. train_locations

2.Test Data

    1. test_customers
    2. test_location
    
📬 Author

Ragipati Sireesha

