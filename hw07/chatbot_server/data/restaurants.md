# Restaurant Database Description

## Overview
This restaurant database was created from the Yelp Academic Dataset, containing restaurant information and customer reviews. The database was designed with a strategic sampling approach to create a diverse, representative dataset for testing the Restaurant Reservation ChatBot.

## Data Source
- **Primary Dataset**: Yelp Academic Dataset (Business and Review files)
- **Initial Filtering**: 
  - Extracted only businesses categorized as "Restaurants"
  - Retained only city-state pairs with at least 5 restaurants to ensure meaningful representation

## Sampling Strategy

The final database was constructed in 3 steps to ensure diversity and challenge:

### 1. Duplicate City Names

**Method**: 
- Identified all city names that appear in multiple states these cities include:
    - Audubon (NJ|PA)
    - Brentwood (MO|TN)
    - Clayton (MO|NJ)
    - Lebanon (IL|TN)
    - Southampton (NJ|PA)
    - Springfield (TN|PA)
- Included **ALL restaurants** (560) from these duplicate-name cities
- This creates natural ambiguity scenarios (e.g., Springfield, PA vs. Springfield, TN)

### 2. Major Metropolitan Areas

**Method**:
- Selected the **top 6 cities** by restaurant count these cities include:
    - Philadelphia PA
    - Tampa FL
    - Indianapolis IN
    - Nashville TN
    - Tucson AZ
    - New Orleans LA
- Included **ALL restaurants** (18910) from these major metropolitan areas

### 3. "Needle in Haystack" Samples

**Method**:
- Randomly sampled **100 restaurants** from remaining locations nationwide in various smaller cities and towns some examples to test for are:
    - Hart's Tavern Lahaska PA
    - Red Robin Gourmet Burgers and Brews Wesley Chapel FL
    - The Country Club Chesterfield	MO
    - Beachwood BBQ & Burger Saint Petersburg FL
- Also included **100 non-restaurant businesses** as noise
    - First Call Demo and Clean Outs Philadelphia PA
    - Ace's Upholstery Sparks NV
    - Nashboro Animal Hospital Nashville TN



## Review Data Integration

**Review Processing**:
- Sampled **up to 10 reviews per restaurant**
- Each restaurant entry includes:
  - Review texts (list of up to 10 reviews)
  - Star ratings (corresponding ratings for each review)
  - Number of sampled reviews
