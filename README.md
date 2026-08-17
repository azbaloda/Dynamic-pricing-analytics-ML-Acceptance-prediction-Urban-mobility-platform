# Dynamic-pricing-analytics-ML-Acceptance-prediction-
# Urban-mobility-platform

## 📌 1. Project Context & Business Overview
In ride-hailing platforms, **Dynamic Pricing (Surge Multiplier)** is the primary mechanism to balance real-time demand and driver supply. When demand spikes (e.g., peak rush hours, rainstorms, concerts) or driver supply drops, prices automatically scale upwards by a multiplier (e.g., 1.2x to 4.5x).

However, excessive surge multipliers can cause customer friction, app abandonment, and loss of long-term customer lifetime value (LTV). Conversely, insufficient surge during high demand leads to unfulfilled requests, long wait times, and driver churn.

This project uses **500,000+ trip records**, **80,000 customer profiles**, **10,000 active drivers**, and **3,500 external surge events** across 6 major Indian metropolitan areas (*Mumbai, Delhi NCR, Bangalore, Hyderabad, Chennai, Pune*) to model price elasticity, optimize surge multipliers for profit maximization, and predict trip acceptance probabilities using Machine Learning.

---

## 📊 2. Detailed Problem Statements

### **Problem Statement 1: Baseline Demand-Supply Mechanics & Spatial-Temporal Surge Distribution**
* **Business Need**: Operations managers need to understand when and where surge pricing triggers most frequently across major Indian metro zones.
* **Detailed Analytics**: Aggregate total requested trips, total completed revenue, average surge multiplier, and available driver supply across 6 Cities, 30 Zones, 24 Hours of the day, and Weekday vs Weekend splits.
* **Key Metrics**: 
  - Total Trip Request Volume & Completed Revenue (₹ Cr)
  - Average Surge Multiplier by Zone & Hour
  - Driver-to-Request Supply Ratio (`supply_available_drivers / trip_requests`)
  - Weather Impact Factor (Clear vs Light Rain vs Heavy Rain)
* **Business Impact**: Pinpoints spatial-temporal supply deficits (e.g., Bangalore Tech Parks at 6 PM during rain) to deploy targeted driver re-location incentives.

---

### **Problem Statement 2: Surge Multiplier Threshold & Cancellation Funnel Analysis**
* **Business Need**: Product managers need to determine the exact surge multiplier threshold where customer conversion breaks down and app abandonment surges.
* **Detailed Analytics**: Build a step-by-step conversion funnel analyzing `trip_accepted` rates across granular surge buckets (`1.0x`, `1.1x–1.4x`, `1.5x–1.9x`, `2.0x–2.4x`, `2.5x–3.0x`, `>3.0x`). Categorize non-accepted trips by primary cancellation reason (`PRICE_TOO_HIGH`, `FOUND_CHEAPER_OPTION`, `DRIVER_TOO_FAR`, `CHANGED_MIND`).
* **Key Metrics**:
  - Conversion Rate per Surge Bucket (%)
  - Cancellation Rate per Surge Bucket (%)
  - Lost Revenue Exposure due to Price Rejections (₹ Lakhs)
  - Dominant Cancellation Reason Distribution (%)
* **Business Impact**: Identifies the "Price Cliff"—the exact surge level beyond which price increases reduce total revenue due to mass cancellations.

---

### **Problem Statement 3: Competitor Price Ratio & Price Elasticity of Demand (PED) Modeling**
* **Business Need**: Executive leadership needs to measure customer sensitivity to price changes relative to market competitors.
* **Detailed Analytics**: 
  1. Calculate the **Competitor Price Ratio** (`our_final_fare / competitor_fare`).
  2. Compute **Price Elasticity of Demand (PED)** across different customer sensitivity cohorts (`LOW`, `MEDIUM`, `HIGH`) and vehicle categories (`BIKE`, `AUTO`, `MINI`, `SEDAN`, `SUV`).
  $$\text{PED} = \frac{\% \Delta \text{ Trip Acceptance Rate}}{\% \Delta \text{ Surge Multiplier}}$$
* **Key Metrics**:
  - Price Elasticity Coefficient ($\epsilon$) per Customer Segment
  - Sensitivity Category Breakdown (Inelastic $|\epsilon| < 1$, Elastic $|\epsilon| > 1$)
  - Win-Loss Rate vs Competitor Fares (%)
* **Business Impact**: Enables risk-adjusted pricing—charging higher surge to price-inelastic corporate commuters while maintaining competitive fares for price-sensitive budget users.

---

### **Problem Statement 4: Revenue Optimization & Optimal Surge Multiplier Discovery**
* **Business Need**: Revenue management teams want to maximize expected gross revenue per zone per hour rather than blindly maximizing the surge multiplier.
* **Detailed Analytics**: For each surge multiplier step $s \in [1.0, 4.5]$, model Expected Revenue $E[R(s)]$ as:
  $$E[R(s)] = \text{Base Fare} \times s \times P(\text{Acceptance} \mid s)$$
  Plot Expected Revenue Curves to identify the optimal multiplier ($s^*$) that achieves peak revenue.
* **Key Metrics**:
  - Empirical Optimal Surge Multiplier ($s^*$) per Zone & Hour
  - Peak Revenue Potential vs Current Revenue (₹ Lakhs)
  - Revenue Leakage from Over-Surging (₹ Lakhs)
* **Business Impact**: Replaces naive linear surge algorithms with revenue-maximizing optimal price points.

---

### **Problem Statement 5: Machine Learning — Trip Acceptance Probability Prediction**
* **Business Need**: The dispatch engine needs real-time predictive intelligence to evaluate whether presenting a specific surge price to a specific user will result in a confirmed booking or an immediate drop-off.
* **Detailed Analytics**: Train and evaluate Machine Learning Classification Models (**Logistic Regression, Random Forest, XGBoost**) to predict the binary target `trip_accepted`.
* **Features Included**: `surge_multiplier`, `distance_km`, `base_fare`, `competitor_fare_ratio`, `weather_condition`, `trip_hour`, `is_weekend`, `supply_available_drivers`, `customer_segment`, `driver_rating`.
* **Key Metrics**:
  - Model Accuracy, Precision, Recall, F1-Score, AUC-ROC Score
  - Feature Importance Ranking (identifying key drivers of acceptance)
  - Confusion Matrix Analysis
* **Business Impact**: Powers real-time personalized pricing engines in production dispatch systems.

---

### **Problem Statement 6: Customer Segmentation & Price Sensitivity Cohort Analysis**
* **Business Need**: Marketing and loyalty teams need to segment riders based on historical booking behavior, surge tolerance, and trip frequency.
* **Detailed Analytics**: Perform customer cohort profiling and **K-Means Clustering** on aggregated customer features (Total Rides, Avg Surge Paid, Acceptance Rate under Surge, Cancellation Rate due to Price).
* **Key Metrics**:
  - Cluster Centroids & Profiles (`Corporate Power Users`, `Budget Commuters`, `Fair-Weather Riders`, `Price-Sensitive Churn Risks`)
  - Segment Revenue Contribution (% of Total Platform Revenue)
  - Segment Price Tolerance Threshold (Max acceptable surge)
* **Business Impact**: Guides loyalty program perks (e.g., "Surge Protection Pass" for Frequent Commuters).

---

### **Problem Statement 7: External Event Surge Amplification & Weather Risk Modeling**
* **Business Need**: Fleet management needs to quantify how extreme events (Heavy Rain, IPL Matches, Concerts, Metro Strikes) impact demand surges and driver availability.
* **Detailed Analytics**: Merge `surge_events` data with `trips` to evaluate demand spikes (%), surge boost factors (+0.5x to +1.5x), and driver supply deficits during adverse events.
* **Key Metrics**:
  - Event Demand Spike Factor (%)
  - Unmet Demand Rate during Rain/Events (%)
  - Incremental Revenue Generated by Surge Events (₹ Lakhs)
* **Business Impact**: Optimizes driver surge guarantees and location-specific bonus allocations during high-demand events.
