# Customer RFM Segmentation Dashboard

This Streamlit application provides an interactive dashboard for customer segmentation using RFM (Recency, Frequency, Monetary) analysis.

## Features
- **Dynamic RFM Calculation:** Adjust the weights for Recency, Frequency, and Monetary scores to see how customer segments change in real-time.
- **Descriptive Analytics:** Get a high-level overview of your customer data with key metrics and visualizations.
- **Interactive Data Table:** Search, sort, and filter your customer data along with their calculated RFM scores and segments.
- **CSV Upload:** Upload your own customer dataset for analysis.

## Setup & Installation
1.  **Clone the repository or download the files.**
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run the App
1.  Make sure your customer data is in a CSV file named `careem_food_enhanced_user_data.csv` and is placed in the same directory as `app.py`. Alternatively, you can upload a different file through the app's sidebar.
2.  Run the following command in your terminal:
    ```bash
    streamlit run app.py
    ```
3.  The application will open in your web browser.
