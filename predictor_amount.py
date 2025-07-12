import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import warnings
import io
from datetime import datetime, timedelta
import logging

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('prophet').setLevel(logging.WARNING)

st.set_page_config(
    page_title="AI Inventory Management Dashboard",
    page_icon="|||",
    layout="wide",
    initial_sidebar_state="expanded"
)

class InventoryForecaster:
    def __init__(self):
        self.forecasts = {}
        self.reorder_recommendations = pd.DataFrame()
    
    def load_and_validate_data(self, df):
        """Load and validate the sales data"""
        required_columns = ['date', 'product_id', 'units_sold', 'stock_left']
        
        # Check if all required columns exist
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            return None
        
        # Convert date column
        try:
            df['date'] = pd.to_datetime(df['date'])
        except:
            st.error("Error converting date column. Please ensure dates are in a valid format.")
            return None
        
        # Ensure numeric columns
        numeric_columns = ['units_sold', 'stock_left']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with invalid data
        df = df.dropna()
        
        if df.empty:
            st.error("No valid data found after cleaning.")
            return None
        
        return df
    
    def prepare_prophet_data(self, product_data):
        """Prepare data for Prophet model"""
        # Aggregate daily sales
        daily_sales = product_data.groupby('date')['units_sold'].sum().reset_index()
        daily_sales.columns = ['ds', 'y']
        
        # Ensure we have enough data points
        if len(daily_sales) < 1:
            return None
        
        return daily_sales
    
    def train_prophet_model(self, prophet_data):
        """Train Prophet model for a product"""
        try:
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0
            )
            
            model.fit(prophet_data)
            return model
        except Exception as e:
            st.warning(f"Error training model: {str(e)}")
            return None
    
    def generate_forecast(self, model, days=30):
        """Generate forecast for next 30 days"""
        future = model.make_future_dataframe(periods=days)
        forecast = model.predict(future)
        return forecast
    
    def calculate_reorder_quantity(self, predicted_demand, current_stock, safety_factor=1.2, lead_time_days=7):
        """Calculate reorder quantity based on forecast"""
        # Add lead time demand to the 30-day forecast
        total_demand = predicted_demand * (1 + lead_time_days/30)
        
        # Apply safety factor
        required_stock = total_demand * safety_factor
        
        # Calculate reorder quantity
        reorder_qty = max(0, required_stock - current_stock)
        
        return int(np.ceil(reorder_qty))
    
    def process_all_products(self, df, safety_factor, lead_time):
        """Process all products and generate forecasts"""
        products = df['product_id'].unique()
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, product_id in enumerate(products):
            status_text.text(f"Processing product {product_id}... ({i+1}/{len(products)})")
            
            # Get product data
            product_data = df[df['product_id'] == product_id].copy()
            
            # Prepare data for Prophet
            prophet_data = self.prepare_prophet_data(product_data)
            
            if prophet_data is None:
                continue
            
            # Train model
            model = self.train_prophet_model(prophet_data)
            
            if model is None:
                continue
            
            # Generate forecast
            forecast = self.generate_forecast(model)
            
            # Calculate metrics
            last_30_days = forecast.tail(30)
            predicted_demand = last_30_days['yhat'].sum()
            current_stock = product_data['stock_left'].iloc[-1]
            
            # Calculate reorder quantity
            reorder_qty = self.calculate_reorder_quantity(
                predicted_demand, current_stock, safety_factor, lead_time
            )
            
            # Store results
            results.append({
                'product_id': product_id,
                'current_stock': current_stock,
                'predicted_demand_30d': max(0, predicted_demand),
                'reorder_quantity': reorder_qty,
                'forecast_data': forecast,
                'historical_data': prophet_data,
                'model': model
            })
            
            # Store forecast for visualization
            self.forecasts[product_id] = {
                'forecast': forecast,
                'historical': prophet_data,
                'model': model
            }
            
            progress_bar.progress((i + 1) / len(products))
        
        status_text.text("Processing complete!")
        progress_bar.empty()
        status_text.empty()
        
        # Create results DataFrame
        self.reorder_recommendations = pd.DataFrame([
            {
                'Product ID': r['product_id'],
                'Current Stock': r['current_stock'],
                'Predicted Demand (30d)': int(r['predicted_demand_30d']),
                'Reorder Quantity': r['reorder_quantity'],
                'Stock Status': 'Low Stock' if r['reorder_quantity'] > 0 else 'Sufficient',
                'Days Until Stockout': int(r['current_stock'] / max(1, r['predicted_demand_30d']/30)) if r['predicted_demand_30d'] > 0 else 999
            }
            for r in results
        ])
        
        return results

def create_forecast_chart(product_id, forecast_data, historical_data):
    """Create forecast visualization chart"""
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_data['ds'],
        y=historical_data['y'],
        mode='markers+lines',
        name='Historical Sales',
        line=dict(color='blue'),
        marker=dict(size=4)
    ))
    
    # Forecast
    forecast_future = forecast_data.tail(30)
    fig.add_trace(go.Scatter(
        x=forecast_future['ds'],
        y=forecast_future['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='red', dash='dash')
    ))
    
    # Confidence intervals
    fig.add_trace(go.Scatter(
        x=forecast_future['ds'],
        y=forecast_future['yhat_upper'],
        fill=None,
        mode='lines',
        line_color='rgba(0,0,0,0)',
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_future['ds'],
        y=forecast_future['yhat_lower'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,0,0,0)',
        name='Confidence Interval',
        fillcolor='rgba(255,0,0,0.2)'
    ))
    
    fig.update_layout(
        title=f'Sales Forecast for Product {product_id}',
        xaxis_title='Date',
        yaxis_title='Units Sold',
        hovermode='x unified',
        height=400
    )
    
    return fig

def create_summary_charts(df):
    """Create summary charts for the dashboard"""
    
    # Stock status pie chart
    stock_status_counts = df['Stock Status'].value_counts()
    fig_pie = px.pie(
        values=stock_status_counts.values,
        names=stock_status_counts.index,
        title="Stock Status Distribution",
        color_discrete_map={'Low Stock': '#ff6b6b', 'Sufficient': '#51cf66'}
    )
    
    # Top products needing reorder
    top_reorder = df.nlargest(10, 'Reorder Quantity')
    fig_bar = px.bar(
        top_reorder,
        x='Product ID',
        y='Reorder Quantity',
        title="Top 10 Products Requiring Reorder",
        color='Reorder Quantity',
        color_continuous_scale='Reds'
    )
    
    return fig_pie, fig_bar

def main():
    st.title("AI Inventory Management Dashboard")
    st.markdown("Upload your sales data to get AI-powered demand forecasts and reorder recommendations")
    
    # Sidebar for parameters
    st.sidebar.header("Configuration")
    safety_factor = st.sidebar.slider("Safety Stock Factor", 1.0, 2.0, 1.2, 0.1,
                                     help="Multiplier for safety stock (higher = more conservative)")
    lead_time = st.sidebar.slider("Lead Time (days)", 1, 30, 7,
                                 help="Time between order and delivery")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Sales Data CSV",
        type=['csv'],
        help="CSV should contain columns: date, product_id, units_sold, stock_left"
    )
    
    if uploaded_file is not None:
        # Load data
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Data loaded successfully! {len(df)} rows, {len(df['product_id'].unique())} unique products")
            
            # Show data preview
            with st.expander("Data Preview"):
                st.dataframe(df.head())
                st.write(f"Date range: {df['date'].min()} to {df['date'].max()}")
                st.write(f"Products: {', '.join(map(str, df['product_id'].unique()[:10]))}" + 
                        (f" and {len(df['product_id'].unique())-10} more..." if len(df['product_id'].unique()) > 10 else ""))
            
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            return
        
        # Initialize forecaster
        forecaster = InventoryForecaster()
        
        # Validate and process data
        clean_df = forecaster.load_and_validate_data(df)
        
        if clean_df is not None:
            # Process forecasts
            if st.button("Generate Forecasts & Recommendations", type="primary"):
                with st.spinner("Training AI models and generating forecasts..."):
                    results = forecaster.process_all_products(clean_df, safety_factor, lead_time)
                
                if len(results) == 0:
                    st.error("No valid forecasts could be generated. Please check your data.")
                    return
                
                # Display results
                st.header("Forecast Results")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Products Analyzed", len(results))
                
                with col2:
                    low_stock_count = len(forecaster.reorder_recommendations[
                        forecaster.reorder_recommendations['Stock Status'] == 'Low Stock'
                    ])
                    st.metric("Products Needing Reorder", low_stock_count)
                
                with col3:
                    total_reorder = forecaster.reorder_recommendations['Reorder Quantity'].sum()
                    st.metric("Total Units to Reorder", int(total_reorder))
                
                with col4:
                    avg_demand = forecaster.reorder_recommendations['Predicted Demand (30d)'].mean()
                    st.metric("Avg. 30-day Demand", int(avg_demand))
                
                # Summary charts
                fig_pie, fig_bar = create_summary_charts(forecaster.reorder_recommendations)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig_pie, use_container_width=True)
                with col2:
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # Detailed results table
                st.header("📋 Detailed Recommendations")
                
                # Filter options
                col1, col2 = st.columns(2)
                with col1:
                    status_filter = st.selectbox("Filter by Stock Status", 
                                               ["All", "Low Stock", "Sufficient"])
                with col2:
                    sort_by = st.selectbox("Sort by", 
                                         ["Reorder Quantity", "Predicted Demand (30d)", "Product ID"])
                
                # Apply filters
                filtered_df = forecaster.reorder_recommendations.copy()
                if status_filter != "All":
                    filtered_df = filtered_df[filtered_df['Stock Status'] == status_filter]
                
                filtered_df = filtered_df.sort_values(sort_by, ascending=False)
                
                # Display table
                st.dataframe(
                    filtered_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Stock Status": st.column_config.TextColumn(
                            "Stock Status",
                            help="Current stock status"
                        ),
                        "Days Until Stockout": st.column_config.NumberColumn(
                            "Days Until Stockout",
                            help="Estimated days until stock runs out",
                            format="%d days"
                        )
                    }
                )
                
                # Download button
                csv_buffer = io.StringIO()
                forecaster.reorder_recommendations.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Download Recommendations CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"reorder_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Individual product forecasts
                st.header("Individual Product Forecasts")
                
                selected_products = st.multiselect(
                    "Select products to view detailed forecasts:",
                    options=list(forecaster.forecasts.keys()),
                    default=list(forecaster.forecasts.keys())[:3] if len(forecaster.forecasts) >= 3 else list(forecaster.forecasts.keys())
                )
                
                for product_id in selected_products:
                    if product_id in forecaster.forecasts:
                        forecast_info = forecaster.forecasts[product_id]
                        fig = create_forecast_chart(
                            product_id,
                            forecast_info['forecast'],
                            forecast_info['historical']
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Show sample data format
        st.info("Please upload a CSV file with your sales data to get started.")
        
        st.subheader("Expected CSV Format:")
        sample_data = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
            'product_id': ['PROD_001', 'PROD_001', 'PROD_002', 'PROD_002'],
            'units_sold': [10, 8, 15, 12],
            'stock_left': [100, 92, 200, 188]
        })
        st.dataframe(sample_data)
        
        st.markdown("""
        **Required Columns:**
        - `date`: Date of sales (YYYY-MM-DD format recommended)
        - `product_id`: Unique identifier for each product
        - `units_sold`: Number of units sold on that date
        - `stock_left`: Remaining stock after sales
        
        **Optional Columns:**
        - Any additional columns will be ignored
        """)

if __name__ == "__main__":
    main()