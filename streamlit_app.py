import streamlit as st
import pandas as pd
import boto3
import json
import requests
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from functools import lru_cache
import os
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AWS Cloud Pricing Optimizer with AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        background-size: 200% 200%;
        animation: gradient 3s ease infinite;
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #4ECDC4;
        margin: 1rem 0;
    }
    
    .ai-recommendation {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .pricing-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: transform 0.2s;
    }
    
    .pricing-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .cost-savings {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class PricingData:
    """Data class for AWS pricing information"""
    service: str
    instance_type: str
    region: str
    price_per_hour: float
    price_per_month: float
    currency: str
    last_updated: datetime

@dataclass
class AIRecommendation:
    """Data class for Claude AI recommendations"""
    recommendation: str
    confidence_score: float
    cost_impact: str
    reasoning: str

class AWSPricingFetcher:
    """Handles AWS Pricing API calls with caching"""
    
    def __init__(self):
        self.pricing_client = None
        self.ec2_client = None
        self._initialize_clients()
        
    def _initialize_clients(self):
        """Initialize AWS clients with error handling"""
        try:
            # Use environment variables or IAM roles for authentication
            session = boto3.Session(
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name='us-east-1'  # Pricing API is only available in us-east-1
            )
            
            self.pricing_client = session.client('pricing', region_name='us-east-1')
            self.ec2_client = session.client('ec2', region_name='us-east-1')
            
        except Exception as e:
            logger.error(f"Failed to initialize AWS clients: {e}")
            st.error("Failed to connect to AWS. Please check your credentials.")
    
    @lru_cache(maxsize=100)
    def get_ec2_pricing(self, instance_type: str, region: str, operating_system: str = "Windows") -> Optional[float]:
        """Fetch EC2 pricing with caching"""
        try:
            if not self.pricing_client:
                return None
                
            response = self.pricing_client.get_products(
                ServiceCode='AmazonEC2',
                Filters=[
                    {'Type': 'TERM_MATCH', 'Field': 'instanceType', 'Value': instance_type},
                    {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': self._get_location_name(region)},
                    {'Type': 'TERM_MATCH', 'Field': 'operatingSystem', 'Value': operating_system},
                    {'Type': 'TERM_MATCH', 'Field': 'preInstalledSw', 'Value': 'SQL Server Standard'},
                    {'Type': 'TERM_MATCH', 'Field': 'tenancy', 'Value': 'Shared'},
                    {'Type': 'TERM_MATCH', 'Field': 'capacitystatus', 'Value': 'Used'}
                ]
            )
            
            for price_item in response['PriceList']:
                price_data = json.loads(price_item)
                terms = price_data.get('terms', {}).get('OnDemand', {})
                
                for term_key, term_value in terms.items():
                    price_dimensions = term_value.get('priceDimensions', {})
                    for pd_key, pd_value in price_dimensions.items():
                        price_per_hour = float(pd_value['pricePerUnit']['USD'])
                        return price_per_hour
                        
        except Exception as e:
            logger.error(f"Error fetching EC2 pricing: {e}")
            return None
        
        return None
    
    def _get_location_name(self, region: str) -> str:
        """Convert AWS region to location name for pricing API"""
        region_mapping = {
            'us-east-1': 'US East (N. Virginia)',
            'us-west-1': 'US West (N. California)',
            'us-west-2': 'US West (Oregon)',
            'eu-west-1': 'Europe (Ireland)',
            'ap-southeast-1': 'Asia Pacific (Singapore)',
            # Add more regions as needed
        }
        return region_mapping.get(region, 'US East (N. Virginia)')
    
    def get_ebs_pricing(self, volume_type: str, region: str) -> Optional[float]:
        """Fetch EBS pricing"""
        try:
            response = self.pricing_client.get_products(
                ServiceCode='AmazonEBS',
                Filters=[
                    {'Type': 'TERM_MATCH', 'Field': 'volumeType', 'Value': volume_type},
                    {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': self._get_location_name(region)}
                ]
            )
            
            for price_item in response['PriceList']:
                price_data = json.loads(price_item)
                terms = price_data.get('terms', {}).get('OnDemand', {})
                
                for term_key, term_value in terms.items():
                    price_dimensions = term_value.get('priceDimensions', {})
                    for pd_key, pd_value in price_dimensions.items():
                        price_per_gb = float(pd_value['pricePerUnit']['USD'])
                        return price_per_gb
                        
        except Exception as e:
            logger.error(f"Error fetching EBS pricing: {e}")
            return None
        
        return None

class ClaudeAIIntegration:
    """Integration with Claude API for AI recommendations"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com/v1/messages"
        
    async def get_optimization_recommendations(self, 
                                             workload_data: Dict, 
                                             pricing_data: List[PricingData]) -> AIRecommendation:
        """Get AI-powered optimization recommendations"""
        try:
            prompt = self._build_optimization_prompt(workload_data, pricing_data)
            
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            }
            
            payload = {
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result['content'][0]['text']
                        return self._parse_ai_response(content)
                    else:
                        logger.error(f"Claude API error: {response.status}")
                        return self._fallback_recommendation()
                        
        except Exception as e:
            logger.error(f"Error getting AI recommendations: {e}")
            return self._fallback_recommendation()
    
    def _build_optimization_prompt(self, workload_data: Dict, pricing_data: List[PricingData]) -> str:
        """Build prompt for Claude AI"""
        return f"""
        As a cloud cost optimization expert, analyze this SQL Server workload and provide recommendations:
        
        Workload Details:
        - CPU Cores: {workload_data.get('cpu_cores', 'N/A')}
        - RAM: {workload_data.get('ram_gb', 'N/A')} GB
        - Storage: {workload_data.get('storage_gb', 'N/A')} GB
        - Peak CPU Utilization: {workload_data.get('peak_cpu', 'N/A')}%
        - Peak RAM Utilization: {workload_data.get('peak_ram', 'N/A')}%
        - Workload Type: {workload_data.get('workload_type', 'N/A')}
        - Region: {workload_data.get('region', 'N/A')}
        
        Current Pricing Options (top 3):
        {self._format_pricing_for_prompt(pricing_data[:3])}
        
        Please provide:
        1. Recommended instance type and reasoning
        2. Cost optimization opportunities
        3. License optimization suggestions
        4. Performance considerations
        5. Risk assessment
        
        Format your response as JSON with fields: recommendation, confidence_score (0-100), cost_impact, reasoning
        """
    
    def _format_pricing_for_prompt(self, pricing_data: List[PricingData]) -> str:
        """Format pricing data for AI prompt"""
        formatted = []
        for pd in pricing_data:
            formatted.append(f"- {pd.instance_type}: ${pd.price_per_hour:.3f}/hour (${pd.price_per_month:.2f}/month)")
        return "\n".join(formatted)
    
    def _parse_ai_response(self, content: str) -> AIRecommendation:
        """Parse Claude AI response"""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return AIRecommendation(
                    recommendation=data.get('recommendation', content),
                    confidence_score=data.get('confidence_score', 75),
                    cost_impact=data.get('cost_impact', 'Medium'),
                    reasoning=data.get('reasoning', 'AI analysis complete')
                )
        except:
            pass
        
        # Fallback to text parsing
        return AIRecommendation(
            recommendation=content[:500],
            confidence_score=70,
            cost_impact="Medium",
            reasoning="Analysis based on workload characteristics"
        )
    
    def _fallback_recommendation(self) -> AIRecommendation:
        """Provide fallback recommendation when AI is unavailable"""
        return AIRecommendation(
            recommendation="Consider right-sizing instances based on actual utilization patterns. Evaluate BYOL options for cost savings.",
            confidence_score=60,
            cost_impact="Medium",
            reasoning="Basic optimization principles applied"
        )

class CloudPricingOptimizer:
    """Main application class"""
    
    def __init__(self):
        self.aws_pricing = AWSPricingFetcher()
        self.claude_ai = None
        self._initialize_session_state()
        
    def _initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'pricing_cache' not in st.session_state:
            st.session_state.pricing_cache = {}
        if 'last_analysis' not in st.session_state:
            st.session_state.last_analysis = None
        if 'claude_enabled' not in st.session_state:
            st.session_state.claude_enabled = False
    
    def setup_claude_integration(self, api_key: str):
        """Setup Claude AI integration"""
        if api_key:
            self.claude_ai = ClaudeAIIntegration(api_key)
            st.session_state.claude_enabled = True
    
    def render_main_interface(self):
        """Render the main Streamlit interface"""
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>🚀 AWS Cloud Pricing Optimizer with AI</h1>
            <p>Real-time AWS pricing with Claude AI-powered recommendations</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar configuration
        self.render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs(["💰 Pricing Analysis", "🤖 AI Recommendations", "📊 Cost Comparison", "📈 Trends & Reports"])
        
        with tab1:
            self.render_pricing_analysis()
        
        with tab2:
            self.render_ai_recommendations()
        
        with tab3:
            self.render_cost_comparison()
        
        with tab4:
            self.render_trends_reports()
    
    def render_sidebar(self):
        """Render sidebar configuration"""
        with st.sidebar:
            st.header("🔧 Configuration")
            
            # AWS Credentials
            st.subheader("AWS Configuration")
            aws_access_key = st.text_input("AWS Access Key ID", type="password", 
                                         value=os.getenv('AWS_ACCESS_KEY_ID', ''))
            aws_secret_key = st.text_input("AWS Secret Access Key", type="password",
                                         value=os.getenv('AWS_SECRET_ACCESS_KEY', ''))
            
            # Claude API Key
            st.subheader("Claude AI Configuration")
            claude_api_key = st.text_input("Claude API Key", type="password",
                                         value=os.getenv('CLAUDE_API_KEY', ''))
            
            if claude_api_key:
                self.setup_claude_integration(claude_api_key)
                st.success("✅ Claude AI Connected")
            else:
                st.warning("⚠️ Claude AI Disabled")
            
            # Workload Configuration
            st.subheader("Workload Parameters")
            
            region = st.selectbox("AWS Region", [
                "us-east-1", "us-west-1", "us-west-2", 
                "eu-west-1", "ap-southeast-1"
            ])
            
            workload_type = st.selectbox("Workload Type", [
                "Production", "Staging", "Development", "Testing"
            ])
            
            cpu_cores = st.slider("CPU Cores", 2, 64, 8)
            ram_gb = st.slider("RAM (GB)", 4, 512, 32)
            storage_gb = st.slider("Storage (GB)", 100, 10000, 500)
            
            peak_cpu = st.slider("Peak CPU Utilization (%)", 20, 100, 70)
            peak_ram = st.slider("Peak RAM Utilization (%)", 20, 100, 80)
            
            sql_edition = st.selectbox("SQL Server Edition", [
                "Standard", "Enterprise", "Developer"
            ])
            
            licensing_model = st.selectbox("Licensing Model", [
                "License Included", "BYOL"
            ])
            
            # Store configuration in session state
            st.session_state.config = {
                'region': region,
                'workload_type': workload_type,
                'cpu_cores': cpu_cores,
                'ram_gb': ram_gb,
                'storage_gb': storage_gb,
                'peak_cpu': peak_cpu,
                'peak_ram': peak_ram,
                'sql_edition': sql_edition,
                'licensing_model': licensing_model
            }
    
    def render_pricing_analysis(self):
        """Render real-time pricing analysis"""
        st.header("💰 Real-Time AWS Pricing Analysis")
        
        if st.button("🔄 Fetch Latest Prices", type="primary"):
            with st.spinner("Fetching real-time pricing data..."):
                pricing_data = self.fetch_pricing_data()
                
                if pricing_data:
                    st.session_state.latest_pricing = pricing_data
                    self.display_pricing_results(pricing_data)
                else:
                    st.error("Failed to fetch pricing data. Please check your AWS credentials.")
        
        # Display cached results if available
        if hasattr(st.session_state, 'latest_pricing'):
            self.display_pricing_results(st.session_state.latest_pricing)
    
    def fetch_pricing_data(self) -> List[PricingData]:
        """Fetch pricing data from AWS"""
        config = st.session_state.config
        pricing_data = []
        
        # Define instance types to check
        instance_types = [
            'm5.large', 'm5.xlarge', 'm5.2xlarge', 'm5.4xlarge',
            'r5.large', 'r5.xlarge', 'r5.2xlarge', 'r5.4xlarge',
            'm6a.large', 'm6a.xlarge', 'm6a.2xlarge',
            'r6a.xlarge', 'r6a.2xlarge'
        ]
        
        for instance_type in instance_types:
            price_per_hour = self.aws_pricing.get_ec2_pricing(
                instance_type, config['region'], "Windows"
            )
            
            if price_per_hour:
                pricing_data.append(PricingData(
                    service="EC2",
                    instance_type=instance_type,
                    region=config['region'],
                    price_per_hour=price_per_hour,
                    price_per_month=price_per_hour * 730,
                    currency="USD",
                    last_updated=datetime.now()
                ))
        
        return sorted(pricing_data, key=lambda x: x.price_per_month)
    
    def display_pricing_results(self, pricing_data: List[PricingData]):
        """Display pricing results in a formatted table"""
        if not pricing_data:
            st.warning("No pricing data available")
            return
        
        # Create DataFrame for display
        df = pd.DataFrame([{
            'Instance Type': pd.instance_type,
            'vCPUs': self.get_instance_specs(pd.instance_type)['vcpus'],
            'RAM (GB)': self.get_instance_specs(pd.instance_type)['ram'],
            'Hourly Cost': f"${pd.price_per_hour:.3f}",
            'Monthly Cost': f"${pd.price_per_month:.2f}",
            'Annual Cost': f"${pd.price_per_month * 12:.2f}"
        } for pd in pricing_data])
        
        st.subheader("💸 Instance Pricing Comparison")
        st.dataframe(df, use_container_width=True)
        
        # Create pricing visualization
        fig = px.bar(
            x=[pd.instance_type for pd in pricing_data[:10]],
            y=[pd.price_per_month for pd in pricing_data[:10]],
            title="Monthly Pricing Comparison (Top 10 Options)",
            labels={'x': 'Instance Type', 'y': 'Monthly Cost ($)'}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    def get_instance_specs(self, instance_type: str) -> Dict:
        """Get instance specifications"""
        specs_mapping = {
            'm5.large': {'vcpus': 2, 'ram': 8},
            'm5.xlarge': {'vcpus': 4, 'ram': 16},
            'm5.2xlarge': {'vcpus': 8, 'ram': 32},
            'm5.4xlarge': {'vcpus': 16, 'ram': 64},
            'r5.large': {'vcpus': 2, 'ram': 16},
            'r5.xlarge': {'vcpus': 4, 'ram': 32},
            'r5.2xlarge': {'vcpus': 8, 'ram': 64},
            'r5.4xlarge': {'vcpus': 16, 'ram': 128},
            'm6a.large': {'vcpus': 2, 'ram': 8},
            'm6a.xlarge': {'vcpus': 4, 'ram': 16},
            'm6a.2xlarge': {'vcpus': 8, 'ram': 32},
            'r6a.xlarge': {'vcpus': 4, 'ram': 32},
            'r6a.2xlarge': {'vcpus': 8, 'ram': 64}
        }
        return specs_mapping.get(instance_type, {'vcpus': 0, 'ram': 0})
    
    def render_ai_recommendations(self):
        """Render AI-powered recommendations"""
        st.header("🤖 Claude AI Recommendations")
        
        if not st.session_state.claude_enabled:
            st.warning("⚠️ Claude AI is not configured. Please add your API key in the sidebar.")
            return
        
        if st.button("🧠 Get AI Recommendations", type="primary"):
            if hasattr(st.session_state, 'latest_pricing'):
                with st.spinner("Analyzing with Claude AI..."):
                    recommendation = asyncio.run(
                        self.claude_ai.get_optimization_recommendations(
                            st.session_state.config,
                            st.session_state.latest_pricing
                        )
                    )
                    
                    self.display_ai_recommendations(recommendation)
            else:
                st.error("Please fetch pricing data first!")
    
    def display_ai_recommendations(self, recommendation: AIRecommendation):
        """Display AI recommendations"""
        st.markdown(f"""
        <div class="ai-recommendation">
            <h3>🎯 AI-Powered Optimization Recommendations</h3>
            <p><strong>Confidence Score:</strong> {recommendation.confidence_score}%</p>
            <p><strong>Cost Impact:</strong> {recommendation.cost_impact}</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📋 Recommendation")
            st.write(recommendation.recommendation)
            
            st.subheader("🧠 AI Reasoning")
            st.write(recommendation.reasoning)
        
        with col2:
            # Confidence meter
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = recommendation.confidence_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Confidence"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    def render_cost_comparison(self):
        """Render cost comparison analysis"""
        st.header("📊 Cost Comparison & Optimization")
        
        if hasattr(st.session_state, 'latest_pricing'):
            pricing_data = st.session_state.latest_pricing
            
            # Current vs Optimized comparison
            col1, col2, col3 = st.columns(3)
            
            current_cost = pricing_data[5].price_per_month if len(pricing_data) > 5 else 1000
            optimized_cost = pricing_data[0].price_per_month if pricing_data else 800
            savings = current_cost - optimized_cost
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Current Monthly Cost</h3>
                    <h2>${current_cost:.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Optimized Monthly Cost</h3>
                    <h2>${optimized_cost:.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="cost-savings">
                    <h3>Monthly Savings</h3>
                    <h2>${savings:.2f}</h2>
                    <p>{(savings/current_cost)*100:.1f}% reduction</p>
                </div>
                """, unsafe_allow_html=True)
            
            # ROI Calculator
            st.subheader("💹 ROI Calculator")
            
            migration_cost = st.number_input("Migration Cost ($)", value=50000, step=1000)
            
            if savings > 0:
                payback_months = migration_cost / savings
                annual_savings = savings * 12
                three_year_roi = (annual_savings * 3 - migration_cost) / migration_cost * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Payback Period", f"{payback_months:.1f} months")
                with col2:
                    st.metric("Annual Savings", f"${annual_savings:,.0f}")
                with col3:
                    st.metric("3-Year ROI", f"{three_year_roi:.1f}%")
    
    def render_trends_reports(self):
        """Render trends and reporting section"""
        st.header("📈 Pricing Trends & Reports")
        
        # Simulated trend data (in real app, this would come from historical data)
        dates = pd.date_range(start='2024-01-01', end='2024-12-01', freq='M')
        trend_data = pd.DataFrame({
            'Date': dates,
            'm5.xlarge': [200 + i*2 + (i%3)*5 for i in range(len(dates))],
            'r5.xlarge': [250 + i*2.5 + (i%4)*6 for i in range(len(dates))],
            'm6a.xlarge': [190 + i*1.8 + (i%5)*4 for i in range(len(dates))]
        })
        
        fig = px.line(trend_data, x='Date', y=['m5.xlarge', 'r5.xlarge', 'm6a.xlarge'],
                     title='AWS Instance Pricing Trends (Simulated Data)')
        st.plotly_chart(fig, use_container_width=True)
        
        # Export functionality
        st.subheader("📊 Export Reports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Export Pricing Report"):
                if hasattr(st.session_state, 'latest_pricing'):
                    csv_data = self.export_pricing_report()
                    st.download_button(
                        label="Download CSV Report",
                        data=csv_data,
                        file_name=f"aws_pricing_report_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
        
        with col2:
            if st.button("📊 Export Analysis Summary"):
                summary_data = self.create_analysis_summary()
                st.download_button(
                    label="Download Summary",
                    data=summary_data,
                    file_name=f"optimization_summary_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
    
    def export_pricing_report(self) -> str:
        """Export pricing data as CSV"""
        if hasattr(st.session_state, 'latest_pricing'):
            df = pd.DataFrame([{
                'Instance Type': pd.instance_type,
                'Region': pd.region,
                'Price Per Hour': pd.price_per_hour,
                'Price Per Month': pd.price_per_month,
                'Last Updated': pd.last_updated.strftime('%Y-%m-%d %H:%M:%S')
            } for pd in st.session_state.latest_pricing])
            
            return df.to_csv(index=False)
        return ""
    
    def create_analysis_summary(self) -> str:
        """Create analysis summary as JSON"""
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'configuration': st.session_state.config if hasattr(st.session_state, 'config') else {},
            'pricing_data_points': len(st.session_state.latest_pricing) if hasattr(st.session_state, 'latest_pricing') else 0,
            'claude_ai_enabled': st.session_state.claude_enabled
        }
        
        return json.dumps(summary, indent=2)

# Application entry point
def main():
    """Main application entry point"""
    optimizer = CloudPricingOptimizer()
    optimizer.render_main_interface()

if __name__ == "__main__":
    main()