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
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AWS Cloud Pricing Optimizer",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1f4e79 0%, #2c5aa0 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(31, 78, 121, 0.3);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 600;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    .metric-card {
        background: white;
        padding: 1.8rem;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.08);
        border-left: 4px solid #4a90e2;
        margin: 1rem 0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.12);
    }
    
    .metric-card h3 {
        color: #343a40;
        margin: 0 0 0.5rem 0;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-card h2 {
        color: #1f4e79;
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    
    .ai-recommendation {
        background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 20px rgba(74, 144, 226, 0.3);
    }
    
    .ai-recommendation h3 {
        margin: 0 0 1rem 0;
        font-size: 1.3rem;
        font-weight: 600;
    }
    
    .pricing-card {
        background: white;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .pricing-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.15);
        border-color: #4a90e2;
    }
    
    .cost-savings {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 1.8rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(40, 167, 69, 0.3);
    }
    
    .cost-savings h3 {
        margin: 0 0 0.5rem 0;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        opacity: 0.9;
    }
    
    .cost-savings h2 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    
    .cost-savings p {
        margin: 0.5rem 0 0 0;
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-success {
        background: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .status-warning {
        background: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    
    .status-error {
        background: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    
    .sidebar .element-container {
        margin-bottom: 1rem;
    }
    
    .section-header {
        color: #1f4e79;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e9ecef;
    }
    
    /* Custom button styling */
    .stButton > button {
        background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(74, 144, 226, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(74, 144, 226, 0.4);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        color: #6c757d;
    }
    
    .stTabs [aria-selected="true"] {
        background: #4a90e2;
        color: white;
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
        """Initialize AWS clients using Streamlit secrets"""
        try:
            # Get credentials from Streamlit secrets
            aws_access_key = st.secrets.get("AWS_ACCESS_KEY_ID")
            aws_secret_key = st.secrets.get("AWS_SECRET_ACCESS_KEY")
            
            if not aws_access_key or not aws_secret_key:
                logger.error("AWS credentials not found in secrets")
                st.error("❌ AWS credentials not configured in secrets. Please contact administrator.")
                return
            
            session = boto3.Session(
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name='us-east-1'  # Pricing API is only available in us-east-1
            )
            
            self.pricing_client = session.client('pricing', region_name='us-east-1')
            self.ec2_client = session.client('ec2', region_name='us-east-1')
            
            logger.info("AWS clients initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AWS clients: {e}")
            st.error(f"❌ Failed to connect to AWS: {str(e)}")
    
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
            'eu-central-1': 'Europe (Frankfurt)',
            'ap-northeast-1': 'Asia Pacific (Tokyo)',
            'us-east-2': 'US East (Ohio)',
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
    
    def __init__(self):
        self.api_key = st.secrets.get("CLAUDE_API_KEY")
        self.base_url = "https://api.anthropic.com/v1/messages"
        
    async def get_optimization_recommendations(self, 
                                             workload_data: Dict, 
                                             pricing_data: List[PricingData]) -> AIRecommendation:
        """Get AI-powered optimization recommendations"""
        try:
            if not self.api_key:
                return self._fallback_recommendation()
                
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
            reasoning="Basic optimization principles applied (AI unavailable)"
        )

class CloudPricingOptimizer:
    """Main application class"""
    
    def __init__(self):
        self.aws_pricing = AWSPricingFetcher()
        self.claude_ai = ClaudeAIIntegration()
        self._initialize_session_state()
        
    def _initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'pricing_cache' not in st.session_state:
            st.session_state.pricing_cache = {}
        if 'last_analysis' not in st.session_state:
            st.session_state.last_analysis = None
    
    def render_main_interface(self):
        """Render the main Streamlit interface"""
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>☁️ AWS Cloud Pricing Optimizer</h1>
            <p>Professional-grade AWS pricing analysis with AI-powered optimization recommendations</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Check connection status
        self.render_connection_status()
        
        # Sidebar configuration
        self.render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs(["💰 Pricing Analysis", "🤖 AI Recommendations", "📊 Cost Comparison", "📈 Reports & Export"])
        
        with tab1:
            self.render_pricing_analysis()
        
        with tab2:
            self.render_ai_recommendations()
        
        with tab3:
            self.render_cost_comparison()
        
        with tab4:
            self.render_trends_reports()
    
    def render_connection_status(self):
        """Display connection status for AWS and Claude AI"""
        col1, col2 = st.columns(2)
        
        with col1:
            aws_status = "✅ Connected" if self.aws_pricing.pricing_client else "❌ Disconnected"
            aws_class = "status-success" if self.aws_pricing.pricing_client else "status-error"
            st.markdown(f'<span class="status-badge {aws_class}">AWS: {aws_status}</span>', unsafe_allow_html=True)
        
        with col2:
            claude_status = "✅ Connected" if self.claude_ai.api_key else "❌ Disconnected"
            claude_class = "status-success" if self.claude_ai.api_key else "status-warning"
            st.markdown(f'<span class="status-badge {claude_class}">Claude AI: {claude_status}</span>', unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar configuration"""
        with st.sidebar:
            st.markdown('<div class="section-header">⚙️ Configuration</div>', unsafe_allow_html=True)
            
            # Workload Configuration
            st.markdown('<div class="section-header">📋 Workload Parameters</div>', unsafe_allow_html=True)
            
            region = st.selectbox("AWS Region", [
                "us-east-1", "us-west-1", "us-west-2", "us-east-2",
                "eu-west-1", "eu-central-1", "ap-southeast-1", "ap-northeast-1"
            ], index=0)
            
            workload_type = st.selectbox("Workload Type", [
                "Production", "Staging", "Development", "Testing"
            ])
            
            st.markdown("**System Requirements**")
            cpu_cores = st.slider("CPU Cores", 2, 64, 8)
            ram_gb = st.slider("RAM (GB)", 4, 512, 32)
            storage_gb = st.slider("Storage (GB)", 100, 10000, 500)
            
            st.markdown("**Performance Metrics**")
            peak_cpu = st.slider("Peak CPU Utilization (%)", 20, 100, 70)
            peak_ram = st.slider("Peak RAM Utilization (%)", 20, 100, 80)
            
            st.markdown("**SQL Server Configuration**")
            sql_edition = st.selectbox("SQL Server Edition", [
                "Standard", "Enterprise", "Developer"
            ])
            
            licensing_model = st.selectbox("Licensing Model", [
                "License Included", "BYOL (Bring Your Own License)"
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
            
            # Help section
            with st.expander("ℹ️ Help & Information"):
                st.markdown("""
                **Getting Started:**
                1. Configure your workload parameters
                2. Click 'Fetch Latest Prices' to get current AWS pricing
                3. Use AI recommendations for optimization insights
                4. Export reports for stakeholder review
                
                **Support:**
                Contact your system administrator for API access issues.
                """)
    
    def render_pricing_analysis(self):
        """Render real-time pricing analysis"""
        st.markdown('<div class="section-header">💰 Real-Time AWS Pricing Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write("Get the latest AWS EC2 pricing for SQL Server instances based on your configuration.")
        
        with col2:
            if st.button("🔄 Fetch Latest Prices", type="primary", use_container_width=True):
                with st.spinner("Fetching real-time pricing data..."):
                    pricing_data = self.fetch_pricing_data()
                    
                    if pricing_data:
                        st.session_state.latest_pricing = pricing_data
                        st.success(f"✅ Fetched {len(pricing_data)} pricing options")
                        self.display_pricing_results(pricing_data)
                    else:
                        st.error("❌ Failed to fetch pricing data. Please check your AWS configuration.")
        
        # Display cached results if available
        if hasattr(st.session_state, 'latest_pricing'):
            self.display_pricing_results(st.session_state.latest_pricing)
    
    def fetch_pricing_data(self) -> List[PricingData]:
        """Fetch pricing data from AWS"""
        config = st.session_state.config
        pricing_data = []
        
        # Define instance types to check based on requirements
        instance_types = [
            'm5.large', 'm5.xlarge', 'm5.2xlarge', 'm5.4xlarge',
            'r5.large', 'r5.xlarge', 'r5.2xlarge', 'r5.4xlarge',
            'm6a.large', 'm6a.xlarge', 'm6a.2xlarge',
            'r6a.xlarge', 'r6a.2xlarge', 'c5.xlarge', 'c5.2xlarge'
        ]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, instance_type in enumerate(instance_types):
            progress_bar.progress((i + 1) / len(instance_types))
            status_text.text(f"Fetching pricing for {instance_type}...")
            
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
        
        progress_bar.empty()
        status_text.empty()
        
        return sorted(pricing_data, key=lambda x: x.price_per_month)
    
    def display_pricing_results(self, pricing_data: List[PricingData]):
        """Display pricing results in a formatted table"""
        if not pricing_data:
            st.warning("⚠️ No pricing data available")
            return
        
        # Create DataFrame for display
        df = pd.DataFrame([{
            'Instance Type': pd.instance_type,
            'vCPUs': self.get_instance_specs(pd.instance_type)['vcpus'],
            'RAM (GB)': self.get_instance_specs(pd.instance_type)['ram'],
            'Hourly Cost': f"${pd.price_per_hour:.3f}",
            'Monthly Cost': f"${pd.price_per_month:.2f}",
            'Annual Cost': f"${pd.price_per_month * 12:,.0f}"
        } for pd in pricing_data])
        
        st.markdown('<div class="section-header">💸 Instance Pricing Comparison</div>', unsafe_allow_html=True)
        
        # Show summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cheapest = min(pricing_data, key=lambda x: x.price_per_month)
            st.markdown(f"""
            <div class="metric-card">
                <h3>Most Economical</h3>
                <h2>{cheapest.instance_type}</h2>
                <p>${cheapest.price_per_month:.2f}/month</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            avg_cost = sum(pd.price_per_month for pd in pricing_data) / len(pricing_data)
            st.markdown(f"""
            <div class="metric-card">
                <h3>Average Cost</h3>
                <h2>${avg_cost:.2f}</h2>
                <p>Monthly across all options</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Options Found</h3>
                <h2>{len(pricing_data)}</h2>
                <p>Instance types analyzed</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Pricing table
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Create pricing visualization
        fig = px.bar(
            x=[pd.instance_type for pd in pricing_data[:10]],
            y=[pd.price_per_month for pd in pricing_data[:10]],
            title="Monthly Pricing Comparison (Top 10 Most Economical Options)",
            labels={'x': 'Instance Type', 'y': 'Monthly Cost ($)'},
            color=[pd.price_per_month for pd in pricing_data[:10]],
            color_continuous_scale=['#28a745', '#ffc107', '#dc3545']
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            plot_bgcolor='white',
            font_color='#343a40'
        )
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
            'r6a.2xlarge': {'vcpus': 8, 'ram': 64},
            'c5.xlarge': {'vcpus': 4, 'ram': 8},
            'c5.2xlarge': {'vcpus': 8, 'ram': 16}
        }
        return specs_mapping.get(instance_type, {'vcpus': 0, 'ram': 0})
    
    def render_ai_recommendations(self):
        """Render AI-powered recommendations"""
        st.markdown('<div class="section-header">🤖 Claude AI Recommendations</div>', unsafe_allow_html=True)
        
        if not self.claude_ai.api_key:
            st.markdown("""
            <div class="status-badge status-warning" style="display: block; text-align: center; margin: 2rem 0;">
                ⚠️ Claude AI is not configured. Please contact administrator to enable AI features.
            </div>
            """, unsafe_allow_html=True)
            return
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write("Get intelligent optimization recommendations based on your workload and current AWS pricing.")
        
        with col2:
            if st.button("🧠 Get AI Analysis", type="primary", use_container_width=True):
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
                    st.error("❌ Please fetch pricing data first!")
    
    def display_ai_recommendations(self, recommendation: AIRecommendation):
        """Display AI recommendations"""
        st.markdown(f"""
        <div class="ai-recommendation">
            <h3>🎯 AI-Powered Optimization Analysis</h3>
            <p><strong>Confidence Score:</strong> {recommendation.confidence_score}%</p>
            <p><strong>Expected Cost Impact:</strong> {recommendation.cost_impact}</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**📋 Recommendations**")
            st.write(recommendation.recommendation)
            
            st.markdown("**🧠 AI Analysis**")
            st.write(recommendation.reasoning)
        
        with col2:
            # Confidence meter
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = recommendation.confidence_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "AI Confidence", 'font': {'color': '#1f4e79'}},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#4a90e2"},
                    'steps': [
                        {'range': [0, 50], 'color': "#f8f9fa"},
                        {'range': [50, 80], 'color': "#fff3cd"},
                        {'range': [80, 100], 'color': "#d4edda"}
                    ],
                    'threshold': {
                        'line': {'color': "#28a745", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=300, font_color='#343a40')
            st.plotly_chart(fig, use_container_width=True)
    
    def render_cost_comparison(self):
        """Render cost comparison analysis"""
        st.markdown('<div class="section-header">📊 Cost Comparison & ROI Analysis</div>', unsafe_allow_html=True)
        
        if hasattr(st.session_state, 'latest_pricing'):
            pricing_data = st.session_state.latest_pricing
            
            # Current vs Optimized comparison
            col1, col2, col3 = st.columns(3)
            
            # Simulate current cost (assume using a mid-range option)
            current_cost = pricing_data[5].price_per_month if len(pricing_data) > 5 else 1000
            optimized_cost = pricing_data[0].price_per_month if pricing_data else 800
            savings = current_cost - optimized_cost
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Current Monthly Cost</h3>
                    <h2>${current_cost:.2f}</h2>
                    <p>Based on current instance</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Optimized Monthly Cost</h3>
                    <h2>${optimized_cost:.2f}</h2>
                    <p>Most economical option</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="cost-savings">
                    <h3>Potential Monthly Savings</h3>
                    <h2>${savings:.2f}</h2>
                    <p>{(savings/current_cost)*100:.1f}% cost reduction</p>
                </div>
                """, unsafe_allow_html=True)
            
            # ROI Calculator
            st.markdown('<div class="section-header">💹 ROI Calculator</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                migration_cost = st.number_input("Migration & Setup Cost ($)", value=50000, step=1000)
                
            with col2:
                operational_overhead = st.number_input("Monthly Operational Overhead ($)", value=500, step=100)
            
            if savings > operational_overhead:
                net_monthly_savings = savings - operational_overhead
                payback_months = migration_cost / net_monthly_savings
                annual_savings = net_monthly_savings * 12
                three_year_roi = (annual_savings * 3 - migration_cost) / migration_cost * 100
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Payback Period", f"{payback_months:.1f} months")
                with col2:
                    st.metric("Net Monthly Savings", f"${net_monthly_savings:,.0f}")
                with col3:
                    st.metric("Annual Savings", f"${annual_savings:,.0f}")
                with col4:
                    st.metric("3-Year ROI", f"{three_year_roi:.1f}%")
            else:
                st.warning("⚠️ Migration may not be cost-effective with current parameters.")
    
    def render_trends_reports(self):
        """Render trends and reporting section"""
        st.markdown('<div class="section-header">📈 Pricing Trends & Reports</div>', unsafe_allow_html=True)
        
        # Simulated trend data (in real app, this would come from historical data)
        dates = pd.date_range(start='2024-01-01', end='2024-12-01', freq='M')
        trend_data = pd.DataFrame({
            'Date': dates,
            'm5.xlarge': [200 + i*2 + (i%3)*5 for i in range(len(dates))],
            'r5.xlarge': [250 + i*2.5 + (i%4)*6 for i in range(len(dates))],
            'm6a.xlarge': [190 + i*1.8 + (i%5)*4 for i in range(len(dates))]
        })
        
        fig = px.line(trend_data, x='Date', y=['m5.xlarge', 'r5.xlarge', 'm6a.xlarge'],
                     title='AWS Instance Pricing Trends (Historical Simulation)',
                     color_discrete_sequence=['#4a90e2', '#28a745', '#ffc107'])
        fig.update_layout(plot_bgcolor='white', font_color='#343a40')
        st.plotly_chart(fig, use_container_width=True)
        
        # Export functionality
        st.markdown('<div class="section-header">📊 Export & Download</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Export Pricing Report", use_container_width=True):
                if hasattr(st.session_state, 'latest_pricing'):
                    csv_data = self.export_pricing_report()
                    if csv_data:
                        st.download_button(
                            label="📥 Download CSV Report",
                            data=csv_data,
                            file_name=f"aws_pricing_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
        
        with col2:
            if st.button("📊 Export Configuration", use_container_width=True):
                config_data = self.export_configuration()
                st.download_button(
                    label="📥 Download Config",
                    data=config_data,
                    file_name=f"workload_config_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        with col3:
            if st.button("📈 Export Full Analysis", use_container_width=True):
                summary_data = self.create_analysis_summary()
                st.download_button(
                    label="📥 Download Analysis",
                    data=summary_data,
                    file_name=f"optimization_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json",
                    use_container_width=True
                )
    
    def export_pricing_report(self) -> str:
        """Export pricing data as CSV"""
        if hasattr(st.session_state, 'latest_pricing'):
            df = pd.DataFrame([{
                'Instance Type': pd.instance_type,
                'Region': pd.region,
                'vCPUs': self.get_instance_specs(pd.instance_type)['vcpus'],
                'RAM (GB)': self.get_instance_specs(pd.instance_type)['ram'],
                'Price Per Hour': pd.price_per_hour,
                'Price Per Month': pd.price_per_month,
                'Price Per Year': pd.price_per_month * 12,
                'Last Updated': pd.last_updated.strftime('%Y-%m-%d %H:%M:%S')
            } for pd in st.session_state.latest_pricing])
            
            return df.to_csv(index=False)
        return ""
    
    def export_configuration(self) -> str:
        """Export current configuration"""
        config = st.session_state.config if hasattr(st.session_state, 'config') else {}
        export_data = {
            'workload_configuration': config,
            'export_timestamp': datetime.now().isoformat(),
            'export_version': '1.0'
        }
        return json.dumps(export_data, indent=2)
    
    def create_analysis_summary(self) -> str:
        """Create comprehensive analysis summary"""
        summary = {
            'analysis_metadata': {
                'generated_at': datetime.now().isoformat(),
                'version': '1.0',
                'aws_region': st.session_state.config.get('region', 'N/A') if hasattr(st.session_state, 'config') else 'N/A'
            },
            'workload_configuration': st.session_state.config if hasattr(st.session_state, 'config') else {},
            'pricing_analysis': {
                'total_options_analyzed': len(st.session_state.latest_pricing) if hasattr(st.session_state, 'latest_pricing') else 0,
                'cheapest_option': st.session_state.latest_pricing[0].instance_type if hasattr(st.session_state, 'latest_pricing') and st.session_state.latest_pricing else 'N/A',
                'cheapest_monthly_cost': st.session_state.latest_pricing[0].price_per_month if hasattr(st.session_state, 'latest_pricing') and st.session_state.latest_pricing else 0
            },
            'service_status': {
                'aws_connected': self.aws_pricing.pricing_client is not None,
                'claude_ai_available': self.claude_ai.api_key is not None
            }
        }
        
        return json.dumps(summary, indent=2)

# Application entry point
def main():
    """Main application entry point"""
    optimizer = CloudPricingOptimizer()
    optimizer.render_main_interface()

if __name__ == "__main__":
    main()