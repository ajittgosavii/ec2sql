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
import numpy as np

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
    
    .section-header {
        color: #1f4e79;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e9ecef;
    }
    
    .error-details {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        font-family: monospace;
        font-size: 0.9rem;
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
    specifications: Dict = None

@dataclass
class AIRecommendation:
    """Data class for Claude AI recommendations"""
    recommendation: str
    confidence_score: float
    cost_impact: str
    reasoning: str

class AWSPricingFetcher:
    """Handles AWS Pricing API calls with improved error handling"""
    
    def __init__(self):
        self.pricing_client = None
        self.ec2_client = None
        self.connection_status = {"connected": False, "error": None}
        self._initialize_clients()
        
    def _initialize_clients(self):
        """Initialize AWS clients with better error handling"""
        try:
            # Check if we're in development mode (no secrets)
            if not hasattr(st, 'secrets') or 'AWS_ACCESS_KEY_ID' not in st.secrets:
                logger.warning("AWS credentials not found in secrets - using demo mode")
                self.connection_status = {
                    "connected": False, 
                    "error": "AWS credentials not configured. Running in demo mode."
                }
                return
            
            aws_access_key = st.secrets.get("AWS_ACCESS_KEY_ID")
            aws_secret_key = st.secrets.get("AWS_SECRET_ACCESS_KEY")
            
            if not aws_access_key or not aws_secret_key:
                self.connection_status = {
                    "connected": False,
                    "error": "AWS credentials incomplete in secrets configuration"
                }
                return
            
            session = boto3.Session(
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name='us-east-1'  # Pricing API is only available in us-east-1
            )
            
            self.pricing_client = session.client('pricing', region_name='us-east-1')
            self.ec2_client = session.client('ec2', region_name='us-east-1')
            
            # Test the connection
            self.pricing_client.describe_services(ServiceCode='AmazonEC2', MaxResults=1)
            
            self.connection_status = {"connected": True, "error": None}
            logger.info("AWS clients initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize AWS clients: {str(e)}"
            logger.error(error_msg)
            self.connection_status = {"connected": False, "error": error_msg}
    
    def get_ec2_pricing_simplified(self, instance_type: str, region: str) -> Optional[float]:
        """Simplified EC2 pricing fetch with fewer filters"""
        try:
            if not self.pricing_client:
                return None
                
            # Use fewer, more flexible filters
            filters = [
                {'Type': 'TERM_MATCH', 'Field': 'instanceType', 'Value': instance_type},
                {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': self._get_location_name(region)},
                {'Type': 'TERM_MATCH', 'Field': 'operatingSystem', 'Value': 'Windows'},
                {'Type': 'TERM_MATCH', 'Field': 'tenancy', 'Value': 'Shared'},
                {'Type': 'TERM_MATCH', 'Field': 'capacitystatus', 'Value': 'Used'}
            ]
            
            response = self.pricing_client.get_products(
                ServiceCode='AmazonEC2',
                Filters=filters,
                MaxResults=10
            )
            
            # Parse pricing from response
            for price_item in response['PriceList']:
                price_data = json.loads(price_item)
                terms = price_data.get('terms', {}).get('OnDemand', {})
                
                for term_key, term_value in terms.items():
                    price_dimensions = term_value.get('priceDimensions', {})
                    for pd_key, pd_value in price_dimensions.items():
                        try:
                            price_per_hour = float(pd_value['pricePerUnit']['USD'])
                            if price_per_hour > 0:  # Valid price found
                                return price_per_hour
                        except (KeyError, ValueError):
                            continue
                            
        except Exception as e:
            logger.error(f"Error fetching EC2 pricing for {instance_type}: {e}")
            return None
        
        return None
    
    def _get_location_name(self, region: str) -> str:
        """Convert AWS region to location name for pricing API"""
        region_mapping = {
            'us-east-1': 'US East (N. Virginia)',
            'us-west-1': 'US West (N. California)',
            'us-west-2': 'US West (Oregon)',
            'us-east-2': 'US East (Ohio)',
            'eu-west-1': 'Europe (Ireland)',
            'eu-central-1': 'Europe (Frankfurt)',
            'ap-southeast-1': 'Asia Pacific (Singapore)',
            'ap-northeast-1': 'Asia Pacific (Tokyo)',
            'ca-central-1': 'Canada (Central)',
            'eu-west-2': 'Europe (London)',
            'ap-south-1': 'Asia Pacific (Mumbai)',
        }
        return region_mapping.get(region, 'US East (N. Virginia)')

class MockPricingData:
    """Provides mock pricing data when AWS API is unavailable"""
    
    @staticmethod
    def get_sample_pricing_data(region: str) -> List[PricingData]:
        """Generate realistic sample pricing data"""
        # Base prices in USD for us-east-1 (adjust for other regions)
        region_multiplier = {
            'us-east-1': 1.0,
            'us-west-1': 1.1,
            'us-west-2': 1.05,
            'us-east-2': 0.95,
            'eu-west-1': 1.15,
            'eu-central-1': 1.12,
            'ap-southeast-1': 1.18,
            'ap-northeast-1': 1.20,
        }.get(region, 1.0)
        
        base_pricing = [
            # Instance Type, Base Hourly Price, vCPUs, RAM
            ('m5.large', 0.192, 2, 8),
            ('m5.xlarge', 0.384, 4, 16),
            ('m5.2xlarge', 0.768, 8, 32),
            ('m5.4xlarge', 1.536, 16, 64),
            ('r5.large', 0.252, 2, 16),
            ('r5.xlarge', 0.504, 4, 32),
            ('r5.2xlarge', 1.008, 8, 64),
            ('r5.4xlarge', 2.016, 16, 128),
            ('m6a.large', 0.173, 2, 8),
            ('m6a.xlarge', 0.346, 4, 16),
            ('m6a.2xlarge', 0.691, 8, 32),
            ('r6a.xlarge', 0.453, 4, 32),
            ('r6a.2xlarge', 0.907, 8, 64),
            ('c5.xlarge', 0.340, 4, 8),
            ('c5.2xlarge', 0.680, 8, 16),
        ]
        
        pricing_data = []
        for instance_type, base_price, vcpus, ram in base_pricing:
            # Add SQL Server licensing cost (approximately 3x base Windows cost)
            sql_server_multiplier = 4.0
            adjusted_price = base_price * sql_server_multiplier * region_multiplier
            
            pricing_data.append(PricingData(
                service="EC2",
                instance_type=instance_type,
                region=region,
                price_per_hour=adjusted_price,
                price_per_month=adjusted_price * 730,
                currency="USD",
                last_updated=datetime.now(),
                specifications={'vcpus': vcpus, 'ram': ram}
            ))
        
        return sorted(pricing_data, key=lambda x: x.price_per_month)

class ClaudeAIIntegration:
    """Updated Claude AI integration with current API format"""
    
    def __init__(self):
        self.api_key = st.secrets.get("CLAUDE_API_KEY") if hasattr(st, 'secrets') else None
        self.base_url = "https://api.anthropic.com/v1/messages"
        
    async def get_optimization_recommendations(self, 
                                             workload_data: Dict, 
                                             pricing_data: List[PricingData]) -> AIRecommendation:
        """Get AI-powered optimization recommendations using updated API"""
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
                "model": "claude-3-haiku-20240307",  # Using Haiku for faster responses
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=headers, json=payload, timeout=30) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result['content'][0]['text']
                        return self._parse_ai_response(content)
                    else:
                        logger.error(f"Claude API error: {response.status}")
                        error_text = await response.text()
                        logger.error(f"Error details: {error_text}")
                        return self._fallback_recommendation()
                        
        except Exception as e:
            logger.error(f"Error getting AI recommendations: {e}")
            return self._fallback_recommendation()
    
    def _build_optimization_prompt(self, workload_data: Dict, pricing_data: List[PricingData]) -> str:
        """Build optimized prompt for Claude AI"""
        return f"""
        As a cloud cost optimization expert, analyze this SQL Server workload configuration and AWS pricing data to provide actionable recommendations.

        WORKLOAD REQUIREMENTS:
        - CPU Cores Required: {workload_data.get('cpu_cores', 'N/A')}
        - RAM Required: {workload_data.get('ram_gb', 'N/A')} GB
        - Storage: {workload_data.get('storage_gb', 'N/A')} GB
        - Peak CPU Utilization: {workload_data.get('peak_cpu', 'N/A')}%
        - Peak RAM Utilization: {workload_data.get('peak_ram', 'N/A')}%
        - Environment: {workload_data.get('workload_type', 'N/A')}
        - Region: {workload_data.get('region', 'N/A')}
        - SQL Server Edition: {workload_data.get('sql_edition', 'N/A')}
        - Licensing: {workload_data.get('licensing_model', 'N/A')}

        TOP 5 PRICING OPTIONS:
        {self._format_pricing_for_prompt(pricing_data[:5])}

        Please provide:
        1. **Recommended Instance**: Best fit instance type with justification
        2. **Cost Optimization**: Specific cost-saving opportunities
        3. **Performance Considerations**: Any performance trade-offs
        4. **Risk Assessment**: Potential risks and mitigation strategies

        Respond in this exact JSON format:
        {{
            "recommendation": "Detailed recommendation with specific instance type and reasoning",
            "confidence_score": 85,
            "cost_impact": "High/Medium/Low",
            "reasoning": "Detailed technical justification for the recommendation"
        }}
        """
    
    def _format_pricing_for_prompt(self, pricing_data: List[PricingData]) -> str:
        """Format pricing data for AI prompt"""
        formatted = []
        for i, pd in enumerate(pricing_data, 1):
            specs = pd.specifications or {}
            formatted.append(
                f"{i}. {pd.instance_type} - {specs.get('vcpus', '?')} vCPUs, "
                f"{specs.get('ram', '?')} GB RAM - ${pd.price_per_hour:.3f}/hour (${pd.price_per_month:.0f}/month)"
            )
        return "\n".join(formatted)
    
    def _parse_ai_response(self, content: str) -> AIRecommendation:
        """Parse Claude AI response with better error handling"""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return AIRecommendation(
                    recommendation=data.get('recommendation', content[:500]),
                    confidence_score=min(100, max(0, data.get('confidence_score', 75))),
                    cost_impact=data.get('cost_impact', 'Medium'),
                    reasoning=data.get('reasoning', 'AI analysis based on workload requirements')
                )
        except Exception as e:
            logger.warning(f"Failed to parse AI response as JSON: {e}")
        
        # Fallback to text parsing
        return AIRecommendation(
            recommendation=content[:500] if content else "Review instance sizing based on utilization patterns",
            confidence_score=70,
            cost_impact="Medium",
            reasoning="Analysis based on standard optimization principles"
        )
    
    def _fallback_recommendation(self) -> AIRecommendation:
        """Enhanced fallback recommendation"""
        return AIRecommendation(
            recommendation="""Based on standard optimization practices:
            1. Right-size instances based on actual CPU/RAM utilization
            2. Consider BYOL if you have existing SQL Server licenses
            3. Use Reserved Instances for predictable workloads (up to 72% savings)
            4. Monitor and adjust instance types quarterly
            5. Consider newer generation instances (m6a, r6a) for better price/performance""",
            confidence_score=65,
            cost_impact="Medium to High",
            reasoning="Standard cloud optimization best practices applied when AI analysis is unavailable"
        )

class CloudPricingOptimizer:
    """Enhanced main application class with better error handling"""
    
    def __init__(self):
        self.aws_pricing = AWSPricingFetcher()
        self.claude_ai = ClaudeAIIntegration()
        self.mock_data = MockPricingData()
        self._initialize_session_state()
        
    def _initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'pricing_cache' not in st.session_state:
            st.session_state.pricing_cache = {}
        if 'last_analysis' not in st.session_state:
            st.session_state.last_analysis = None
        if 'demo_mode' not in st.session_state:
            st.session_state.demo_mode = not self.aws_pricing.connection_status["connected"]
    
    def render_main_interface(self):
        """Render the main Streamlit interface"""
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>☁️ AWS Cloud Pricing Optimizer</h1>
            <p>Professional-grade AWS pricing analysis with AI-powered optimization recommendations</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Check connection status and show warnings
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
        """Enhanced connection status display"""
        col1, col2 = st.columns(2)
        
        with col1:
            if self.aws_pricing.connection_status["connected"]:
                st.markdown('<span class="status-badge status-success">AWS: ✅ Connected</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-badge status-error">AWS: ❌ Demo Mode</span>', unsafe_allow_html=True)
                
                # Show error details in expander
                with st.expander("ℹ️ AWS Connection Details"):
                    st.markdown(f"""
                    <div class="error-details">
                    <strong>Status:</strong> {self.aws_pricing.connection_status["error"]}<br>
                    <strong>Impact:</strong> Using realistic sample pricing data<br>
                    <strong>Solution:</strong> Configure AWS credentials in Streamlit secrets
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            claude_status = "✅ Connected" if self.claude_ai.api_key else "❌ Unavailable"
            claude_class = "status-success" if self.claude_ai.api_key else "status-warning"
            st.markdown(f'<span class="status-badge {claude_class}">Claude AI: {claude_status}</span>', unsafe_allow_html=True)
            
            if not self.claude_ai.api_key:
                with st.expander("ℹ️ Claude AI Details"):
                    st.markdown("""
                    <div class="error-details">
                    <strong>Status:</strong> API key not configured<br>
                    <strong>Impact:</strong> Using rule-based recommendations<br>
                    <strong>Solution:</strong> Add CLAUDE_API_KEY to Streamlit secrets
                    </div>
                    """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar configuration"""
        with st.sidebar:
            st.markdown('<div class="section-header">⚙️ Configuration</div>', unsafe_allow_html=True)
            
            # Demo mode indicator
            if st.session_state.demo_mode:
                st.markdown("""
                <div style="background: #fff3cd; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                    <strong>🚀 Demo Mode</strong><br>
                    Using sample data for demonstration
                </div>
                """, unsafe_allow_html=True)
            
            # Workload Configuration
            st.markdown('<div class="section-header">📋 Workload Parameters</div>', unsafe_allow_html=True)
            
            region = st.selectbox("AWS Region", [
                "us-east-1", "us-west-1", "us-west-2", "us-east-2",
                "eu-west-1", "eu-central-1", "ap-southeast-1", "ap-northeast-1"
            ], index=0, key="region_selector")
            
            workload_type = st.selectbox("Workload Type", [
                "Production", "Staging", "Development", "Testing"
            ], key="workload_type_selector")
            
            st.markdown("**System Requirements**")
            cpu_cores = st.slider("CPU Cores", 2, 64, 8, key="cpu_cores_slider")
            ram_gb = st.slider("RAM (GB)", 4, 512, 32, key="ram_gb_slider")
            storage_gb = st.slider("Storage (GB)", 100, 10000, 500, key="storage_gb_slider")
            
            st.markdown("**Performance Metrics**")
            peak_cpu = st.slider("Peak CPU Utilization (%)", 20, 100, 70, key="peak_cpu_slider")
            peak_ram = st.slider("Peak RAM Utilization (%)", 20, 100, 80, key="peak_ram_slider")
            
            st.markdown("**SQL Server Configuration**")
            sql_edition = st.selectbox("SQL Server Edition", [
                "Standard", "Enterprise", "Developer"
            ], key="sql_edition_selector")
            
            licensing_model = st.selectbox("Licensing Model", [
                "License Included", "BYOL (Bring Your Own License)"
            ], key="licensing_model_selector")
            
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
        """Enhanced pricing analysis with better error handling"""
        st.markdown('<div class="section-header">💰 Real-Time AWS Pricing Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.session_state.demo_mode:
                st.write("📊 Displaying realistic sample pricing data for demonstration. Configure AWS credentials for live data.")
            else:
                st.write("Get the latest AWS EC2 pricing for SQL Server instances based on your configuration.")
        
        with col2:
            if st.button("🔄 Fetch Latest Prices", type="primary", use_container_width=True, key="fetch_prices_btn"):
                with st.spinner("Fetching pricing data..."):
                    pricing_data = self.fetch_pricing_data()
                    
                    if pricing_data:
                        st.session_state.latest_pricing = pricing_data
                        if st.session_state.demo_mode:
                            st.info(f"📊 Generated {len(pricing_data)} sample pricing options")
                        else:
                            st.success(f"✅ Fetched {len(pricing_data)} live pricing options")
                        self.display_pricing_results(pricing_data)
                    else:
                        st.error("❌ Failed to fetch pricing data")
        
        # Display cached results if available
        if hasattr(st.session_state, 'latest_pricing'):
            self.display_pricing_results(st.session_state.latest_pricing)
    
    def fetch_pricing_data(self) -> List[PricingData]:
        """Enhanced pricing data fetch with fallback"""
        config = st.session_state.config
        
        # Use mock data if in demo mode or AWS unavailable
        if st.session_state.demo_mode or not self.aws_pricing.connection_status["connected"]:
            return self.mock_data.get_sample_pricing_data(config['region'])
        
        # Try to fetch real AWS data
        pricing_data = []
        instance_types = [
            'm5.large', 'm5.xlarge', 'm5.2xlarge', 'm5.4xlarge',
            'r5.large', 'r5.xlarge', 'r5.2xlarge', 'r5.4xlarge',
            'm6a.large', 'm6a.xlarge', 'm6a.2xlarge',
            'r6a.xlarge', 'r6a.2xlarge', 'c5.xlarge', 'c5.2xlarge'
        ]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        successful_fetches = 0
        
        for i, instance_type in enumerate(instance_types):
            progress_bar.progress((i + 1) / len(instance_types))
            status_text.text(f"Fetching pricing for {instance_type}...")
            
            price_per_hour = self.aws_pricing.get_ec2_pricing_simplified(
                instance_type, config['region']
            )
            
            if price_per_hour:
                successful_fetches += 1
                specs = self.get_instance_specs(instance_type)
                pricing_data.append(PricingData(
                    service="EC2",
                    instance_type=instance_type,
                    region=config['region'],
                    price_per_hour=price_per_hour,
                    price_per_month=price_per_hour * 730,
                    currency="USD",
                    last_updated=datetime.now(),
                    specifications=specs
                ))
            
            # Add small delay to avoid rate limiting
            time.sleep(0.1)
        
        progress_bar.empty()
        status_text.empty()
        
        # If we got less than 3 successful fetches, fall back to mock data
        if successful_fetches < 3:
            st.warning("⚠️ Limited live data available. Supplementing with sample data.")
            return self.mock_data.get_sample_pricing_data(config['region'])
        
        return sorted(pricing_data, key=lambda x: x.price_per_month)
    
    def get_instance_specs(self, instance_type: str) -> Dict:
        """Enhanced instance specifications mapping"""
        specs_mapping = {
            # M5 instances
            'm5.large': {'vcpus': 2, 'ram': 8, 'network': 'Up to 10 Gbps'},
            'm5.xlarge': {'vcpus': 4, 'ram': 16, 'network': 'Up to 10 Gbps'},
            'm5.2xlarge': {'vcpus': 8, 'ram': 32, 'network': 'Up to 10 Gbps'},
            'm5.4xlarge': {'vcpus': 16, 'ram': 64, 'network': 'Up to 10 Gbps'},
            
            # R5 instances (memory optimized)
            'r5.large': {'vcpus': 2, 'ram': 16, 'network': 'Up to 10 Gbps'},
            'r5.xlarge': {'vcpus': 4, 'ram': 32, 'network': 'Up to 10 Gbps'},
            'r5.2xlarge': {'vcpus': 8, 'ram': 64, 'network': 'Up to 10 Gbps'},
            'r5.4xlarge': {'vcpus': 16, 'ram': 128, 'network': 'Up to 10 Gbps'},
            
            # M6a instances (AMD)
            'm6a.large': {'vcpus': 2, 'ram': 8, 'network': 'Up to 12.5 Gbps'},
            'm6a.xlarge': {'vcpus': 4, 'ram': 16, 'network': 'Up to 12.5 Gbps'},
            'm6a.2xlarge': {'vcpus': 8, 'ram': 32, 'network': 'Up to 12.5 Gbps'},
            
            # R6a instances (AMD memory optimized)
            'r6a.xlarge': {'vcpus': 4, 'ram': 32, 'network': 'Up to 12.5 Gbps'},
            'r6a.2xlarge': {'vcpus': 8, 'ram': 64, 'network': 'Up to 12.5 Gbps'},
            
            # C5 instances (compute optimized)
            'c5.xlarge': {'vcpus': 4, 'ram': 8, 'network': 'Up to 10 Gbps'},
            'c5.2xlarge': {'vcpus': 8, 'ram': 16, 'network': 'Up to 10 Gbps'},
        }
        return specs_mapping.get(instance_type, {'vcpus': 0, 'ram': 0, 'network': 'Unknown'})
    
    def display_pricing_results(self, pricing_data: List[PricingData]):
        """Enhanced pricing results display"""
        if not pricing_data:
            st.warning("⚠️ No pricing data available")
            return
        
        # Create DataFrame for display
        df = pd.DataFrame([{
            'Instance Type': pd.instance_type,
            'vCPUs': pd.specifications.get('vcpus', 'N/A') if pd.specifications else 'N/A',
            'RAM (GB)': pd.specifications.get('ram', 'N/A') if pd.specifications else 'N/A',
            'Hourly Cost': f"${pd.price_per_hour:.3f}",
            'Monthly Cost': f"${pd.price_per_month:.2f}",
            'Annual Cost': f"${pd.price_per_month * 12:,.0f}",
            'Cost per vCPU/month': f"${pd.price_per_month / pd.specifications.get('vcpus', 1):.2f}" if pd.specifications else 'N/A'
        } for pd in pricing_data])
        
        st.markdown('<div class="section-header">💸 Instance Pricing Comparison</div>', unsafe_allow_html=True)
        
        # Show summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
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
            most_expensive = max(pricing_data, key=lambda x: x.price_per_month)
            st.markdown(f"""
            <div class="metric-card">
                <h3>Highest Performance</h3>
                <h2>{most_expensive.instance_type}</h2>
                <p>${most_expensive.price_per_month:.2f}/month</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Options Found</h3>
                <h2>{len(pricing_data)}</h2>
                <p>Instance types analyzed</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Pricing table with better formatting
        st.dataframe(
            df, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "Instance Type": st.column_config.TextColumn(width="medium"),
                "vCPUs": st.column_config.NumberColumn(format="%d"),
                "RAM (GB)": st.column_config.NumberColumn(format="%d"),
            }
        )
        
        # Create enhanced pricing visualization
        fig = px.bar(
            x=[pd.instance_type for pd in pricing_data[:10]],
            y=[pd.price_per_month for pd in pricing_data[:10]],
            title="Monthly Pricing Comparison (Top 10 Most Economical Options)",
            labels={'x': 'Instance Type', 'y': 'Monthly Cost ($)'},
            color=[pd.price_per_month for pd in pricing_data[:10]],
            color_continuous_scale='Viridis',
            text=[f"${pd.price_per_month:.0f}" for pd in pricing_data[:10]]
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            xaxis_tickangle=-45,
            plot_bgcolor='white',
            font_color='#343a40',
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True, key="pricing_comparison_chart")
    
    def render_ai_recommendations(self):
        """Enhanced AI recommendations with better fallback"""
        st.markdown('<div class="section-header">🤖 Intelligent Optimization Recommendations</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if self.claude_ai.api_key:
                st.write("🧠 Get Claude AI-powered optimization recommendations based on your workload requirements and current AWS pricing.")
            else:
                st.write("📋 Get intelligent optimization recommendations based on industry best practices and cost optimization principles.")
        
        with col2:
            if st.button("🧠 Get AI Analysis", type="primary", use_container_width=True, key="ai_analysis_btn"):
                if hasattr(st.session_state, 'latest_pricing'):
                    with st.spinner("Analyzing with AI..."):
                        if self.claude_ai.api_key:
                            recommendation = asyncio.run(
                                self.claude_ai.get_optimization_recommendations(
                                    st.session_state.config,
                                    st.session_state.latest_pricing
                                )
                            )
                        else:
                            # Use enhanced fallback recommendation
                            recommendation = self.generate_rule_based_recommendation()
                        
                        st.session_state.current_recommendation = recommendation
                        st.success("✅ Analysis complete!")
                else:
                    st.error("❌ Please fetch pricing data first!")
        
        # Display cached recommendation if available
        if hasattr(st.session_state, 'current_recommendation'):
            self.display_ai_recommendations(st.session_state.current_recommendation)
        else:
            # Show helpful message when no analysis has been run
            st.info("👆 Click 'Get AI Analysis' above to generate optimization recommendations based on your configuration.")
            
            # Show current configuration summary
            if hasattr(st.session_state, 'config'):
                config = st.session_state.config
                st.markdown("**Current Configuration Summary:**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    **System Requirements:**
                    - CPU: {config.get('cpu_cores', 'N/A')} cores
                    - RAM: {config.get('ram_gb', 'N/A')} GB
                    - Storage: {config.get('storage_gb', 'N/A')} GB
                    """)
                
                with col2:
                    st.markdown(f"""
                    **Performance:**
                    - Peak CPU: {config.get('peak_cpu', 'N/A')}%
                    - Peak RAM: {config.get('peak_ram', 'N/A')}%
                    - Type: {config.get('workload_type', 'N/A')}
                    """)
                
                with col3:
                    st.markdown(f"""
                    **SQL Server:**
                    - Edition: {config.get('sql_edition', 'N/A')}
                    - Licensing: {config.get('licensing_model', 'N/A')}
                    - Region: {config.get('region', 'N/A')}
                    """)
    
    def generate_rule_based_recommendation(self) -> AIRecommendation:
        """Generate intelligent rule-based recommendation when AI is unavailable"""
        config = st.session_state.config
        pricing_data = st.session_state.latest_pricing
        
        # Analyze requirements
        cpu_req = config.get('cpu_cores', 4)
        ram_req = config.get('ram_gb', 16)
        peak_cpu = config.get('peak_cpu', 70)
        peak_ram = config.get('peak_ram', 80)
        workload_type = config.get('workload_type', 'Production')
        
        # Find suitable instances
        suitable_instances = []
        for pd in pricing_data:
            specs = pd.specifications or {}
            if specs.get('vcpus', 0) >= cpu_req and specs.get('ram', 0) >= ram_req:
                suitable_instances.append(pd)
        
        if not suitable_instances:
            return AIRecommendation(
                recommendation="No instances meet your minimum requirements. Consider reducing requirements or checking larger instance types.",
                confidence_score=90,
                cost_impact="High",
                reasoning="Requirements exceed available instance specifications"
            )
        
        # Get the most cost-effective suitable instance
        recommended = min(suitable_instances, key=lambda x: x.price_per_month)
        
        # Build recommendation
        recommendations = []
        recommendations.append(f"**Recommended Instance: {recommended.instance_type}**")
        recommendations.append(f"- Monthly cost: ${recommended.price_per_month:.2f}")
        recommendations.append(f"- Specifications: {recommended.specifications.get('vcpus')} vCPUs, {recommended.specifications.get('ram')} GB RAM")
        
        # Add optimization suggestions
        if peak_cpu < 60:
            recommendations.append("\n**CPU Optimization:** Your peak CPU utilization is low. Consider smaller instances or consolidating workloads.")
        
        if peak_ram < 70:
            recommendations.append("**Memory Optimization:** RAM utilization suggests you might benefit from compute-optimized instances instead of memory-optimized.")
        
        if workload_type in ['Development', 'Testing']:
            recommendations.append("**Environment-based Savings:** Consider using Spot instances for development/testing (up to 90% savings).")
        
        if config.get('licensing_model') == 'License Included':
            recommendations.append("**Licensing Optimization:** If you have existing SQL Server licenses, switch to BYOL for significant savings.")
        
        recommendations.append("\n**Additional Optimizations:**")
        recommendations.append("- Use Reserved Instances for predictable workloads (up to 72% savings)")
        recommendations.append("- Consider newer generation instances (m6a, r6a) for better price/performance")
        recommendations.append("- Implement automated start/stop schedules for non-production environments")
        
        cost_impact = "Medium"
        if len([r for r in recommendations if "savings" in r.lower()]) >= 2:
            cost_impact = "High"
        
        return AIRecommendation(
            recommendation="\n".join(recommendations),
            confidence_score=85,
            cost_impact=cost_impact,
            reasoning="Analysis based on workload requirements, utilization patterns, and AWS best practices for cost optimization"
        )
    
    def display_ai_recommendations(self, recommendation: AIRecommendation):
        """Enhanced AI recommendations display"""
        st.markdown(f"""
        <div class="ai-recommendation">
            <h3>🎯 Optimization Analysis</h3>
            <p><strong>Confidence Score:</strong> {recommendation.confidence_score}%</p>
            <p><strong>Expected Cost Impact:</strong> {recommendation.cost_impact}</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**📋 Recommendations**")
            st.markdown(recommendation.recommendation)
            
            st.markdown("**🧠 Analysis Details**")
            st.markdown(recommendation.reasoning)
        
        with col2:
            # Enhanced confidence meter
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = recommendation.confidence_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Confidence Score", 'font': {'color': '#1f4e79', 'size': 16}},
                delta = {'reference': 80, 'position': "top"},
                gauge = {
                    'axis': {'range': [None, 100], 'tickcolor': "#1f4e79"},
                    'bar': {'color': "#4a90e2", 'thickness': 0.8},
                    'steps': [
                        {'range': [0, 50], 'color': "#f8f9fa"},
                        {'range': [50, 70], 'color': "#fff3cd"},
                        {'range': [70, 85], 'color': "#d1ecf1"},
                        {'range': [85, 100], 'color': "#d4edda"}
                    ],
                    'threshold': {
                        'line': {'color': "#28a745", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=350, font_color='#343a40', margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True, key="confidence_gauge_chart")
            
            # Cost impact indicator
            impact_colors = {
                "Low": "#28a745",
                "Medium": "#ffc107", 
                "High": "#dc3545"
            }
            st.markdown(f"""
            <div style="text-align: center; margin-top: 1rem;">
                <div style="background: {impact_colors.get(recommendation.cost_impact, '#6c757d')}; 
                           color: white; padding: 0.5rem; border-radius: 8px; font-weight: bold;">
                    💰 {recommendation.cost_impact} Cost Impact
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def render_cost_comparison(self):
        """Enhanced cost comparison with more detailed analysis"""
        st.markdown('<div class="section-header">📊 Cost Comparison & ROI Analysis</div>', unsafe_allow_html=True)
        
        if not hasattr(st.session_state, 'latest_pricing'):
            st.warning("⚠️ Please fetch pricing data first to see cost comparison.")
            return
            
        pricing_data = st.session_state.latest_pricing
        
        # Get current configuration
        config = st.session_state.config
        cpu_req = config.get('cpu_cores', 4)
        ram_req = config.get('ram_gb', 16)
        
        # Find current suitable instance (mid-range option)
        suitable_instances = [pd for pd in pricing_data 
                            if pd.specifications and 
                            pd.specifications.get('vcpus', 0) >= cpu_req and 
                            pd.specifications.get('ram', 0) >= ram_req]
        
        if not suitable_instances:
            st.error("❌ No instances meet your minimum requirements.")
            return
        
        current_instance = suitable_instances[len(suitable_instances)//2] if len(suitable_instances) > 2 else suitable_instances[0]
        optimized_instance = suitable_instances[0]  # Most economical
        premium_instance = suitable_instances[-1] if len(suitable_instances) > 1 else current_instance
        
        # Cost comparison
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Current Option</h3>
                <h2>{current_instance.instance_type}</h2>
                <p>${current_instance.price_per_month:.2f}/month</p>
                <small>{current_instance.specifications.get('vcpus', 'N/A')} vCPUs, {current_instance.specifications.get('ram', 'N/A')} GB RAM</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Optimized Option</h3>
                <h2>{optimized_instance.instance_type}</h2>
                <p>${optimized_instance.price_per_month:.2f}/month</p>
                <small>{optimized_instance.specifications.get('vcpus', 'N/A')} vCPUs, {optimized_instance.specifications.get('ram', 'N/A')} GB RAM</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            savings = current_instance.price_per_month - optimized_instance.price_per_month
            savings_pct = (savings / current_instance.price_per_month) * 100 if current_instance.price_per_month > 0 else 0
            st.markdown(f"""
            <div class="cost-savings">
                <h3>Monthly Savings</h3>
                <h2>${savings:.2f}</h2>
                <p>{savings_pct:.1f}% cost reduction</p>
            </div>
            """, unsafe_allow_html=True)
        
        # ROI Calculator
        st.markdown('<div class="section-header">💹 Advanced ROI Calculator</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            migration_cost = st.number_input("Migration & Setup Cost ($)", value=25000, step=1000, 
                                           help="One-time cost for migration and setup", key="migration_cost_input")
            
        with col2:
            operational_overhead = st.number_input("Monthly Operational Overhead ($)", value=200, step=50, 
                                                 help="Additional monthly operational costs", key="operational_overhead_input")
            
        with col3:
            reservation_discount = st.selectbox("Reserved Instance Discount", 
                                              ["No Reservation (0%)", "1-Year Term (40%)", "3-Year Term (60%)"],
                                              help="Additional savings from Reserved Instances", key="reservation_discount_selector")
        
        # Calculate reservation discount
        discount_map = {"No Reservation (0%)": 0, "1-Year Term (40%)": 0.4, "3-Year Term (60%)": 0.6}
        discount_rate = discount_map[reservation_discount]
        
        # Apply reservation discount to optimized cost
        optimized_cost_with_reservation = optimized_instance.price_per_month * (1 - discount_rate)
        total_monthly_savings = current_instance.price_per_month - optimized_cost_with_reservation - operational_overhead
        
        if total_monthly_savings > 0:
            payback_months = migration_cost / total_monthly_savings
            annual_savings = total_monthly_savings * 12
            three_year_total_savings = annual_savings * 3 - migration_cost
            three_year_roi = (three_year_total_savings / migration_cost) * 100 if migration_cost > 0 else 0
            
            # Display results
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Payback Period", f"{payback_months:.1f} months", help="Time to recover migration costs")
            with col2:
                st.metric("Net Monthly Savings", f"${total_monthly_savings:,.0f}", help="Monthly savings after all costs")
            with col3:
                st.metric("Annual Savings", f"${annual_savings:,.0f}", help="Total annual cost savings")
            with col4:
                roi_color = "normal" if three_year_roi < 100 else "inverse"
                st.metric("3-Year ROI", f"{three_year_roi:.1f}%", help="Return on investment over 3 years")
            
            # ROI Timeline Chart
            months = list(range(0, 37))  # 3 years
            cumulative_savings = []
            
            for month in months:
                if month == 0:
                    cumulative_savings.append(-migration_cost)
                else:
                    previous_savings = cumulative_savings[-1]
                    cumulative_savings.append(previous_savings + total_monthly_savings)
            
            fig = px.line(
                x=months, 
                y=cumulative_savings,
                title="Cumulative Savings Over Time",
                labels={'x': 'Months', 'y': 'Cumulative Savings ($)'},
                color_discrete_sequence=['#28a745']
            )
            
            # Add break-even line
            fig.add_hline(y=0, line_dash="dash", line_color="red", 
                         annotation_text="Break-even point")
            
            fig.update_layout(plot_bgcolor='white', font_color='#343a40')
            st.plotly_chart(fig, use_container_width=True, key="roi_timeline_chart")
            
        else:
            st.warning("⚠️ Current configuration may not provide positive ROI with the selected parameters.")
            st.info("💡 Try reducing migration costs or operational overhead, or consider Reserved Instances for better savings.")
        
        # Scenario Analysis
        st.markdown('<div class="section-header">🎯 Scenario Analysis</div>', unsafe_allow_html=True)
        
        scenarios_df = pd.DataFrame({
            'Scenario': ['Current Setup', 'Optimized Instance', 'Optimized + 1-Year RI', 'Optimized + 3-Year RI'],
            'Instance Type': [current_instance.instance_type, optimized_instance.instance_type, 
                            optimized_instance.instance_type, optimized_instance.instance_type],
            'Monthly Cost': [f"${current_instance.price_per_month:.2f}",
                           f"${optimized_instance.price_per_month:.2f}",
                           f"${optimized_instance.price_per_month * 0.6:.2f}",
                           f"${optimized_instance.price_per_month * 0.4:.2f}"],
            'Annual Cost': [f"${current_instance.price_per_month * 12:,.0f}",
                          f"${optimized_instance.price_per_month * 12:,.0f}",
                          f"${optimized_instance.price_per_month * 0.6 * 12:,.0f}",
                          f"${optimized_instance.price_per_month * 0.4 * 12:,.0f}"],
            'Annual Savings': ["$0",
                             f"${(current_instance.price_per_month - optimized_instance.price_per_month) * 12:,.0f}",
                             f"${(current_instance.price_per_month - optimized_instance.price_per_month * 0.6) * 12:,.0f}",
                             f"${(current_instance.price_per_month - optimized_instance.price_per_month * 0.4) * 12:,.0f}"]
        })
        
        st.dataframe(scenarios_df, use_container_width=True, hide_index=True)
    
    def render_trends_reports(self):
        """Enhanced trends and reporting section"""
        st.markdown('<div class="section-header">📈 Pricing Trends & Advanced Reports</div>', unsafe_allow_html=True)
        
        # Generate enhanced trend data
        dates = pd.date_range(start='2024-01-01', end='2025-01-01', freq='M')
        
        # More realistic pricing trends with seasonality
        try:
            base_trend = pd.DataFrame({
                'Date': dates,
                'm5.xlarge': [190 + i*1.5 + 10*np.sin(i*0.5) + np.random.normal(0, 5) for i in range(len(dates))],
                'r5.xlarge': [240 + i*2 + 15*np.sin(i*0.6) + np.random.normal(0, 8) for i in range(len(dates))],
                'm6a.xlarge': [175 + i*1.2 + 8*np.sin(i*0.4) + np.random.normal(0, 4) for i in range(len(dates))],
                'r6a.xlarge': [220 + i*1.8 + 12*np.sin(i*0.55) + np.random.normal(0, 6) for i in range(len(dates))]
            })
            
            # Ensure positive values
            for col in base_trend.columns[1:]:
                base_trend[col] = base_trend[col].clip(lower=50)
        except Exception as e:
            # Fallback to simple trend without randomness
            base_trend = pd.DataFrame({
                'Date': dates,
                'm5.xlarge': [190 + i*1.5 for i in range(len(dates))],
                'r5.xlarge': [240 + i*2 for i in range(len(dates))],
                'm6a.xlarge': [175 + i*1.2 for i in range(len(dates))],
                'r6a.xlarge': [220 + i*1.8 for i in range(len(dates))]
            })
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.line(base_trend, x='Date', y=['m5.xlarge', 'r5.xlarge', 'm6a.xlarge', 'r6a.xlarge'],
                         title='AWS Instance Pricing Trends (Monthly)',
                         color_discrete_sequence=['#4a90e2', '#28a745', '#ffc107', '#dc3545'])
            fig.update_layout(
                plot_bgcolor='white', 
                font_color='#343a40',
                yaxis_title="Monthly Cost ($)",
                legend_title="Instance Types"
            )
            st.plotly_chart(fig, use_container_width=True, key="pricing_trends_chart")
        
        with col2:
            # Trend summary
            latest_prices = base_trend.iloc[-1]
            six_months_ago = base_trend.iloc[-7] if len(base_trend) >= 7 else base_trend.iloc[0]
            
            st.markdown("**📊 Trend Summary (6 months)**")
            for instance in ['m5.xlarge', 'r5.xlarge', 'm6a.xlarge']:
                change = latest_prices[instance] - six_months_ago[instance]
                change_pct = (change / six_months_ago[instance]) * 100
                direction = "📈" if change > 0 else "📉"
                st.metric(instance, f"${latest_prices[instance]:.2f}", f"{direction} {change_pct:+.1f}%")
        
        # Price volatility analysis
        st.markdown('<div class="section-header">📊 Price Volatility Analysis</div>', unsafe_allow_html=True)
        
        volatility_data = []
        for instance in ['m5.xlarge', 'r5.xlarge', 'm6a.xlarge', 'r6a.xlarge']:
            prices = base_trend[instance]
            volatility = prices.std() / prices.mean() * 100
            volatility_data.append({
                'Instance Type': instance,
                'Average Price': f"${prices.mean():.2f}",
                'Price Volatility': f"{volatility:.1f}%",
                'Min Price': f"${prices.min():.2f}",
                'Max Price': f"${prices.max():.2f}"
            })
        
        volatility_df = pd.DataFrame(volatility_data)
        st.dataframe(volatility_df, use_container_width=True, hide_index=True)
        
        # Export functionality
        st.markdown('<div class="section-header">📊 Export & Download Options</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📄 Pricing Report", use_container_width=True, key="export_pricing"):
                if hasattr(st.session_state, 'latest_pricing'):
                    csv_data = self.export_pricing_report()
                    if csv_data:
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_data,
                            file_name=f"aws_pricing_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key="download_pricing"
                        )
                else:
                    st.error("No pricing data to export")
        
        with col2:
            if st.button("📊 Configuration", use_container_width=True, key="export_config"):
                config_data = self.export_configuration()
                st.download_button(
                    label="📥 Download JSON",
                    data=config_data,
                    file_name=f"workload_config_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json",
                    use_container_width=True,
                    key="download_config"
                )
        
        with col3:
            if st.button("📈 Full Analysis", use_container_width=True, key="export_analysis"):
                summary_data = self.create_analysis_summary()
                st.download_button(
                    label="📥 Download Report",
                    data=summary_data,
                    file_name=f"optimization_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json",
                    use_container_width=True,
                    key="download_analysis"
                )
        
        with col4:
            if st.button("📊 Trend Data", use_container_width=True, key="export_trends"):
                trend_csv = base_trend.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=trend_csv,
                    file_name=f"pricing_trends_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_trends"
                )
    
    def export_pricing_report(self) -> str:
        """Enhanced pricing report export"""
        if not hasattr(st.session_state, 'latest_pricing'):
            return ""
            
        config = st.session_state.config
        
        report_data = []
        for pd in st.session_state.latest_pricing:
            specs = pd.specifications or {}
            report_data.append({
                'Instance Type': pd.instance_type,
                'Region': pd.region,
                'vCPUs': specs.get('vcpus', 'N/A'),
                'RAM (GB)': specs.get('ram', 'N/A'),
                'Network Performance': specs.get('network', 'N/A'),
                'Price Per Hour': pd.price_per_hour,
                'Price Per Month': pd.price_per_month,
                'Price Per Year': pd.price_per_month * 12,
                'Cost per vCPU/Month': pd.price_per_month / specs.get('vcpus', 1) if specs.get('vcpus', 0) > 0 else 'N/A',
                'Meets CPU Requirement': 'Yes' if specs.get('vcpus', 0) >= config.get('cpu_cores', 0) else 'No',
                'Meets RAM Requirement': 'Yes' if specs.get('ram', 0) >= config.get('ram_gb', 0) else 'No',
                'Data Source': 'AWS API' if not st.session_state.demo_mode else 'Sample Data',
                'Last Updated': pd.last_updated.strftime('%Y-%m-%d %H:%M:%S')
            })
        
        df = pd.DataFrame(report_data)
        
        # Add summary information as header comments
        summary_info = [
            f"# AWS Cloud Pricing Report",
            f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Region: {config.get('region', 'N/A')}",
            f"# Workload Type: {config.get('workload_type', 'N/A')}",
            f"# CPU Requirement: {config.get('cpu_cores', 'N/A')} cores",
            f"# RAM Requirement: {config.get('ram_gb', 'N/A')} GB",
            f"# Total Options Analyzed: {len(df)}",
            f"#",
        ]
        
        return "\n".join(summary_info) + "\n" + df.to_csv(index=False)
    
    def export_configuration(self) -> str:
        """Enhanced configuration export"""
        config = st.session_state.config if hasattr(st.session_state, 'config') else {}
        
        export_data = {
            'metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'export_version': '2.0',
                'application': 'AWS Cloud Pricing Optimizer',
                'data_source': 'AWS API' if not st.session_state.demo_mode else 'Sample Data'
            },
            'workload_configuration': config,
            'connection_status': {
                'aws_connected': self.aws_pricing.connection_status["connected"],
                'claude_ai_available': self.claude_ai.api_key is not None,
                'demo_mode': st.session_state.demo_mode
            },
            'analysis_summary': {
                'total_options_analyzed': len(st.session_state.latest_pricing) if hasattr(st.session_state, 'latest_pricing') else 0,
                'recommendation_available': hasattr(st.session_state, 'current_recommendation')
            }
        }
        
        return json.dumps(export_data, indent=2)
    
    def create_analysis_summary(self) -> str:
        """Enhanced analysis summary export"""
        config = st.session_state.config if hasattr(st.session_state, 'config') else {}
        pricing_data = st.session_state.latest_pricing if hasattr(st.session_state, 'latest_pricing') else []
        recommendation = st.session_state.current_recommendation if hasattr(st.session_state, 'current_recommendation') else None
        
        # Find suitable instances
        suitable_instances = []
        if pricing_data:
            cpu_req = config.get('cpu_cores', 4)
            ram_req = config.get('ram_gb', 16)
            
            for pd in pricing_data:
                specs = pd.specifications or {}
                if specs.get('vcpus', 0) >= cpu_req and specs.get('ram', 0) >= ram_req:
                    suitable_instances.append(pd)
        
        summary = {
            'analysis_metadata': {
                'generated_at': datetime.now().isoformat(),
                'version': '2.0',
                'aws_region': config.get('region', 'N/A'),
                'data_source': 'AWS API' if not st.session_state.demo_mode else 'Sample Data'
            },
            'workload_requirements': {
                'cpu_cores': config.get('cpu_cores', 'N/A'),
                'ram_gb': config.get('ram_gb', 'N/A'),
                'storage_gb': config.get('storage_gb', 'N/A'),
                'peak_cpu_utilization': f"{config.get('peak_cpu', 'N/A')}%",
                'peak_ram_utilization': f"{config.get('peak_ram', 'N/A')}%",
                'workload_type': config.get('workload_type', 'N/A'),
                'sql_edition': config.get('sql_edition', 'N/A'),
                'licensing_model': config.get('licensing_model', 'N/A')
            },
            'pricing_analysis': {
                'total_options_analyzed': len(pricing_data),
                'suitable_options': len(suitable_instances),
                'cheapest_suitable_option': {
                    'instance_type': suitable_instances[0].instance_type if suitable_instances else 'None',
                    'monthly_cost': suitable_instances[0].price_per_month if suitable_instances else 0,
                    'specifications': suitable_instances[0].specifications if suitable_instances else {}
                },
                'most_expensive_option': {
                    'instance_type': pricing_data[-1].instance_type if pricing_data else 'None',
                    'monthly_cost': pricing_data[-1].price_per_month if pricing_data else 0
                },
                'average_monthly_cost': sum(pd.price_per_month for pd in pricing_data) / len(pricing_data) if pricing_data else 0
            },
            'ai_recommendation': {
                'available': recommendation is not None,
                'confidence_score': recommendation.confidence_score if recommendation else 0,
                'cost_impact': recommendation.cost_impact if recommendation else 'N/A',
                'recommendation_text': recommendation.recommendation if recommendation else 'N/A'
            },
            'service_status': {
                'aws_connected': self.aws_pricing.connection_status["connected"],
                'aws_error': self.aws_pricing.connection_status.get("error"),
                'claude_ai_available': self.claude_ai.api_key is not None,
                'demo_mode': st.session_state.demo_mode
            }
        }
        
        return json.dumps(summary, indent=2)

# Application entry point
def main():
    """Main application entry point"""
    try:
        optimizer = CloudPricingOptimizer()
        optimizer.render_main_interface()
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        st.info("🔧 Please check your configuration and try again.")
        logger.error(f"Application error: {e}")

if __name__ == "__main__":
    main()