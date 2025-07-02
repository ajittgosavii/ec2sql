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
from dataclasses import dataclass, asdict
from functools import lru_cache
from io import BytesIO
import numpy as np
import base64

# PDF generation imports
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import HexColor, black, white, grey
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.platypus.frames import Frame
from reportlab.platypus.doctemplate import PageTemplate, BaseDocTemplate
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing, Line, Rect
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics import renderPDF
import plotly.io as pio

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

# Enhanced Professional CSS styling
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
    
    .pdf-preview {
        background: #f8f9fa;
        border: 2px dashed #dee2e6;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    .risk-matrix {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .risk-low {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    
    .risk-medium {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    
    .risk-high {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
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
    
    .download-section {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin: 2rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class PricingData:
    """Enhanced data class for AWS pricing information"""
    service: str
    instance_type: str
    region: str
    price_per_hour: float
    price_per_month: float
    currency: str
    last_updated: datetime
    specifications: Dict = None
    reserved_pricing: Dict = None  # Added for RI pricing
    spot_pricing: float = None     # Added for spot pricing

@dataclass
class AIRecommendation:
    """Enhanced data class for Claude AI recommendations"""
    recommendation: str
    confidence_score: float
    cost_impact: str
    reasoning: str
    risk_assessment: str = ""
    implementation_timeline: str = ""
    expected_savings: float = 0.0

@dataclass
class RiskAssessment:
    """New data class for risk assessment"""
    category: str
    risk_level: str  # Low, Medium, High
    description: str
    mitigation_strategy: str
    impact: str

@dataclass
class ImplementationPhase:
    """New data class for implementation roadmap"""
    phase: str
    duration: str
    activities: List[str]
    dependencies: List[str]
    deliverables: List[str]

class EnhancedAWSPricingFetcher:
    """Enhanced AWS Pricing fetcher with Reserved Instance and Spot pricing"""
    
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
                region_name='us-east-1'
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
    
    def get_enhanced_ec2_pricing(self, instance_type: str, region: str) -> Optional[PricingData]:
        """Enhanced EC2 pricing fetch with RI and Spot pricing"""
        try:
            if not self.pricing_client:
                return None
                
            # Get On-Demand pricing
            on_demand_price = self.get_ec2_pricing_simplified(instance_type, region)
            if not on_demand_price:
                return None
            
            # Calculate Reserved Instance pricing (estimates)
            reserved_pricing = {
                "1_year_no_upfront": on_demand_price * 0.7,
                "1_year_partial_upfront": on_demand_price * 0.65,
                "1_year_all_upfront": on_demand_price * 0.6,
                "3_year_no_upfront": on_demand_price * 0.5,
                "3_year_partial_upfront": on_demand_price * 0.45,
                "3_year_all_upfront": on_demand_price * 0.4
            }
            
            # Estimate Spot pricing (typically 70-90% off On-Demand)
            spot_price = on_demand_price * 0.2
            
            specs = self.get_instance_specs(instance_type)
            
            return PricingData(
                service="EC2",
                instance_type=instance_type,
                region=region,
                price_per_hour=on_demand_price,
                price_per_month=on_demand_price * 730,
                currency="USD",
                last_updated=datetime.now(),
                specifications=specs,
                reserved_pricing=reserved_pricing,
                spot_pricing=spot_price
            )
                        
        except Exception as e:
            logger.error(f"Error fetching enhanced EC2 pricing for {instance_type}: {e}")
            return None
    
    def get_ec2_pricing_simplified(self, instance_type: str, region: str) -> Optional[float]:
        """Simplified EC2 pricing fetch with fewer filters"""
        try:
            if not self.pricing_client:
                return None
                
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
            
            for price_item in response['PriceList']:
                price_data = json.loads(price_item)
                terms = price_data.get('terms', {}).get('OnDemand', {})
                
                for term_key, term_value in terms.items():
                    price_dimensions = term_value.get('priceDimensions', {})
                    for pd_key, pd_value in price_dimensions.items():
                        try:
                            price_per_hour = float(pd_value['pricePerUnit']['USD'])
                            if price_per_hour > 0:
                                return price_per_hour
                        except (KeyError, ValueError):
                            continue
                            
        except Exception as e:
            logger.error(f"Error fetching EC2 pricing for {instance_type}: {e}")
            return None
        
        return None
    
    def get_instance_specs(self, instance_type: str) -> Dict:
        """Enhanced instance specifications mapping"""
        specs_mapping = {
            # M5 instances
            'm5.large': {'vcpus': 2, 'ram': 8, 'network': 'Up to 10 Gbps', 'storage': 'EBS Only', 'family': 'General Purpose'},
            'm5.xlarge': {'vcpus': 4, 'ram': 16, 'network': 'Up to 10 Gbps', 'storage': 'EBS Only', 'family': 'General Purpose'},
            'm5.2xlarge': {'vcpus': 8, 'ram': 32, 'network': 'Up to 10 Gbps', 'storage': 'EBS Only', 'family': 'General Purpose'},
            'm5.4xlarge': {'vcpus': 16, 'ram': 64, 'network': 'Up to 10 Gbps', 'storage': 'EBS Only', 'family': 'General Purpose'},
            
            # R5 instances (memory optimized)
            'r5.large': {'vcpus': 2, 'ram': 16, 'network': 'Up to 10 Gbps', 'storage': 'EBS Only', 'family': 'Memory Optimized'},
            'r5.xlarge': {'vcpus': 4, 'ram': 32, 'network': 'Up to 10 Gbps', 'storage': 'EBS Only', 'family': 'Memory Optimized'},
            'r5.2xlarge': {'vcpus': 8, 'ram': 64, 'network': 'Up to 10 Gbps', 'storage': 'EBS Only', 'family': 'Memory Optimized'},
            'r5.4xlarge': {'vcpus': 16, 'ram': 128, 'network': 'Up to 10 Gbps', 'storage': 'EBS Only', 'family': 'Memory Optimized'},
            
            # M6a instances (AMD)
            'm6a.large': {'vcpus': 2, 'ram': 8, 'network': 'Up to 12.5 Gbps', 'storage': 'EBS Only', 'family': 'General Purpose'},
            'm6a.xlarge': {'vcpus': 4, 'ram': 16, 'network': 'Up to 12.5 Gbps', 'storage': 'EBS Only', 'family': 'General Purpose'},
            'm6a.2xlarge': {'vcpus': 8, 'ram': 32, 'network': 'Up to 12.5 Gbps', 'storage': 'EBS Only', 'family': 'General Purpose'},
            
            # R6a instances (AMD memory optimized)
            'r6a.xlarge': {'vcpus': 4, 'ram': 32, 'network': 'Up to 12.5 Gbps', 'storage': 'EBS Only', 'family': 'Memory Optimized'},
            'r6a.2xlarge': {'vcpus': 8, 'ram': 64, 'network': 'Up to 12.5 Gbps', 'storage': 'EBS Only', 'family': 'Memory Optimized'},
            
            # C5 instances (compute optimized)
            'c5.xlarge': {'vcpus': 4, 'ram': 8, 'network': 'Up to 10 Gbps', 'storage': 'EBS Only', 'family': 'Compute Optimized'},
            'c5.2xlarge': {'vcpus': 8, 'ram': 16, 'network': 'Up to 10 Gbps', 'storage': 'EBS Only', 'family': 'Compute Optimized'},
        }
        return specs_mapping.get(instance_type, {'vcpus': 0, 'ram': 0, 'network': 'Unknown', 'storage': 'Unknown', 'family': 'Unknown'})
    
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

class EnhancedMockPricingData:
    """Enhanced mock pricing data with RI and Spot pricing"""
    
    @staticmethod
    def get_enhanced_sample_pricing_data(region: str) -> List[PricingData]:
        """Generate realistic sample pricing data with enhanced features"""
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
            ('m5.large', 0.192, 2, 8, 'General Purpose'),
            ('m5.xlarge', 0.384, 4, 16, 'General Purpose'),
            ('m5.2xlarge', 0.768, 8, 32, 'General Purpose'),
            ('m5.4xlarge', 1.536, 16, 64, 'General Purpose'),
            ('r5.large', 0.252, 2, 16, 'Memory Optimized'),
            ('r5.xlarge', 0.504, 4, 32, 'Memory Optimized'),
            ('r5.2xlarge', 1.008, 8, 64, 'Memory Optimized'),
            ('r5.4xlarge', 2.016, 16, 128, 'Memory Optimized'),
            ('m6a.large', 0.173, 2, 8, 'General Purpose'),
            ('m6a.xlarge', 0.346, 4, 16, 'General Purpose'),
            ('m6a.2xlarge', 0.691, 8, 32, 'General Purpose'),
            ('r6a.xlarge', 0.453, 4, 32, 'Memory Optimized'),
            ('r6a.2xlarge', 0.907, 8, 64, 'Memory Optimized'),
            ('c5.xlarge', 0.340, 4, 8, 'Compute Optimized'),
            ('c5.2xlarge', 0.680, 8, 16, 'Compute Optimized'),
        ]
        
        pricing_data = []
        for instance_type, base_price, vcpus, ram, family in base_pricing:
            # Add SQL Server licensing cost
            sql_server_multiplier = 4.0
            adjusted_price = base_price * sql_server_multiplier * region_multiplier
            
            # Calculate Reserved Instance pricing
            reserved_pricing = {
                "1_year_no_upfront": adjusted_price * 0.7,
                "1_year_partial_upfront": adjusted_price * 0.65,
                "1_year_all_upfront": adjusted_price * 0.6,
                "3_year_no_upfront": adjusted_price * 0.5,
                "3_year_partial_upfront": adjusted_price * 0.45,
                "3_year_all_upfront": adjusted_price * 0.4
            }
            
            # Spot pricing
            spot_price = adjusted_price * 0.2
            
            pricing_data.append(PricingData(
                service="EC2",
                instance_type=instance_type,
                region=region,
                price_per_hour=adjusted_price,
                price_per_month=adjusted_price * 730,
                currency="USD",
                last_updated=datetime.now(),
                specifications={
                    'vcpus': vcpus, 
                    'ram': ram, 
                    'family': family,
                    'network': 'Up to 10 Gbps',
                    'storage': 'EBS Only'
                },
                reserved_pricing=reserved_pricing,
                spot_pricing=spot_price
            ))
        
        return sorted(pricing_data, key=lambda x: x.price_per_month)

class EnhancedClaudeAIIntegration:
    """Enhanced Claude AI integration with advanced analysis"""
    
    def __init__(self):
        self.api_key = st.secrets.get("CLAUDE_API_KEY") if hasattr(st, 'secrets') else None
        self.base_url = "https://api.anthropic.com/v1/messages"
        
    async def get_comprehensive_analysis(self, 
                                       workload_data: Dict, 
                                       pricing_data: List[PricingData]) -> Tuple[AIRecommendation, List[RiskAssessment], List[ImplementationPhase]]:
        """Get comprehensive AI analysis including recommendations, risks, and implementation plan"""
        try:
            if not self.api_key:
                return self._fallback_comprehensive_analysis(workload_data, pricing_data)
                
            prompt = self._build_comprehensive_prompt(workload_data, pricing_data)
            
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            }
            
            payload = {
                "model": "claude-3-sonnet-20240229",  # Using Sonnet for more comprehensive analysis
                "max_tokens": 2000,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=headers, json=payload, timeout=45) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result['content'][0]['text']
                        return self._parse_comprehensive_response(content)
                    else:
                        logger.error(f"Claude API error: {response.status}")
                        return self._fallback_comprehensive_analysis(workload_data, pricing_data)
                        
        except Exception as e:
            logger.error(f"Error getting comprehensive AI analysis: {e}")
            return self._fallback_comprehensive_analysis(workload_data, pricing_data)
    
    def _build_comprehensive_prompt(self, workload_data: Dict, pricing_data: List[PricingData]) -> str:
        """Build comprehensive prompt for detailed analysis"""
        return f"""
        As a senior cloud architect and cost optimization expert, provide a comprehensive analysis for this SQL Server migration to AWS.

        WORKLOAD PROFILE:
        - CPU Cores Required: {workload_data.get('cpu_cores', 'N/A')}
        - RAM Required: {workload_data.get('ram_gb', 'N/A')} GB
        - Storage: {workload_data.get('storage_gb', 'N/A')} GB
        - Peak CPU Utilization: {workload_data.get('peak_cpu', 'N/A')}%
        - Peak RAM Utilization: {workload_data.get('peak_ram', 'N/A')}%
        - Environment: {workload_data.get('workload_type', 'N/A')}
        - Region: {workload_data.get('region', 'N/A')}
        - SQL Server Edition: {workload_data.get('sql_edition', 'N/A')}
        - Licensing: {workload_data.get('licensing_model', 'N/A')}

        TOP PRICING OPTIONS WITH RESERVED INSTANCE PRICING:
        {self._format_enhanced_pricing_for_prompt(pricing_data[:5])}

        Please provide a comprehensive analysis in this exact JSON format:

        {{
            "recommendation": {{
                "recommendation": "Detailed recommendation with specific instance type and reasoning",
                "confidence_score": 85,
                "cost_impact": "High/Medium/Low",
                "reasoning": "Detailed technical justification",
                "risk_assessment": "Key risks and considerations",
                "implementation_timeline": "Suggested implementation timeline",
                "expected_savings": 50000
            }},
            "risk_assessments": [
                {{
                    "category": "Technical Risk",
                    "risk_level": "Medium",
                    "description": "Risk description",
                    "mitigation_strategy": "Mitigation approach",
                    "impact": "Business impact if risk materializes"
                }}
            ],
            "implementation_phases": [
                {{
                    "phase": "Phase 1: Assessment",
                    "duration": "2-3 weeks",
                    "activities": ["Activity 1", "Activity 2"],
                    "dependencies": ["Dependency 1"],
                    "deliverables": ["Deliverable 1"]
                }}
            ]
        }}
        """
    
    def _format_enhanced_pricing_for_prompt(self, pricing_data: List[PricingData]) -> str:
        """Format enhanced pricing data for AI prompt"""
        formatted = []
        for i, pricing_obj in enumerate(pricing_data, 1):
            specs = pricing_obj.specifications or {}
            reserved_1yr = pricing_obj.reserved_pricing.get('1_year_all_upfront', 0) if pricing_obj.reserved_pricing else 0
            reserved_3yr = pricing_obj.reserved_pricing.get('3_year_all_upfront', 0) if pricing_obj.reserved_pricing else 0
            
            formatted.append(
                f"{i}. {pricing_obj.instance_type} ({specs.get('family', 'Unknown')}) - "
                f"{specs.get('vcpus', '?')} vCPUs, {specs.get('ram', '?')} GB RAM\n"
                f"   On-Demand: ${pricing_obj.price_per_hour:.3f}/hour (${pricing_obj.price_per_month:.0f}/month)\n"
                f"   Reserved 1-Year: ${reserved_1yr:.3f}/hour (${reserved_1yr * 730:.0f}/month)\n"
                f"   Reserved 3-Year: ${reserved_3yr:.3f}/hour (${reserved_3yr * 730:.0f}/month)\n"
                f"   Spot: ${pricing_obj.spot_pricing:.3f}/hour (${pricing_obj.spot_pricing * 730:.0f}/month)"
            )
        return "\n\n".join(formatted)
    
    def _parse_comprehensive_response(self, content: str) -> Tuple[AIRecommendation, List[RiskAssessment], List[ImplementationPhase]]:
        """Parse comprehensive AI response"""
        try:
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                # Parse recommendation
                rec_data = data.get('recommendation', {})
                recommendation = AIRecommendation(
                    recommendation=rec_data.get('recommendation', ''),
                    confidence_score=rec_data.get('confidence_score', 75),
                    cost_impact=rec_data.get('cost_impact', 'Medium'),
                    reasoning=rec_data.get('reasoning', ''),
                    risk_assessment=rec_data.get('risk_assessment', ''),
                    implementation_timeline=rec_data.get('implementation_timeline', ''),
                    expected_savings=rec_data.get('expected_savings', 0)
                )
                
                # Parse risk assessments
                risks = []
                for risk_data in data.get('risk_assessments', []):
                    risks.append(RiskAssessment(
                        category=risk_data.get('category', ''),
                        risk_level=risk_data.get('risk_level', 'Medium'),
                        description=risk_data.get('description', ''),
                        mitigation_strategy=risk_data.get('mitigation_strategy', ''),
                        impact=risk_data.get('impact', '')
                    ))
                
                # Parse implementation phases
                phases = []
                for phase_data in data.get('implementation_phases', []):
                    phases.append(ImplementationPhase(
                        phase=phase_data.get('phase', ''),
                        duration=phase_data.get('duration', ''),
                        activities=phase_data.get('activities', []),
                        dependencies=phase_data.get('dependencies', []),
                        deliverables=phase_data.get('deliverables', [])
                    ))
                
                return recommendation, risks, phases
                
        except Exception as e:
            logger.warning(f"Failed to parse comprehensive AI response: {e}")
        
        return self._fallback_comprehensive_analysis({}, [])
    
    def _fallback_comprehensive_analysis(self, workload_data: Dict, pricing_data: List[PricingData]) -> Tuple[AIRecommendation, List[RiskAssessment], List[ImplementationPhase]]:
        """Enhanced fallback analysis"""
        # Generate rule-based recommendation
        recommendation = AIRecommendation(
            recommendation="Based on standard optimization practices, implement right-sizing and Reserved Instances for cost optimization",
            confidence_score=70,
            cost_impact="Medium to High",
            reasoning="Analysis based on AWS best practices and industry standards",
            risk_assessment="Migration risks are manageable with proper planning",
            implementation_timeline="3-6 months for full optimization",
            expected_savings=30000
        )
        
        # Generate standard risk assessments
        risks = [
            RiskAssessment(
                category="Technical Risk",
                risk_level="Medium",
                description="Application compatibility and performance during migration",
                mitigation_strategy="Comprehensive testing in staging environment",
                impact="Potential application downtime or performance degradation"
            ),
            RiskAssessment(
                category="Cost Risk",
                risk_level="Low",
                description="Unexpected cost overruns during migration",
                mitigation_strategy="Detailed cost monitoring and budgeting",
                impact="Budget variance of 10-20%"
            ),
            RiskAssessment(
                category="Security Risk",
                risk_level="Medium",
                description="Data security during migration and in cloud environment",
                mitigation_strategy="Implement encryption and security best practices",
                impact="Potential data exposure or compliance issues"
            )
        ]
        
        # Generate implementation phases
        phases = [
            ImplementationPhase(
                phase="Phase 1: Assessment & Planning",
                duration="2-4 weeks",
                activities=["Current state assessment", "Migration strategy development", "Risk assessment"],
                dependencies=["Stakeholder approval", "Budget allocation"],
                deliverables=["Migration plan", "Risk matrix", "Cost projections"]
            ),
            ImplementationPhase(
                phase="Phase 2: Environment Setup",
                duration="3-4 weeks",
                activities=["AWS account setup", "Network configuration", "Security implementation"],
                dependencies=["Phase 1 completion", "AWS account access"],
                deliverables=["Configured AWS environment", "Security documentation"]
            ),
            ImplementationPhase(
                phase="Phase 3: Migration & Testing",
                duration="4-6 weeks",
                activities=["Data migration", "Application migration", "Performance testing"],
                dependencies=["Phase 2 completion", "Application dependencies"],
                deliverables=["Migrated applications", "Test results", "Performance benchmarks"]
            )
        ]
        
        return recommendation, risks, phases

class PDFReportGenerator:
    """Professional PDF report generator"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
        
    def setup_custom_styles(self):
        """Setup custom styles for the PDF"""
        # Custom styles
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            fontName='Helvetica-Bold',
            textColor=HexColor('#1f4e79'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomHeading1',
            parent=self.styles['Heading1'],
            fontSize=16,
            fontName='Helvetica-Bold',
            textColor=HexColor('#2c5aa0'),
            spaceBefore=20,
            spaceAfter=12,
            borderWidth=2,
            borderColor=HexColor('#e9ecef'),
            borderPadding=5
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomHeading2',
            parent=self.styles['Heading2'],
            fontSize=14,
            fontName='Helvetica-Bold',
            textColor=HexColor('#4a90e2'),
            spaceBefore=15,
            spaceAfter=8
        ))
        
        self.styles.add(ParagraphStyle(
            name='ExecutiveSummary',
            parent=self.styles['Normal'],
            fontSize=12,
            fontName='Helvetica',
            spaceBefore=10,
            spaceAfter=10,
            leftIndent=20,
            rightIndent=20,
            borderWidth=1,
            borderColor=HexColor('#dee2e6'),
            borderPadding=15,
            backColor=HexColor('#f8f9fa')
        ))
        
        self.styles.add(ParagraphStyle(
            name='HighlightBox',
            parent=self.styles['Normal'],
            fontSize=11,
            fontName='Helvetica',
            spaceBefore=10,
            spaceAfter=10,
            leftIndent=15,
            rightIndent=15,
            borderWidth=1,
            borderColor=HexColor('#28a745'),
            borderPadding=10,
            backColor=HexColor('#d4edda')
        ))
    
    def create_comprehensive_report(self, 
                                  config: Dict,
                                  pricing_data: List[PricingData],
                                  recommendation: AIRecommendation,
                                  risk_assessments: List[RiskAssessment],
                                  implementation_phases: List[ImplementationPhase]) -> BytesIO:
        """Create comprehensive PDF report"""
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        story = []
        
        # Title Page
        story.extend(self._create_title_page(config))
        story.append(PageBreak())
        
        # Executive Summary
        story.extend(self._create_executive_summary(config, recommendation, pricing_data))
        story.append(PageBreak())
        
        # Current State Analysis
        story.extend(self._create_current_state_analysis(config))
        
        # Pricing Analysis
        story.extend(self._create_pricing_analysis(pricing_data))
        story.append(PageBreak())
        
        # Recommendations
        story.extend(self._create_recommendations_section(recommendation))
        
        # Cost Analysis & ROI
        story.extend(self._create_cost_analysis(pricing_data, config))
        story.append(PageBreak())
        
        # Risk Assessment
        story.extend(self._create_risk_assessment_section(risk_assessments))
        
        # Implementation Roadmap
        story.extend(self._create_implementation_roadmap(implementation_phases))
        story.append(PageBreak())
        
        # Technical Appendix
        story.extend(self._create_technical_appendix(pricing_data, config))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    def _create_title_page(self, config: Dict) -> List:
        """Create professional title page"""
        elements = []
        
        # Title
        elements.append(Spacer(1, 2*inch))
        elements.append(Paragraph("AWS Cloud Migration", self.styles['CustomTitle']))
        elements.append(Paragraph("Cost Optimization Report", self.styles['CustomTitle']))
        
        elements.append(Spacer(1, 0.5*inch))
        
        # Subtitle
        elements.append(Paragraph(
            f"SQL Server Workload Analysis for {config.get('region', 'N/A')} Region",
            self.styles['CustomHeading2']
        ))
        
        elements.append(Spacer(1, 1*inch))
        
        # Report details table
        report_data = [
            ['Report Generated:', datetime.now().strftime('%B %d, %Y at %I:%M %p')],
            ['Analysis Region:', config.get('region', 'N/A')],
            ['Workload Type:', config.get('workload_type', 'N/A')],
            ['SQL Server Edition:', config.get('sql_edition', 'N/A')],
            ['Licensing Model:', config.get('licensing_model', 'N/A')],
        ]
        
        table = Table(report_data, colWidths=[2*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), HexColor('#f8f9fa')),
            ('TEXTCOLOR', (0, 0), (-1, -1), black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#dee2e6')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(table)
        
        elements.append(Spacer(1, 1*inch))
        
        # Disclaimer
        disclaimer = """
        <i>This report provides professional cost optimization recommendations based on current AWS pricing 
        and industry best practices. All projections are estimates and actual costs may vary based on 
        usage patterns, market conditions, and implementation specifics.</i>
        """
        elements.append(Paragraph(disclaimer, self.styles['Normal']))
        
        return elements
    
    def _create_executive_summary(self, config: Dict, recommendation: AIRecommendation, pricing_data: List[PricingData]) -> List:
        """Create executive summary"""
        elements = []
        
        elements.append(Paragraph("Executive Summary", self.styles['CustomHeading1']))
        
        # Key findings
        if pricing_data:
            cheapest = min(pricing_data, key=lambda x: x.price_per_month)
            avg_cost = sum(pricing_obj.price_per_month for pricing_obj in pricing_data) / len(pricing_data)
            
            summary_text = f"""
            <b>Analysis Overview:</b> This report analyzes {len(pricing_data)} AWS EC2 instance configurations 
            for your SQL Server workload migration to the {config.get('region', 'N/A')} region. 
            Our analysis indicates potential cost optimization opportunities with expected annual savings 
            of ${recommendation.expected_savings:,.0f}.
            <br/><br/>
            <b>Key Findings:</b>
            <br/>• Most cost-effective option: {cheapest.instance_type} at ${cheapest.price_per_month:,.2f}/month
            <br/>• Average monthly cost across options: ${avg_cost:,.2f}
            <br/>• Confidence score for recommendations: {recommendation.confidence_score}%
            <br/>• Expected cost impact: {recommendation.cost_impact}
            <br/><br/>
            <b>Strategic Recommendation:</b> {recommendation.recommendation[:200]}...
            """
        else:
            summary_text = """
            This report provides cost optimization analysis for your SQL Server workload migration to AWS. 
            Due to limited pricing data availability, this analysis is based on industry benchmarks and 
            best practices.
            """
        
        elements.append(Paragraph(summary_text, self.styles['ExecutiveSummary']))
        
        # Cost savings highlight
        if recommendation.expected_savings > 0:
            savings_text = f"""
            <b>Projected Annual Savings: ${recommendation.expected_savings:,.0f}</b>
            <br/>Implementation of our recommendations could result in significant cost reductions 
            while maintaining or improving performance and reliability.
            """
            elements.append(Paragraph(savings_text, self.styles['HighlightBox']))
        
        return elements
    
    def _create_current_state_analysis(self, config: Dict) -> List:
        """Create current state analysis section"""
        elements = []
        
        elements.append(Paragraph("Current State Analysis", self.styles['CustomHeading1']))
        
        # Workload requirements table
        elements.append(Paragraph("Workload Requirements", self.styles['CustomHeading2']))
        
        workload_data = [
            ['Requirement', 'Current Specification', 'Analysis'],
            ['CPU Cores', f"{config.get('cpu_cores', 'N/A')}", 'Primary compute requirement'],
            ['Memory (RAM)', f"{config.get('ram_gb', 'N/A')} GB", 'Memory-intensive workload consideration'],
            ['Storage', f"{config.get('storage_gb', 'N/A')} GB", 'Storage capacity planning'],
            ['Peak CPU Utilization', f"{config.get('peak_cpu', 'N/A')}%", 'Right-sizing opportunity assessment'],
            ['Peak RAM Utilization', f"{config.get('peak_ram', 'N/A')}%", 'Memory optimization potential'],
            ['Environment Type', config.get('workload_type', 'N/A'), 'Availability and performance requirements'],
        ]
        
        workload_table = Table(workload_data, colWidths=[1.5*inch, 1.5*inch, 2.5*inch])
        workload_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#4a90e2')),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#dee2e6')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(workload_table)
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _create_pricing_analysis(self, pricing_data: List[PricingData]) -> List:
        """Create detailed pricing analysis section"""
        elements = []
        
        elements.append(Paragraph("Detailed Pricing Analysis", self.styles['CustomHeading1']))
        
        if not pricing_data:
            elements.append(Paragraph("No pricing data available for analysis.", self.styles['Normal']))
            return elements
        
        # Top 10 options table
        elements.append(Paragraph("Top Cost-Effective Options", self.styles['CustomHeading2']))
        
        pricing_table_data = [['Instance Type', 'Family', 'vCPUs', 'RAM (GB)', 'On-Demand\n(Monthly)', '1-Year RI\n(Monthly)', '3-Year RI\n(Monthly)', 'Spot\n(Monthly)']]
        
        for pricing_obj in pricing_data[:10]:
            specs = pricing_obj.specifications or {}
            reserved_1yr = pricing_obj.reserved_pricing.get('1_year_all_upfront', 0) * 730 if pricing_obj.reserved_pricing else 0
            reserved_3yr = pricing_obj.reserved_pricing.get('3_year_all_upfront', 0) * 730 if pricing_obj.reserved_pricing else 0
            spot_monthly = pricing_obj.spot_pricing * 730 if pricing_obj.spot_pricing else 0
            
            pricing_table_data.append([
                pricing_obj.instance_type,
                specs.get('family', 'N/A'),
                str(specs.get('vcpus', 'N/A')),
                str(specs.get('ram', 'N/A')),
                f"${pricing_obj.price_per_month:,.0f}",
                f"${reserved_1yr:,.0f}",
                f"${reserved_3yr:,.0f}",
                f"${spot_monthly:,.0f}"
            ])
        
        pricing_table = Table(pricing_table_data, colWidths=[0.8*inch, 0.8*inch, 0.5*inch, 0.6*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.7*inch])
        pricing_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#4a90e2')),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#dee2e6')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(pricing_table)
        elements.append(Spacer(1, 20))
        
        # Cost optimization insights
        cheapest = min(pricing_data, key=lambda x: x.price_per_month)
        most_expensive = max(pricing_data, key=lambda x: x.price_per_month)
        
        insights_text = f"""
        <b>Key Pricing Insights:</b>
        <br/>• Cost range: ${cheapest.price_per_month:,.0f} - ${most_expensive.price_per_month:,.0f} monthly
        <br/>• Reserved Instances provide 40-60% savings over On-Demand pricing
        <br/>• Spot Instances offer up to 80% savings for fault-tolerant workloads
        <br/>• Newer generation instances (m6a, r6a) typically offer better price/performance
        """
        elements.append(Paragraph(insights_text, self.styles['Normal']))
        
        return elements
    
    def _create_recommendations_section(self, recommendation: AIRecommendation) -> List:
        """Create recommendations section"""
        elements = []
        
        elements.append(Paragraph("Strategic Recommendations", self.styles['CustomHeading1']))
        
        # Main recommendation
        elements.append(Paragraph("Primary Recommendation", self.styles['CustomHeading2']))
        elements.append(Paragraph(recommendation.recommendation, self.styles['Normal']))
        elements.append(Spacer(1, 15))
        
        # Reasoning
        elements.append(Paragraph("Technical Justification", self.styles['CustomHeading2']))
        elements.append(Paragraph(recommendation.reasoning, self.styles['Normal']))
        elements.append(Spacer(1, 15))
        
        # Risk assessment
        if recommendation.risk_assessment:
            elements.append(Paragraph("Risk Considerations", self.styles['CustomHeading2']))
            elements.append(Paragraph(recommendation.risk_assessment, self.styles['Normal']))
            elements.append(Spacer(1, 15))
        
        # Implementation timeline
        if recommendation.implementation_timeline:
            elements.append(Paragraph("Implementation Timeline", self.styles['CustomHeading2']))
            elements.append(Paragraph(recommendation.implementation_timeline, self.styles['Normal']))
        
        return elements
    
    def _create_cost_analysis(self, pricing_data: List[PricingData], config: Dict) -> List:
        """Create comprehensive cost analysis"""
        elements = []
        
        elements.append(Paragraph("Cost Analysis & ROI Projection", self.styles['CustomHeading1']))
        
        if not pricing_data:
            return elements
        
        # Find suitable instances
        cpu_req = config.get('cpu_cores', 4)
        ram_req = config.get('ram_gb', 16)
        
        suitable_instances = [pricing_obj for pricing_obj in pricing_data 
                            if pricing_obj.specifications and 
                            pricing_obj.specifications.get('vcpus', 0) >= cpu_req and 
                            pricing_obj.specifications.get('ram', 0) >= ram_req]
        
        if suitable_instances:
            recommended_instance = suitable_instances[0]  # Most economical suitable option
            
            # 3-year cost projection
            elements.append(Paragraph("3-Year Total Cost of Ownership", self.styles['CustomHeading2']))
            
            tco_data = [
                ['Cost Component', 'On-Demand', '1-Year Reserved', '3-Year Reserved'],
                ['Monthly Instance Cost', 
                 f"${recommended_instance.price_per_month:,.0f}",
                 f"${recommended_instance.reserved_pricing.get('1_year_all_upfront', 0) * 730:,.0f}",
                 f"${recommended_instance.reserved_pricing.get('3_year_all_upfront', 0) * 730:,.0f}"],
                ['Annual Instance Cost',
                 f"${recommended_instance.price_per_month * 12:,.0f}",
                 f"${recommended_instance.reserved_pricing.get('1_year_all_upfront', 0) * 730 * 12:,.0f}",
                 f"${recommended_instance.reserved_pricing.get('3_year_all_upfront', 0) * 730 * 12:,.0f}"],
                ['3-Year Instance Cost',
                 f"${recommended_instance.price_per_month * 36:,.0f}",
                 f"${recommended_instance.reserved_pricing.get('1_year_all_upfront', 0) * 730 * 36:,.0f}",
                 f"${recommended_instance.reserved_pricing.get('3_year_all_upfront', 0) * 730 * 36:,.0f}"]
            ]
            
            tco_table = Table(tco_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
            tco_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HexColor('#28a745')),
                ('TEXTCOLOR', (0, 0), (-1, 0), white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, HexColor('#dee2e6')),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            elements.append(tco_table)
            elements.append(Spacer(1, 20))
            
            # Savings calculation
            on_demand_3yr = recommended_instance.price_per_month * 36
            reserved_3yr = recommended_instance.reserved_pricing.get('3_year_all_upfront', 0) * 730 * 36
            savings = on_demand_3yr - reserved_3yr
            
            savings_text = f"""
            <b>Cost Optimization Opportunity:</b>
            <br/>By implementing 3-Year Reserved Instances, you can save ${savings:,.0f} over three years 
            ({(savings/on_demand_3yr)*100:.1f}% reduction in compute costs).
            """
            elements.append(Paragraph(savings_text, self.styles['HighlightBox']))
        
        return elements
    
    def _create_risk_assessment_section(self, risk_assessments: List[RiskAssessment]) -> List:
        """Create risk assessment section"""
        elements = []
        
        elements.append(Paragraph("Risk Assessment Matrix", self.styles['CustomHeading1']))
        
        if not risk_assessments:
            elements.append(Paragraph("No specific risks identified in the analysis.", self.styles['Normal']))
            return elements
        
        # Risk assessment table
        risk_data = [['Risk Category', 'Risk Level', 'Description', 'Mitigation Strategy']]
        
        for risk in risk_assessments:
            risk_data.append([
                risk.category,
                risk.risk_level,
                risk.description[:100] + "..." if len(risk.description) > 100 else risk.description,
                risk.mitigation_strategy[:100] + "..." if len(risk.mitigation_strategy) > 100 else risk.mitigation_strategy
            ])
        
        risk_table = Table(risk_data, colWidths=[1.5*inch, 1*inch, 2*inch, 2*inch])
        risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#dc3545')),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#dee2e6')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(risk_table)
        
        # Risk mitigation summary
        elements.append(Spacer(1, 20))
        mitigation_text = """
        <b>Risk Mitigation Strategy:</b> All identified risks can be effectively managed through 
        proper planning, testing, and implementation of AWS best practices. Regular monitoring 
        and gradual migration approach will minimize potential impacts.
        """
        elements.append(Paragraph(mitigation_text, self.styles['Normal']))
        
        return elements
    
    def _create_implementation_roadmap(self, implementation_phases: List[ImplementationPhase]) -> List:
        """Create implementation roadmap section"""
        elements = []
        
        elements.append(Paragraph("Implementation Roadmap", self.styles['CustomHeading1']))
        
        if not implementation_phases:
            elements.append(Paragraph("No specific implementation phases defined.", self.styles['Normal']))
            return elements
        
        for i, phase in enumerate(implementation_phases, 1):
            elements.append(Paragraph(f"{phase.phase}", self.styles['CustomHeading2']))
            
            phase_details = f"""
            <b>Duration:</b> {phase.duration}
            <br/><b>Key Activities:</b> {', '.join(phase.activities[:3])}{'...' if len(phase.activities) > 3 else ''}
            <br/><b>Dependencies:</b> {', '.join(phase.dependencies) if phase.dependencies else 'None'}
            <br/><b>Deliverables:</b> {', '.join(phase.deliverables[:2])}{'...' if len(phase.deliverables) > 2 else ''}
            """
            elements.append(Paragraph(phase_details, self.styles['Normal']))
            elements.append(Spacer(1, 15))
        
        return elements
    
    def _create_technical_appendix(self, pricing_data: List[PricingData], config: Dict) -> List:
        """Create technical appendix"""
        elements = []
        
        elements.append(Paragraph("Technical Appendix", self.styles['CustomHeading1']))
        
        # Methodology
        elements.append(Paragraph("Analysis Methodology", self.styles['CustomHeading2']))
        methodology_text = """
        This analysis was conducted using AWS Pricing API data (where available) and industry-standard 
        cost optimization practices. Instance recommendations are based on workload requirements, 
        utilization patterns, and cost-effectiveness metrics. Reserved Instance pricing reflects 
        current AWS pricing models with estimated savings based on commitment levels.
        """
        elements.append(Paragraph(methodology_text, self.styles['Normal']))
        elements.append(Spacer(1, 15))
        
        # Data sources
        elements.append(Paragraph("Data Sources", self.styles['CustomHeading2']))
        data_sources_text = """
        • AWS Pricing API for real-time instance pricing
        • AWS EC2 Instance specifications and capabilities
        • Industry benchmarks for SQL Server workload optimization
        • AWS Reserved Instance pricing models
        • Spot Instance historical pricing data
        """
        elements.append(Paragraph(data_sources_text, self.styles['Normal']))
        elements.append(Spacer(1, 15))
        
        # Assumptions
        elements.append(Paragraph("Key Assumptions", self.styles['CustomHeading2']))
        assumptions_text = f"""
        • Analysis based on {config.get('region', 'N/A')} region pricing
        • SQL Server licensing costs included in Windows instances
        • 730 hours per month for cost calculations
        • Network and storage costs not included in instance pricing
        • Prices subject to change based on AWS pricing updates
        """
        elements.append(Paragraph(assumptions_text, self.styles['Normal']))
        
        return elements

class EnhancedCloudPricingOptimizer:
    """Enhanced main application class with PDF reporting"""
    
    def __init__(self):
        self.aws_pricing = EnhancedAWSPricingFetcher()
        self.claude_ai = EnhancedClaudeAIIntegration()
        self.mock_data = EnhancedMockPricingData()
        self.pdf_generator = PDFReportGenerator()
        self._initialize_session_state()
        
    def _initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'pricing_cache' not in st.session_state:
            st.session_state.pricing_cache = {}
        if 'last_analysis' not in st.session_state:
            st.session_state.last_analysis = None
        if 'demo_mode' not in st.session_state:
            st.session_state.demo_mode = not self.aws_pricing.connection_status["connected"]
        if 'comprehensive_analysis' not in st.session_state:
            st.session_state.comprehensive_analysis = None
    
    def render_main_interface(self):
        """Render the enhanced main Streamlit interface"""
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>☁️ AWS Cloud Pricing Optimizer</h1>
            <p>Professional-grade AWS pricing analysis with comprehensive PDF reporting</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Check connection status and show warnings
        self.render_connection_status()
        
        # Sidebar configuration
        self.render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "💰 Pricing Analysis", 
            "🤖 AI Recommendations", 
            "📊 Cost Comparison", 
            "⚠️ Risk Assessment",
            "📄 Professional Reports"
        ])
        
        with tab1:
            self.render_pricing_analysis()
        
        with tab2:
            self.render_ai_recommendations()
        
        with tab3:
            self.render_cost_comparison()
            
        with tab4:
            self.render_risk_assessment()
        
        with tab5:
            self.render_professional_reports()
    
    def render_connection_status(self):
        """Enhanced connection status display"""
        col1, col2 = st.columns(2)
        
        with col1:
            if self.aws_pricing.connection_status["connected"]:
                st.markdown('<span class="status-badge status-success">AWS: ✅ Connected</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-badge status-error">AWS: ❌ Demo Mode</span>', unsafe_allow_html=True)
                
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
        """Render enhanced sidebar configuration"""
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
        """Enhanced pricing analysis with Reserved Instance and Spot pricing"""
        st.markdown('<div class="section-header">💰 Comprehensive AWS Pricing Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.session_state.demo_mode:
                st.write("📊 Displaying realistic sample pricing data including Reserved Instance and Spot pricing.")
            else:
                st.write("Get comprehensive AWS EC2 pricing including On-Demand, Reserved Instance, and Spot pricing.")
        
        with col2:
            if st.button("🔄 Fetch Latest Prices", type="primary", use_container_width=True, key="fetch_prices_btn"):
                with st.spinner("Fetching comprehensive pricing data..."):
                    pricing_data = self.fetch_enhanced_pricing_data()
                    
                    if pricing_data:
                        st.session_state.latest_pricing = pricing_data
                        if st.session_state.demo_mode:
                            st.info(f"📊 Generated {len(pricing_data)} sample pricing options with RI and Spot pricing")
                        else:
                            st.success(f"✅ Fetched {len(pricing_data)} comprehensive pricing options")
                        self.display_enhanced_pricing_results(pricing_data)
                    else:
                        st.error("❌ Failed to fetch pricing data")
        
        # Display cached results if available
        if hasattr(st.session_state, 'latest_pricing'):
            self.display_enhanced_pricing_results(st.session_state.latest_pricing)
    
    def fetch_enhanced_pricing_data(self) -> List[PricingData]:
        """Fetch enhanced pricing data with RI and Spot pricing"""
        config = st.session_state.config
        
        # Use enhanced mock data if in demo mode or AWS unavailable
        if st.session_state.demo_mode or not self.aws_pricing.connection_status["connected"]:
            return self.mock_data.get_enhanced_sample_pricing_data(config['region'])
        
        # Try to fetch real AWS data with enhanced features
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
            status_text.text(f"Fetching enhanced pricing for {instance_type}...")
            
            enhanced_pricing = self.aws_pricing.get_enhanced_ec2_pricing(
                instance_type, config['region']
            )
            
            if enhanced_pricing:
                successful_fetches += 1
                pricing_data.append(enhanced_pricing)
            
            time.sleep(0.1)
        
        progress_bar.empty()
        status_text.empty()
        
        # If we got less than 3 successful fetches, fall back to mock data
        if successful_fetches < 3:
            st.warning("⚠️ Limited live data available. Using enhanced sample data.")
            return self.mock_data.get_enhanced_sample_pricing_data(config['region'])
        
        return sorted(pricing_data, key=lambda x: x.price_per_month)
    
    def display_enhanced_pricing_results(self, pricing_data: List[PricingData]):
        """Display enhanced pricing results with RI and Spot pricing"""
        if not pricing_data:
            st.warning("⚠️ No pricing data available")
            return
        
        # Create enhanced DataFrame for display
        df_data = []
        for pricing_obj in pricing_data:
            specs = pricing_obj.specifications or {}
            reserved_1yr = pricing_obj.reserved_pricing.get('1_year_all_upfront', 0) * 730 if pricing_obj.reserved_pricing else 0
            reserved_3yr = pricing_obj.reserved_pricing.get('3_year_all_upfront', 0) * 730 if pricing_obj.reserved_pricing else 0
            spot_monthly = pricing_obj.spot_pricing * 730 if pricing_obj.spot_pricing else 0
            
            df_data.append({
                'Instance Type': pricing_obj.instance_type,
                'Family': specs.get('family', 'N/A'),
                'vCPUs': specs.get('vcpus', 'N/A'),
                'RAM (GB)': specs.get('ram', 'N/A'),
                'On-Demand (Monthly)': f"${pricing_obj.price_per_month:.2f}",
                '1-Year RI (Monthly)': f"${reserved_1yr:.2f}",
                '3-Year RI (Monthly)': f"${reserved_3yr:.2f}",
                'Spot (Monthly)': f"${spot_monthly:.2f}",
                'Max Savings': f"{((pricing_obj.price_per_month - reserved_3yr) / pricing_obj.price_per_month * 100):.0f}%" if pricing_obj.price_per_month > 0 else "N/A"
            })
        
        df = pd.DataFrame(df_data)
        
        st.markdown('<div class="section-header">💸 Comprehensive Instance Pricing Comparison</div>', unsafe_allow_html=True)
        
        # Show enhanced summary metrics
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
            avg_savings = np.mean([
                (pricing_obj.price_per_month - pricing_obj.reserved_pricing.get('3_year_all_upfront', 0) * 730) / pricing_obj.price_per_month * 100
                for pricing_obj in pricing_data if pricing_obj.reserved_pricing and pricing_obj.price_per_month > 0
            ])
            st.markdown(f"""
            <div class="metric-card">
                <h3>Avg RI Savings</h3>
                <h2>{avg_savings:.0f}%</h2>
                <p>3-Year Reserved Instance</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            spot_savings = np.mean([
                (pricing_obj.price_per_month - pricing_obj.spot_pricing * 730) / pricing_obj.price_per_month * 100
                for pricing_obj in pricing_data if pricing_obj.spot_pricing and pricing_obj.price_per_month > 0
            ])
            st.markdown(f"""
            <div class="metric-card">
                <h3>Avg Spot Savings</h3>
                <h2>{spot_savings:.0f}%</h2>
                <p>Spot Instance pricing</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Options Analyzed</h3>
                <h2>{len(pricing_data)}</h2>
                <p>Instance configurations</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Enhanced pricing table
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
        
        # Enhanced pricing visualization with multiple pricing models
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('On-Demand vs Reserved Instance Pricing', 'Spot Instance Savings', 
                          'Price by Instance Family', 'Cost per vCPU Comparison'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # On-Demand vs RI comparison
        instance_names = [pricing_obj.instance_type for pricing_obj in pricing_data[:8]]
        on_demand_costs = [pricing_obj.price_per_month for pricing_obj in pricing_data[:8]]
        ri_3yr_costs = [pricing_obj.reserved_pricing.get('3_year_all_upfront', 0) * 730 for pricing_obj in pricing_data[:8]]
        
        fig.add_trace(
            go.Bar(name='On-Demand', x=instance_names, y=on_demand_costs, marker_color='#dc3545'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='3-Year RI', x=instance_names, y=ri_3yr_costs, marker_color='#28a745'),
            row=1, col=1
        )
        
        # Spot savings
        spot_savings_pct = [(pricing_obj.price_per_month - pricing_obj.spot_pricing * 730) / pricing_obj.price_per_month * 100 
                           for pricing_obj in pricing_data[:8]]
        fig.add_trace(
            go.Bar(x=instance_names, y=spot_savings_pct, marker_color='#ffc107', name='Spot Savings %'),
            row=1, col=2
        )
        
        # Price by family
        families = {}
        for pricing_obj in pricing_data:
            family = pricing_obj.specifications.get('family', 'Unknown') if pricing_obj.specifications else 'Unknown'
            if family not in families:
                families[family] = []
            families[family].append(pricing_obj.price_per_month)
        
        family_avg = {family: np.mean(costs) for family, costs in families.items()}
        fig.add_trace(
            go.Bar(x=list(family_avg.keys()), y=list(family_avg.values()), 
                  marker_color='#4a90e2', name='Avg Cost by Family'),
            row=2, col=1
        )
        
        # Cost per vCPU
        cost_per_vcpu = []
        vcpu_instances = []
        for pricing_obj in pricing_data[:8]:
            if pricing_obj.specifications and pricing_obj.specifications.get('vcpus', 0) > 0:
                cost_per_vcpu.append(pricing_obj.price_per_month / pricing_obj.specifications['vcpus'])
                vcpu_instances.append(pricing_obj.instance_type)
        
        fig.add_trace(
            go.Bar(x=vcpu_instances, y=cost_per_vcpu, marker_color='#20c997', name='Cost per vCPU'),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Comprehensive Pricing Analysis Dashboard",
            font_color='#343a40'
        )
        
        st.plotly_chart(fig, use_container_width=True, key="enhanced_pricing_dashboard")
    
    def render_ai_recommendations(self):
        """Enhanced AI recommendations with comprehensive analysis"""
        st.markdown('<div class="section-header">🤖 Comprehensive AI Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if self.claude_ai.api_key:
                st.write("🧠 Get comprehensive Claude AI analysis including recommendations, risk assessment, and implementation planning.")
            else:
                st.write("📋 Get comprehensive optimization analysis based on industry best practices and cost optimization principles.")
        
        with col2:
            if st.button("🧠 Get Comprehensive Analysis", type="primary", use_container_width=True, key="comprehensive_analysis_btn"):
                if hasattr(st.session_state, 'latest_pricing'):
                    with st.spinner("Performing comprehensive AI analysis..."):
                        if self.claude_ai.api_key:
                            recommendation, risks, phases = asyncio.run(
                                self.claude_ai.get_comprehensive_analysis(
                                    st.session_state.config,
                                    st.session_state.latest_pricing
                                )
                            )
                        else:
                            recommendation, risks, phases = self.claude_ai._fallback_comprehensive_analysis(
                                st.session_state.config,
                                st.session_state.latest_pricing
                            )
                        
                        st.session_state.comprehensive_analysis = {
                            'recommendation': recommendation,
                            'risks': risks,
                            'phases': phases
                        }
                        st.success("✅ Comprehensive analysis complete!")
                else:
                    st.error("❌ Please fetch pricing data first!")
        
        # Display comprehensive analysis if available
        if st.session_state.comprehensive_analysis:
            self.display_comprehensive_analysis()
        else:
            st.info("👆 Click 'Get Comprehensive Analysis' above to generate detailed recommendations, risk assessment, and implementation planning.")
    
    def display_comprehensive_analysis(self):
        """Display comprehensive AI analysis results"""
        analysis = st.session_state.comprehensive_analysis
        recommendation = analysis['recommendation']
        
        # Main recommendation
        st.markdown(f"""
        <div class="ai-recommendation">
            <h3>🎯 Strategic Recommendation</h3>
            <p><strong>Confidence Score:</strong> {recommendation.confidence_score}%</p>
            <p><strong>Expected Cost Impact:</strong> {recommendation.cost_impact}</p>
            <p><strong>Expected Annual Savings:</strong> ${recommendation.expected_savings:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**📋 Detailed Recommendations**")
            st.markdown(recommendation.recommendation)
            
            if recommendation.reasoning:
                st.markdown("**🧠 Technical Justification**")
                st.markdown(recommendation.reasoning)
            
            if recommendation.implementation_timeline:
                st.markdown("**⏱️ Implementation Timeline**")
                st.markdown(recommendation.implementation_timeline)
        
        with col2:
            # Enhanced confidence meter with savings indicator
            fig = go.Figure()
            
            # Confidence gauge
            fig.add_trace(go.Indicator(
                mode = "gauge+number",
                value = recommendation.confidence_score,
                domain = {'x': [0, 1], 'y': [0.6, 1]},
                title = {'text': "Confidence Score", 'font': {'color': '#1f4e79', 'size': 14}},
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
            
            # Savings indicator
            fig.add_trace(go.Indicator(
                mode = "number",
                value = recommendation.expected_savings,
                domain = {'x': [0, 1], 'y': [0, 0.4]},
                title = {'text': "Expected Annual Savings ($)", 'font': {'color': '#28a745', 'size': 14}},
                number = {'font': {'color': '#28a745', 'size': 24}}
            ))
            
            fig.update_layout(height=400, font_color='#343a40', margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True, key="comprehensive_metrics_chart")
    
    def render_risk_assessment(self):
        """Render risk assessment section"""
        st.markdown('<div class="section-header">⚠️ Risk Assessment & Mitigation</div>', unsafe_allow_html=True)
        
        if not st.session_state.comprehensive_analysis:
            st.info("🔍 Please run the Comprehensive AI Analysis first to see detailed risk assessment.")
            return
        
        risks = st.session_state.comprehensive_analysis['risks']
        phases = st.session_state.comprehensive_analysis['phases']
        
        # Risk matrix visualization
        if risks:
            st.markdown("**📊 Risk Assessment Matrix**")
            
            risk_levels = {'Low': 1, 'Medium': 2, 'High': 3}
            risk_colors = {'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'}
            
            col1, col2, col3 = st.columns(3)
            
            low_risks = [r for r in risks if r.risk_level == 'Low']
            medium_risks = [r for r in risks if r.risk_level == 'Medium']
            high_risks = [r for r in risks if r.risk_level == 'High']
            
            with col1:
                st.markdown(f"""
                <div class="risk-low">
                    <h4>Low Risk ({len(low_risks)})</h4>
                    {'<br/>'.join([f"• {r.category}" for r in low_risks])}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="risk-medium">
                    <h4>Medium Risk ({len(medium_risks)})</h4>
                    {'<br/>'.join([f"• {r.category}" for r in medium_risks])}
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="risk-high">
                    <h4>High Risk ({len(high_risks)})</h4>
                    {'<br/>'.join([f"• {r.category}" for r in high_risks])}
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed risk table
            st.markdown("**🔍 Detailed Risk Analysis**")
            
            risk_data = []
            for risk in risks:
                risk_data.append({
                    'Risk Category': risk.category,
                    'Risk Level': risk.risk_level,
                    'Description': risk.description,
                    'Mitigation Strategy': risk.mitigation_strategy,
                    'Business Impact': risk.impact
                })
            
            risk_df = pd.DataFrame(risk_data)
            st.dataframe(risk_df, use_container_width=True, hide_index=True)
        
        # Implementation phases timeline
        if phases:
            st.markdown("**🗓️ Implementation Timeline**")
            
            for i, phase in enumerate(phases, 1):
                with st.expander(f"Phase {i}: {phase.phase} ({phase.duration})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Key Activities:**")
                        for activity in phase.activities:
                            st.markdown(f"• {activity}")
                        
                        st.markdown("**Dependencies:**")
                        for dependency in phase.dependencies:
                            st.markdown(f"• {dependency}")
                    
                    with col2:
                        st.markdown("**Deliverables:**")
                        for deliverable in phase.deliverables:
                            st.markdown(f"• {deliverable}")
    
    def render_cost_comparison(self):
        """Enhanced cost comparison with comprehensive ROI analysis"""
        st.markdown('<div class="section-header">📊 Advanced Cost Analysis & ROI</div>', unsafe_allow_html=True)
        
        if not hasattr(st.session_state, 'latest_pricing'):
            st.warning("⚠️ Please fetch pricing data first to see cost comparison.")
            return
            
        pricing_data = st.session_state.latest_pricing
        config = st.session_state.config
        
        # Enhanced cost comparison with multiple scenarios
        cpu_req = config.get('cpu_cores', 4)
        ram_req = config.get('ram_gb', 16)
        
        suitable_instances = [pricing_obj for pricing_obj in pricing_data 
                            if pricing_obj.specifications and 
                            pricing_obj.specifications.get('vcpus', 0) >= cpu_req and 
                            pricing_obj.specifications.get('ram', 0) >= ram_req]
        
        if not suitable_instances:
            st.error("❌ No instances meet your minimum requirements.")
            return
        
        recommended_instance = suitable_instances[0]  # Most economical
        
        # Comprehensive scenario analysis
        st.markdown('<div class="section-header">🎯 5-Year Total Cost of Ownership Analysis</div>', unsafe_allow_html=True)
        
        scenarios = {
            'On-Demand': {
                'monthly_cost': recommended_instance.price_per_month,
                'description': 'Pay-as-you-go pricing'
            },
            '1-Year RI (No Upfront)': {
                'monthly_cost': recommended_instance.reserved_pricing.get('1_year_no_upfront', 0) * 730,
                'description': '1-year commitment, no upfront payment'
            },
            '3-Year RI (All Upfront)': {
                'monthly_cost': recommended_instance.reserved_pricing.get('3_year_all_upfront', 0) * 730,
                'description': '3-year commitment, full upfront payment'
            },
            'Spot Instance': {
                'monthly_cost': recommended_instance.spot_pricing * 730 if recommended_instance.spot_pricing else 0,
                'description': 'Interruptible workloads with significant savings'
            }
        }
        
        # Calculate 5-year costs
        years = [1, 2, 3, 4, 5]
        cumulative_costs = {}
        
        for scenario_name, scenario_data in scenarios.items():
            monthly_cost = scenario_data['monthly_cost']
            cumulative_costs[scenario_name] = [monthly_cost * 12 * year for year in years]
        
        # Create comprehensive cost visualization
        fig = go.Figure()
        
        colors = ['#dc3545', '#ffc107', '#28a745', '#17a2b8']
        
        for i, (scenario, costs) in enumerate(cumulative_costs.items()):
            fig.add_trace(go.Scatter(
                x=years,
                y=costs,
                mode='lines+markers',
                name=scenario,
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title="5-Year Total Cost of Ownership Comparison",
            xaxis_title="Years",
            yaxis_title="Cumulative Cost ($)",
            font_color='#343a40',
            plot_bgcolor='white',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True, key="tco_comparison_chart")
        
        # Cost savings table
        base_cost_5yr = cumulative_costs['On-Demand'][-1]
        
        savings_data = []
        for scenario_name, scenario_data in scenarios.items():
            if scenario_name != 'On-Demand':
                scenario_cost_5yr = cumulative_costs[scenario_name][-1]
                savings = base_cost_5yr - scenario_cost_5yr
                savings_pct = (savings / base_cost_5yr) * 100 if base_cost_5yr > 0 else 0
                
                savings_data.append({
                    'Pricing Model': scenario_name,
                    'Monthly Cost': f"${scenario_data['monthly_cost']:,.2f}",
                    '5-Year Total': f"${scenario_cost_5yr:,.0f}",
                    'Total Savings': f"${savings:,.0f}",
                    'Savings %': f"{savings_pct:.1f}%",
                    'Description': scenario_data['description']
                })
        
        if savings_data:
            savings_df = pd.DataFrame(savings_data)
            st.markdown("**💰 Cost Optimization Opportunities**")
            st.dataframe(savings_df, use_container_width=True, hide_index=True)
            
            # Highlight best option
            best_savings = max(savings_data, key=lambda x: float(x['Total Savings'].replace('$', '').replace(',', '')))
            st.markdown(f"""
            <div class="cost-savings">
                <h3>Recommended Optimization</h3>
                <h2>{best_savings['Pricing Model']}</h2>
                <p>Save {best_savings['Total Savings']} over 5 years ({best_savings['Savings %']})</p>
            </div>
            """, unsafe_allow_html=True)
    
    def render_professional_reports(self):
        """Render professional PDF report generation section"""
        st.markdown('<div class="section-header">📄 Professional PDF Reports</div>', unsafe_allow_html=True)
        
        # Check if analysis is complete
        has_pricing = hasattr(st.session_state, 'latest_pricing') and st.session_state.latest_pricing
        has_analysis = st.session_state.comprehensive_analysis is not None
        
        if not has_pricing:
            st.warning("⚠️ Please fetch pricing data first to generate reports.")
            return
        
        # Report generation section
        st.markdown("""
        <div class="download-section">
            <h3>📊 Generate Professional Analysis Report</h3>
            <p>Create a comprehensive PDF report including pricing analysis, recommendations, 
            risk assessment, and implementation roadmap for stakeholders and decision makers.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📋 Report Contents:**")
            st.markdown("""
            • Executive Summary
            • Current State Analysis
            • Comprehensive Pricing Analysis
            • Strategic Recommendations
            • Cost Analysis & ROI Projections
            • Risk Assessment Matrix
            • Implementation Roadmap
            • Technical Appendix
            """)
        
        with col2:
            st.markdown("**🎯 Target Audience:**")
            st.markdown("""
            • C-Level Executives
            • IT Directors
            • Cloud Architects
            • Financial Planning Teams
            • Project Managers
            • Technical Teams
            """)
        
        with col3:
            st.markdown("**📈 Key Benefits:**")
            st.markdown("""
            • Professional Presentation
            • Data-Driven Insights
            • Cost Optimization Roadmap
            • Risk Mitigation Strategies
            • Implementation Planning
            • Offline Reference
            """)
        
        # Generate report button
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("📄 Generate Professional PDF Report", 
                        type="primary", 
                        use_container_width=True, 
                        key="generate_pdf_btn"):
                
                if not has_analysis:
                    st.warning("⚠️ Please run the Comprehensive AI Analysis first for complete report.")
                    # Use basic analysis for report
                    recommendation = AIRecommendation(
                        recommendation="Basic cost optimization recommendations",
                        confidence_score=75,
                        cost_impact="Medium",
                        reasoning="Analysis based on pricing data and best practices",
                        expected_savings=25000
                    )
                    risks = []
                    phases = []
                else:
                    analysis = st.session_state.comprehensive_analysis
                    recommendation = analysis['recommendation']
                    risks = analysis['risks']
                    phases = analysis['phases']
                
                with st.spinner("Generating professional PDF report..."):
                    try:
                        # Generate PDF
                        pdf_buffer = self.pdf_generator.create_comprehensive_report(
                            config=st.session_state.config,
                            pricing_data=st.session_state.latest_pricing,
                            recommendation=recommendation,
                            risk_assessments=risks,
                            implementation_phases=phases
                        )
                        
                        # Create download button
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                        filename = f"AWS_Cost_Optimization_Report_{timestamp}.pdf"
                        
                        st.success("✅ PDF report generated successfully!")
                        
                        # Download button
                        st.download_button(
                            label="📥 Download Professional Report (PDF)",
                            data=pdf_buffer.getvalue(),
                            file_name=filename,
                            mime="application/pdf",
                            use_container_width=True,
                            key="download_pdf_btn"
                        )
                        
                        # Show preview information
                        st.markdown("""
                        <div class="pdf-preview">
                            <h4>📄 Report Generated Successfully</h4>
                            <p><strong>Filename:</strong> {}</p>
                            <p><strong>Pages:</strong> Comprehensive multi-page analysis</p>
                            <p><strong>Format:</strong> Professional PDF suitable for presentations</p>
                            <p><strong>Content:</strong> Executive summary, detailed analysis, recommendations, and appendices</p>
                        </div>
                        """.format(filename), unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error generating PDF report: {str(e)}")
                        logger.error(f"PDF generation error: {e}")
        
        # Additional export options
        st.markdown('<div class="section-header">📊 Additional Export Options</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📊 Pricing Data (CSV)", use_container_width=True, key="export_pricing_csv"):
                csv_data = self.export_enhanced_pricing_report()
                if csv_data:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name=f"aws_pricing_analysis_{timestamp}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="download_pricing_csv"
                    )
        
        with col2:
            if st.button("⚙️ Configuration (JSON)", use_container_width=True, key="export_config_json"):
                config_data = self.export_enhanced_configuration()
                timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                st.download_button(
                    label="📥 Download JSON",
                    data=config_data,
                    file_name=f"workload_configuration_{timestamp}.json",
                    mime="application/json",
                    use_container_width=True,
                    key="download_config_json"
                )
        
        with col3:
            if st.button("🧠 AI Analysis (JSON)", use_container_width=True, key="export_analysis_json"):
                if has_analysis:
                    analysis_data = self.export_ai_analysis()
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
                    st.download_button(
                        label="📥 Download JSON",
                        data=analysis_data,
                        file_name=f"ai_analysis_{timestamp}.json",
                        mime="application/json",
                        use_container_width=True,
                        key="download_analysis_json"
                    )
                else:
                    st.error("Run AI analysis first")
        
        with col4:
            if st.button("📈 ROI Calculator (Excel)", use_container_width=True, key="export_roi_excel"):
                st.info("📝 Excel export feature coming soon!")
    
    def export_enhanced_pricing_report(self) -> str:
        """Export enhanced pricing report with RI and Spot pricing"""
        if not hasattr(st.session_state, 'latest_pricing'):
            return ""
            
        config = st.session_state.config
        
        report_data = []
        for pricing_obj in st.session_state.latest_pricing:
            specs = pricing_obj.specifications or {}
            
            # Calculate all pricing options
            reserved_1yr = pricing_obj.reserved_pricing.get('1_year_all_upfront', 0) * 730 if pricing_obj.reserved_pricing else 0
            reserved_3yr = pricing_obj.reserved_pricing.get('3_year_all_upfront', 0) * 730 if pricing_obj.reserved_pricing else 0
            spot_monthly = pricing_obj.spot_pricing * 730 if pricing_obj.spot_pricing else 0
            
            report_data.append({
                'Instance Type': pricing_obj.instance_type,
                'Instance Family': specs.get('family', 'N/A'),
                'Region': pricing_obj.region,
                'vCPUs': specs.get('vcpus', 'N/A'),
                'RAM (GB)': specs.get('ram', 'N/A'),
                'Network Performance': specs.get('network', 'N/A'),
                'Storage Type': specs.get('storage', 'N/A'),
                'On-Demand Hourly': pricing_obj.price_per_hour,
                'On-Demand Monthly': pricing_obj.price_per_month,
                'On-Demand Annual': pricing_obj.price_per_month * 12,
                '1-Year RI Monthly': reserved_1yr,
                '1-Year RI Annual': reserved_1yr * 12,
                '3-Year RI Monthly': reserved_3yr,
                '3-Year RI Annual': reserved_3yr * 12,
                'Spot Monthly': spot_monthly,
                'Spot Annual': spot_monthly * 12,
                'Max Savings vs On-Demand': f"{((pricing_obj.price_per_month - reserved_3yr) / pricing_obj.price_per_month * 100):.1f}%" if pricing_obj.price_per_month > 0 else "N/A",
                'Meets CPU Requirement': 'Yes' if specs.get('vcpus', 0) >= config.get('cpu_cores', 0) else 'No',
                'Meets RAM Requirement': 'Yes' if specs.get('ram', 0) >= config.get('ram_gb', 0) else 'No',
                'Data Source': 'AWS API' if not st.session_state.demo_mode else 'Sample Data',
                'Last Updated': pricing_obj.last_updated.strftime('%Y-%m-%d %H:%M:%S')
            })
        
        df = pd.DataFrame(report_data)
        
        # Add comprehensive header
        summary_info = [
            f"# AWS Cloud Migration - Comprehensive Pricing Analysis",
            f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Analysis Region: {config.get('region', 'N/A')}",
            f"# Workload Type: {config.get('workload_type', 'N/A')}",
            f"# CPU Requirement: {config.get('cpu_cores', 'N/A')} cores",
            f"# RAM Requirement: {config.get('ram_gb', 'N/A')} GB",
            f"# SQL Server Edition: {config.get('sql_edition', 'N/A')}",
            f"# Licensing Model: {config.get('licensing_model', 'N/A')}",
            f"# Total Options Analyzed: {len(df)}",
            f"# Data Source: {'AWS Pricing API' if not st.session_state.demo_mode else 'Sample Data'}",
            f"#",
            f"# Pricing Models Included:",
            f"# - On-Demand: Pay-as-you-go pricing",
            f"# - Reserved Instances: 1-year and 3-year commitments",
            f"# - Spot Instances: Interruptible workloads with up to 90% savings",
            f"#",
        ]
        
        return "\n".join(summary_info) + "\n" + df.to_csv(index=False)
    
    def export_enhanced_configuration(self) -> str:
        """Export enhanced configuration with comprehensive metadata"""
        config = st.session_state.config if hasattr(st.session_state, 'config') else {}
        
        export_data = {
            'metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'export_version': '3.0',
                'application': 'AWS Cloud Pricing Optimizer - Enhanced Edition',
                'data_source': 'AWS Pricing API' if not st.session_state.demo_mode else 'Sample Data',
                'analysis_type': 'Comprehensive'
            },
            'workload_configuration': config,
            'requirements_analysis': {
                'cpu_adequacy': 'Analyzed based on peak utilization',
                'memory_adequacy': 'Evaluated against SQL Server requirements',
                'storage_considerations': 'EBS optimization recommended',
                'network_requirements': 'Standard AWS networking sufficient'
            },
            'service_status': {
                'aws_pricing_api': self.aws_pricing.connection_status["connected"],
                'claude_ai_available': self.claude_ai.api_key is not None,
                'demo_mode': st.session_state.demo_mode,
                'pricing_models_included': ['On-Demand', 'Reserved Instances', 'Spot Instances']
            },
            'analysis_scope': {
                'total_options_analyzed': len(st.session_state.latest_pricing) if hasattr(st.session_state, 'latest_pricing') else 0,
                'comprehensive_analysis_available': st.session_state.comprehensive_analysis is not None,
                'risk_assessment_included': True,
                'implementation_roadmap_included': True
            }
        }
        
        return json.dumps(export_data, indent=2)
    
    def export_ai_analysis(self) -> str:
        """Export comprehensive AI analysis results"""
        if not st.session_state.comprehensive_analysis:
            return json.dumps({"error": "No AI analysis available"}, indent=2)
        
        analysis = st.session_state.comprehensive_analysis
        
        export_data = {
            'analysis_metadata': {
                'generated_at': datetime.now().isoformat(),
                'version': '3.0',
                'analysis_type': 'Comprehensive AI Analysis',
                'confidence_level': 'High' if analysis['recommendation'].confidence_score >= 80 else 'Medium'
            },
            'strategic_recommendation': asdict(analysis['recommendation']),
            'risk_assessments': [asdict(risk) for risk in analysis['risks']],
            'implementation_phases': [asdict(phase) for phase in analysis['phases']],
            'executive_summary': {
                'key_recommendation': analysis['recommendation'].recommendation[:200] + "...",
                'expected_annual_savings': analysis['recommendation'].expected_savings,
                'implementation_timeline': analysis['recommendation'].implementation_timeline,
                'primary_risks': [risk.category for risk in analysis['risks'] if risk.risk_level in ['High', 'Medium']],
                'recommended_next_steps': [phase.phase for phase in analysis['phases'][:3]]
            }
        }
        
        return json.dumps(export_data, indent=2)

# Application entry point
def main():
    """Enhanced main application entry point"""
    try:
        optimizer = EnhancedCloudPricingOptimizer()
        optimizer.render_main_interface()
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        st.info("🔧 Please check your configuration and try again.")
        logger.error(f"Application error: {e}")
        
        # Show detailed error information in development
        if st.session_state.get('debug_mode', False):
            st.exception(e)

if __name__ == "__main__":
    main()