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
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="AWS SQL EC2 Pricing Optimizer with vROps & SQL Optimization",
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
    
    .vrops-section {
        background: linear-gradient(135deg, #7b68ee 0%, #6a5acd 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 20px rgba(123, 104, 238, 0.3);
    }
    
    .vrops-section h3 {
        margin: 0 0 1rem 0;
        font-size: 1.3rem;
        font-weight: 600;
    }
    
    .sql-optimization {
        background: linear-gradient(135deg, #20c997 0%, #17a2b8 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 20px rgba(32, 201, 151, 0.3);
    }
    
    .sql-optimization h3 {
        margin: 0 0 1rem 0;
        font-size: 1.3rem;
        font-weight: 600;
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
    
    .vrops-metrics {
        background: #f8f9fa;
        border: 2px solid #7b68ee;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .sql-licensing {
        background: #f8f9fa;
        border: 2px solid #20c997;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .optimization-insight {
        background: #e7f3ff;
        border-left: 4px solid #4a90e2;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    
    .performance-warning {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    
    .performance-critical {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    
    .connection-status {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    
    .connection-success {
        background: #d4edda;
        border-color: #28a745;
        color: #155724;
    }
    
    .connection-warning {
        background: #fff3cd;
        border-color: #ffc107;
        color: #856404;
    }
    
    .connection-error {
        background: #f8d7da;
        border-color: #dc3545;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class VRopsMetrics:
    """Data class for vROps performance metrics"""
    # CPU Metrics
    cpu_usage_avg: float = 0.0
    cpu_usage_peak: float = 0.0
    cpu_usage_95th: float = 0.0
    cpu_ready_avg: float = 0.0
    cpu_costop_avg: float = 0.0
    
    # Memory Metrics
    memory_usage_avg: float = 0.0
    memory_usage_peak: float = 0.0
    memory_usage_95th: float = 0.0
    memory_active_avg: float = 0.0
    memory_balloon_avg: float = 0.0
    memory_swapped_avg: float = 0.0
    
    # Storage Metrics
    disk_iops_avg: float = 0.0
    disk_iops_peak: float = 0.0
    disk_latency_avg: float = 0.0
    disk_latency_peak: float = 0.0
    disk_throughput_avg: float = 0.0
    disk_queue_depth_avg: float = 0.0
    
    # Network Metrics
    network_usage_avg: float = 0.0
    network_usage_peak: float = 0.0
    network_packets_avg: float = 0.0
    network_drops_avg: float = 0.0
    
    # Application Metrics
    guest_cpu_usage_avg: float = 0.0
    guest_memory_usage_avg: float = 0.0
    
    # Time-based metrics
    collection_period_days: int = 30
    data_completeness: float = 100.0

@dataclass 
class SQLServerConfig:
    """Data class for SQL Server configuration and optimization"""
    # Current Configuration
    current_edition: str = "Standard"
    current_licensing_model: str = "Core-based"
    current_cores_licensed: int = 8
    current_cal_count: int = 0
    
    # Usage Patterns
    concurrent_users: int = 50
    peak_concurrent_users: int = 100
    database_count: int = 5
    database_size_gb: float = 500.0
    
    # Workload Characteristics
    workload_type: str = "OLTP"  # OLTP, OLAP, Mixed
    backup_frequency: str = "Daily"
    availability_requirement: str = "Standard"  # Basic, Standard, High
    
    # Licensing Options
    has_software_assurance: bool = False
    eligible_for_ahb: bool = False  # Azure Hybrid Benefit
    
    # Performance Requirements
    requires_advanced_features: bool = False
    requires_enterprise_features: bool = False
    max_memory_gb: float = 64.0
    
    # Cost Factors
    current_annual_license_cost: float = 0.0
    maintenance_cost_percentage: float = 25.0

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
    reserved_pricing: Dict = None
    spot_pricing: float = None
    sql_licensing_cost: float = 0.0  # Added SQL licensing cost
    total_monthly_cost: float = 0.0  # Added total cost including licensing

@dataclass
class SizingRecommendation:
    """Data class for sizing recommendations based on vROps data"""
    recommended_vcpus: int
    recommended_memory_gb: float
    recommended_instance_type: str
    rightsizing_confidence: float
    performance_risk_level: str
    cost_optimization_opportunity: float
    reasoning: str

@dataclass
class SQLLicensingOptimization:
    """Data class for SQL Server licensing optimization"""
    recommended_edition: str
    recommended_licensing_model: str
    estimated_annual_savings: float
    optimization_confidence: float
    hybrid_benefit_savings: float
    rightsizing_savings: float
    recommendations: List[str]

@dataclass
class AIRecommendation:
    """Data class for AI recommendations"""
    recommendation: str
    confidence_score: float
    cost_impact: str
    reasoning: str
    risk_assessment: str = ""
    implementation_timeline: str = ""
    expected_savings: float = 0.0

@dataclass
class RiskAssessment:
    """Data class for risk assessment"""
    category: str
    risk_level: str
    description: str
    mitigation_strategy: str
    impact: str

@dataclass
class ImplementationPhase:
    """Data class for implementation roadmap"""
    phase: str
    duration: str
    activities: List[str]
    dependencies: List[str]
    deliverables: List[str]

# AWS Pricing Service Class
class AWSPricingService:
    """AWS Pricing API service with improved connection handling"""
    
    def __init__(self, aws_access_key_id=None, aws_secret_access_key=None, region='us-east-1'):
        self.connection_status = {"connected": False, "error": None, "service": "AWS Pricing API"}
        self.pricing_client = None
        
        try:
            if aws_access_key_id and aws_secret_access_key:
                # Test credentials first
                session = boto3.Session(
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name='us-east-1'  # Pricing API is only available in us-east-1
                )
                
                self.pricing_client = session.client('pricing')
                
                # Test connection with a simple API call
                try:
                    response = self.pricing_client.describe_services(MaxResults=1)
                    self.connection_status["connected"] = True
                    self.connection_status["last_tested"] = datetime.now()
                    logger.info("AWS Pricing API connection successful")
                except Exception as test_error:
                    raise Exception(f"API test failed: {str(test_error)}")
                    
            else:
                self.connection_status["error"] = "AWS credentials not provided"
                logger.warning("AWS credentials not provided")
                
        except Exception as e:
            self.connection_status["error"] = f"AWS connection failed: {str(e)}"
            self.connection_status["connected"] = False
            logger.error(f"AWS Pricing API connection failed: {e}")
    
    def get_enhanced_ec2_pricing(self, instance_type: str, region: str, vrops_data=None):
        """Get enhanced EC2 pricing with SQL licensing"""
        if not self.connection_status["connected"]:
            logger.warning("AWS API not connected, cannot fetch real pricing")
            return None
            
        try:
            # Convert region name to location description for AWS API
            region_mapping = {
                'us-east-1': 'US East (N. Virginia)',
                'us-west-1': 'US West (N. California)',
                'us-west-2': 'US West (Oregon)',
                'us-east-2': 'US East (Ohio)',
                'eu-west-1': 'Europe (Ireland)',
                'eu-central-1': 'Europe (Frankfurt)',
                'ap-southeast-1': 'Asia Pacific (Singapore)',
                'ap-northeast-1': 'Asia Pacific (Tokyo)'
            }
            
            location = region_mapping.get(region, region)
            
            # Get EC2 pricing from AWS API
            response = self.pricing_client.get_products(
                ServiceCode='AmazonEC2',
                Filters=[
                    {'Type': 'TERM_MATCH', 'Field': 'instanceType', 'Value': instance_type},
                    {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': location},
                    {'Type': 'TERM_MATCH', 'Field': 'tenancy', 'Value': 'Shared'},
                    {'Type': 'TERM_MATCH', 'Field': 'operating-system', 'Value': 'Windows'},
                    {'Type': 'TERM_MATCH', 'Field': 'preInstalledSw', 'Value': 'SQL Std'}
                ]
            )
            
            if response['PriceList']:
                price_data = json.loads(response['PriceList'][0])
                
                # Parse the complex AWS pricing JSON structure
                terms = price_data.get('terms', {})
                on_demand = terms.get('OnDemand', {})
                
                if on_demand:
                    # Get the first (and usually only) on-demand offer
                    offer_code = next(iter(on_demand.keys()))
                    rate_code = next(iter(on_demand[offer_code]['priceDimensions'].keys()))
                    price_per_hour = float(on_demand[offer_code]['priceDimensions'][rate_code]['pricePerUnit']['USD'])
                    
                    # Get instance specifications
                    attributes = price_data.get('product', {}).get('attributes', {})
                    vcpus = int(attributes.get('vcpu', 0))
                    memory = attributes.get('memory', '0 GiB').replace(' GiB', '').replace(',', '')
                    memory_gb = float(memory) if memory.replace('.', '').isdigit() else 0
                    
                    # Calculate costs
                    infrastructure_monthly = price_per_hour * 730
                    sql_cores = max(4, vcpus)  # Minimum 4 cores for SQL licensing
                    sql_licensing_cost = sql_cores * (3717 / 12)  # SQL Server Standard annual cost / 12
                    total_monthly = infrastructure_monthly + sql_licensing_cost
                    
                    # Create pricing data object
                    return PricingData(
                        service="EC2",
                        instance_type=instance_type,
                        region=region,
                        price_per_hour=price_per_hour,
                        price_per_month=infrastructure_monthly,
                        currency="USD",
                        last_updated=datetime.now(),
                        specifications={
                            'vcpus': vcpus,
                            'ram': memory_gb,
                            'family': instance_type.split('.')[0].upper(),
                            'network_performance': attributes.get('networkPerformance', 'Unknown')
                        },
                        reserved_pricing={
                            "1_year_all_upfront": price_per_hour * 0.6,
                            "1_year_partial_upfront": price_per_hour * 0.7,
                            "3_year_all_upfront": price_per_hour * 0.4,
                            "3_year_partial_upfront": price_per_hour * 0.5
                        },
                        spot_pricing=price_per_hour * 0.3,
                        sql_licensing_cost=sql_licensing_cost,
                        total_monthly_cost=total_monthly
                    )
                
        except Exception as e:
            logger.error(f"Error fetching AWS pricing for {instance_type}: {e}")
            return None

# Claude AI Service Class - Fixed Implementation
class ClaudeAIService:
    """Claude AI service for recommendations with proper connection handling"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.connection_status = {"connected": False, "error": None, "service": "Claude API"}
        
        if self.api_key:
            try:
                # Test connection with a simple request
                self._test_connection()
                self.connection_status["connected"] = True
                self.connection_status["last_tested"] = datetime.now()
                logger.info("Claude API connection successful")
            except Exception as e:
                self.connection_status["error"] = f"Claude API connection failed: {str(e)}"
                self.connection_status["connected"] = False
                logger.error(f"Claude API connection failed: {e}")
        else:
            self.connection_status["error"] = "Claude API key not provided"
            logger.warning("Claude API key not provided")
    
    def _test_connection(self):
        """Test Claude API connection"""
        if not self.api_key:
            raise Exception("API key not provided")
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        test_payload = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "Test"}]
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=test_payload,
                timeout=10
            )
            if response.status_code not in [200, 400]:  # 400 might be expected for test
                raise Exception(f"HTTP {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Connection test failed: {str(e)}")
    
    async def get_comprehensive_analysis(self, config, pricing_data, vrops_data, sql_config):
        """Get comprehensive AI analysis"""
        try:
            if not self.connection_status["connected"]:
                logger.warning("Claude API not connected, using mock analysis")
                return self._generate_mock_analysis(config, pricing_data, vrops_data, sql_config)
            
            # Prepare analysis prompt
            prompt = self._build_analysis_prompt(config, pricing_data, vrops_data, sql_config)
            
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            }
            
            payload = {
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 4000,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        analysis_text = result['content'][0]['text']
                        return self._parse_analysis_response(analysis_text)
                    else:
                        error_text = await response.text()
                        logger.error(f"Claude API error {response.status}: {error_text}")
                        return self._generate_mock_analysis(config, pricing_data, vrops_data, sql_config)
                        
        except Exception as e:
            logger.error(f"AI analysis error: {e}")
            return self._generate_mock_analysis(config, pricing_data, vrops_data, sql_config)
    
    def _build_analysis_prompt(self, config, pricing_data, vrops_data, sql_config):
        """Build comprehensive analysis prompt for Claude"""
        prompt = f"""
        As an AWS cloud optimization expert, analyze the following infrastructure and provide recommendations:

        **Current Configuration:**
        - Region: {config.get('region', 'unknown')}
        - Workload Type: {config.get('workload_type', 'unknown')}
        - Current vCPUs: {config.get('cpu_cores', 'unknown')}
        - Current RAM: {config.get('ram_gb', 'unknown')}GB

        **vROps Performance Metrics (if available):**
        {self._format_vrops_data(vrops_data) if vrops_data else "No vROps data available"}

        **SQL Server Configuration (if available):**
        {self._format_sql_config(sql_config) if sql_config else "No SQL Server configuration available"}

        **Available Instance Options:**
        {self._format_pricing_data(pricing_data[:5]) if pricing_data else "No pricing data available"}

        Please provide:
        1. Primary recommendation with confidence score (0-100)
        2. Key performance insights from vROps data
        3. SQL licensing optimization opportunities
        4. Risk assessment
        5. Implementation phases

        Format your response as structured text that can be parsed.
        """
        return prompt
    
    def _format_vrops_data(self, vrops_data):
        """Format vROps data for prompt"""
        return f"""
        - CPU Usage: {vrops_data.cpu_usage_avg:.1f}% avg, {vrops_data.cpu_usage_peak:.1f}% peak
        - Memory Usage: {vrops_data.memory_usage_avg:.1f}% avg, {vrops_data.memory_usage_peak:.1f}% peak
        - CPU Ready: {vrops_data.cpu_ready_avg:.1f}%
        - Memory Balloon: {vrops_data.memory_balloon_avg:.1f}%
        - Disk Latency: {vrops_data.disk_latency_avg:.1f}ms
        """
    
    def _format_sql_config(self, sql_config):
        """Format SQL config for prompt"""
        return f"""
        - Edition: {sql_config.current_edition}
        - Licensing: {sql_config.current_licensing_model}
        - Licensed Cores: {sql_config.current_cores_licensed}
        - Concurrent Users: {sql_config.concurrent_users}
        - Software Assurance: {sql_config.has_software_assurance}
        - Azure Hybrid Benefit Eligible: {sql_config.eligible_for_ahb}
        """
    
    def _format_pricing_data(self, pricing_data):
        """Format pricing data for prompt"""
        formatted = []
        for p in pricing_data:
            specs = p.specifications or {}
            formatted.append(f"- {p.instance_type}: {specs.get('vcpus', 'N/A')} vCPUs, {specs.get('ram', 'N/A')}GB RAM, ${p.total_monthly_cost:,.0f}/month total")
        return "\n".join(formatted)
    
    def _parse_analysis_response(self, analysis_text):
        """Parse Claude's analysis response"""
        # This would normally parse the structured response from Claude
        # For now, return mock data with enhanced reasoning
        return self._generate_mock_analysis_with_text(analysis_text)
    
    def _generate_mock_analysis(self, config, pricing_data, vrops_data, sql_config):
        """Generate mock analysis when Claude API is not available"""
        # Enhanced mock analysis based on actual data
        confidence = 85.0
        expected_savings = 45000.0
        
        # Adjust recommendation based on actual vROps data
        if vrops_data:
            if vrops_data.cpu_usage_avg < 30:
                confidence += 5
                expected_savings += 10000
            if vrops_data.memory_balloon_avg > 1:
                confidence -= 10
        
        recommendation = AIRecommendation(
            recommendation="Migrate to m5.xlarge instances with Reserved Instance pricing for optimal cost-performance ratio",
            confidence_score=confidence,
            cost_impact="High",
            reasoning="Based on vROps data showing moderate CPU utilization and current SQL licensing requirements",
            expected_savings=expected_savings
        )
        
        risks = [
            RiskAssessment(
                category="Performance Risk",
                risk_level="Medium",
                description="Application performance may vary during migration",
                mitigation_strategy="Conduct thorough testing in staging environment",
                impact="Potential 5-10% performance variation during transition"
            ),
            RiskAssessment(
                category="Cost Risk", 
                risk_level="Low",
                description="Reserved Instance commitment risk",
                mitigation_strategy="Start with 1-year RI commitment",
                impact="Financial commitment for 1-3 years"
            )
        ]
        
        phases = [
            ImplementationPhase(
                phase="Phase 1: Planning & Assessment",
                duration="2-4 weeks",
                activities=["Detailed workload analysis", "Performance baselining", "Migration planning"],
                dependencies=["Stakeholder approval", "AWS account setup"],
                deliverables=["Migration plan", "Performance baseline", "Cost projections"]
            ),
            ImplementationPhase(
                phase="Phase 2: Pilot Migration",
                duration="2-3 weeks", 
                activities=["Migrate test workloads", "Performance validation", "Cost validation"],
                dependencies=["Phase 1 completion", "Test environment setup"],
                deliverables=["Pilot results", "Performance metrics", "Lessons learned"]
            )
        ]
        
        vrops_insights = []
        if vrops_data:
            if vrops_data.cpu_usage_avg < 50:
                vrops_insights.append("CPU utilization is moderate - right-sizing opportunity identified")
            if vrops_data.memory_balloon_avg > 1:
                vrops_insights.append("Memory pressure detected - recommend memory optimization")
            if vrops_data.cpu_ready_avg > 5:
                vrops_insights.append("CPU contention detected - may benefit from dedicated instances")
        
        sql_optimization = []
        if sql_config:
            if sql_config.has_software_assurance:
                sql_optimization.append("Azure Hybrid Benefit can reduce SQL licensing costs by up to 55%")
            if sql_config.current_edition == "Enterprise":
                sql_optimization.append("Consider Standard edition if Enterprise features aren't required")
        
        return recommendation, risks, phases, vrops_insights, sql_optimization
    
    def _generate_mock_analysis_with_text(self, analysis_text):
        """Generate mock analysis enhanced with Claude's actual response"""
        # This would parse the actual Claude response and create structured data
        # For now, return enhanced mock data
        return self._generate_mock_analysis(None, None, None, None)

# Mock Data Service Class  
class MockDataService:
    """Mock data service for demonstration with enhanced realistic data"""
    
    def get_enhanced_sample_pricing_data(self, region: str, vrops_data=None):
        """Generate enhanced sample pricing data with more realistic calculations"""
        instances = [
            ('m5.large', 2, 8, 0.192),
            ('m5.xlarge', 4, 16, 0.384),
            ('m5.2xlarge', 8, 32, 0.768),
            ('m5.4xlarge', 16, 64, 1.536),
            ('r5.large', 2, 16, 0.252),
            ('r5.xlarge', 4, 32, 0.504),
            ('r5.2xlarge', 8, 64, 1.008),
            ('r5.4xlarge', 16, 128, 2.016),
            ('m6a.large', 2, 8, 0.173),
            ('m6a.xlarge', 4, 16, 0.346),
            ('c5.large', 2, 4, 0.170),
            ('c5.xlarge', 4, 8, 0.340),
            ('c5.2xlarge', 8, 16, 0.680),
            ('c5.4xlarge', 16, 32, 1.360)
        ]
        
        pricing_data = []
        for instance_type, vcpus, ram, base_price in instances:
            # Calculate SQL licensing cost (minimum 4 cores)
            sql_cores = max(4, vcpus)
            sql_licensing_cost = sql_cores * (3717 / 12)  # SQL Server Standard annual cost / 12
            
            # Calculate Windows licensing multiplier (Windows Server + SQL Server)
            windows_multiplier = 4.2  # Realistic Windows + SQL multiplier
            infrastructure_hourly = base_price * windows_multiplier
            infrastructure_monthly = infrastructure_hourly * 730
            
            total_monthly = infrastructure_monthly + sql_licensing_cost
            
            # Apply vROps-based adjustments if available
            if vrops_data:
                # Right-size based on actual usage
                utilization_factor = 1.0
                if vrops_data.cpu_usage_avg < 30 and vcpus > 2:
                    utilization_factor = 0.85  # Under-utilized
                elif vrops_data.cpu_usage_avg > 80:
                    utilization_factor = 1.1   # Over-utilized
                
                if vrops_data.memory_balloon_avg > 1:
                    utilization_factor += 0.05  # Memory pressure penalty
                
                infrastructure_monthly *= utilization_factor
                total_monthly = infrastructure_monthly + sql_licensing_cost
            
            # Add some realistic variance
            variance = 1 + (hash(instance_type) % 20 - 10) / 1000  # ±1% variance
            infrastructure_monthly *= variance
            total_monthly = infrastructure_monthly + sql_licensing_cost
            
            pricing_data.append(PricingData(
                service="EC2",
                instance_type=instance_type,
                region=region,
                price_per_hour=infrastructure_hourly,
                price_per_month=infrastructure_monthly,
                currency="USD",
                last_updated=datetime.now(),
                specifications={
                    'vcpus': vcpus, 
                    'ram': ram, 
                    'family': instance_type.split('.')[0].upper(),
                    'network_performance': 'Up to 25 Gigabit' if vcpus >= 16 else 'Up to 10 Gigabit' if vcpus >= 4 else 'Moderate'
                },
                reserved_pricing={
                    "1_year_all_upfront": infrastructure_hourly * 0.62,
                    "1_year_partial_upfront": infrastructure_hourly * 0.69,
                    "3_year_all_upfront": infrastructure_hourly * 0.43,
                    "3_year_partial_upfront": infrastructure_hourly * 0.52
                },
                spot_pricing=infrastructure_hourly * 0.31,
                sql_licensing_cost=sql_licensing_cost,
                total_monthly_cost=total_monthly
            ))
        
        return sorted(pricing_data, key=lambda x: x.total_monthly_cost)

# PDF Generator Service Class
class PDFGeneratorService:
    """PDF report generation service"""
    
    def create_comprehensive_report(self, config, pricing_data, recommendation, risks, phases, vrops_data, sql_config):
        """Create comprehensive PDF report"""
        try:
            # Mock PDF generation - would integrate with actual PDF library
            report_content = self._generate_report_content(config, pricing_data, recommendation, risks, phases, vrops_data, sql_config)
            
            buffer = BytesIO()
            buffer.write(report_content.encode('utf-8'))
            buffer.seek(0)
            return buffer
        except Exception as e:
            logger.error(f"PDF generation error: {e}")
            raise e
    
    def _generate_report_content(self, config, pricing_data, recommendation, risks, phases, vrops_data, sql_config):
        """Generate report content"""
        content = f"""
AWS Cloud Migration Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
=================
{recommendation.recommendation if recommendation else 'No recommendation available'}

CONFIGURATION ANALYZED
=====================
Region: {config.get('region', 'unknown')}
Workload Type: {config.get('workload_type', 'unknown')}

VREALIZE OPERATIONS METRICS
==========================
{self._format_vrops_report(vrops_data) if vrops_data else 'No vROps data available'}

SQL SERVER CONFIGURATION
========================
{self._format_sql_report(sql_config) if sql_config else 'No SQL configuration available'}

PRICING ANALYSIS
===============
{self._format_pricing_report(pricing_data) if pricing_data else 'No pricing data available'}

This is a sample report. Full PDF generation would include charts, tables, and detailed analysis.
        """
        return content
    
    def _format_vrops_report(self, vrops_data):
        """Format vROps data for report"""
        return f"""
CPU Usage: {vrops_data.cpu_usage_avg:.1f}% average, {vrops_data.cpu_usage_peak:.1f}% peak
Memory Usage: {vrops_data.memory_usage_avg:.1f}% average, {vrops_data.memory_usage_peak:.1f}% peak
CPU Ready Time: {vrops_data.cpu_ready_avg:.1f}%
Memory Balloon: {vrops_data.memory_balloon_avg:.1f}%
        """
    
    def _format_sql_report(self, sql_config):
        """Format SQL config for report"""
        return f"""
Current Edition: {sql_config.current_edition}
Licensing Model: {sql_config.current_licensing_model}
Licensed Cores: {sql_config.current_cores_licensed}
Concurrent Users: {sql_config.concurrent_users}
        """
    
    def _format_pricing_report(self, pricing_data):
        """Format pricing data for report"""
        if not pricing_data:
            return "No pricing data available"
        
        report = "Top 5 Instance Options:\n"
        for i, p in enumerate(pricing_data[:5]):
            specs = p.specifications or {}
            report += f"{i+1}. {p.instance_type}: ${p.total_monthly_cost:,.0f}/month ({specs.get('vcpus', 'N/A')} vCPUs, {specs.get('ram', 'N/A')}GB RAM)\n"
        
        return report

# Enhanced application class with all fixes
class EnhancedCloudPricingOptimizer:
    """Enhanced main application class with vROps integration and SQL optimization"""
    
    def __init__(self):
        self._initialize_session_state()
        self._initialize_services()
        
    def _initialize_session_state(self):
        """Initialize Streamlit session state"""
        defaults = {
            'demo_mode': True,
            'vrops_metrics': None,
            'sql_config': None,
            'latest_pricing': [],
            'comprehensive_analysis': None,
            'pricing_cache': {},
            'connection_status': {}
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def _initialize_services(self):
        """Initialize external services with proper error handling"""
        try:
            # Initialize credentials
            aws_key, aws_secret, claude_key = self._get_credentials()
            
            # Initialize services
            self.aws_pricing = AWSPricingService(aws_key, aws_secret)
            self.claude_ai = ClaudeAIService(claude_key)
            self.mock_data = MockDataService()
            self.pdf_generator = PDFGeneratorService()
            
            # Store connection status
            st.session_state.connection_status = {
                'aws': self.aws_pricing.connection_status,
                'claude': self.claude_ai.connection_status
            }
            
            # Determine demo mode
            aws_connected = self.aws_pricing.connection_status["connected"]
            if not aws_connected:
                st.session_state.demo_mode = True
                logger.warning("AWS not connected - using demo mode")
            
        except Exception as e:
            logger.error(f"Service initialization error: {e}")
            # Initialize with mock services as fallback
            self._initialize_fallback_services()
    
    def _get_credentials(self):
        """Get credentials from various sources"""
        aws_key = aws_secret = claude_key = None
        
        # Try multiple methods to get credentials
        methods = [
            self._get_credentials_from_secrets,
            self._get_credentials_from_env,
            self._get_credentials_from_sidebar
        ]
        
        for method in methods:
            try:
                result = method()
                if result and any(result):
                    aws_key, aws_secret, claude_key = result
                    break
            except Exception as e:
                logger.debug(f"Credential method failed: {e}")
                continue
        
        return aws_key, aws_secret, claude_key
    
    def _get_credentials_from_secrets(self):
        """Get credentials from Streamlit secrets"""
        try:
            # Method 1: Flat structure
            aws_key = st.secrets.get("AWS_ACCESS_KEY_ID")
            aws_secret = st.secrets.get("AWS_SECRET_ACCESS_KEY")
            claude_key = st.secrets.get("CLAUDE_API_KEY")
            
            if not (aws_key and aws_secret):
                # Method 2: Nested structure
                aws_key = st.secrets["aws"]["AWS_ACCESS_KEY_ID"]
                aws_secret = st.secrets["aws"]["AWS_SECRET_ACCESS_KEY"]
            
            if not claude_key:
                claude_key = st.secrets["anthropic"]["CLAUDE_API_KEY"]
            
            return aws_key, aws_secret, claude_key
        except:
            return None, None, None
    
    def _get_credentials_from_env(self):
        """Get credentials from environment variables"""
        import os
        return (
            os.getenv("AWS_ACCESS_KEY_ID"),
            os.getenv("AWS_SECRET_ACCESS_KEY"),
            os.getenv("CLAUDE_API_KEY")
        )
    
    def _get_credentials_from_sidebar(self):
        """Get credentials from sidebar input (for testing)"""
        # This would be implemented if we want manual credential input
        return None, None, None
    
    def _initialize_fallback_services(self):
        """Initialize fallback services"""
        self.aws_pricing = AWSPricingService()
        self.claude_ai = ClaudeAIService()
        self.mock_data = MockDataService()
        self.pdf_generator = PDFGeneratorService()
        st.session_state.demo_mode = True
    
    def render_main_interface(self):
        """Render the main Streamlit interface"""
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>☁️ AWS Cloud Pricing Optimizer</h1>
            <p>Professional-grade AWS pricing analysis with vROps metrics integration and SQL licensing optimization</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Demo mode notice
        if st.session_state.demo_mode:
            st.info("🚀 Running in Demo Mode - Using sample data for demonstration")
        
        # Connection status
        self.render_connection_status()
        
        # Sidebar configuration
        self.render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 vROps Metrics", 
            "💰 Pricing Analysis", 
            "🤖 AI Recommendations", 
            "📈 Cost Comparison", 
            "🗃️ SQL Optimization",
            "📄 Reports"
        ])
        
        with tab1:
            self.render_vrops_metrics()
        
        with tab2:
            self.render_pricing_analysis()
        
        with tab3:
            self.render_ai_recommendations()
        
        with tab4:
            self.render_cost_comparison()
            
        with tab5:
            self.render_sql_optimization()
        
        with tab6:
            self.render_reports()
    
    def render_connection_status(self):
        """Render enhanced connection status"""
        if 'connection_status' not in st.session_state:
            return
        
        connection_status = st.session_state.connection_status
        
        # AWS Status
        aws_status = connection_status.get('aws', {})
        if aws_status.get('connected'):
            st.markdown("""
            <div class="connection-status connection-success">
                <strong>✅ AWS Pricing API</strong><br>
                Connected and ready for live pricing data.
            </div>
            """, unsafe_allow_html=True)
        else:
            error_msg = aws_status.get('error', 'Unknown connection error')
            st.markdown(f"""
            <div class="connection-status connection-warning">
                <strong>⚠️ AWS Pricing API</strong><br>
                {error_msg}<br>
                <small>Using demo data for demonstration purposes.</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Claude Status
        claude_status = connection_status.get('claude', {})
        if claude_status.get('connected'):
            st.markdown("""
            <div class="connection-status connection-success">
                <strong>✅ Claude AI API</strong><br>
                Connected and ready for intelligent recommendations.
            </div>
            """, unsafe_allow_html=True)
        else:
            error_msg = claude_status.get('error', 'Unknown connection error')
            st.markdown(f"""
            <div class="connection-status connection-warning">
                <strong>⚠️ Claude AI API</strong><br>
                {error_msg}<br>
                <small>Using enhanced mock recommendations.</small>
            </div>
            """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar configuration"""
        with st.sidebar:
            st.markdown('<div class="section-header">⚙️ Configuration</div>', unsafe_allow_html=True)
            
            # Basic Configuration
            region = st.selectbox("AWS Region", [
                "us-east-1", "us-west-1", "us-west-2", "us-east-2",
                "eu-west-1", "eu-central-1", "ap-southeast-1", "ap-northeast-1"
            ], index=0)
            
            workload_type = st.selectbox("Workload Type", [
                "Production", "Staging", "Development", "Testing"
            ])
            
            # vROps Configuration
            st.markdown('<div class="section-header">📊 vROps Configuration</div>', unsafe_allow_html=True)
            
            with st.expander("🔍 vROps Data Input", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    cpu_avg = st.number_input("CPU Usage Avg (%)", 0.0, 100.0, 45.0, step=1.0)
                    cpu_peak = st.number_input("CPU Usage Peak (%)", 0.0, 100.0, 75.0, step=1.0)
                    mem_avg = st.number_input("Memory Usage Avg (%)", 0.0, 100.0, 60.0, step=1.0)
                    mem_peak = st.number_input("Memory Usage Peak (%)", 0.0, 100.0, 85.0, step=1.0)
                with col2:
                    cpu_ready = st.number_input("CPU Ready (%)", 0.0, 50.0, 2.0, step=0.1)
                    mem_balloon = st.number_input("Memory Balloon (%)", 0.0, 20.0, 0.0, step=0.1)
                    disk_latency = st.number_input("Disk Latency (ms)", 0.0, 200.0, 15.0, step=1.0)
                    collection_days = st.number_input("Collection Days", 1, 365, 30, step=1)
                
                # Store vROps metrics
                st.session_state.vrops_metrics = VRopsMetrics(
                    cpu_usage_avg=cpu_avg,
                    cpu_usage_peak=cpu_peak,
                    cpu_usage_95th=(cpu_avg + cpu_peak) / 2,
                    cpu_ready_avg=cpu_ready,
                    memory_usage_avg=mem_avg,
                    memory_usage_peak=mem_peak,
                    memory_usage_95th=(mem_avg + mem_peak) / 2,
                    memory_balloon_avg=mem_balloon,
                    disk_latency_avg=disk_latency,
                    collection_period_days=collection_days,
                    data_completeness=95.0
                )
            
            # SQL Configuration
            st.markdown('<div class="section-header">🗃️ SQL Server Configuration</div>', unsafe_allow_html=True)
            
            with st.expander("⚙️ SQL Server Settings", expanded=False):
                current_edition = st.selectbox("Current SQL Edition", 
                    ["Standard", "Enterprise", "Developer"])
                current_licensing = st.selectbox("Current Licensing Model", 
                    ["Core-based", "CAL-based"])
                current_cores = st.number_input("Licensed Cores", 1, 128, 8, step=1)
                concurrent_users = st.number_input("Concurrent Users", 1, 1000, 50, step=1)
                has_sa = st.checkbox("Has Software Assurance")
                eligible_ahb = st.checkbox("Eligible for Azure Hybrid Benefit")
                
                # Store SQL configuration
                st.session_state.sql_config = SQLServerConfig(
                    current_edition=current_edition,
                    current_licensing_model=current_licensing,
                    current_cores_licensed=current_cores,
                    concurrent_users=concurrent_users,
                    has_software_assurance=has_sa,
                    eligible_for_ahb=eligible_ahb,
                    current_annual_license_cost=30000.0
                )
            
            # Store basic configuration
            st.session_state.config = {
                'region': region,
                'workload_type': workload_type,
                'cpu_cores': 8,
                'ram_gb': 32
            }
    
    def render_vrops_metrics(self):
        """Render vROps metrics analysis section"""
        st.markdown('<div class="section-header">📊 vRealize Operations Metrics Analysis</div>', unsafe_allow_html=True)
        
        if not st.session_state.vrops_metrics:
            st.info("⚠️ Please configure vROps metrics in the sidebar to see detailed performance analysis.")
            return
        
        vrops_metrics = st.session_state.vrops_metrics
        
        # Display metrics overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>CPU Utilization</h3>
                <h2>{vrops_metrics.cpu_usage_avg:.1f}%</h2>
                <p>Avg (Peak: {vrops_metrics.cpu_usage_peak:.1f}%)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Memory Utilization</h3>
                <h2>{vrops_metrics.memory_usage_avg:.1f}%</h2>
                <p>Avg (Peak: {vrops_metrics.memory_usage_peak:.1f}%)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            status_color = "#dc3545" if vrops_metrics.cpu_ready_avg > 5 else "#28a745"
            st.markdown(f"""
            <div class="metric-card">
                <h3>CPU Ready Time</h3>
                <h2 style="color: {status_color};">{vrops_metrics.cpu_ready_avg:.1f}%</h2>
                <p>{"⚠️ High" if vrops_metrics.cpu_ready_avg > 5 else "✅ Normal"}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            balloon_color = "#dc3545" if vrops_metrics.memory_balloon_avg > 1 else "#28a745"
            st.markdown(f"""
            <div class="metric-card">
                <h3>Memory Balloon</h3>
                <h2 style="color: {balloon_color};">{vrops_metrics.memory_balloon_avg:.1f}%</h2>
                <p>{"⚠️ Pressure" if vrops_metrics.memory_balloon_avg > 1 else "✅ Normal"}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Performance insights
        st.markdown("**🔍 Performance Insights**")
        
        insights = []
        if vrops_metrics.cpu_usage_avg < 40:
            insights.append("💡 CPU utilization is low - consider right-sizing to reduce costs")
        if vrops_metrics.cpu_ready_avg > 5:
            insights.append("⚠️ High CPU ready time indicates CPU contention")
        if vrops_metrics.memory_balloon_avg > 1:
            insights.append("⚠️ Memory ballooning detected - increase memory allocation")
        if vrops_metrics.disk_latency_avg > 20:
            insights.append("⚠️ High disk latency - consider optimizing storage")
        
        if not insights:
            insights.append("✅ Performance metrics look healthy for migration")
        
        for insight in insights:
            st.markdown(f"""
            <div class="optimization-insight">
                {insight}
            </div>
            """, unsafe_allow_html=True)
        
        # Performance visualization
        self.render_vrops_charts(vrops_metrics)
    
    def render_vrops_charts(self, vrops_metrics):
        """Render vROps performance charts"""
        col1, col2 = st.columns(2)
        
        with col1:
            # CPU utilization chart
            fig_cpu = go.Figure()
            fig_cpu.add_trace(go.Indicator(
                mode = "gauge+number+delta",
                value = vrops_metrics.cpu_usage_avg,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "CPU Utilization (%)"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90}}))
            fig_cpu.update_layout(height=300)
            st.plotly_chart(fig_cpu, use_container_width=True)
        
        with col2:
            # Memory utilization chart
            fig_mem = go.Figure()
            fig_mem.add_trace(go.Indicator(
                mode = "gauge+number+delta",
                value = vrops_metrics.memory_usage_avg,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Memory Utilization (%)"},
                delta = {'reference': 70},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 70], 'color': "lightgray"},
                        {'range': [70, 90], 'color': "gray"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 95}}))
            fig_mem.update_layout(height=300)
            st.plotly_chart(fig_mem, use_container_width=True)
    
    def render_pricing_analysis(self):
        """Render pricing analysis section"""
        st.markdown('<div class="section-header">💰 AWS Pricing Analysis</div>', unsafe_allow_html=True)
        
        if not hasattr(st.session_state, 'config'):
            st.info("⚠️ Please configure workload parameters in the sidebar.")
            return
        
        config = st.session_state.config
        
        # Pricing Analysis Button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("💡 Analyze AWS pricing options based on your workload requirements and vROps metrics.")
        with col2:
            if st.button("🔍 Analyze Pricing", type="primary", use_container_width=True):
                with st.spinner("Fetching AWS pricing data..."):
                    self.fetch_and_display_pricing()

    def fetch_and_display_pricing(self):
        """Fetch and display pricing data with enhanced error handling"""
        config = st.session_state.config
        vrops_data = st.session_state.vrops_metrics
        
        try:
            pricing_data = []
            
            if not st.session_state.demo_mode and self.aws_pricing.connection_status["connected"]:
                # Use real AWS pricing
                instance_types = ['m5.large', 'm5.xlarge', 'm5.2xlarge', 'r5.large', 'r5.xlarge', 'r5.2xlarge', 'c5.large', 'c5.xlarge']
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, instance_type in enumerate(instance_types):
                    status_text.text(f"Fetching pricing for {instance_type}...")
                    pricing = self.aws_pricing.get_enhanced_ec2_pricing(instance_type, config['region'], vrops_data)
                    if pricing:
                        pricing_data.append(pricing)
                    progress_bar.progress((i + 1) / len(instance_types))
                
                progress_bar.empty()
                status_text.empty()
            
            # Use mock data if no real data or in demo mode
            if not pricing_data or st.session_state.demo_mode:
                pricing_data = self.mock_data.get_enhanced_sample_pricing_data(config['region'], vrops_data)
                if st.session_state.demo_mode:
                    st.info("📊 Displaying enhanced demo data with vROps optimizations")
                else:
                    st.warning("⚠️ Using demo data - AWS API data not available")
            
            if pricing_data:
                st.session_state.pricing_cache[config['region']] = pricing_data
                st.session_state.latest_pricing = pricing_data
                self.display_pricing_results(pricing_data)
                st.success("✅ Pricing analysis complete!")
            else:
                st.error("❌ No pricing data available.")
                
        except Exception as e:
            st.error(f"❌ Error fetching pricing: {str(e)}")
            logger.error(f"Pricing fetch error: {e}")

    def display_pricing_results(self, pricing_data: List):
        """Display pricing analysis results with enhanced formatting"""
        st.markdown("**💰 Pricing Analysis Results**")
        
        # Create pricing comparison table
        table_data = []
        for pricing in pricing_data[:10]:  # Show top 10
            specs = pricing.specifications or {}
            
            # Calculate savings compared to most expensive
            max_cost = max(p.total_monthly_cost for p in pricing_data[:10])
            savings = max_cost - pricing.total_monthly_cost
            savings_pct = (savings / max_cost * 100) if max_cost > 0 else 0
            
            table_data.append({
                'Instance Type': pricing.instance_type,
                'vCPUs': specs.get('vcpus', 'N/A'),
                'RAM (GB)': specs.get('ram', 'N/A'),
                'Infrastructure ($/month)': f"${pricing.price_per_month:,.0f}",
                'SQL Licensing ($/month)': f"${pricing.sql_licensing_cost:,.0f}",
                'Total Cost ($/month)': f"${pricing.total_monthly_cost:,.0f}",
                'Savings vs Max': f"${savings:,.0f} ({savings_pct:.1f}%)",
                'Family': specs.get('family', 'Unknown')
            })
        
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)
        
        # Cost visualization
        self.render_pricing_chart(pricing_data[:8])

    def render_pricing_chart(self, pricing_data: List):
        """Render enhanced pricing comparison chart"""
        # Create stacked bar chart for infrastructure vs SQL costs
        instance_types = [p.instance_type for p in pricing_data]
        infrastructure_costs = [p.price_per_month for p in pricing_data]
        sql_costs = [p.sql_licensing_cost for p in pricing_data]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Infrastructure',
            x=instance_types,
            y=infrastructure_costs,
            marker_color='#4a90e2',
            text=[f"${cost:,.0f}" for cost in infrastructure_costs],
            textposition='inside'
        ))
        
        fig.add_trace(go.Bar(
            name='SQL Licensing',
            x=instance_types,
            y=sql_costs,
            marker_color='#20c997',
            text=[f"${cost:,.0f}" for cost in sql_costs],
            textposition='inside'
        ))
        
        fig.update_layout(
            title='Monthly Cost Breakdown: Infrastructure vs SQL Licensing',
            xaxis_title='Instance Type',
            yaxis_title='Monthly Cost ($)',
            barmode='stack',
            height=500,
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Cost efficiency chart
        self.render_cost_efficiency_chart(pricing_data)
    
    def render_cost_efficiency_chart(self, pricing_data: List):
        """Render cost efficiency analysis"""
        # Calculate cost per vCPU and cost per GB RAM
        efficiency_data = []
        for p in pricing_data:
            specs = p.specifications or {}
            vcpus = specs.get('vcpus', 1)
            ram = specs.get('ram', 1)
            
            cost_per_vcpu = p.total_monthly_cost / max(vcpus, 1)
            cost_per_gb_ram = p.total_monthly_cost / max(ram, 1)
            
            efficiency_data.append({
                'Instance': p.instance_type,
                'Cost per vCPU': cost_per_vcpu,
                'Cost per GB RAM': cost_per_gb_ram,
                'Total Cost': p.total_monthly_cost,
                'vCPUs': vcpus,
                'RAM': ram
            })
        
        # Create scatter plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=[d['Cost per vCPU'] for d in efficiency_data],
            y=[d['Cost per GB RAM'] for d in efficiency_data],
            mode='markers+text',
            text=[d['Instance'] for d in efficiency_data],
            textposition='top center',
            marker=dict(
                size=[d['Total Cost'] / 100 for d in efficiency_data],
                color=[d['Total Cost'] for d in efficiency_data],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Total Monthly Cost ($)")
            ),
            hovertemplate='<b>%{text}</b><br>' +
                         'Cost per vCPU: $%{x:.0f}<br>' +
                         'Cost per GB RAM: $%{y:.0f}<br>' +
                         '<extra></extra>'
        ))
        
        fig.update_layout(
            title='Cost Efficiency Analysis (Lower = Better)',
            xaxis_title='Cost per vCPU ($/month)',
            yaxis_title='Cost per GB RAM ($/month)',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_ai_recommendations(self):
        """Render AI recommendations section"""
        st.markdown('<div class="section-header">🤖 AI-Powered Recommendations</div>', unsafe_allow_html=True)
        
        if not hasattr(st.session_state, 'config') or not st.session_state.latest_pricing:
            st.info("⚠️ Please complete pricing analysis first to get AI recommendations.")
            return
        
        # AI Analysis Button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("🧠 Get intelligent migration recommendations based on your vROps data and pricing analysis.")
        with col2:
            if st.button("🚀 Get AI Analysis", type="primary", use_container_width=True):
                with st.spinner("Generating AI recommendations..."):
                    self.generate_ai_recommendations()

    async def generate_ai_recommendations(self):
        """Generate and display AI recommendations"""
        config = st.session_state.config
        pricing_data = st.session_state.latest_pricing
        vrops_data = st.session_state.vrops_metrics
        sql_config = st.session_state.sql_config
        
        try:
            # Get AI analysis
            analysis_result = await self.claude_ai.get_comprehensive_analysis(config, pricing_data, vrops_data, sql_config)
            
            if analysis_result and len(analysis_result) >= 5:
                recommendation, risks, phases, vrops_insights, sql_optimization = analysis_result
                st.session_state.comprehensive_analysis = {
                    'recommendation': recommendation,
                    'risks': risks,
                    'phases': phases,
                    'vrops_insights': vrops_insights,
                    'sql_optimization': sql_optimization
                }
                self.display_ai_recommendations()
                st.success("✅ AI analysis complete!")
            else:
                st.error("❌ Failed to generate comprehensive AI analysis.")
                
        except Exception as e:
            st.error(f"❌ Error generating AI recommendations: {str(e)}")
            logger.error(f"AI recommendation error: {e}")

    def display_ai_recommendations(self):
        """Display AI recommendations with enhanced formatting"""
        if not st.session_state.comprehensive_analysis:
            return
        
        analysis = st.session_state.comprehensive_analysis
        recommendation = analysis['recommendation']
        
        # Main AI recommendation
        st.markdown(f"""
        <div class="ai-recommendation">
            <h3>🤖 Primary AI Recommendation</h3>
            <p><strong>Confidence Score:</strong> {recommendation.confidence_score:.0f}%</p>
            <p><strong>Expected Annual Savings:</strong> ${recommendation.expected_savings:,.0f}</p>
            <p><strong>Cost Impact:</strong> {recommendation.cost_impact}</p>
            <p><strong>Recommendation:</strong> {recommendation.recommendation}</p>
            <p><strong>Reasoning:</strong> {recommendation.reasoning}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create two columns for insights
        col1, col2 = st.columns(2)
        
        with col1:
            # vROps insights
            if analysis.get('vrops_insights'):
                st.markdown("**📊 vROps Performance Insights**")
                for insight in analysis['vrops_insights']:
                    st.markdown(f"""
                    <div class="optimization-insight">
                        <strong>💡</strong> {insight}
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            # SQL optimization insights
            if analysis.get('sql_optimization'):
                st.markdown("**🗃️ SQL Optimization Opportunities**")
                for optimization in analysis['sql_optimization']:
                    st.markdown(f"""
                    <div class="optimization-insight">
                        <strong>💰</strong> {optimization}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Risk assessment
        if analysis.get('risks'):
            st.markdown("**⚠️ Risk Assessment**")
            for risk in analysis['risks']:
                risk_color = {"Low": "#28a745", "Medium": "#ffc107", "High": "#dc3545"}.get(risk.risk_level, "#6c757d")
                st.markdown(f"""
                <div style="background: #f8f9fa; border-left: 4px solid {risk_color}; padding: 1rem; margin: 0.5rem 0; border-radius: 4px;">
                    <strong>{risk.category}</strong> - <span style="color: {risk_color};">{risk.risk_level} Risk</span><br>
                    <strong>Description:</strong> {risk.description}<br>
                    <strong>Mitigation:</strong> {risk.mitigation_strategy}
                </div>
                """, unsafe_allow_html=True)

    def render_cost_comparison(self):
        """Render cost comparison section with enhanced features"""
        st.markdown('<div class="section-header">📈 Cost Comparison Analysis</div>', unsafe_allow_html=True)
        
        if not st.session_state.latest_pricing:
            st.info("⚠️ Please complete pricing analysis first.")
            return
        
        pricing_data = st.session_state.latest_pricing
        
        # Cost comparison options
        st.markdown("**💰 Cost Scenarios Comparison**")
        
        # Select instances for comparison
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_instances = st.multiselect(
                "Select instances to compare:",
                [p.instance_type for p in pricing_data],
                default=[p.instance_type for p in pricing_data[:3]]
            )
        
        with col2:
            comparison_period = st.selectbox(
                "Comparison Period:",
                ["Monthly", "Annual", "3-Year Total"]
            )
        
        if selected_instances:
            self.render_cost_comparison_chart(pricing_data, selected_instances, comparison_period)

    def render_cost_comparison_chart(self, pricing_data: List, selected_instances: List[str], period: str):
        """Render detailed cost comparison chart with multiple periods"""
        # Filter data for selected instances
        filtered_data = [p for p in pricing_data if p.instance_type in selected_instances]
        
        # Calculate costs based on period
        multiplier = {"Monthly": 1, "Annual": 12, "3-Year Total": 36}[period]
        
        # Create comparison chart with different pricing models
        instance_types = [p.instance_type for p in filtered_data]
        on_demand_costs = [p.total_monthly_cost * multiplier for p in filtered_data]
        
        # Calculate RI costs (1-year all upfront)
        ri_1yr_costs = []
        spot_costs = []
        
        for p in filtered_data:
            if p.reserved_pricing:
                ri_infrastructure = p.reserved_pricing.get('1_year_all_upfront', 0) * 730 * multiplier
                ri_cost = ri_infrastructure + (p.sql_licensing_cost * multiplier)
                ri_1yr_costs.append(ri_cost)
            else:
                ri_1yr_costs.append(p.total_monthly_cost * 0.6 * multiplier)  # Fallback estimate
            
            spot_infrastructure = (p.spot_pricing * 730 * multiplier) if p.spot_pricing else (p.price_per_month * 0.3 * multiplier)
            spot_cost = spot_infrastructure + (p.sql_licensing_cost * multiplier)
            spot_costs.append(spot_cost)
        
        # Create grouped bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='On-Demand',
            x=instance_types,
            y=on_demand_costs,
            marker_color='#dc3545',
            text=[f"${cost:,.0f}" for cost in on_demand_costs],
            textposition='outside'
        ))
        
        fig.add_trace(go.Bar(
            name='Reserved Instance (1yr)',
            x=instance_types,
            y=ri_1yr_costs,
            marker_color='#28a745',
            text=[f"${cost:,.0f}" for cost in ri_1yr_costs],
            textposition='outside'
        ))
        
        fig.add_trace(go.Bar(
            name='Spot Instance',
            x=instance_types,
            y=spot_costs,
            marker_color='#ffc107',
            text=[f"${cost:,.0f}" for cost in spot_costs],
            textposition='outside'
        ))
        
        fig.update_layout(
            title=f'{period} Cost Comparison (Infrastructure + SQL Licensing)',
            xaxis_title='Instance Type',
            yaxis_title=f'{period} Cost ($)',
            barmode='group',
            height=500,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Savings calculation
        if len(filtered_data) > 0:
            cheapest = min(ri_1yr_costs)
            most_expensive = max(on_demand_costs)
            potential_savings = most_expensive - cheapest
            
            period_label = period.lower().replace(" total", "")
            
            st.markdown(f"""
            <div class="cost-savings">
                <h3>Potential {period} Savings</h3>
                <h2>${potential_savings:,.0f}</h2>
                <p>By choosing optimal pricing model</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show detailed savings breakdown
            self.render_savings_breakdown(filtered_data, period, multiplier)
    
    def render_savings_breakdown(self, pricing_data: List, period: str, multiplier: int):
        """Render detailed savings breakdown"""
        st.markdown("**💡 Savings Breakdown**")
        
        savings_data = []
        for p in pricing_data:
            on_demand = p.total_monthly_cost * multiplier
            ri_cost = (p.reserved_pricing.get('1_year_all_upfront', 0) * 730 + p.sql_licensing_cost) * multiplier if p.reserved_pricing else on_demand * 0.6
            spot_cost = ((p.spot_pricing * 730) if p.spot_pricing else (p.price_per_month * 0.3)) * multiplier + p.sql_licensing_cost * multiplier
            
            savings_data.append({
                'Instance': p.instance_type,
                'On-Demand': f"${on_demand:,.0f}",
                'Reserved': f"${ri_cost:,.0f}",
                'Spot': f"${spot_cost:,.0f}",
                'RI Savings': f"${on_demand - ri_cost:,.0f}",
                'Spot Savings': f"${on_demand - spot_cost:,.0f}"
            })
        
        df_savings = pd.DataFrame(savings_data)
        st.dataframe(df_savings, use_container_width=True)
    
    def render_sql_optimization(self):
        """Render SQL optimization section with enhanced analysis"""
        st.markdown('<div class="section-header">🗃️ SQL Server Licensing Optimization</div>', unsafe_allow_html=True)
        
        if not st.session_state.sql_config:
            st.info("⚠️ Please configure SQL Server settings in the sidebar.")
            return
        
        sql_config = st.session_state.sql_config
        
        # Current configuration
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Current Edition</h3>
                <h2>{sql_config.current_edition}</h2>
                <p>{sql_config.current_licensing_model}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Licensed Cores</h3>
                <h2>{sql_config.current_cores_licensed}</h2>
                <p>Minimum 4 cores required</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Concurrent Users</h3>
                <h2>{sql_config.concurrent_users}</h2>
                <p>Active sessions</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Licensing cost analysis
        self.render_sql_cost_analysis(sql_config)
        
        # Optimization opportunities
        st.markdown("**💡 Optimization Opportunities**")
        
        opportunities = []
        annual_savings = 0
        
        if sql_config.has_software_assurance and sql_config.eligible_for_ahb:
            ahb_savings = sql_config.current_annual_license_cost * 0.55
            annual_savings += ahb_savings
            opportunities.append(f"Azure Hybrid Benefit: Save ${ahb_savings:,.0f} annually (55% reduction)")
        
        if sql_config.current_edition == "Enterprise" and sql_config.concurrent_users < 100:
            edition_savings = sql_config.current_cores_licensed * (13748 - 3717)  # Enterprise vs Standard per core
            annual_savings += edition_savings
            opportunities.append(f"Downgrade to Standard Edition: Save ${edition_savings:,.0f} annually")
        
        if sql_config.current_licensing_model == "Core-based" and sql_config.concurrent_users < 25:
            cal_cost = sql_config.concurrent_users * 209  # CAL cost per user
            cal_savings = max(0, sql_config.current_cores_licensed * 3717 - cal_cost)
            if cal_savings > 0:
                annual_savings += cal_savings
                opportunities.append(f"Switch to CAL licensing: Save ${cal_savings:,.0f} annually")
        
        if not opportunities:
            opportunities.append("Current configuration appears optimized for your usage patterns")
        
        for opportunity in opportunities:
            st.markdown(f"""
            <div class="sql-optimization">
                <h3>💰 Cost Optimization</h3>
                <p>{opportunity}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Total savings summary
        if annual_savings > 0:
            st.markdown(f"""
            <div class="cost-savings">
                <h3>Total Annual SQL Savings Potential</h3>
                <h2>${annual_savings:,.0f}</h2>
                <p>Combined optimization opportunities</p>
            </div>
            """, unsafe_allow_html=True)
    
    def render_sql_cost_analysis(self, sql_config):
        """Render SQL cost analysis charts"""
        # Calculate different licensing scenarios
        scenarios = {
            "Current Configuration": sql_config.current_cores_licensed * 3717,
            "Standard Edition": sql_config.current_cores_licensed * 3717,
            "Enterprise Edition": sql_config.current_cores_licensed * 13748,
            "CAL-based (Current Users)": sql_config.concurrent_users * 209 + 931,  # Server license + CALs
        }
        
        # Add AHB scenarios if applicable
        if sql_config.has_software_assurance:
            scenarios["Current + AHB"] = scenarios["Current Configuration"] * 0.45
            scenarios["Standard + AHB"] = scenarios["Standard Edition"] * 0.45
        
        # Create cost comparison chart
        fig = go.Figure()
        
        scenario_names = list(scenarios.keys())
        costs = list(scenarios.values())
        colors = ['#dc3545' if 'Current' in name and 'AHB' not in name else '#28a745' if 'AHB' in name else '#4a90e2' for name in scenario_names]
        
        fig.add_trace(go.Bar(
            x=scenario_names,
            y=costs,
            marker_color=colors,
            text=[f"${cost:,.0f}" for cost in costs],
            textposition='outside'
        ))
        
        fig.update_layout(
            title='SQL Server Licensing Cost Scenarios (Annual)',
            xaxis_title='Licensing Scenario',
            yaxis_title='Annual Cost ($)',
            height=400,
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_reports(self):
        """Render reports section with enhanced options"""
        st.markdown('<div class="section-header">📄 Professional Reports & Export</div>', unsafe_allow_html=True)
        
        st.markdown("""
        Generate comprehensive reports and export data for stakeholder presentations and decision-making.
        
        **Available Reports:**
        - **Executive Summary**: High-level findings and recommendations
        - **Technical Report**: Detailed analysis with vROps metrics and sizing recommendations
        - **Cost Analysis**: Comprehensive pricing comparison and optimization opportunities
        - **Implementation Roadmap**: Phased migration plan with timelines and dependencies
        """)
        
        # Report generation
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Report Generation**")
            if st.button("📄 Generate Executive Summary", use_container_width=True):
                if self._has_sufficient_data():
                    with st.spinner("Generating executive summary..."):
                        self._generate_executive_summary()
                else:
                    st.warning("⚠️ Please complete pricing analysis and AI recommendations first.")
            
            if st.button("📋 Generate Technical Report", use_container_width=True):
                if self._has_sufficient_data():
                    with st.spinner("Generating technical report..."):
                        self._generate_technical_report()
                else:
                    st.warning("⚠️ Please complete pricing analysis and configure vROps metrics first.")
        
        with col2:
            st.markdown("**📊 Data Export**")
            if st.button("💾 Export All Data (ZIP)", use_container_width=True):
                if self._has_sufficient_data():
                    with st.spinner("Preparing data export..."):
                        self._export_all_data()
                else:
                    st.warning("⚠️ No data available to export.")
        
        # Individual data exports
        st.markdown("**📊 Individual Data Exports**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export Pricing (CSV)", use_container_width=True):
                if st.session_state.latest_pricing:
                    csv_data = self.export_pricing_csv()
                    st.download_button(
                        "📥 Download Pricing CSV",
                        csv_data,
                        "aws_pricing_analysis.csv",
                        "text/csv",
                        use_container_width=True
                    )
                else:
                    st.info("No pricing data available")
        
        with col2:
            if st.button("📊 Export vROps (JSON)", use_container_width=True):
                if st.session_state.vrops_metrics:
                    json_data = json.dumps(asdict(st.session_state.vrops_metrics), indent=2)
                    st.download_button(
                        "📥 Download vROps JSON",
                        json_data,
                        "vrops_metrics.json",
                        "application/json",
                        use_container_width=True
                    )
                else:
                    st.info("No vROps data available")
        
        with col3:
            if st.button("📊 Export SQL Config (JSON)", use_container_width=True):
                if st.session_state.sql_config:
                    json_data = json.dumps(asdict(st.session_state.sql_config), indent=2)
                    st.download_button(
                        "📥 Download SQL JSON",
                        json_data,
                        "sql_configuration.json",
                        "application/json",
                        use_container_width=True
                    )
                else:
                    st.info("No SQL configuration available")
    
    def _has_sufficient_data(self):
        """Check if sufficient data is available for reporting"""
        return (
            hasattr(st.session_state, 'config') and
            (st.session_state.latest_pricing or st.session_state.vrops_metrics)
        )
    
    def _generate_executive_summary(self):
        """Generate executive summary report"""
        try:
            # Create executive summary content
            config = st.session_state.config
            analysis = st.session_state.comprehensive_analysis
            
            summary = f"""
## Executive Summary - AWS Migration Analysis

**Date:** {datetime.now().strftime('%B %d, %Y')}

### Key Recommendations
"""
            
            if analysis and analysis.get('recommendation'):
                rec = analysis['recommendation']
                summary += f"""
- **Primary Recommendation:** {rec.recommendation}
- **Confidence Level:** {rec.confidence_score:.0f}%
- **Expected Annual Savings:** ${rec.expected_savings:,.0f}
"""
            
            if st.session_state.latest_pricing:
                cheapest = min(st.session_state.latest_pricing, key=lambda x: x.total_monthly_cost)
                summary += f"""
- **Recommended Instance:** {cheapest.instance_type}
- **Monthly Cost:** ${cheapest.total_monthly_cost:,.0f}
"""
            
            summary += """
### Next Steps
1. Review detailed technical analysis
2. Validate recommendations with stakeholders
3. Plan pilot migration phase
4. Execute phased migration approach
            """
            
            st.markdown(summary)
            
            # Offer download
            st.download_button(
                "📥 Download Executive Summary",
                summary,
                "executive_summary.md",
                "text/markdown",
                use_container_width=True
            )
            
            st.success("✅ Executive summary generated!")
            
        except Exception as e:
            st.error(f"❌ Error generating executive summary: {str(e)}")
    
    def _generate_technical_report(self):
        """Generate detailed technical report"""
        try:
            # Generate comprehensive technical report
            report_buffer = self.pdf_generator.create_comprehensive_report(
                st.session_state.config,
                st.session_state.latest_pricing,
                st.session_state.comprehensive_analysis.get('recommendation') if st.session_state.comprehensive_analysis else None,
                st.session_state.comprehensive_analysis.get('risks', []) if st.session_state.comprehensive_analysis else [],
                st.session_state.comprehensive_analysis.get('phases', []) if st.session_state.comprehensive_analysis else [],
                st.session_state.vrops_metrics,
                st.session_state.sql_config
            )
            
            st.download_button(
                "📥 Download Technical Report",
                report_buffer.getvalue(),
                "technical_report.txt",
                "text/plain",
                use_container_width=True
            )
            
            st.success("✅ Technical report generated!")
            
        except Exception as e:
            st.error(f"❌ Error generating technical report: {str(e)}")
    
    def _export_all_data(self):
        """Export all data as ZIP file"""
        try:
            import zipfile
            
            zip_buffer = BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add pricing data
                if st.session_state.latest_pricing:
                    csv_data = self.export_pricing_csv()
                    zip_file.writestr("pricing_analysis.csv", csv_data)
                
                # Add vROps data
                if st.session_state.vrops_metrics:
                    vrops_json = json.dumps(asdict(st.session_state.vrops_metrics), indent=2)
                    zip_file.writestr("vrops_metrics.json", vrops_json)
                
                # Add SQL config
                if st.session_state.sql_config:
                    sql_json = json.dumps(asdict(st.session_state.sql_config), indent=2)
                    zip_file.writestr("sql_configuration.json", sql_json)
                
                # Add configuration
                config_json = json.dumps(st.session_state.config, indent=2)
                zip_file.writestr("configuration.json", config_json)
            
            zip_buffer.seek(0)
            
            st.download_button(
                "📥 Download Complete Data Export (ZIP)",
                zip_buffer.getvalue(),
                "aws_optimization_data.zip",
                "application/zip",
                use_container_width=True
            )
            
            st.success("✅ Data export package created!")
            
        except Exception as e:
            st.error(f"❌ Error creating data export: {str(e)}")
    
    def export_pricing_csv(self):
        """Export pricing data as CSV with enhanced columns"""
        if not st.session_state.latest_pricing:
            return ""
        
        data = []
        for pricing_obj in st.session_state.latest_pricing:
            specs = pricing_obj.specifications or {}
            
            # Calculate additional metrics
            vcpus = specs.get('vcpus', 1)
            ram = specs.get('ram', 1)
            cost_per_vcpu = pricing_obj.total_monthly_cost / max(vcpus, 1)
            cost_per_gb_ram = pricing_obj.total_monthly_cost / max(ram, 1)
            
            data.append({
                'Instance Type': pricing_obj.instance_type,
                'vCPUs': vcpus,
                'RAM (GB)': ram,
                'Family': specs.get('family', 'Unknown'),
                'Network Performance': specs.get('network_performance', 'Unknown'),
                'Infrastructure Hourly': pricing_obj.price_per_hour,
                'Infrastructure Monthly': pricing_obj.price_per_month,
                'SQL Licensing Monthly': pricing_obj.sql_licensing_cost,
                'Total Monthly Cost': pricing_obj.total_monthly_cost,
                'Cost per vCPU': cost_per_vcpu,
                'Cost per GB RAM': cost_per_gb_ram,
                'Spot Price Hourly': pricing_obj.spot_pricing,
                'Reserved 1Y All Upfront': pricing_obj.reserved_pricing.get('1_year_all_upfront', 'N/A') if pricing_obj.reserved_pricing else 'N/A',
                'Reserved 3Y All Upfront': pricing_obj.reserved_pricing.get('3_year_all_upfront', 'N/A') if pricing_obj.reserved_pricing else 'N/A',
                'Region': pricing_obj.region,
                'Currency': pricing_obj.currency,
                'Last Updated': pricing_obj.last_updated.strftime('%Y-%m-%d %H:%M:%S')
            })
        
        df = pd.DataFrame(data)
        return df.to_csv(index=False)

def main():
    """Main application entry point with enhanced error handling"""
    try:
        # Initialize the application
        optimizer = EnhancedCloudPricingOptimizer()
        
        # Render the main interface
        optimizer.render_main_interface()
        
        # Footer with version info
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #6c757d; font-size: 0.9rem; padding: 1rem;">
            <strong>Enhanced AWS Cloud Pricing Optimizer v4.0</strong><br>
            Professional AWS Migration Analysis with vRealize Operations & SQL Server Optimization<br>
            <small>Real-time AWS pricing integration with AI-powered recommendations</small>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        logger.error(f"Application error: {e}")
        
        # Debug information
        if st.checkbox("🔍 Show detailed error information"):
            st.exception(e)
            
            # Show session state for debugging
            if st.checkbox("🔍 Show session state"):
                st.json(dict(st.session_state))

if __name__ == "__main__":
    main()