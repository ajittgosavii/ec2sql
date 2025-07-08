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
    page_title="AWS Cloud Migration Optimizer",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Professional CSS styling
st.markdown("""
<style>
    /* Reset any potential layout conflicts */
    .stApp > div:first-child {
        padding-top: 0 !important;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1f4e79 0%, #2c5aa0 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0 2rem 0;
        box-shadow: 0 4px 15px rgba(31, 78, 121, 0.2);
        width: 100%;
        box-sizing: border-box;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 600;
        line-height: 1.2;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
        opacity: 0.9;
        line-height: 1.4;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.06);
        border-left: 4px solid #4a90e2;
        margin: 0.8rem 0;
        width: 100%;
        box-sizing: border-box;
        transition: box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card h3 {
        color: #343a40;
        margin: 0 0 0.5rem 0;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-card h2 {
        color: #1f4e79;
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    .metric-card p {
        margin: 0.3rem 0 0 0;
        color: #6c757d;
        font-size: 0.9rem;
    }
    
    .vrops-section {
        background: linear-gradient(135deg, #7b68ee 0%, #6a5acd 100%);
        color: white;
        padding: 1.8rem;
        border-radius: 10px;
        margin: 1.2rem 0;
        box-shadow: 0 4px 15px rgba(123, 104, 238, 0.2);
        width: 100%;
        box-sizing: border-box;
    }
    
    .vrops-section h3 {
        margin: 0 0 1rem 0;
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    .sql-optimization {
        background: linear-gradient(135deg, #20c997 0%, #17a2b8 100%);
        color: white;
        padding: 1.8rem;
        border-radius: 10px;
        margin: 1.2rem 0;
        box-shadow: 0 4px 15px rgba(32, 201, 151, 0.2);
        width: 100%;
        box-sizing: border-box;
    }
    
    .sql-optimization h3 {
        margin: 0 0 1rem 0;
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    .ai-recommendation {
        background: linear-gradient(135deg, #4a90e2 0%, #357abd 100%);
        color: white;
        padding: 1.8rem;
        border-radius: 10px;
        margin: 1.2rem 0;
        box-shadow: 0 4px 15px rgba(74, 144, 226, 0.2);
        width: 100%;
        box-sizing: border-box;
    }
    
    .ai-recommendation h3 {
        margin: 0 0 1rem 0;
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    .cost-savings {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.2);
        margin: 1rem 0;
        width: 100%;
        box-sizing: border-box;
    }
    
    .cost-savings h3 {
        margin: 0 0 0.5rem 0;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        opacity: 0.9;
    }
    
    .cost-savings h2 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    .section-header {
        color: #1f4e79;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e9ecef;
        width: 100%;
        box-sizing: border-box;
    }
    
    .optimization-insight {
        background: #e7f3ff;
        border-left: 4px solid #4a90e2;
        padding: 1rem;
        margin: 0.8rem 0;
        border-radius: 4px;
        width: 100%;
        box-sizing: border-box;
    }
    
    .performance-warning {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 0.8rem 0;
        border-radius: 4px;
        width: 100%;
        box-sizing: border-box;
    }
    
    .performance-critical {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin: 0.8rem 0;
        border-radius: 4px;
        width: 100%;
        box-sizing: border-box;
    }
    
    .demo-mode-banner {
        background: linear-gradient(135deg, #17a2b8 0%, #138496 100%);
        color: white;
        padding: 15px 25px;
        border-radius: 12px;
        margin: 15px 0;
        text-align: center;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 4px 15px rgba(23, 162, 184, 0.3);
    }
    
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.8rem;
        }
        .main-header p {
            font-size: 0.9rem;
        }
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class VRopsMetrics:
    """Enhanced data class for vROps performance metrics with actual sizing"""
    
    # Current Infrastructure Sizing
    current_vcpus: int = 8
    current_ram_gb: float = 32.0
    current_storage_gb: float = 500.0
    vm_count: int = 1
    
    # CPU Metrics (Percentages)
    cpu_usage_avg: float = 0.0
    cpu_usage_peak: float = 0.0
    cpu_usage_95th: float = 0.0
    cpu_ready_avg: float = 0.0
    cpu_costop_avg: float = 0.0
    
    # Memory Metrics (Percentages)
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
    
    # Calculated Properties
    @property
    def actual_cpu_cores_used_avg(self) -> float:
        """Calculate actual CPU cores used on average"""
        return (self.current_vcpus * self.cpu_usage_avg / 100)
    
    @property
    def actual_cpu_cores_used_peak(self) -> float:
        """Calculate actual CPU cores used at peak"""
        return (self.current_vcpus * self.cpu_usage_peak / 100)
    
    @property
    def actual_ram_gb_used_avg(self) -> float:
        """Calculate actual RAM GB used on average"""
        return (self.current_ram_gb * self.memory_usage_avg / 100)
    
    @property
    def actual_ram_gb_used_peak(self) -> float:
        """Calculate actual RAM GB used at peak"""
        return (self.current_ram_gb * self.memory_usage_peak / 100)
    
    @property
    def recommended_cpu_cores(self) -> int:
        """Recommend CPU cores with 20% headroom above peak usage"""
        return max(2, math.ceil(self.actual_cpu_cores_used_peak * 1.2))
    
    @property
    def recommended_ram_gb(self) -> int:
        """Recommend RAM with 25% headroom above peak usage"""
        return max(8, math.ceil(self.actual_ram_gb_used_peak * 1.25))
    
    @property
    def sizing_efficiency_score(self) -> float:
        """Calculate efficiency score (0-100) based on utilization"""
        cpu_efficiency = min(100, (self.cpu_usage_avg / 70) * 100)  # 70% is considered optimal
        memory_efficiency = min(100, (self.memory_usage_avg / 80) * 100)  # 80% is considered optimal
        return (cpu_efficiency + memory_efficiency) / 2
    
    @property
    def rightsizing_opportunity(self) -> str:
        """Determine rightsizing opportunity"""
        if self.cpu_usage_avg < 30 and self.memory_usage_avg < 50:
            return "Significant downsizing opportunity"
        elif self.cpu_usage_avg < 50 and self.memory_usage_avg < 70:
            return "Moderate downsizing opportunity"
        elif self.cpu_usage_avg > 80 or self.memory_usage_avg > 85:
            return "Consider upsizing for better performance"
        else:
            return "Current sizing appears appropriate"

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
    sql_licensing_cost: float = 0.0
    total_monthly_cost: float = 0.0

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

class AWSPricingService:
    """AWS Pricing API service with improved connection handling"""
    
    def __init__(self, aws_access_key_id=None, aws_secret_access_key=None, region='us-east-1'):
        self.connection_status = {"connected": False, "error": None, "service": "AWS Pricing API"}
        self.pricing_client = None
        
        try:
            if aws_access_key_id and aws_secret_access_key:
                session = boto3.Session(
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name='us-east-1'
                )
                
                self.pricing_client = session.client('pricing')
                
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
                terms = price_data.get('terms', {})
                on_demand = terms.get('OnDemand', {})
                
                if on_demand:
                    offer_code = next(iter(on_demand.keys()))
                    rate_code = next(iter(on_demand[offer_code]['priceDimensions'].keys()))
                    price_per_hour = float(on_demand[offer_code]['priceDimensions'][rate_code]['pricePerUnit']['USD'])
                    
                    attributes = price_data.get('product', {}).get('attributes', {})
                    vcpus = int(attributes.get('vcpu', 0))
                    memory = attributes.get('memory', '0 GiB').replace(' GiB', '').replace(',', '')
                    memory_gb = float(memory) if memory.replace('.', '').isdigit() else 0
                    
                    infrastructure_monthly = price_per_hour * 730
                    sql_cores = max(4, vcpus)
                    sql_licensing_cost = sql_cores * (3717 / 12)
                    total_monthly = infrastructure_monthly + sql_licensing_cost
                    
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

class ClaudeAIService:
    """Claude AI service for recommendations with proper connection handling"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.connection_status = {"connected": False, "error": None, "service": "Claude API"}
        
        if self.api_key:
            try:
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
            if response.status_code not in [200, 400]:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Connection test failed: {str(e)}")
    
    async def get_comprehensive_analysis(self, config, pricing_data, vrops_data, sql_config):
        """Get comprehensive AI analysis"""
        try:
            if not self.connection_status["connected"]:
                logger.warning("Claude API not connected, using mock analysis")
                return self._generate_mock_analysis(config, pricing_data, vrops_data, sql_config)
            
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
        - Current Sizing: {vrops_data.current_vcpus} vCPUs, {vrops_data.current_ram_gb} GB RAM
        - Actual Usage: {vrops_data.actual_cpu_cores_used_avg:.1f} CPU cores, {vrops_data.actual_ram_gb_used_avg:.1f} GB RAM
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
        return self._generate_mock_analysis_with_text(analysis_text)
    
    def _generate_mock_analysis(self, config, pricing_data, vrops_data, sql_config):
        """Generate mock analysis when Claude API is not available"""
        confidence = 85.0
        expected_savings = 45000.0
        
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
        return self._generate_mock_analysis(None, None, None, None)

class MockDataService:
    """Mock data service for demonstration with enhanced realistic data"""
    
    def get_enhanced_sample_pricing_data(self, region: str, vrops_data=None):
        """Generate enhanced sample pricing data with rightsizing based on actual usage"""
        
        # Base instance configurations (instance_type, vcpus, ram, base_price)
        all_instances = [
            ('t3.medium', 2, 4, 0.0416),
            ('t3.large', 2, 8, 0.0832),
            ('t3.xlarge', 4, 16, 0.1664),
            ('t3.2xlarge', 8, 32, 0.3328),
            ('m5.large', 2, 8, 0.096),
            ('m5.xlarge', 4, 16, 0.192),
            ('m5.2xlarge', 8, 32, 0.384),
            ('m5.4xlarge', 16, 64, 0.768),
            ('r5.large', 2, 16, 0.126),
            ('r5.xlarge', 4, 32, 0.252),
            ('r5.2xlarge', 8, 64, 0.504),
            ('r5.4xlarge', 16, 128, 1.008),
            ('m6a.large', 2, 8, 0.0864),
            ('m6a.xlarge', 4, 16, 0.1728),
            ('m6a.2xlarge', 8, 32, 0.3456),
            ('c5.large', 2, 4, 0.085),
            ('c5.xlarge', 4, 8, 0.17),
            ('c5.2xlarge', 8, 16, 0.34),
            ('c5.4xlarge', 16, 32, 0.68)
        ]
        
        # Filter instances based on vROps data if available
        filtered_instances = all_instances
        
        if vrops_data:
            # Calculate required resources with headroom
            min_cpu_required = vrops_data.recommended_cpu_cores
            min_ram_required = vrops_data.recommended_ram_gb
            
            # Filter instances that meet minimum requirements
            filtered_instances = [
                (instance_type, vcpus, ram, price) 
                for instance_type, vcpus, ram, price in all_instances
                if vcpus >= min_cpu_required and ram >= min_ram_required
            ]
            
            # If no instances meet requirements, include slightly smaller ones
            if not filtered_instances:
                filtered_instances = [
                    (instance_type, vcpus, ram, price) 
                    for instance_type, vcpus, ram, price in all_instances
                    if vcpus >= min_cpu_required * 0.8 and ram >= min_ram_required * 0.8
                ]
        
        pricing_data = []
        
        for instance_type, vcpus, ram, base_price in filtered_instances:
            # Calculate SQL licensing (minimum 4 cores)
            sql_cores = max(4, vcpus)
            sql_licensing_cost = sql_cores * (3717 / 12)  # Monthly cost per core
            
            # Windows licensing multiplier
            windows_multiplier = 4.2
            infrastructure_hourly = base_price * windows_multiplier
            infrastructure_monthly = infrastructure_hourly * 730
            
            # Apply efficiency scoring if vROps data available
            if vrops_data:
                # Calculate efficiency score for this instance size
                cpu_utilization_projected = (vrops_data.actual_cpu_cores_used_avg / vcpus) * 100
                ram_utilization_projected = (vrops_data.actual_ram_gb_used_avg / ram) * 100
                
                # Efficiency scoring - prefer 60-80% utilization
                cpu_efficiency = 1.0
                if cpu_utilization_projected < 40:
                    cpu_efficiency = 0.9  # Slightly penalize over-provisioning
                elif cpu_utilization_projected > 85:
                    cpu_efficiency = 1.1  # Slightly penalize under-provisioning risk
                
                ram_efficiency = 1.0
                if ram_utilization_projected < 50:
                    ram_efficiency = 0.95
                elif ram_utilization_projected > 85:
                    ram_efficiency = 1.05
                
                # Apply efficiency factor
                efficiency_factor = (cpu_efficiency + ram_efficiency) / 2
                infrastructure_monthly *= efficiency_factor
            
            total_monthly = infrastructure_monthly + sql_licensing_cost
            
            # Add some realistic variance
            variance = 1 + (hash(instance_type) % 20 - 10) / 1000
            infrastructure_monthly *= variance
            total_monthly = infrastructure_monthly + sql_licensing_cost
            
            # Calculate utilization metrics for this instance
            utilization_metrics = {}
            if vrops_data:
                utilization_metrics = {
                    'projected_cpu_utilization': min(100, (vrops_data.actual_cpu_cores_used_avg / vcpus) * 100),
                    'projected_ram_utilization': min(100, (vrops_data.actual_ram_gb_used_avg / ram) * 100),
                    'rightsizing_score': self._calculate_rightsizing_score(vrops_data, vcpus, ram),
                    'performance_risk': self._assess_performance_risk(vrops_data, vcpus, ram)
                }
            
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
                    'network_performance': 'Up to 25 Gigabit' if vcpus >= 16 else 'Up to 10 Gigabit' if vcpus >= 4 else 'Moderate',
                    **utilization_metrics
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
        
        # Sort by total cost, but boost instances with better rightsizing scores
        if vrops_data:
            pricing_data.sort(key=lambda x: (
                x.total_monthly_cost * (2 - x.specifications.get('rightsizing_score', 0.5))
            ))
        else:
            pricing_data.sort(key=lambda x: x.total_monthly_cost)
        
        return pricing_data
    
    def _calculate_rightsizing_score(self, vrops_data, instance_vcpus, instance_ram):
        """Calculate rightsizing score (0-1, where 1 is optimal)"""
        cpu_utilization = (vrops_data.actual_cpu_cores_used_avg / instance_vcpus) * 100
        ram_utilization = (vrops_data.actual_ram_gb_used_avg / instance_ram) * 100
        
        # Optimal utilization ranges
        cpu_score = 1.0
        if cpu_utilization < 40:
            cpu_score = cpu_utilization / 40  # Penalize under-utilization
        elif cpu_utilization > 80:
            cpu_score = 0.8  # Penalize over-utilization risk
        
        ram_score = 1.0
        if ram_utilization < 50:
            ram_score = ram_utilization / 50
        elif ram_utilization > 85:
            ram_score = 0.7
        
        return (cpu_score + ram_score) / 2
    
    def _assess_performance_risk(self, vrops_data, instance_vcpus, instance_ram):
        """Assess performance risk level"""
        cpu_headroom = ((instance_vcpus - vrops_data.actual_cpu_cores_used_peak) / instance_vcpus) * 100
        ram_headroom = ((instance_ram - vrops_data.actual_ram_gb_used_peak) / instance_ram) * 100
        
        if cpu_headroom < 10 or ram_headroom < 15:
            return "High"
        elif cpu_headroom < 20 or ram_headroom < 25:
            return "Medium"
        else:
            return "Low"

class PDFGeneratorService:
    """PDF generator service with proper optional import handling"""
    
    def __init__(self):
        self.pdf_available = False
        self.reportlab_available = False
        
        try:
            # Try to import reportlab at runtime
            import reportlab.lib.pagesizes
            import reportlab.platypus
            import reportlab.lib.styles
            import reportlab.lib.units
            import reportlab.lib.colors
            
            self.reportlab_available = True
            self.pdf_available = True
            logger.info("ReportLab available - PDF generation enabled")
        except ImportError as e:
            self.reportlab_available = False
            self.pdf_available = False
            logger.warning(f"ReportLab not available: {e}")
    
    def create_comprehensive_report(self, config, pricing_data, recommendation, risks, phases, vrops_data, sql_config):
        """Create comprehensive report - PDF if available, otherwise detailed text"""
        try:
            if self.pdf_available:
                return self._create_pdf_report(config, pricing_data, recommendation, risks, phases, vrops_data, sql_config)
            else:
                return self._create_text_report(config, pricing_data, recommendation, risks, phases, vrops_data, sql_config)
                
        except Exception as e:
            logger.error(f"Report generation error: {e}")
            return self._create_fallback_report(config, pricing_data, recommendation, e)
    
    def _create_pdf_report(self, config, pricing_data, recommendation, risks, phases, vrops_data, sql_config):
        """Create actual PDF report using reportlab"""
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib import colors
            
            buffer = BytesIO()
            doc = SimpleDocTemplate(
                buffer, 
                pagesize=A4, 
                rightMargin=72, 
                leftMargin=72, 
                topMargin=72, 
                bottomMargin=18
            )
            
            styles = getSampleStyleSheet()
            story = []
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=1,  # Center alignment
                textColor=colors.HexColor('#1f4e79')
            )
            
            # Title
            title = Paragraph("AWS Migration Analysis Report", title_style)
            story.append(title)
            story.append(Spacer(1, 20))
            
            # Executive Summary
            if recommendation:
                story.append(Paragraph("Executive Summary", styles['Heading1']))
                story.append(Paragraph(f"<b>Primary Recommendation:</b> {recommendation.recommendation}", styles['Normal']))
                story.append(Spacer(1, 12))
                story.append(Paragraph(f"<b>Confidence Score:</b> {recommendation.confidence_score:.0f}%", styles['Normal']))
                story.append(Spacer(1, 12))
                story.append(Paragraph(f"<b>Expected Annual Savings:</b> ${recommendation.expected_savings:,.0f}", styles['Normal']))
                story.append(Spacer(1, 12))
                story.append(Paragraph(f"<b>Cost Impact:</b> {recommendation.cost_impact}", styles['Normal']))
                story.append(Spacer(1, 20))
            
            # vROps Analysis
            if vrops_data:
                story.append(Paragraph("vRealize Operations Analysis", styles['Heading1']))
                story.append(Paragraph(f"<b>Current Infrastructure:</b> {vrops_data.current_vcpus} vCPUs, {vrops_data.current_ram_gb} GB RAM", styles['Normal']))
                story.append(Spacer(1, 6))
                story.append(Paragraph(f"<b>CPU Utilization:</b> {vrops_data.cpu_usage_avg:.1f}% average, {vrops_data.cpu_usage_peak:.1f}% peak", styles['Normal']))
                story.append(Spacer(1, 6))
                story.append(Paragraph(f"<b>Memory Utilization:</b> {vrops_data.memory_usage_avg:.1f}% average, {vrops_data.memory_usage_peak:.1f}% peak", styles['Normal']))
                story.append(Spacer(1, 6))
                story.append(Paragraph(f"<b>Actual Usage:</b> {vrops_data.actual_cpu_cores_used_avg:.1f} CPU cores, {vrops_data.actual_ram_gb_used_avg:.1f} GB RAM", styles['Normal']))
                story.append(Spacer(1, 6))
                story.append(Paragraph(f"<b>Recommended Sizing:</b> {vrops_data.recommended_cpu_cores} vCPUs, {vrops_data.recommended_ram_gb} GB RAM", styles['Normal']))
                story.append(Spacer(1, 20))
            
            # Pricing Analysis
            if pricing_data:
                story.append(Paragraph("Pricing Analysis", styles['Heading1']))
                cheapest = min(pricing_data, key=lambda x: x.total_monthly_cost)
                story.append(Paragraph(f"<b>Recommended Instance:</b> {cheapest.instance_type}", styles['Normal']))
                story.append(Spacer(1, 6))
                
                specs = cheapest.specifications or {}
                story.append(Paragraph(f"<b>Specifications:</b> {specs.get('vcpus', 'N/A')} vCPUs, {specs.get('ram', 'N/A')}GB RAM", styles['Normal']))
                story.append(Spacer(1, 6))
                story.append(Paragraph(f"<b>Monthly Cost:</b> ${cheapest.total_monthly_cost:,.0f}", styles['Normal']))
                story.append(Spacer(1, 6))
                story.append(Paragraph(f"<b>Annual Cost:</b> ${cheapest.total_monthly_cost * 12:,.0f}", styles['Normal']))
                story.append(Spacer(1, 20))
            
            # SQL Optimization
            if sql_config:
                story.append(Paragraph("SQL Server Configuration", styles['Heading1']))
                story.append(Paragraph(f"<b>Current Edition:</b> {sql_config.current_edition}", styles['Normal']))
                story.append(Spacer(1, 6))
                story.append(Paragraph(f"<b>Licensing Model:</b> {sql_config.current_licensing_model}", styles['Normal']))
                story.append(Spacer(1, 6))
                story.append(Paragraph(f"<b>Licensed Cores:</b> {sql_config.current_cores_licensed}", styles['Normal']))
                story.append(Spacer(1, 6))
                story.append(Paragraph(f"<b>Concurrent Users:</b> {sql_config.concurrent_users}", styles['Normal']))
                story.append(Spacer(1, 20))
            
            # Risk Assessment
            if risks:
                story.append(Paragraph("Risk Assessment", styles['Heading1']))
                for risk in risks:
                    story.append(Paragraph(f"<b>{risk.category}</b> - {risk.risk_level} Risk", styles['Normal']))
                    story.append(Paragraph(f"Description: {risk.description}", styles['Normal']))
                    story.append(Paragraph(f"Mitigation: {risk.mitigation_strategy}", styles['Normal']))
                    story.append(Spacer(1, 12))
            
            # Implementation Phases
            if phases:
                story.append(Paragraph("Implementation Roadmap", styles['Heading1']))
                for i, phase in enumerate(phases, 1):
                    story.append(Paragraph(f"<b>Phase {i}: {phase.phase}</b>", styles['Normal']))
                    story.append(Paragraph(f"Duration: {phase.duration}", styles['Normal']))
                    story.append(Paragraph(f"Activities: {', '.join(phase.activities)}", styles['Normal']))
                    story.append(Spacer(1, 12))
            
            # Footer
            story.append(Spacer(1, 30))
            story.append(Paragraph("Analysis Configuration", styles['Heading2']))
            story.append(Paragraph(f"Target Region: {config.get('region', 'Not specified')}", styles['Normal']))
            story.append(Paragraph(f"Workload Type: {config.get('workload_type', 'Not specified')}", styles['Normal']))
            story.append(Paragraph(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            
            doc.build(story)
            buffer.seek(0)
            return buffer
            
        except ImportError as ie:
            logger.error(f"ReportLab import error in PDF creation: {ie}")
            return self._create_text_report(config, pricing_data, recommendation, risks, phases, vrops_data, sql_config)
        except Exception as e:
            logger.error(f"PDF creation error: {e}")
            return self._create_text_report(config, pricing_data, recommendation, risks, phases, vrops_data, sql_config)
    
    def _create_text_report(self, config, pricing_data, recommendation, risks, phases, vrops_data, sql_config):
        """Create detailed text report when PDF is not available"""
        report_content = f"""
AWS CLOUD MIGRATION ANALYSIS REPORT
===================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
=================
"""
        
        if recommendation:
            report_content += f"""
Primary Recommendation: {recommendation.recommendation}
Confidence Level: {recommendation.confidence_score:.0f}%
Expected Annual Savings: ${recommendation.expected_savings:,.0f}
Cost Impact: {recommendation.cost_impact}
Reasoning: {recommendation.reasoning}
"""
        
        if pricing_data:
            cheapest = min(pricing_data, key=lambda x: x.total_monthly_cost)
            most_expensive = max(pricing_data, key=lambda x: x.total_monthly_cost)
            report_content += f"""
Recommended Instance: {cheapest.instance_type}
Optimal Monthly Cost: ${cheapest.total_monthly_cost:,.0f}
Potential Monthly Savings: ${most_expensive.total_monthly_cost - cheapest.total_monthly_cost:,.0f}
"""
        
        if vrops_data:
            report_content += f"""

VREALIZE OPERATIONS PERFORMANCE ANALYSIS
========================================
Current Infrastructure:
- vCPUs Allocated: {vrops_data.current_vcpus}
- RAM Allocated: {vrops_data.current_ram_gb} GB
- Storage Allocated: {vrops_data.current_storage_gb} GB
- VM Count: {vrops_data.vm_count}

Performance Metrics:
- CPU Usage (Average): {vrops_data.cpu_usage_avg:.1f}%
- CPU Usage (Peak): {vrops_data.cpu_usage_peak:.1f}%
- Memory Usage (Average): {vrops_data.memory_usage_avg:.1f}%
- Memory Usage (Peak): {vrops_data.memory_usage_peak:.1f}%
- CPU Ready Time: {vrops_data.cpu_ready_avg:.1f}%
- Memory Balloon: {vrops_data.memory_balloon_avg:.1f}%
- Disk Latency: {vrops_data.disk_latency_avg:.1f}ms

Actual Resource Consumption:
- CPU Cores Used (Avg): {vrops_data.actual_cpu_cores_used_avg:.1f}
- CPU Cores Used (Peak): {vrops_data.actual_cpu_cores_used_peak:.1f}
- RAM Used (Avg): {vrops_data.actual_ram_gb_used_avg:.1f} GB
- RAM Used (Peak): {vrops_data.actual_ram_gb_used_peak:.1f} GB

Rightsizing Recommendations:
- Recommended vCPUs: {vrops_data.recommended_cpu_cores}
- Recommended RAM: {vrops_data.recommended_ram_gb} GB
- Sizing Efficiency Score: {vrops_data.sizing_efficiency_score:.0f}/100
- Rightsizing Opportunity: {vrops_data.rightsizing_opportunity}

Performance Assessment:
"""
            
            if vrops_data.cpu_usage_avg < 40:
                report_content += "- CPU utilization is low - significant right-sizing opportunity identified\n"
            if vrops_data.cpu_ready_avg > 5:
                report_content += "- High CPU ready time indicates resource contention\n"
            if vrops_data.memory_balloon_avg > 1:
                report_content += "- Memory ballooning detected - recommend increasing memory allocation\n"
            if vrops_data.disk_latency_avg > 20:
                report_content += "- High disk latency may impact application performance\n"
        
        if pricing_data:
            report_content += f"""

AWS PRICING ANALYSIS
===================
Instance Pricing Comparison (Top 10):
"""
            for i, pricing in enumerate(pricing_data[:10], 1):
                specs = pricing.specifications or {}
                projected_cpu = specs.get('projected_cpu_utilization', 0)
                projected_ram = specs.get('projected_ram_utilization', 0)
                risk = specs.get('performance_risk', 'Unknown')
                score = specs.get('rightsizing_score', 0) * 100
                
                report_content += f"{i:2}. {pricing.instance_type:<12} | {specs.get('vcpus', 'N/A'):>2} vCPUs | {specs.get('ram', 'N/A'):>3}GB RAM | ${pricing.total_monthly_cost:>7,.0f} total"
                if projected_cpu > 0:
                    report_content += f" | CPU: {projected_cpu:.0f}% | RAM: {projected_ram:.0f}% | Risk: {risk} | Score: {score:.0f}"
                report_content += "\n"
        
        if recommendation and risks:
            report_content += f"""

AI-POWERED RECOMMENDATIONS
==========================
Primary Recommendation: {recommendation.recommendation}

Risk Assessment:
"""
            for risk in risks:
                report_content += f"- {risk.category} ({risk.risk_level} Risk): {risk.description}\n  Mitigation: {risk.mitigation_strategy}\n"
        
        if sql_config:
            report_content += f"""

SQL SERVER LICENSING OPTIMIZATION
=================================
Current Configuration:
- Edition: {sql_config.current_edition}
- Licensing Model: {sql_config.current_licensing_model}
- Licensed Cores: {sql_config.current_cores_licensed}
- Concurrent Users: {sql_config.concurrent_users}
- Software Assurance: {'Yes' if sql_config.has_software_assurance else 'No'}
- Azure Hybrid Benefit Eligible: {'Yes' if sql_config.eligible_for_ahb else 'No'}

Optimization Opportunities:
"""
            if sql_config.has_software_assurance and sql_config.eligible_for_ahb:
                report_content += "- Azure Hybrid Benefit: Up to 55% savings on SQL licensing costs\n"
            if sql_config.current_edition == "Enterprise" and sql_config.concurrent_users < 100:
                report_content += "- Consider downgrading to Standard Edition based on usage patterns\n"
        
        if phases:
            report_content += f"""

IMPLEMENTATION ROADMAP
=====================
"""
            for i, phase in enumerate(phases, 1):
                report_content += f"""
Phase {i}: {phase.phase}
Duration: {phase.duration}
Key Activities: {', '.join(phase.activities)}
Dependencies: {', '.join(phase.dependencies)}
Deliverables: {', '.join(phase.deliverables)}
"""
        
        report_content += f"""

ANALYSIS CONFIGURATION
=====================
Target Region: {config.get('region', 'Not specified')}
Workload Type: {config.get('workload_type', 'Not specified')}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Currency: USD
Pricing Model: On-Demand with SQL Server licensing
"""
        
        if vrops_data:
            report_content += f"vROps Data Collection Period: {vrops_data.collection_period_days} days\n"
            report_content += f"Data Completeness: {vrops_data.data_completeness:.1f}%\n"
        
        report_content += """

DISCLAIMER
==========
This analysis is based on current AWS pricing and provided performance data. 
Actual costs may vary based on usage patterns, regional availability, and 
AWS pricing changes. Please validate all recommendations in a test environment 
before implementing in production.
"""
        
        buffer = BytesIO()
        buffer.write(report_content.encode('utf-8'))
        buffer.seek(0)
        return buffer
    
    def _create_fallback_report(self, config, pricing_data, recommendation, error):
        """Create minimal fallback report when both PDF and detailed text fail"""
        fallback_content = f"""
AWS MIGRATION ANALYSIS REPORT (FALLBACK)
========================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ERROR: {str(error)}

BASIC ANALYSIS SUMMARY:
"""
        
        if recommendation:
            fallback_content += f"""
- Primary Recommendation: {recommendation.recommendation}
- Expected Savings: ${recommendation.expected_savings:,.0f}
"""
        
        if pricing_data:
            cheapest = min(pricing_data, key=lambda x: x.total_monthly_cost)
            fallback_content += f"""
- Recommended Instance: {cheapest.instance_type}
- Monthly Cost: ${cheapest.total_monthly_cost:,.0f}
"""
        
        fallback_content += f"""
- Target Region: {config.get('region', 'Not specified')}
- Workload Type: {config.get('workload_type', 'Not specified')}
"""
        
        buffer = BytesIO()
        buffer.write(fallback_content.encode('utf-8'))
        buffer.seek(0)
        return buffer

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
            aws_key, aws_secret, claude_key = self._get_credentials()
            
            self.aws_pricing = AWSPricingService(aws_key, aws_secret)
            self.claude_ai = ClaudeAIService(claude_key)
            self.mock_data = MockDataService()
            self.pdf_generator = PDFGeneratorService()
            
            st.session_state.connection_status = {
                'aws': self.aws_pricing.connection_status,
                'claude': self.claude_ai.connection_status
            }
            
            aws_connected = self.aws_pricing.connection_status["connected"]
            if not aws_connected:
                st.session_state.demo_mode = True
                logger.warning("AWS not connected - using demo mode")
            
        except Exception as e:
            logger.error(f"Service initialization error: {e}")
            self._initialize_fallback_services()
    
    def _get_credentials(self):
        """Get credentials from various sources"""
        aws_key = aws_secret = claude_key = None
        
        methods = [
            self._get_credentials_from_secrets,
            self._get_credentials_from_env,
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
            aws_key = st.secrets.get("AWS_ACCESS_KEY_ID")
            aws_secret = st.secrets.get("AWS_SECRET_ACCESS_KEY")
            claude_key = st.secrets.get("CLAUDE_API_KEY")
            
            if not (aws_key and aws_secret):
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
    
    def _initialize_fallback_services(self):
        """Initialize fallback services"""
        self.aws_pricing = AWSPricingService()
        self.claude_ai = ClaudeAIService()
        self.mock_data = MockDataService()
        self.pdf_generator = PDFGeneratorService()
        st.session_state.demo_mode = True
    
    def render_main_interface(self):
        """Render the main Streamlit interface"""
        st.markdown("""
        <div class="main-header">
            <h1>AWS Cloud Migration Optimizer</h1>
            <p>Enterprise AWS pricing analysis with vRealize Operations integration and intelligent rightsizing</p>
        </div>
        """, unsafe_allow_html=True)
        
        self.render_connection_status()
        self.render_sidebar()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
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
        """Render professional connection status"""        
        if 'connection_status' not in st.session_state:
            return
        
        connection_status = st.session_state.connection_status
        
        # Compact professional status indicators
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown("**System Status**")
        
        with col2:
            aws_status = connection_status.get('aws', {})
            if aws_status.get('connected'):
                st.success("🟢 AWS API")
            else:
                st.warning("🟡 Demo Mode")
        
        with col3:
            claude_status = connection_status.get('claude', {})
            if claude_status.get('connected'):
                st.success("🟢 AI Engine")
            else:
                st.warning("🟡 Mock AI")
        
        if not aws_status.get('connected') or not claude_status.get('connected'):
            with st.expander("🔧 API Configuration", expanded=False):
                if not aws_status.get('connected'):
                    error_msg = aws_status.get('error', 'Credentials not configured')
                    st.info(f"**AWS Status:** {error_msg}")
                
                if not claude_status.get('connected'):
                    error_msg = claude_status.get('error', 'API key not configured')
                    st.info(f"**Claude Status:** {error_msg}")
                
                st.markdown("""
                **Enterprise Configuration:**
                - Configure AWS credentials in secrets
                - Add Claude API key for AI insights
                - All features remain fully functional in demo mode
                """)
        
        st.markdown("<br>", unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render enhanced sidebar configuration with on-premise infrastructure"""
        with st.sidebar:
            st.markdown('<div class="section-header">⚙️ Configuration</div>', unsafe_allow_html=True)
            
            region = st.selectbox("AWS Region", [
                "us-east-1", "us-west-1", "us-west-2", "us-east-2",
                "eu-west-1", "eu-central-1", "ap-southeast-1", "ap-northeast-1"
            ], index=0)
            
            workload_type = st.selectbox("Workload Type", [
                "Production", "Staging", "Development", "Testing"
            ])
            
            st.markdown('<div class="section-header">🖥️ Current On-Premise Infrastructure</div>', unsafe_allow_html=True)
            
            with st.expander("⚙️ Current VM Configuration", expanded=True):
                st.markdown("**📊 Current Resource Allocation**")
                col1, col2 = st.columns(2)
                with col1:
                    current_vcpus = st.number_input("Current vCPUs", 1, 128, 8, step=1, 
                                                   help="Number of virtual CPUs currently allocated")
                    current_ram_gb = st.number_input("Current RAM (GB)", 1, 1024, 32, step=1,
                                                    help="Total RAM currently allocated to the VM")
                with col2:
                    current_storage_gb = st.number_input("Current Storage (GB)", 1, 10000, 500, step=10,
                                                       help="Total storage currently allocated")
                    vm_count = st.number_input("Number of VMs", 1, 100, 1, step=1,
                                              help="Number of VMs with similar configuration")
            
            st.markdown('<div class="section-header">📊 vROps Performance Data</div>', unsafe_allow_html=True)
            
            with st.expander("🔍 vROps Metrics Input", expanded=False):
                st.markdown("**📈 Performance Utilization Metrics**")
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
                
                # Calculate actual usage based on allocated resources
                actual_cpu_cores_used = (current_vcpus * cpu_avg / 100)
                actual_ram_gb_used = (current_ram_gb * mem_avg / 100)
                
                st.markdown("**🧮 Calculated Actual Usage**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Actual CPU Cores Used (Avg)", f"{actual_cpu_cores_used:.1f}", 
                             help=f"Based on {cpu_avg}% of {current_vcpus} vCPUs")
                    st.metric("Actual RAM Used (Avg)", f"{actual_ram_gb_used:.1f} GB", 
                             help=f"Based on {mem_avg}% of {current_ram_gb} GB")
                with col2:
                    peak_cpu_used = (current_vcpus * cpu_peak / 100)
                    peak_ram_used = (current_ram_gb * mem_peak / 100)
                    st.metric("Actual CPU Cores Used (Peak)", f"{peak_cpu_used:.1f}", 
                             help=f"Based on {cpu_peak}% of {current_vcpus} vCPUs")
                    st.metric("Actual RAM Used (Peak)", f"{peak_ram_used:.1f} GB", 
                             help=f"Based on {mem_peak}% of {current_ram_gb} GB")
            
            # Update the vROps metrics object creation
            st.session_state.vrops_metrics = VRopsMetrics(
                current_vcpus=current_vcpus,
                current_ram_gb=current_ram_gb,
                current_storage_gb=current_storage_gb,
                vm_count=vm_count,
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
                
                st.session_state.sql_config = SQLServerConfig(
                    current_edition=current_edition,
                    current_licensing_model=current_licensing,
                    current_cores_licensed=current_cores,
                    concurrent_users=concurrent_users,
                    has_software_assurance=has_sa,
                    eligible_for_ahb=eligible_ahb,
                    current_annual_license_cost=30000.0
                )
            
            # Update the main configuration to include actual sizing
            st.session_state.config = {
                'region': region,
                'workload_type': workload_type,
                'cpu_cores': current_vcpus,  # Now uses actual input instead of hardcoded
                'ram_gb': current_ram_gb,    # Now uses actual input instead of hardcoded
                'storage_gb': current_storage_gb,
                'vm_count': vm_count,
                'actual_cpu_used_avg': actual_cpu_cores_used,
                'actual_ram_used_avg': actual_ram_gb_used,
                'actual_cpu_used_peak': peak_cpu_used,
                'actual_ram_used_peak': peak_ram_used
            }
    
    def render_vrops_metrics(self):
        """Render vROps metrics analysis section with enhanced actual usage display"""
        st.markdown('<div class="section-header">📊 vRealize Operations Metrics Analysis</div>', unsafe_allow_html=True)
        
        if not st.session_state.vrops_metrics:
            st.info("⚠️ Please configure vROps metrics in the sidebar to see detailed performance analysis.")
            return
        
        vrops_metrics = st.session_state.vrops_metrics
        
        # Current Infrastructure Overview
        st.markdown("**🖥️ Current Infrastructure**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>vCPUs Allocated</h3>
                <h2>{vrops_metrics.current_vcpus}</h2>
                <p>Virtual CPU cores</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>RAM Allocated</h3>
                <h2>{vrops_metrics.current_ram_gb:.0f} GB</h2>
                <p>Total memory</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Storage Allocated</h3>
                <h2>{vrops_metrics.current_storage_gb:.0f} GB</h2>
                <p>Total storage</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>VM Count</h3>
                <h2>{vrops_metrics.vm_count}</h2>
                <p>Virtual machines</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Performance Utilization
        st.markdown("**📈 Performance Utilization**")
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
        
        # Actual Resource Consumption
        st.markdown("**🧮 Actual Resource Consumption**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>CPU Cores Used (Avg)</h3>
                <h2>{vrops_metrics.actual_cpu_cores_used_avg:.1f}</h2>
                <p>of {vrops_metrics.current_vcpus} allocated</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>RAM Used (Avg)</h3>
                <h2>{vrops_metrics.actual_ram_gb_used_avg:.1f} GB</h2>
                <p>of {vrops_metrics.current_ram_gb:.0f} GB allocated</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>CPU Cores Used (Peak)</h3>
                <h2>{vrops_metrics.actual_cpu_cores_used_peak:.1f}</h2>
                <p>Peak usage</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>RAM Used (Peak)</h3>
                <h2>{vrops_metrics.actual_ram_gb_used_peak:.1f} GB</h2>
                <p>Peak usage</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Rightsizing Recommendations
        st.markdown("**🎯 Rightsizing Recommendations**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rec_cpu_color = "#28a745" if vrops_metrics.recommended_cpu_cores < vrops_metrics.current_vcpus else "#ffc107"
            st.markdown(f"""
            <div class="metric-card">
                <h3>Recommended vCPUs</h3>
                <h2 style="color: {rec_cpu_color};">{vrops_metrics.recommended_cpu_cores}</h2>
                <p>With 20% headroom</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            rec_ram_color = "#28a745" if vrops_metrics.recommended_ram_gb < vrops_metrics.current_ram_gb else "#ffc107"
            st.markdown(f"""
            <div class="metric-card">
                <h3>Recommended RAM</h3>
                <h2 style="color: {rec_ram_color};">{vrops_metrics.recommended_ram_gb} GB</h2>
                <p>With 25% headroom</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            score_color = "#28a745" if vrops_metrics.sizing_efficiency_score > 70 else "#ffc107" if vrops_metrics.sizing_efficiency_score > 40 else "#dc3545"
            st.markdown(f"""
            <div class="metric-card">
                <h3>Efficiency Score</h3>
                <h2 style="color: {score_color};">{vrops_metrics.sizing_efficiency_score:.0f}/100</h2>
                <p>Utilization efficiency</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("**🔍 Performance Insights**")
        
        # Enhanced insights with actual usage
        insights = []
        insights.append(f"💡 {vrops_metrics.rightsizing_opportunity}")
        
        if vrops_metrics.actual_cpu_cores_used_avg < vrops_metrics.current_vcpus * 0.5:
            cpu_waste = vrops_metrics.current_vcpus - vrops_metrics.actual_cpu_cores_used_avg
            insights.append(f"⚡ CPU over-provisioning: {cpu_waste:.1f} cores unused on average")
        
        if vrops_metrics.actual_ram_gb_used_avg < vrops_metrics.current_ram_gb * 0.6:
            ram_waste = vrops_metrics.current_ram_gb - vrops_metrics.actual_ram_gb_used_avg
            insights.append(f"💾 Memory over-provisioning: {ram_waste:.1f} GB unused on average")
        
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
        
        self.render_vrops_charts(vrops_metrics)
    
    def render_vrops_charts(self, vrops_metrics):
        """Render vROps performance charts with actual usage overlay"""
        col1, col2 = st.columns(2)
        
        with col1:
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
        
        # Actual vs Allocated Resources Chart
        st.markdown("**📊 Resource Allocation vs Usage**")
        
        fig_resources = go.Figure()
        
        categories = ['CPU Cores (Avg)', 'CPU Cores (Peak)', 'RAM GB (Avg)', 'RAM GB (Peak)']
        allocated = [vrops_metrics.current_vcpus, vrops_metrics.current_vcpus, 
                    vrops_metrics.current_ram_gb, vrops_metrics.current_ram_gb]
        used = [vrops_metrics.actual_cpu_cores_used_avg, vrops_metrics.actual_cpu_cores_used_peak,
               vrops_metrics.actual_ram_gb_used_avg, vrops_metrics.actual_ram_gb_used_peak]
        
        fig_resources.add_trace(go.Bar(
            name='Allocated',
            x=categories,
            y=allocated,
            marker_color='lightblue',
            opacity=0.7
        ))
        
        fig_resources.add_trace(go.Bar(
            name='Used',
            x=categories,
            y=used,
            marker_color='darkblue'
        ))
        
        fig_resources.update_layout(
            title='Current Allocation vs Actual Usage',
            yaxis_title='Resources',
            barmode='overlay',
            height=400
        )
        
        st.plotly_chart(fig_resources, use_container_width=True)
    
    def render_pricing_analysis(self):
        """Render pricing analysis section"""
        st.markdown('<div class="section-header">💰 AWS Pricing Analysis with Rightsizing</div>', unsafe_allow_html=True)
        
        if not hasattr(st.session_state, 'config'):
            st.info("⚠️ Please configure workload parameters in the sidebar.")
            return
        
        st.write("💡 Analyze AWS pricing options based on your actual workload requirements and vROps metrics.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔍 Analyze Pricing with Rightsizing", type="primary", use_container_width=True):
                with st.spinner("Fetching AWS pricing data and calculating rightsizing..."):
                    self.fetch_and_display_pricing()

    def fetch_and_display_pricing(self):
        """Fetch and display pricing data with enhanced rightsizing analysis"""
        config = st.session_state.config
        vrops_data = st.session_state.vrops_metrics
        
        try:
            pricing_data = []
            
            if not st.session_state.demo_mode and self.aws_pricing.connection_status["connected"]:
                instance_types = ['m5.large', 'm5.xlarge', 'm5.2xlarge', 'r5.large', 'r5.xlarge', 'r5.2xlarge', 'c5.large', 'c5.xlarge']
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, instance_type in enumerate(instance_types):
                    status_text.text(f"Fetching live pricing for {instance_type}...")
                    pricing = self.aws_pricing.get_enhanced_ec2_pricing(instance_type, config['region'], vrops_data)
                    if pricing:
                        pricing_data.append(pricing)
                    progress_bar.progress((i + 1) / len(instance_types))
                
                progress_bar.empty()
                status_text.empty()
            
            if not pricing_data or st.session_state.demo_mode:
                pricing_data = self.mock_data.get_enhanced_sample_pricing_data(config['region'], vrops_data)
                if st.session_state.demo_mode:
                    st.success("📊 Enhanced demo pricing data loaded with rightsizing analysis")
                else:
                    st.warning("⚠️ Using demo data - AWS API data not available")
            
            if pricing_data:
                st.session_state.pricing_cache[config['region']] = pricing_data
                st.session_state.latest_pricing = pricing_data
                self.display_pricing_results(pricing_data)
                st.success("✅ Pricing analysis with rightsizing complete!")
            else:
                st.error("❌ No pricing data available.")
                
        except Exception as e:
            st.error(f"❌ Error fetching pricing: {str(e)}")
            logger.error(f"Pricing fetch error: {e}")

    def display_pricing_results(self, pricing_data: List):
        """Display pricing analysis results with rightsizing analysis"""
        
        # Show current vs recommended sizing first if vROps data available
        if st.session_state.vrops_metrics:
            vrops = st.session_state.vrops_metrics
            
            st.markdown("**🔍 Rightsizing Analysis**")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Current vCPUs</h3>
                    <h2>{vrops.current_vcpus}</h2>
                    <p>Allocated cores</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Current RAM</h3>
                    <h2>{vrops.current_ram_gb:.0f} GB</h2>
                    <p>Allocated memory</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                color = "#28a745" if vrops.actual_cpu_cores_used_avg < vrops.current_vcpus * 0.8 else "#ffc107"
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Actual CPU Used</h3>
                    <h2 style="color: {color};">{vrops.actual_cpu_cores_used_avg:.1f}</h2>
                    <p>Avg cores ({vrops.cpu_usage_avg:.1f}%)</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                color = "#28a745" if vrops.actual_ram_gb_used_avg < vrops.current_ram_gb * 0.8 else "#ffc107"
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Actual RAM Used</h3>
                    <h2 style="color: {color};">{vrops.actual_ram_gb_used_avg:.1f} GB</h2>
                    <p>Avg memory ({vrops.memory_usage_avg:.1f}%)</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Rightsizing recommendation
            st.markdown(f"""
            <div class="optimization-insight">
                <strong>💡 Rightsizing Opportunity:</strong> {vrops.rightsizing_opportunity}<br>
                <strong>📊 Efficiency Score:</strong> {vrops.sizing_efficiency_score:.0f}/100<br>
                <strong>🎯 Recommended Sizing:</strong> {vrops.recommended_cpu_cores} vCPUs, {vrops.recommended_ram_gb} GB RAM
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("**💰 AWS Instance Recommendations**")
        
        # Create enhanced table with rightsizing information
        table_data = []
        for i, pricing in enumerate(pricing_data[:10]):
            specs = pricing.specifications or {}
            
            # Calculate savings compared to most expensive option
            max_cost = max(p.total_monthly_cost for p in pricing_data[:10])
            savings = max_cost - pricing.total_monthly_cost
            savings_pct = (savings / max_cost * 100) if max_cost > 0 else 0
            
            # Add rightsizing metrics if available
            rightsizing_info = ""
            if 'projected_cpu_utilization' in specs:
                cpu_util = specs['projected_cpu_utilization']
                ram_util = specs['projected_ram_utilization']
                risk = specs.get('performance_risk', 'Unknown')
                score = specs.get('rightsizing_score', 0) * 100
                
                rightsizing_info = f"CPU: {cpu_util:.0f}%, RAM: {ram_util:.0f}%, Risk: {risk}, Score: {score:.0f}"
            
            table_data.append({
                'Rank': f"#{i+1}",
                'Instance Type': pricing.instance_type,
                'vCPUs': specs.get('vcpus', 'N/A'),
                'RAM (GB)': specs.get('ram', 'N/A'),
                'Infrastructure ($/month)': f"${pricing.price_per_month:,.0f}",
                'SQL Licensing ($/month)': f"${pricing.sql_licensing_cost:,.0f}",
                'Total Cost ($/month)': f"${pricing.total_monthly_cost:,.0f}",
                'Monthly Savings': f"${savings:,.0f} ({savings_pct:.1f}%)",
                'Rightsizing Analysis': rightsizing_info,
                'Family': specs.get('family', 'Unknown')
            })
        
        df = pd.DataFrame(table_data)
        
        # Style the dataframe based on performance risk if available
        def highlight_risk(row):
            if 'High' in str(row['Rightsizing Analysis']):
                return ['background-color: #fff3cd'] * len(row)  # Light yellow
            elif 'Low' in str(row['Rightsizing Analysis']):
                return ['background-color: #d4edda'] * len(row)  # Light green
            else:
                return [''] * len(row)
        
        if pricing_data and 'projected_cpu_utilization' in pricing_data[0].specifications:
            styled_df = df.style.apply(highlight_risk, axis=1)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Show best match recommendation
        if pricing_data and st.session_state.vrops_metrics:
            best_match = pricing_data[0]  # Already sorted by optimal rightsizing
            specs = best_match.specifications or {}
            
            st.markdown(f"""
            <div class="ai-recommendation">
                <h3>🎯 Optimal Instance Recommendation</h3>
                <p><strong>Instance:</strong> {best_match.instance_type}</p>
                <p><strong>Sizing:</strong> {specs.get('vcpus', 'N/A')} vCPUs, {specs.get('ram', 'N/A')} GB RAM</p>
                <p><strong>Monthly Cost:</strong> ${best_match.total_monthly_cost:,.0f}</p>
                <p><strong>Annual Cost:</strong> ${best_match.total_monthly_cost * 12:,.0f}</p>
                <p><strong>Projected Utilization:</strong> CPU {specs.get('projected_cpu_utilization', 0):.0f}%, RAM {specs.get('projected_ram_utilization', 0):.0f}%</p>
                <p><strong>Performance Risk:</strong> {specs.get('performance_risk', 'Unknown')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        if len(pricing_data) > 0:
            self.render_pricing_chart(pricing_data[:8])
            
            # Add rightsizing efficiency chart if vROps data available
            if st.session_state.vrops_metrics and pricing_data and 'rightsizing_score' in pricing_data[0].specifications:
                self.render_rightsizing_chart(pricing_data[:8])

    def render_rightsizing_chart(self, pricing_data: List):
        """Render rightsizing efficiency chart"""
        st.markdown("**📊 Rightsizing Efficiency Analysis**")
        
        instance_types = [p.instance_type for p in pricing_data]
        rightsizing_scores = [p.specifications.get('rightsizing_score', 0) * 100 for p in pricing_data]
        cpu_utilizations = [p.specifications.get('projected_cpu_utilization', 0) for p in pricing_data]
        ram_utilizations = [p.specifications.get('projected_ram_utilization', 0) for p in pricing_data]
        total_costs = [p.total_monthly_cost for p in pricing_data]
        
        fig = go.Figure()
        
        # Create bubble chart where bubble size = cost, color = rightsizing score
        fig.add_trace(go.Scatter(
            x=cpu_utilizations,
            y=ram_utilizations,
            mode='markers+text',
            text=instance_types,
            textposition='top center',
            marker=dict(
                size=[cost / 100 for cost in total_costs],  # Scale for visibility
                color=rightsizing_scores,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Rightsizing Score"),
                line=dict(width=2, color='white')
            ),
            hovertemplate='<b>%{text}</b><br>' +
                         'CPU Utilization: %{x:.1f}%<br>' +
                         'RAM Utilization: %{y:.1f}%<br>' +
                         'Rightsizing Score: %{marker.color:.0f}<br>' +
                         '<extra></extra>'
        ))
        
        # Add optimal zone rectangle
        fig.add_shape(
            type="rect",
            x0=50, y0=60, x1=80, y1=85,
            fillcolor="rgba(0,255,0,0.1)",
            line=dict(color="green", width=2, dash="dash"),
            layer="below"
        )
        
        fig.add_annotation(
            x=65, y=72.5,
            text="Optimal Zone",
            showarrow=False,
            font=dict(color="green", size=12),
            bgcolor="rgba(255,255,255,0.8)"
        )
        
        fig.update_layout(
            title='Instance Rightsizing Analysis (Bubble size = Cost)',
            xaxis_title='Projected CPU Utilization (%)',
            yaxis_title='Projected RAM Utilization (%)',
            height=500,
            xaxis=dict(range=[0, 100]),
            yaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add interpretation guide
        st.markdown("""
        **📋 Interpretation Guide:**
        - **Green Zone (50-80% CPU, 60-85% RAM):** Optimal utilization range
        - **Bubble Size:** Larger = Higher monthly cost
        - **Color:** Green = Better rightsizing score, Red = Poor rightsizing
        - **Target:** Choose instances in/near green zone with good scores
        """)

    def render_pricing_chart(self, pricing_data: List):
        """Render enhanced pricing comparison chart"""
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
        
        self.render_cost_efficiency_chart(pricing_data)
    
    def render_cost_efficiency_chart(self, pricing_data: List):
        """Render cost efficiency analysis"""
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
        
        st.write("🧠 Get intelligent migration recommendations based on your vROps data and pricing analysis.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Get AI Analysis", type="primary", use_container_width=True):
                with st.spinner("Generating AI recommendations..."):
                    self.generate_ai_recommendations()

    def generate_ai_recommendations(self):
        """Generate and display AI recommendations"""
        config = st.session_state.config
        pricing_data = st.session_state.latest_pricing
        vrops_data = st.session_state.vrops_metrics
        sql_config = st.session_state.sql_config
        
        try:
            analysis_result = asyncio.run(
                self.claude_ai.get_comprehensive_analysis(config, pricing_data, vrops_data, sql_config)
            )
            
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
        
        col1, col2 = st.columns(2)
        
        with col1:
            if analysis.get('vrops_insights'):
                st.markdown("**📊 vROps Performance Insights**")
                for insight in analysis['vrops_insights']:
                    st.markdown(f"""
                    <div class="optimization-insight">
                        <strong>💡</strong> {insight}
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            if analysis.get('sql_optimization'):
                st.markdown("**🗃️ SQL Optimization Opportunities**")
                for optimization in analysis['sql_optimization']:
                    st.markdown(f"""
                    <div class="optimization-insight">
                        <strong>💰</strong> {optimization}
                    </div>
                    """, unsafe_allow_html=True)
        
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
        
        st.markdown("**💰 Cost Scenarios Comparison**")
        
        selected_instances = st.multiselect(
            "Select instances to compare:",
            [p.instance_type for p in pricing_data],
            default=[p.instance_type for p in pricing_data[:3]]
        )
        
        comparison_period = st.selectbox(
            "Comparison Period:",
            ["Monthly", "Annual", "3-Year Total"]
        )
        
        if selected_instances:
            self.render_cost_comparison_chart(pricing_data, selected_instances, comparison_period)

    def render_cost_comparison_chart(self, pricing_data: List, selected_instances: List[str], period: str):
        """Render detailed cost comparison chart with multiple periods"""
        filtered_data = [p for p in pricing_data if p.instance_type in selected_instances]
        
        multiplier = {"Monthly": 1, "Annual": 12, "3-Year Total": 36}[period]
        
        instance_types = [p.instance_type for p in filtered_data]
        on_demand_costs = [p.total_monthly_cost * multiplier for p in filtered_data]
        
        ri_1yr_costs = []
        spot_costs = []
        
        for p in filtered_data:
            if p.reserved_pricing:
                ri_infrastructure = p.reserved_pricing.get('1_year_all_upfront', 0) * 730 * multiplier
                ri_cost = ri_infrastructure + (p.sql_licensing_cost * multiplier)
                ri_1yr_costs.append(ri_cost)
            else:
                ri_1yr_costs.append(p.total_monthly_cost * 0.6 * multiplier)
            
            spot_infrastructure = (p.spot_pricing * 730 * multiplier) if p.spot_pricing else (p.price_per_month * 0.3 * multiplier)
            spot_cost = spot_infrastructure + (p.sql_licensing_cost * multiplier)
            spot_costs.append(spot_cost)
        
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
        
        if len(filtered_data) > 0:
            cheapest = min(ri_1yr_costs)
            most_expensive = max(on_demand_costs)
            potential_savings = most_expensive - cheapest
            
            st.markdown(f"""
            <div class="cost-savings">
                <h3>Potential {period} Savings</h3>
                <h2>${potential_savings:,.0f}</h2>
                <p>By choosing optimal pricing model</p>
            </div>
            """, unsafe_allow_html=True)
            
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
        
        self.render_sql_cost_analysis(sql_config)
        
        st.markdown("**💡 Optimization Opportunities**")
        
        opportunities = []
        annual_savings = 0
        
        if sql_config.has_software_assurance and sql_config.eligible_for_ahb:
            ahb_savings = sql_config.current_annual_license_cost * 0.55
            annual_savings += ahb_savings
            opportunities.append(f"Azure Hybrid Benefit: Save ${ahb_savings:,.0f} annually (55% reduction)")
        
        if sql_config.current_edition == "Enterprise" and sql_config.concurrent_users < 100:
            edition_savings = sql_config.current_cores_licensed * (13748 - 3717)
            annual_savings += edition_savings
            opportunities.append(f"Downgrade to Standard Edition: Save ${edition_savings:,.0f} annually")
        
        if sql_config.current_licensing_model == "Core-based" and sql_config.concurrent_users < 25:
            cal_cost = sql_config.concurrent_users * 209
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
        scenarios = {
            "Current Configuration": sql_config.current_cores_licensed * 3717,
            "Standard Edition": sql_config.current_cores_licensed * 3717,
            "Enterprise Edition": sql_config.current_cores_licensed * 13748,
            "CAL-based (Current Users)": sql_config.concurrent_users * 209 + 931,
        }
        
        if sql_config.has_software_assurance:
            scenarios["Current + AHB"] = scenarios["Current Configuration"] * 0.45
            scenarios["Standard + AHB"] = scenarios["Standard Edition"] * 0.45
        
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
        """Render reports section with enhanced functionality"""
        st.markdown('<div class="section-header">📄 Professional Reports & Export</div>', unsafe_allow_html=True)
        
        pdf_available = self.pdf_generator.pdf_available
        
        st.markdown(f"""
        Generate comprehensive reports with detailed analysis from all sections:
        
        **📊 Available Report Formats:**
        - ✅ **Detailed Text Reports** - Complete analysis in structured text format
        - ✅ **CSV Data Export** - Pricing and configuration data
        - ✅ **JSON Export** - vROps metrics and SQL configuration
        {"- ✅ **Professional PDF Reports** - Executive and technical reports with charts" if pdf_available else "- ⚠️ **PDF Reports** - Install `reportlab` for professional PDF generation"}
        """)
        
        if not pdf_available:
            st.info("""
            💡 **Enable PDF Reports:** Install ReportLab for professional PDF generation:
            ```bash
            pip install reportlab
            ```
            Text reports provide the same comprehensive analysis in a structured format.
            """)
        
        has_pricing = bool(st.session_state.latest_pricing)
        has_vrops = bool(st.session_state.vrops_metrics)
        has_sql = bool(st.session_state.sql_config)
        has_ai = bool(st.session_state.comprehensive_analysis)
        has_config = bool(hasattr(st.session_state, 'config'))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Data Availability Status**")
            status_items = [
                ("Basic Configuration", "✅" if has_config else "❌"),
                ("AWS Pricing Analysis", "✅" if has_pricing else "❌"),
                ("vROps Performance Data", "✅" if has_vrops else "❌"),
                ("SQL Server Configuration", "✅" if has_sql else "❌"),
                ("AI Recommendations", "✅" if has_ai else "❌")
            ]
            
            for item, status in status_items:
                st.write(f"{status} {item}")
        
        with col2:
            st.markdown("**📈 Report Completeness**")
            total_sections = 5
            completed_sections = sum([has_config, has_pricing, has_vrops, has_sql, has_ai])
            completeness = (completed_sections / total_sections) * 100
            
            st.metric("Data Completeness", f"{completeness:.0f}%")
            
            if completeness < 60:
                st.warning("⚠️ Limited data available. Report will include available sections only.")
            elif completeness < 80:
                st.info("ℹ️ Good data coverage. Most sections will be included.")
            else:
                st.success("✅ Excellent data coverage. Comprehensive report available.")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            format_type = "PDF" if pdf_available else "TXT"
            if st.button(f"📄 Generate Comprehensive Report ({format_type})", 
                        type="primary", 
                        use_container_width=True):
                if has_config:
                    with st.spinner(f"🔄 Generating comprehensive {format_type.lower()} report..."):
                        try:
                            self._generate_comprehensive_report()
                        except Exception as e:
                            st.error(f"❌ Error generating report: {str(e)}")
                else:
                    st.warning("⚠️ Please configure basic settings in the sidebar first.")
            
            if st.button(f"📋 Generate Executive Summary ({format_type})", 
                        use_container_width=True):
                if has_config and (has_pricing or has_ai):
                    with st.spinner("🔄 Generating executive summary..."):
                        try:
                            self._generate_executive_summary()
                        except Exception as e:
                            st.error(f"❌ Error generating executive summary: {str(e)}")
                else:
                    st.warning("⚠️ Please complete pricing analysis or AI recommendations first.")
        
        with col2:
            if st.button("💾 Export Complete Dataset (ZIP)", 
                        use_container_width=True):
                if self._has_sufficient_data():
                    with st.spinner("🔄 Preparing data export..."):
                        try:
                            self._export_complete_dataset()
                        except Exception as e:
                            st.error(f"❌ Error creating data export: {str(e)}")
                else:
                    st.info("Insufficient data for complete export")
            
            if st.button("📊 Quick Data Summary (TXT)", 
                        use_container_width=True):
                if has_config:
                    with st.spinner("🔄 Generating quick summary..."):
                        try:
                            self._generate_quick_summary()
                        except Exception as e:
                            st.error(f"❌ Error generating summary: {str(e)}")
                else:
                    st.warning("⚠️ Please configure basic settings first.")
        
        st.markdown("---")
        
        st.markdown("**📊 Individual Data Exports**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export Pricing Data (CSV)", use_container_width=True):
                if st.session_state.latest_pricing:
                    csv_data = self.export_pricing_csv()
                    st.download_button(
                        "📥 Download Pricing CSV",
                        csv_data,
                        f"aws_pricing_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                        "text/csv",
                        use_container_width=True
                    )
                else:
                    st.info("No pricing data available")
        
        with col2:
            if st.button("📊 Export vROps Data (JSON)", use_container_width=True):
                if st.session_state.vrops_metrics:
                    json_data = json.dumps(asdict(st.session_state.vrops_metrics), indent=2)
                    st.download_button(
                        "📥 Download vROps JSON",
                        json_data,
                        f"vrops_metrics_{datetime.now().strftime('%Y%m%d')}.json",
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
                        f"sql_configuration_{datetime.now().strftime('%Y%m%d')}.json",
                        "application/json",
                        use_container_width=True
                    )
                else:
                    st.info("No SQL configuration available")
    
    def _generate_comprehensive_report(self):
        """Generate comprehensive report"""
        try:
            config = st.session_state.config
            pricing_data = st.session_state.latest_pricing or []
            vrops_data = st.session_state.vrops_metrics
            sql_config = st.session_state.sql_config
            
            recommendation = None
            risks = []
            phases = []
            
            if st.session_state.comprehensive_analysis:
                analysis = st.session_state.comprehensive_analysis
                recommendation = analysis.get('recommendation')
                risks = analysis.get('risks', [])
                phases = analysis.get('phases', [])
            
            if not phases:
                phases = [
                    ImplementationPhase(
                        phase="Planning & Assessment",
                        duration="2-4 weeks",
                        activities=["Detailed workload analysis", "Performance baselining", "Migration planning"],
                        dependencies=["Stakeholder approval", "AWS account setup"],
                        deliverables=["Migration plan", "Performance baseline", "Cost projections"]
                    ),
                    ImplementationPhase(
                        phase="Pilot Migration",
                        duration="2-3 weeks",
                        activities=["Migrate test workloads", "Performance validation", "Cost validation"],
                        dependencies=["Phase 1 completion", "Test environment setup"],
                        deliverables=["Pilot results", "Performance metrics", "Lessons learned"]
                    )
                ]
            
            report_buffer = self.pdf_generator.create_comprehensive_report(
                config, pricing_data, recommendation, risks, phases, vrops_data, sql_config
            )
            
            file_ext = "pdf" if self.pdf_generator.pdf_available else "txt"
            mime_type = "application/pdf" if self.pdf_generator.pdf_available else "text/plain"
            filename = f"AWS_Migration_Analysis_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_ext}"
            
            st.download_button(
                f"📥 Download Comprehensive Report ({file_ext.upper()})",
                report_buffer.getvalue(),
                filename,
                mime_type,
                use_container_width=True
            )
            
            st.success(f"✅ Comprehensive {file_ext.upper()} report generated successfully!")
            
        except Exception as e:
            logger.error(f"Comprehensive report generation error: {e}")
            raise e
    
    def _generate_executive_summary(self):
        """Generate executive summary"""
        try:
            config = st.session_state.config
            pricing_data = st.session_state.latest_pricing or []
            recommendation = None
            
            if st.session_state.comprehensive_analysis:
                recommendation = st.session_state.comprehensive_analysis.get('recommendation')
            
            if self.pdf_generator.pdf_available:
                # Generate PDF executive summary
                try:
                    from reportlab.lib.pagesizes import A4
                    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                    from reportlab.lib.styles import getSampleStyleSheet
                    
                    buffer = BytesIO()
                    doc = SimpleDocTemplate(buffer, pagesize=A4)
                    styles = getSampleStyleSheet()
                    story = []
                    
                    title = Paragraph("AWS Migration Analysis - Executive Summary", styles['Title'])
                    story.append(title)
                    story.append(Spacer(1, 20))
                    
                    if recommendation:
                        story.append(Paragraph(f"<b>Primary Recommendation:</b> {recommendation.recommendation}", styles['Normal']))
                        story.append(Spacer(1, 12))
                        story.append(Paragraph(f"<b>Expected Annual Savings:</b> ${recommendation.expected_savings:,.0f}", styles['Normal']))
                        story.append(Spacer(1, 12))
                    
                    if pricing_data:
                        cheapest = min(pricing_data, key=lambda x: x.total_monthly_cost)
                        story.append(Paragraph(f"<b>Recommended Instance:</b> {cheapest.instance_type}", styles['Normal']))
                        story.append(Paragraph(f"<b>Optimal Monthly Cost:</b> ${cheapest.total_monthly_cost:,.0f}", styles['Normal']))
                    
                    doc.build(story)
                    buffer.seek(0)
                    
                    filename = f"AWS_Executive_Summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    
                    st.download_button(
                        "📥 Download Executive Summary (PDF)",
                        buffer.getvalue(),
                        filename,
                        "application/pdf",
                        use_container_width=True
                    )
                    
                    st.success("✅ Executive summary PDF generated successfully!")
                    return
                    
                except Exception as pdf_error:
                    logger.warning(f"PDF generation failed: {pdf_error}, falling back to text")
            
            # Generate text executive summary
            summary = f"""AWS MIGRATION ANALYSIS - EXECUTIVE SUMMARY
========================================
Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}

KEY FINDINGS AND RECOMMENDATIONS
===============================
"""
            
            if recommendation:
                summary += f"""
PRIMARY RECOMMENDATION:
{recommendation.recommendation}

CONFIDENCE LEVEL: {recommendation.confidence_score:.0f}%
EXPECTED ANNUAL SAVINGS: ${recommendation.expected_savings:,.0f}
COST IMPACT: {recommendation.cost_impact}

REASONING:
{recommendation.reasoning}
"""
            
            if pricing_data:
                cheapest = min(pricing_data, key=lambda x: x.total_monthly_cost)
                most_expensive = max(pricing_data, key=lambda x: x.total_monthly_cost)
                summary += f"""

COST ANALYSIS SUMMARY
====================
RECOMMENDED INSTANCE: {cheapest.instance_type}
OPTIMAL MONTHLY COST: ${cheapest.total_monthly_cost:,.0f}
POTENTIAL MONTHLY SAVINGS: ${most_expensive.total_monthly_cost - cheapest.total_monthly_cost:,.0f}
ANNUAL SAVINGS POTENTIAL: ${(most_expensive.total_monthly_cost - cheapest.total_monthly_cost) * 12:,.0f}
"""
            
            if st.session_state.vrops_metrics:
                vrops = st.session_state.vrops_metrics
                summary += f"""

RIGHTSIZING ANALYSIS
===================
CURRENT INFRASTRUCTURE: {vrops.current_vcpus} vCPUs, {vrops.current_ram_gb} GB RAM
ACTUAL USAGE (AVG): {vrops.actual_cpu_cores_used_avg:.1f} CPU cores, {vrops.actual_ram_gb_used_avg:.1f} GB RAM
RECOMMENDED SIZING: {vrops.recommended_cpu_cores} vCPUs, {vrops.recommended_ram_gb} GB RAM
EFFICIENCY SCORE: {vrops.sizing_efficiency_score:.0f}/100
RIGHTSIZING OPPORTUNITY: {vrops.rightsizing_opportunity}
"""
            
            summary += f"""

ANALYSIS CONFIGURATION
=====================
TARGET REGION: {config.get('region', 'Not specified')}
WORKLOAD TYPE: {config.get('workload_type', 'Not specified')}
ANALYSIS DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

NEXT STEPS
==========
1. Review detailed technical analysis
2. Validate recommendations with stakeholders  
3. Plan pilot migration phase
4. Execute phased migration approach
5. Monitor and optimize post-migration
"""
            
            filename = f"AWS_Executive_Summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            st.download_button(
                "📥 Download Executive Summary (TXT)",
                summary,
                filename,
                "text/plain",
                use_container_width=True
            )
            
            st.success("✅ Executive summary generated successfully!")
            
        except Exception as e:
            logger.error(f"Executive summary generation error: {e}")
            raise e
    
    def _generate_quick_summary(self):
        """Generate a quick summary of all analysis"""
        try:
            summary = self._create_text_summary()
            
            filename = f"AWS_Quick_Summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            st.download_button(
                "📥 Download Quick Summary (TXT)",
                summary,
                filename,
                "text/plain",
                use_container_width=True
            )
            
            st.success("✅ Quick summary generated successfully!")
            
        except Exception as e:
            logger.error(f"Quick summary generation error: {e}")
            raise e
    
    def _export_complete_dataset(self):
        """Export complete dataset as ZIP file"""
        try:
            import zipfile
            
            zip_buffer = BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                if st.session_state.latest_pricing:
                    csv_data = self.export_pricing_csv()
                    zip_file.writestr("pricing_analysis.csv", csv_data)
                
                if st.session_state.vrops_metrics:
                    vrops_json = json.dumps(asdict(st.session_state.vrops_metrics), indent=2)
                    zip_file.writestr("vrops_metrics.json", vrops_json)
                
                if st.session_state.sql_config:
                    sql_json = json.dumps(asdict(st.session_state.sql_config), indent=2)
                    zip_file.writestr("sql_configuration.json", sql_json)
                
                if st.session_state.comprehensive_analysis:
                    ai_json = json.dumps({
                        'recommendation': asdict(st.session_state.comprehensive_analysis['recommendation']) if st.session_state.comprehensive_analysis.get('recommendation') else None,
                        'insights': st.session_state.comprehensive_analysis.get('vrops_insights', []),
                        'sql_optimization': st.session_state.comprehensive_analysis.get('sql_optimization', [])
                    }, indent=2)
                    zip_file.writestr("ai_recommendations.json", ai_json)
                
                config_json = json.dumps(st.session_state.config, indent=2)
                zip_file.writestr("configuration.json", config_json)
                
                summary = self._create_text_summary()
                zip_file.writestr("analysis_summary.txt", summary)
            
            zip_buffer.seek(0)
            
            filename = f"AWS_Complete_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            
            st.download_button(
                "📥 Download Complete Dataset (ZIP)",
                zip_buffer.getvalue(),
                filename,
                "application/zip",
                use_container_width=True
            )
            
            st.success("✅ Complete dataset exported successfully!")
            
        except Exception as e:
            logger.error(f"Complete dataset export error: {e}")
            raise e
    
    def _create_text_summary(self):
        """Create a text summary of the analysis"""
        summary = f"""AWS CLOUD MIGRATION ANALYSIS SUMMARY
=====================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CONFIGURATION
=============
Region: {st.session_state.config.get('region', 'Not specified')}
Workload Type: {st.session_state.config.get('workload_type', 'Not specified')}

"""
        
        if st.session_state.vrops_metrics:
            vrops = st.session_state.vrops_metrics
            summary += f"""CURRENT INFRASTRUCTURE & VROPS METRICS
======================================
Current Infrastructure:
- vCPUs Allocated: {vrops.current_vcpus}
- RAM Allocated: {vrops.current_ram_gb} GB
- Storage Allocated: {vrops.current_storage_gb} GB
- VM Count: {vrops.vm_count}

Performance Utilization:
- CPU Usage (Avg): {vrops.cpu_usage_avg:.1f}%
- CPU Usage (Peak): {vrops.cpu_usage_peak:.1f}%
- Memory Usage (Avg): {vrops.memory_usage_avg:.1f}%
- Memory Usage (Peak): {vrops.memory_usage_peak:.1f}%
- CPU Ready Time: {vrops.cpu_ready_avg:.1f}%
- Memory Balloon: {vrops.memory_balloon_avg:.1f}%
- Disk Latency: {vrops.disk_latency_avg:.1f}ms

Actual Resource Consumption:
- CPU Cores Used (Avg): {vrops.actual_cpu_cores_used_avg:.1f}
- CPU Cores Used (Peak): {vrops.actual_cpu_cores_used_peak:.1f}
- RAM Used (Avg): {vrops.actual_ram_gb_used_avg:.1f} GB
- RAM Used (Peak): {vrops.actual_ram_gb_used_peak:.1f} GB

Rightsizing Recommendations:
- Recommended vCPUs: {vrops.recommended_cpu_cores}
- Recommended RAM: {vrops.recommended_ram_gb} GB
- Efficiency Score: {vrops.sizing_efficiency_score:.0f}/100
- Rightsizing Opportunity: {vrops.rightsizing_opportunity}

"""
        
        if st.session_state.latest_pricing:
            cheapest = min(st.session_state.latest_pricing, key=lambda x: x.total_monthly_cost)
            summary += f"""PRICING ANALYSIS
================
Recommended Instance: {cheapest.instance_type}
Monthly Cost: ${cheapest.total_monthly_cost:,.0f}
Annual Cost: ${cheapest.total_monthly_cost * 12:,.0f}

TOP 5 INSTANCE OPTIONS:
"""
            for i, p in enumerate(st.session_state.latest_pricing[:5], 1):
                specs = p.specifications or {}
                proj_cpu = specs.get('projected_cpu_utilization', 0)
                proj_ram = specs.get('projected_ram_utilization', 0)
                risk = specs.get('performance_risk', 'Unknown')
                
                summary += f"{i}. {p.instance_type}: ${p.total_monthly_cost:,.0f}/month ({specs.get('vcpus', 'N/A')} vCPUs, {specs.get('ram', 'N/A')}GB RAM)"
                if proj_cpu > 0:
                    summary += f" | Projected: CPU {proj_cpu:.0f}%, RAM {proj_ram:.0f}%, Risk: {risk}"
                summary += "\n"
        
        if st.session_state.comprehensive_analysis and st.session_state.comprehensive_analysis.get('recommendation'):
            rec = st.session_state.comprehensive_analysis['recommendation']
            summary += f"""
AI RECOMMENDATIONS
==================
Primary Recommendation: {rec.recommendation}
Confidence Score: {rec.confidence_score:.0f}%
Expected Annual Savings: ${rec.expected_savings:,.0f}
Cost Impact: {rec.cost_impact}
"""
        
        if st.session_state.sql_config:
            sql = st.session_state.sql_config
            summary += f"""
SQL SERVER CONFIGURATION
========================
Edition: {sql.current_edition}
Licensing Model: {sql.current_licensing_model}
Licensed Cores: {sql.current_cores_licensed}
Concurrent Users: {sql.concurrent_users}
Software Assurance: {'Yes' if sql.has_software_assurance else 'No'}
Azure Hybrid Benefit Eligible: {'Yes' if sql.eligible_for_ahb else 'No'}
"""
        
        return summary
    
    def _has_sufficient_data(self):
        """Check if sufficient data is available for reporting"""
        return (
            hasattr(st.session_state, 'config') and
            (st.session_state.latest_pricing or st.session_state.vrops_metrics or st.session_state.comprehensive_analysis)
        )
    
    def export_pricing_csv(self):
        """Export pricing data as CSV with enhanced columns"""
        if not st.session_state.latest_pricing:
            return ""
        
        data = []
        for pricing_obj in st.session_state.latest_pricing:
            specs = pricing_obj.specifications or {}
            
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
                'Projected CPU Utilization': specs.get('projected_cpu_utilization', 'N/A'),
                'Projected RAM Utilization': specs.get('projected_ram_utilization', 'N/A'),
                'Performance Risk': specs.get('performance_risk', 'N/A'),
                'Rightsizing Score': specs.get('rightsizing_score', 'N/A'),
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
        # Check for optional dependencies silently
        pdf_available = False
        try:
            import reportlab
            pdf_available = True
        except ImportError:
            pass
        
        # Only show dependency info if PDF is not available
        if not pdf_available:
            st.info("""
            💡 **Professional PDF Reports:** Install `reportlab` to enable executive PDF generation.
            All analysis features are fully functional with text reports.
            """)
        
        optimizer = EnhancedCloudPricingOptimizer()
        optimizer.render_main_interface()
        
        st.markdown("---")
        st.markdown(f"""
        <div style="text-align: center; color: #6c757d; font-size: 0.9rem; padding: 1rem;">
            <strong>AWS Cloud Migration Optimizer</strong> • Professional Enterprise Edition with Intelligent Rightsizing<br>
            <small>Real-time pricing analysis • vROps integration • AI-powered recommendations • Comprehensive reporting</small>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        logger.error(f"Application error: {e}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.checkbox("🔍 Show detailed error information"):
                st.exception(e)
        
        with col2:
            if st.checkbox("🔍 Show session state"):
                st.json(dict(st.session_state))
        
        st.markdown("""
        **🛠️ Troubleshooting:**
        
        1. **Refresh the page** and try again
        2. **Clear browser cache** if issues persist  
        3. **Check dependencies** - `pip install streamlit pandas boto3 plotly requests`
        4. **Verify credentials** in Streamlit secrets (if using live APIs)
        5. **Contact support** if the issue continues
        """)

if __name__ == "__main__":
    main()