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
    page_title="AWS Cloud Pricing Optimizer with vROps & SQL Optimization",
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

class VRopsAnalyzer:
    """vRealize Operations metrics analyzer for EC2 sizing"""
    
    def __init__(self):
        self.cpu_safety_margin = 0.2  # 20% safety margin
        self.memory_safety_margin = 0.15  # 15% safety margin
        self.performance_thresholds = {
            'cpu_ready_warning': 5.0,
            'cpu_ready_critical': 10.0,
            'memory_balloon_warning': 1.0,
            'memory_balloon_critical': 5.0,
            'disk_latency_warning': 20.0,
            'disk_latency_critical': 50.0
        }
    
    def analyze_performance_metrics(self, vrops_metrics: VRopsMetrics) -> Tuple[SizingRecommendation, List[str]]:
        """Analyze vROps metrics and provide sizing recommendations"""
        warnings = []
        
        # Analyze CPU requirements
        cpu_analysis = self._analyze_cpu_metrics(vrops_metrics, warnings)
        
        # Analyze memory requirements  
        memory_analysis = self._analyze_memory_metrics(vrops_metrics, warnings)
        
        # Analyze storage performance
        storage_analysis = self._analyze_storage_metrics(vrops_metrics, warnings)
        
        # Analyze network requirements
        network_analysis = self._analyze_network_metrics(vrops_metrics, warnings)
        
        # Calculate recommended specifications
        recommended_vcpus = self._calculate_cpu_requirement(vrops_metrics)
        recommended_memory_gb = self._calculate_memory_requirement(vrops_metrics)
        
        # Determine instance type
        recommended_instance = self._recommend_instance_type(
            recommended_vcpus, recommended_memory_gb, vrops_metrics
        )
        
        # Calculate confidence and risk levels
        confidence = self._calculate_confidence(vrops_metrics)
        risk_level = self._assess_performance_risk(vrops_metrics, warnings)
        cost_opportunity = self._calculate_cost_optimization(vrops_metrics)
        
        # Generate reasoning
        reasoning = self._generate_sizing_reasoning(
            vrops_metrics, recommended_vcpus, recommended_memory_gb, warnings
        )
        
        sizing_recommendation = SizingRecommendation(
            recommended_vcpus=recommended_vcpus,
            recommended_memory_gb=recommended_memory_gb,
            recommended_instance_type=recommended_instance,
            rightsizing_confidence=confidence,
            performance_risk_level=risk_level,
            cost_optimization_opportunity=cost_opportunity,
            reasoning=reasoning
        )
        
        return sizing_recommendation, warnings
    
    def _analyze_cpu_metrics(self, metrics: VRopsMetrics, warnings: List[str]) -> Dict:
        """Analyze CPU performance metrics"""
        analysis = {}
        
        # Check CPU ready time
        if metrics.cpu_ready_avg > self.performance_thresholds['cpu_ready_critical']:
            warnings.append(f"CRITICAL: High CPU ready time ({metrics.cpu_ready_avg:.1f}%) indicates CPU contention")
            analysis['cpu_contention'] = 'critical'
        elif metrics.cpu_ready_avg > self.performance_thresholds['cpu_ready_warning']:
            warnings.append(f"WARNING: Elevated CPU ready time ({metrics.cpu_ready_avg:.1f}%) may impact performance")
            analysis['cpu_contention'] = 'warning'
        else:
            analysis['cpu_contention'] = 'normal'
        
        # Check CPU Co-Stop
        if metrics.cpu_costop_avg > 2.0:
            warnings.append(f"WARNING: CPU Co-Stop detected ({metrics.cpu_costop_avg:.1f}%), consider reducing vCPUs")
            analysis['cpu_costop'] = 'warning'
        
        # Analyze utilization patterns
        if metrics.cpu_usage_peak > 90 and metrics.cpu_usage_avg < 30:
            warnings.append("CPU usage shows high peaks with low average - consider burstable instances")
            analysis['utilization_pattern'] = 'bursty'
        elif metrics.cpu_usage_avg > 80:
            analysis['utilization_pattern'] = 'sustained_high'
        else:
            analysis['utilization_pattern'] = 'normal'
        
        return analysis
    
    def _analyze_memory_metrics(self, metrics: VRopsMetrics, warnings: List[str]) -> Dict:
        """Analyze memory performance metrics"""
        analysis = {}
        
        # Check memory ballooning
        if metrics.memory_balloon_avg > self.performance_thresholds['memory_balloon_critical']:
            warnings.append(f"CRITICAL: High memory ballooning ({metrics.memory_balloon_avg:.1f}%) indicates memory pressure")
            analysis['memory_pressure'] = 'critical'
        elif metrics.memory_balloon_avg > self.performance_thresholds['memory_balloon_warning']:
            warnings.append(f"WARNING: Memory ballooning detected ({metrics.memory_balloon_avg:.1f}%)")
            analysis['memory_pressure'] = 'warning'
        else:
            analysis['memory_pressure'] = 'normal'
        
        # Check memory swapping
        if metrics.memory_swapped_avg > 0:
            warnings.append(f"WARNING: Memory swapping detected ({metrics.memory_swapped_avg:.1f} MB)")
            analysis['memory_swapping'] = True
        else:
            analysis['memory_swapping'] = False
        
        # Analyze memory utilization
        if metrics.memory_usage_peak > 95:
            warnings.append("Memory usage frequently exceeds 95% - increase memory allocation")
            analysis['memory_adequacy'] = 'insufficient'
        elif metrics.memory_usage_avg < 40 and metrics.memory_usage_peak < 60:
            analysis['memory_adequacy'] = 'oversized'
        else:
            analysis['memory_adequacy'] = 'appropriate'
        
        return analysis
    
    def _analyze_storage_metrics(self, metrics: VRopsMetrics, warnings: List[str]) -> Dict:
        """Analyze storage performance metrics"""
        analysis = {}
        
        # Check disk latency
        if metrics.disk_latency_avg > self.performance_thresholds['disk_latency_critical']:
            warnings.append(f"CRITICAL: High disk latency ({metrics.disk_latency_avg:.1f}ms)")
            analysis['storage_performance'] = 'critical'
        elif metrics.disk_latency_avg > self.performance_thresholds['disk_latency_warning']:
            warnings.append(f"WARNING: Elevated disk latency ({metrics.disk_latency_avg:.1f}ms)")
            analysis['storage_performance'] = 'warning'
        else:
            analysis['storage_performance'] = 'normal'
        
        # Analyze IOPS patterns
        if metrics.disk_iops_peak > metrics.disk_iops_avg * 5:
            analysis['iops_pattern'] = 'bursty'
        else:
            analysis['iops_pattern'] = 'steady'
        
        # Queue depth analysis
        if metrics.disk_queue_depth_avg > 10:
            warnings.append(f"High disk queue depth ({metrics.disk_queue_depth_avg:.1f}) may indicate storage bottleneck")
        
        return analysis
    
    def _analyze_network_metrics(self, metrics: VRopsMetrics, warnings: List[str]) -> Dict:
        """Analyze network performance metrics"""
        analysis = {}
        
        # Check for packet drops
        if metrics.network_drops_avg > 0:
            warnings.append(f"Network packet drops detected ({metrics.network_drops_avg:.1f}/sec)")
            analysis['network_issues'] = True
        else:
            analysis['network_issues'] = False
        
        # Analyze network utilization
        if metrics.network_usage_peak > 80:
            analysis['network_utilization'] = 'high'
        else:
            analysis['network_utilization'] = 'normal'
        
        return analysis
    
    def _calculate_cpu_requirement(self, metrics: VRopsMetrics) -> int:
        """Calculate recommended CPU cores based on vROps metrics"""
        # Use 95th percentile with safety margin
        target_utilization = metrics.cpu_usage_95th * (1 + self.cpu_safety_margin)
        
        # Adjust for CPU ready time
        if metrics.cpu_ready_avg > 5.0:
            target_utilization *= 1.2  # Additional overhead for contention
        
        # Calculate required cores (assuming current allocation)
        current_cores = max(1, metrics.cpu_usage_peak / 100 * 4)  # Estimate current cores
        required_cores = math.ceil(current_cores * target_utilization / 70)  # Target 70% utilization
        
        return max(2, min(required_cores, 64))  # Min 2, max 64 cores
    
    def _calculate_memory_requirement(self, metrics: VRopsMetrics) -> float:
        """Calculate recommended memory based on vROps metrics"""
        # Use 95th percentile with safety margin
        target_utilization = metrics.memory_usage_95th * (1 + self.memory_safety_margin)
        
        # Adjust for memory pressure indicators
        if metrics.memory_balloon_avg > 1.0:
            target_utilization *= 1.3
        if metrics.memory_swapped_avg > 0:
            target_utilization *= 1.2
        
        # Calculate required memory (estimate current allocation)
        current_memory = max(4, metrics.memory_usage_peak / 100 * 16)  # Estimate current memory
        required_memory = current_memory * target_utilization / 80  # Target 80% utilization
        
        # Round to common memory sizes
        memory_sizes = [4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
        for size in memory_sizes:
            if required_memory <= size:
                return float(size)
        
        return 1024.0  # Max out at 1TB
    
    def _recommend_instance_type(self, vcpus: int, memory_gb: float, metrics: VRopsMetrics) -> str:
        """Recommend AWS instance type based on requirements"""
        # Calculate memory per vCPU ratio
        memory_per_cpu = memory_gb / vcpus
        
        # Determine instance family based on workload characteristics
        if memory_per_cpu > 12:  # Memory intensive
            if metrics.guest_cpu_usage_avg > 70:
                family = "r5"  # Memory optimized with good CPU
            else:
                family = "r6a"  # Memory optimized AMD (cost effective)
        elif metrics.disk_iops_avg > 10000:  # Compute intensive with high IOPS
            family = "c5"  # Compute optimized
        elif metrics.guest_cpu_usage_avg < 40:  # Low CPU utilization
            family = "m6a"  # General purpose AMD (cost effective)
        else:
            family = "m5"  # General purpose Intel
        
        # Select appropriate size
        if vcpus <= 2:
            size = "large"
        elif vcpus <= 4:
            size = "xlarge"
        elif vcpus <= 8:
            size = "2xlarge"
        elif vcpus <= 16:
            size = "4xlarge"
        elif vcpus <= 32:
            size = "8xlarge"
        elif vcpus <= 48:
            size = "12xlarge"
        else:
            size = "16xlarge"
        
        return f"{family}.{size}"
    
    def _calculate_confidence(self, metrics: VRopsMetrics) -> float:
        """Calculate confidence score for sizing recommendation"""
        confidence = 100.0
        
        # Reduce confidence for insufficient data
        if metrics.collection_period_days < 7:
            confidence -= 30
        elif metrics.collection_period_days < 14:
            confidence -= 15
        
        if metrics.data_completeness < 90:
            confidence -= 20
        elif metrics.data_completeness < 95:
            confidence -= 10
        
        # Reduce confidence for performance issues
        if metrics.cpu_ready_avg > 10:
            confidence -= 15
        if metrics.memory_balloon_avg > 5:
            confidence -= 15
        if metrics.disk_latency_avg > 50:
            confidence -= 10
        
        return max(50.0, confidence)
    
    def _assess_performance_risk(self, metrics: VRopsMetrics, warnings: List[str]) -> str:
        """Assess performance risk level"""
        critical_warnings = [w for w in warnings if "CRITICAL" in w]
        warning_alerts = [w for w in warnings if "WARNING" in w and "CRITICAL" not in w]
        
        if len(critical_warnings) >= 2:
            return "High"
        elif len(critical_warnings) >= 1 or len(warning_alerts) >= 3:
            return "Medium"
        else:
            return "Low"
    
    def _calculate_cost_optimization(self, metrics: VRopsMetrics) -> float:
        """Calculate potential cost optimization percentage"""
        optimization = 0.0
        
        # CPU optimization
        if metrics.cpu_usage_avg < 30 and metrics.cpu_usage_peak < 60:
            optimization += 25.0  # Significant CPU oversizing
        elif metrics.cpu_usage_avg < 50:
            optimization += 15.0  # Moderate CPU oversizing
        
        # Memory optimization
        if metrics.memory_usage_avg < 40 and metrics.memory_usage_peak < 60:
            optimization += 20.0  # Significant memory oversizing
        elif metrics.memory_usage_avg < 60:
            optimization += 10.0  # Moderate memory oversizing
        
        return min(optimization, 50.0)  # Cap at 50% potential optimization
    
    def _generate_sizing_reasoning(self, metrics: VRopsMetrics, vcpus: int, memory_gb: float, warnings: List[str]) -> str:
        """Generate detailed reasoning for sizing recommendation"""
        reasoning_parts = []
        
        # CPU reasoning
        reasoning_parts.append(f"CPU Analysis: Based on 95th percentile utilization of {metrics.cpu_usage_95th:.1f}% "
                             f"with {self.cpu_safety_margin*100:.0f}% safety margin, recommending {vcpus} vCPUs.")
        
        if metrics.cpu_ready_avg > 5:
            reasoning_parts.append(f"CPU ready time of {metrics.cpu_ready_avg:.1f}% indicates contention, "
                                 f"additional headroom provided.")
        
        # Memory reasoning
        reasoning_parts.append(f"Memory Analysis: Based on 95th percentile utilization of {metrics.memory_usage_95th:.1f}% "
                             f"with {self.memory_safety_margin*100:.0f}% safety margin, recommending {memory_gb:.0f}GB.")
        
        if metrics.memory_balloon_avg > 1:
            reasoning_parts.append(f"Memory ballooning of {metrics.memory_balloon_avg:.1f}% detected, "
                                 f"increased allocation to prevent memory pressure.")
        
        # Performance considerations
        if warnings:
            reasoning_parts.append(f"Performance warnings addressed: {len(warnings)} issues identified and "
                                 f"mitigated in sizing recommendation.")
        
        return " ".join(reasoning_parts)

class SQLServerOptimizer:
    """SQL Server licensing and configuration optimizer"""
    
    def __init__(self):
        self.licensing_costs = {
            'Standard': {'core': 3717, 'cal': 209},  # Per core/CAL annual
            'Enterprise': {'core': 14256, 'cal': 209},
            'Developer': {'core': 0, 'cal': 0}
        }
        self.ahb_discount = 0.55  # 55% discount with Azure Hybrid Benefit
        
    def optimize_sql_licensing(self, sql_config: SQLServerConfig, vrops_metrics: VRopsMetrics) -> SQLLicensingOptimization:
        """Optimize SQL Server licensing based on usage patterns and performance data"""
        
        # Analyze current licensing efficiency
        current_efficiency = self._analyze_current_licensing(sql_config)
        
        # Recommend optimal edition
        recommended_edition = self._recommend_sql_edition(sql_config, vrops_metrics)
        
        # Recommend optimal licensing model
        recommended_licensing = self._recommend_licensing_model(sql_config)
        
        # Calculate potential savings
        savings_analysis = self._calculate_licensing_savings(
            sql_config, recommended_edition, recommended_licensing, vrops_metrics
        )
        
        # Generate optimization recommendations
        recommendations = self._generate_licensing_recommendations(
            sql_config, recommended_edition, recommended_licensing, savings_analysis
        )
        
        return SQLLicensingOptimization(
            recommended_edition=recommended_edition,
            recommended_licensing_model=recommended_licensing,
            estimated_annual_savings=savings_analysis['total_savings'],
            optimization_confidence=savings_analysis['confidence'],
            hybrid_benefit_savings=savings_analysis['ahb_savings'],
            rightsizing_savings=savings_analysis['rightsizing_savings'],
            recommendations=recommendations
        )
    
    def _analyze_current_licensing(self, sql_config: SQLServerConfig) -> Dict:
        """Analyze current SQL Server licensing efficiency"""
        analysis = {}
        
        # Calculate current annual cost
        if sql_config.current_licensing_model == "Core-based":
            current_cost = (sql_config.current_cores_licensed * 
                          self.licensing_costs[sql_config.current_edition]['core'])
        else:  # CAL-based
            current_cost = (sql_config.current_cal_count * 
                          self.licensing_costs[sql_config.current_edition]['cal'])
        
        analysis['current_annual_cost'] = current_cost
        analysis['licensing_model'] = sql_config.current_licensing_model
        analysis['edition'] = sql_config.current_edition
        
        return analysis
    
    def _recommend_sql_edition(self, sql_config: SQLServerConfig, vrops_metrics: VRopsMetrics) -> str:
        """Recommend optimal SQL Server edition based on requirements"""
        
        # Check if Enterprise features are required
        if sql_config.requires_enterprise_features:
            return "Enterprise"
        
        # Check advanced features requirement
        if sql_config.requires_advanced_features:
            return "Standard"
        
        # Check memory requirements (Standard limited to 128GB)
        if sql_config.max_memory_gb > 128:
            return "Enterprise"
        
        # Check availability requirements
        if sql_config.availability_requirement == "High":
            return "Enterprise"  # For Always On, advanced backup features
        
        # For most workloads, Standard is sufficient
        return "Standard"
    
    def _recommend_licensing_model(self, sql_config: SQLServerConfig) -> str:
        """Recommend optimal licensing model (Core vs CAL)"""
        
        # Calculate cost for both models
        standard_core_cost = self.licensing_costs['Standard']['core']
        standard_cal_cost = self.licensing_costs['Standard']['cal']
        
        # Core-based licensing cost (minimum 4 cores, sold in 2-core packs)
        min_cores = max(4, sql_config.current_cores_licensed)
        core_annual_cost = min_cores * standard_core_cost
        
        # CAL-based licensing cost (server license + CALs)
        server_license_cost = 931  # Annual server license cost
        cal_annual_cost = server_license_cost + (sql_config.peak_concurrent_users * standard_cal_cost)
        
        if core_annual_cost < cal_annual_cost:
            return "Core-based"
        else:
            return "CAL-based"
    
    def _calculate_licensing_savings(self, sql_config: SQLServerConfig, 
                                   recommended_edition: str, 
                                   recommended_licensing: str,
                                   vrops_metrics: VRopsMetrics) -> Dict:
        """Calculate potential licensing cost savings"""
        
        savings_analysis = {}
        
        # Current costs
        current_analysis = self._analyze_current_licensing(sql_config)
        current_cost = current_analysis['current_annual_cost']
        
        # Recommended costs
        if recommended_licensing == "Core-based":
            min_cores = max(4, sql_config.current_cores_licensed)
            recommended_cost = min_cores * self.licensing_costs[recommended_edition]['core']
        else:
            server_cost = 931
            cal_cost = sql_config.peak_concurrent_users * self.licensing_costs[recommended_edition]['cal']
            recommended_cost = server_cost + cal_cost
        
        # Edition optimization savings
        edition_savings = current_cost - recommended_cost
        
        # Azure Hybrid Benefit savings
        ahb_savings = 0
        if sql_config.has_software_assurance and sql_config.eligible_for_ahb:
            ahb_savings = recommended_cost * self.ahb_discount
        
        # Right-sizing savings based on vROps data
        rightsizing_savings = 0
        if vrops_metrics.cpu_usage_avg < 50:  # Underutilized
            potential_core_reduction = max(1, int(sql_config.current_cores_licensed * 0.3))
            if recommended_licensing == "Core-based":
                rightsizing_savings = potential_core_reduction * self.licensing_costs[recommended_edition]['core']
        
        total_savings = edition_savings + ahb_savings + rightsizing_savings
        
        # Calculate confidence based on data quality
        confidence = 85.0
        if vrops_metrics.collection_period_days < 14:
            confidence -= 15
        if vrops_metrics.data_completeness < 90:
            confidence -= 10
        
        savings_analysis.update({
            'current_cost': current_cost,
            'recommended_cost': recommended_cost,
            'edition_savings': edition_savings,
            'ahb_savings': ahb_savings,
            'rightsizing_savings': rightsizing_savings,
            'total_savings': total_savings,
            'confidence': confidence
        })
        
        return savings_analysis
    
    def _generate_licensing_recommendations(self, sql_config: SQLServerConfig,
                                          recommended_edition: str,
                                          recommended_licensing: str,
                                          savings_analysis: Dict) -> List[str]:
        """Generate specific licensing optimization recommendations"""
        
        recommendations = []
        
        # Edition recommendations
        if recommended_edition != sql_config.current_edition:
            if recommended_edition == "Standard":
                recommendations.append(
                    f"Downgrade from {sql_config.current_edition} to Standard edition "
                    f"(saves ${savings_analysis['edition_savings']:,.0f} annually)"
                )
            else:
                recommendations.append(
                    f"Current {sql_config.current_edition} edition is appropriate for requirements"
                )
        
        # Licensing model recommendations
        if recommended_licensing != sql_config.current_licensing_model:
            recommendations.append(
                f"Switch to {recommended_licensing} licensing model for cost optimization"
            )
        
        # Azure Hybrid Benefit
        if sql_config.has_software_assurance and not sql_config.eligible_for_ahb:
            recommendations.append(
                f"Activate Azure Hybrid Benefit to save ${savings_analysis['ahb_savings']:,.0f} annually"
            )
        elif not sql_config.has_software_assurance:
            recommendations.append(
                "Consider Software Assurance to enable Azure Hybrid Benefit for additional savings"
            )
        
        # Right-sizing recommendations
        if savings_analysis['rightsizing_savings'] > 0:
            recommendations.append(
                f"Right-size core allocation based on actual utilization "
                f"(potential savings: ${savings_analysis['rightsizing_savings']:,.0f})"
            )
        
        # High-level optimization strategies
        if sql_config.database_count > 10:
            recommendations.append(
                "Consider database consolidation to reduce licensing complexity and costs"
            )
        
        if sql_config.workload_type == "OLAP" and sql_config.current_edition == "Enterprise":
            recommendations.append(
                "Evaluate if Analysis Services features justify Enterprise edition costs"
            )
        
        return recommendations

# Update the enhanced application class
class EnhancedCloudPricingOptimizer:
    """Enhanced main application class with vROps integration and SQL optimization"""
    
    def __init__(self):
        self.aws_pricing = EnhancedAWSPricingFetcher()
        self.claude_ai = EnhancedClaudeAIIntegration()
        self.mock_data = EnhancedMockPricingData()
        self.pdf_generator = PDFReportGenerator()
        self.vrops_analyzer = VRopsAnalyzer()
        self.sql_optimizer = SQLServerOptimizer()
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
        if 'vrops_metrics' not in st.session_state:
            st.session_state.vrops_metrics = None
        if 'sql_config' not in st.session_state:
            st.session_state.sql_config = None
        if 'sizing_recommendation' not in st.session_state:
            st.session_state.sizing_recommendation = None
        if 'sql_optimization' not in st.session_state:
            st.session_state.sql_optimization = None
    
    def render_main_interface(self):
        """Render the enhanced main Streamlit interface"""
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>☁️ AWS Cloud Pricing Optimizer</h1>
            <p>Professional-grade AWS pricing analysis with vROps metrics integration and SQL licensing optimization</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Check connection status and show warnings
        self.render_connection_status()
        
        # Sidebar configuration
        self.render_enhanced_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 vROps Metrics", 
            "💰 Pricing Analysis", 
            "🤖 AI Recommendations", 
            "📈 Cost Comparison", 
            "⚠️ Risk Assessment",
            "🗃️ SQL Optimization",
            "📄 Professional Reports"
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
            self.render_risk_assessment()
            
        with tab6:
            self.render_sql_optimization()
        
        with tab7:
            self.render_professional_reports()
    
    def render_enhanced_sidebar(self):
        """Render enhanced sidebar with vROps and SQL configuration"""
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
            
            # Basic Workload Configuration
            st.markdown('<div class="section-header">📋 Workload Parameters</div>', unsafe_allow_html=True)
            
            region = st.selectbox("AWS Region", [
                "us-east-1", "us-west-1", "us-west-2", "us-east-2",
                "eu-west-1", "eu-central-1", "ap-southeast-1", "ap-northeast-1"
            ], index=0, key="region_selector")
            
            workload_type = st.selectbox("Workload Type", [
                "Production", "Staging", "Development", "Testing"
            ], key="workload_type_selector")
            
            # vROps Integration Section
            st.markdown('<div class="section-header">📊 vROps Integration</div>', unsafe_allow_html=True)
            
            with st.expander("🔍 vROps Data Input", expanded=False):
                st.markdown("**CPU Performance Metrics**")
                col1, col2 = st.columns(2)
                with col1:
                    cpu_avg = st.number_input("CPU Usage Avg (%)", 0.0, 100.0, 45.0, key="cpu_avg")
                    cpu_peak = st.number_input("CPU Usage Peak (%)", 0.0, 100.0, 75.0, key="cpu_peak")
                    cpu_95th = st.number_input("CPU Usage 95th (%)", 0.0, 100.0, 65.0, key="cpu_95th")
                with col2:
                    cpu_ready = st.number_input("CPU Ready (%)", 0.0, 50.0, 2.0, key="cpu_ready")
                    cpu_costop = st.number_input("CPU Co-Stop (%)", 0.0, 10.0, 0.5, key="cpu_costop")
                
                st.markdown("**Memory Performance Metrics**")
                col1, col2 = st.columns(2)
                with col1:
                    mem_avg = st.number_input("Memory Usage Avg (%)", 0.0, 100.0, 60.0, key="mem_avg")
                    mem_peak = st.number_input("Memory Usage Peak (%)", 0.0, 100.0, 85.0, key="mem_peak")
                    mem_95th = st.number_input("Memory Usage 95th (%)", 0.0, 100.0, 75.0, key="mem_95th")
                with col2:
                    mem_balloon = st.number_input("Memory Balloon (%)", 0.0, 20.0, 0.0, key="mem_balloon")
                    mem_swapped = st.number_input("Memory Swapped (MB)", 0.0, 1000.0, 0.0, key="mem_swapped")
                    mem_active = st.number_input("Memory Active (%)", 0.0, 100.0, 50.0, key="mem_active")
                
                st.markdown("**Storage Performance Metrics**")
                col1, col2 = st.columns(2)
                with col1:
                    disk_iops_avg = st.number_input("Disk IOPS Avg", 0.0, 50000.0, 1500.0, key="disk_iops_avg")
                    disk_iops_peak = st.number_input("Disk IOPS Peak", 0.0, 100000.0, 3000.0, key="disk_iops_peak")
                    disk_latency_avg = st.number_input("Disk Latency Avg (ms)", 0.0, 200.0, 15.0, key="disk_latency_avg")
                with col2:
                    disk_latency_peak = st.number_input("Disk Latency Peak (ms)", 0.0, 500.0, 45.0, key="disk_latency_peak")
                    disk_throughput = st.number_input("Disk Throughput (MB/s)", 0.0, 1000.0, 50.0, key="disk_throughput")
                    disk_queue_depth = st.number_input("Disk Queue Depth", 0.0, 50.0, 3.0, key="disk_queue_depth")
                
                st.markdown("**Network Performance Metrics**")
                col1, col2 = st.columns(2)
                with col1:
                    net_usage_avg = st.number_input("Network Usage Avg (%)", 0.0, 100.0, 25.0, key="net_usage_avg")
                    net_usage_peak = st.number_input("Network Usage Peak (%)", 0.0, 100.0, 60.0, key="net_usage_peak")
                with col2:
                    net_packets = st.number_input("Network Packets/sec", 0.0, 100000.0, 5000.0, key="net_packets")
                    net_drops = st.number_input("Network Drops/sec", 0.0, 1000.0, 0.0, key="net_drops")
                
                st.markdown("**Guest OS Metrics**")
                col1, col2 = st.columns(2)
                with col1:
                    guest_cpu = st.number_input("Guest CPU Usage (%)", 0.0, 100.0, 40.0, key="guest_cpu")
                    guest_memory = st.number_input("Guest Memory Usage (%)", 0.0, 100.0, 55.0, key="guest_memory")
                with col2:
                    collection_days = st.number_input("Collection Period (days)", 1, 365, 30, key="collection_days")
                    data_completeness = st.number_input("Data Completeness (%)", 50.0, 100.0, 95.0, key="data_completeness")
                
                # Store vROps metrics
                st.session_state.vrops_metrics = VRopsMetrics(
                    cpu_usage_avg=cpu_avg,
                    cpu_usage_peak=cpu_peak,
                    cpu_usage_95th=cpu_95th,
                    cpu_ready_avg=cpu_ready,
                    cpu_costop_avg=cpu_costop,
                    memory_usage_avg=mem_avg,
                    memory_usage_peak=mem_peak,
                    memory_usage_95th=mem_95th,
                    memory_active_avg=mem_active,
                    memory_balloon_avg=mem_balloon,
                    memory_swapped_avg=mem_swapped,
                    disk_iops_avg=disk_iops_avg,
                    disk_iops_peak=disk_iops_peak,
                    disk_latency_avg=disk_latency_avg,
                    disk_latency_peak=disk_latency_peak,
                    disk_throughput_avg=disk_throughput,
                    disk_queue_depth_avg=disk_queue_depth,
                    network_usage_avg=net_usage_avg,
                    network_usage_peak=net_usage_peak,
                    network_packets_avg=net_packets,
                    network_drops_avg=net_drops,
                    guest_cpu_usage_avg=guest_cpu,
                    guest_memory_usage_avg=guest_memory,
                    collection_period_days=collection_days,
                    data_completeness=data_completeness
                )
            
            # SQL Server Configuration Section
            st.markdown('<div class="section-header">🗃️ SQL Server Configuration</div>', unsafe_allow_html=True)
            
            with st.expander("⚙️ SQL Server Settings", expanded=False):
                st.markdown("**Current Licensing**")
                current_edition = st.selectbox("Current SQL Edition", 
                    ["Standard", "Enterprise", "Developer"], key="current_edition")
                current_licensing = st.selectbox("Current Licensing Model", 
                    ["Core-based", "CAL-based"], key="current_licensing")
                
                col1, col2 = st.columns(2)
                with col1:
                    current_cores = st.number_input("Licensed Cores", 1, 128, 8, key="current_cores")
                    current_cals = st.number_input("CAL Count", 0, 1000, 50, key="current_cals")
                with col2:
                    concurrent_users = st.number_input("Concurrent Users", 1, 1000, 50, key="concurrent_users")
                    peak_users = st.number_input("Peak Users", 1, 2000, 100, key="peak_users")
                
                st.markdown("**Database Configuration**")
                col1, col2 = st.columns(2)
                with col1:
                    db_count = st.number_input("Database Count", 1, 100, 5, key="db_count")
                    db_size = st.number_input("Total DB Size (GB)", 1.0, 10000.0, 500.0, key="db_size")
                    max_memory = st.number_input("Max Memory (GB)", 1.0, 1024.0, 64.0, key="max_memory")
                with col2:
                    workload_pattern = st.selectbox("Workload Type", 
                        ["OLTP", "OLAP", "Mixed"], key="workload_pattern")
                    availability_req = st.selectbox("Availability Requirement", 
                        ["Basic", "Standard", "High"], key="availability_req")
                    backup_freq = st.selectbox("Backup Frequency", 
                        ["Hourly", "Daily", "Weekly"], key="backup_freq")
                
                st.markdown("**Features & Licensing Benefits**")
                col1, col2 = st.columns(2)
                with col1:
                    has_sa = st.checkbox("Has Software Assurance", key="has_sa")
                    eligible_ahb = st.checkbox("Eligible for Azure Hybrid Benefit", key="eligible_ahb")
                    advanced_features = st.checkbox("Requires Advanced Features", key="advanced_features")
                with col2:
                    enterprise_features = st.checkbox("Requires Enterprise Features", key="enterprise_features")
                    current_license_cost = st.number_input("Current Annual License Cost ($)", 
                        0.0, 500000.0, 30000.0, key="current_license_cost")
                
                # Store SQL configuration
                st.session_state.sql_config = SQLServerConfig(
                    current_edition=current_edition,
                    current_licensing_model=current_licensing,
                    current_cores_licensed=current_cores,
                    current_cal_count=current_cals,
                    concurrent_users=concurrent_users,
                    peak_concurrent_users=peak_users,
                    database_count=db_count,
                    database_size_gb=db_size,
                    workload_type=workload_pattern,
                    backup_frequency=backup_freq,
                    availability_requirement=availability_req,
                    has_software_assurance=has_sa,
                    eligible_for_ahb=eligible_ahb,
                    requires_advanced_features=advanced_features,
                    requires_enterprise_features=enterprise_features,
                    max_memory_gb=max_memory,
                    current_annual_license_cost=current_license_cost
                )
            
            # Store basic configuration
            st.session_state.config = {
                'region': region,
                'workload_type': workload_type,
                'cpu_cores': 8,  # Will be overridden by vROps analysis
                'ram_gb': 32,   # Will be overridden by vROps analysis
                'storage_gb': 500,
                'peak_cpu': cpu_peak if st.session_state.vrops_metrics else 75,
                'peak_ram': mem_peak if st.session_state.vrops_metrics else 85,
                'sql_edition': current_edition,
                'licensing_model': current_licensing
            }
    
    def render_vrops_metrics(self):
        """Render vROps metrics analysis section"""
        st.markdown('<div class="section-header">📊 vRealize Operations Metrics Analysis</div>', unsafe_allow_html=True)
        
        if not st.session_state.vrops_metrics:
            st.info("⚠️ Please configure vROps metrics in the sidebar to see detailed performance analysis.")
            return
        
        vrops_metrics = st.session_state.vrops_metrics
        
        # Performance Analysis Button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("🔍 Analyze vROps performance metrics to generate intelligent EC2 sizing recommendations.")
        with col2:
            if st.button("🧠 Analyze Performance", type="primary", use_container_width=True):
                with st.spinner("Analyzing vROps performance metrics..."):
                    sizing_rec, warnings = self.vrops_analyzer.analyze_performance_metrics(vrops_metrics)
                    st.session_state.sizing_recommendation = sizing_rec
                    st.session_state.performance_warnings = warnings
                    st.success("✅ Performance analysis complete!")
        
        # Display current metrics overview
        st.markdown("**📈 Current Performance Metrics Overview**")
        
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
            st.markdown(f"""
            <div class="metric-card">
                <h3>Disk Latency</h3>
                <h2>{vrops_metrics.disk_latency_avg:.1f}ms</h2>
                <p>Avg (Peak: {vrops_metrics.disk_latency_peak:.1f}ms)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Data Collection</h3>
                <h2>{vrops_metrics.collection_period_days}</h2>
                <p>Days ({vrops_metrics.data_completeness:.1f}% complete)</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Performance visualizations
        self.render_performance_charts(vrops_metrics)
        
        # Display sizing recommendations if available
        if hasattr(st.session_state, 'sizing_recommendation') and st.session_state.sizing_recommendation:
            self.display_sizing_recommendations()
    
    def render_performance_charts(self, vrops_metrics: VRopsMetrics):
        """Render performance metrics charts"""
        
        # Create performance dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('CPU Performance Profile', 'Memory Utilization Pattern', 
                          'Storage Performance Metrics', 'Network Utilization'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": False}]]
        )
        
        # CPU metrics
        cpu_metrics = ['Average', '95th Percentile', 'Peak']
        cpu_values = [vrops_metrics.cpu_usage_avg, vrops_metrics.cpu_usage_95th, vrops_metrics.cpu_usage_peak]
        cpu_colors = ['#4a90e2', '#ffc107', '#dc3545']
        
        fig.add_trace(
            go.Bar(x=cpu_metrics, y=cpu_values, name='CPU Usage', 
                  marker_color=cpu_colors, showlegend=False),
            row=1, col=1
        )
        
        # Memory metrics
        mem_metrics = ['Average', '95th Percentile', 'Peak']
        mem_values = [vrops_metrics.memory_usage_avg, vrops_metrics.memory_usage_95th, vrops_metrics.memory_usage_peak]
        
        fig.add_trace(
            go.Bar(x=mem_metrics, y=mem_values, name='Memory Usage',
                  marker_color='#20c997', showlegend=False),
            row=1, col=2
        )
        
        # Storage metrics (dual axis)
        storage_metrics = ['IOPS Avg', 'IOPS Peak', 'Latency Avg', 'Latency Peak']
        iops_values = [vrops_metrics.disk_iops_avg, vrops_metrics.disk_iops_peak, 0, 0]
        latency_values = [0, 0, vrops_metrics.disk_latency_avg, vrops_metrics.disk_latency_peak]
        
        fig.add_trace(
            go.Bar(x=storage_metrics[:2], y=iops_values[:2], name='IOPS',
                  marker_color='#17a2b8', showlegend=False),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(x=storage_metrics[2:], y=latency_values[2:], name='Latency (ms)',
                  marker_color='#6f42c1', showlegend=False, yaxis='y4'),
            row=2, col=1, secondary_y=True
        )
        
        # Network metrics
        net_metrics = ['Average Usage', 'Peak Usage']
        net_values = [vrops_metrics.network_usage_avg, vrops_metrics.network_usage_peak]
        
        fig.add_trace(
            go.Bar(x=net_metrics, y=net_values, name='Network Usage',
                  marker_color='#fd7e14', showlegend=False),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            title_text="vROps Performance Metrics Dashboard",
            font_color='#343a40'
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="CPU Usage (%)", row=1, col=1)
        fig.update_yaxes(title_text="Memory Usage (%)", row=1, col=2)
        fig.update_yaxes(title_text="IOPS", row=2, col=1)
        fig.update_yaxes(title_text="Latency (ms)", row=2, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Network Usage (%)", row=2, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance warnings section
        if hasattr(st.session_state, 'performance_warnings') and st.session_state.performance_warnings:
            st.markdown("**⚠️ Performance Warnings Detected**")
            
            for warning in st.session_state.performance_warnings:
                if "CRITICAL" in warning:
                    st.markdown(f"""
                    <div class="performance-critical">
                        <strong>CRITICAL:</strong> {warning.replace("CRITICAL: ", "")}
                    </div>
                    """, unsafe_allow_html=True)
                elif "WARNING" in warning:
                    st.markdown(f"""
                    <div class="performance-warning">
                        <strong>WARNING:</strong> {warning.replace("WARNING: ", "")}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="optimization-insight">
                        <strong>INSIGHT:</strong> {warning}
                    </div>
                    """, unsafe_allow_html=True)
    
    def display_sizing_recommendations(self):
        """Display vROps-based sizing recommendations"""
        sizing_rec = st.session_state.sizing_recommendation
        
        st.markdown("**🎯 Intelligent Sizing Recommendations**")
        
        # Main recommendation card
        st.markdown(f"""
        <div class="vrops-section">
            <h3>💡 Recommended EC2 Configuration</h3>
            <p><strong>Instance Type:</strong> {sizing_rec.recommended_instance_type}</p>
            <p><strong>vCPUs:</strong> {sizing_rec.recommended_vcpus} | <strong>Memory:</strong> {sizing_rec.recommended_memory_gb:.0f} GB</p>
            <p><strong>Confidence Score:</strong> {sizing_rec.rightsizing_confidence:.1f}% | <strong>Performance Risk:</strong> {sizing_rec.performance_risk_level}</p>
            <p><strong>Cost Optimization Opportunity:</strong> {sizing_rec.cost_optimization_opportunity:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed reasoning
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**🧠 Sizing Rationale**")
            st.markdown(sizing_rec.reasoning)
        
        with col2:
            # Confidence gauge
            fig = go.Figure()
            
            fig.add_trace(go.Indicator(
                mode = "gauge+number",
                value = sizing_rec.rightsizing_confidence,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Sizing Confidence", 'font': {'color': '#1f4e79', 'size': 14}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickcolor': "#1f4e79"},
                    'bar': {'color': "#7b68ee", 'thickness': 0.8},
                    'steps': [
                        {'range': [0, 60], 'color': "#f8f9fa"},
                        {'range': [60, 80], 'color': "#fff3cd"},
                        {'range': [80, 90], 'color': "#d1ecf1"},
                        {'range': [90, 100], 'color': "#d4edda"}
                    ],
                    'threshold': {
                        'line': {'color': "#28a745", 'width': 4},
                        'thickness': 0.75,
                        'value': 85
                    }
                }
            ))
            
            fig.update_layout(height=300, font_color='#343a40', margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
        
        # Update session config with vROps recommendations
        if hasattr(st.session_state, 'config'):
            st.session_state.config.update({
                'cpu_cores': sizing_rec.recommended_vcpus,
                'ram_gb': sizing_rec.recommended_memory_gb,
                'recommended_instance': sizing_rec.recommended_instance_type
            })
    
    def render_sql_optimization(self):
        """Render SQL Server licensing optimization section"""
        st.markdown('<div class="section-header">🗃️ SQL Server Licensing Optimization</div>', unsafe_allow_html=True)
        
        if not st.session_state.sql_config:
            st.info("⚠️ Please configure SQL Server settings in the sidebar to see licensing optimization.")
            return
        
        sql_config = st.session_state.sql_config
        
        # SQL Optimization Analysis Button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("💰 Analyze SQL Server licensing to identify cost optimization opportunities.")
        with col2:
            if st.button("🔍 Optimize SQL Licensing", type="primary", use_container_width=True):
                with st.spinner("Analyzing SQL Server licensing optimization..."):
                    vrops_metrics = st.session_state.vrops_metrics or VRopsMetrics()
                    sql_optimization = self.sql_optimizer.optimize_sql_licensing(sql_config, vrops_metrics)
                    st.session_state.sql_optimization = sql_optimization
                    st.success("✅ SQL licensing optimization complete!")
        
        # Display current SQL configuration
        st.markdown("**⚙️ Current SQL Server Configuration**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Current Edition</h3>
                <h2>{sql_config.current_edition}</h2>
                <p>{sql_config.current_licensing_model}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            licensed_cores = sql_config.current_cores_licensed if sql_config.current_licensing_model == "Core-based" else "N/A"
            cal_count = sql_config.current_cal_count if sql_config.current_licensing_model == "CAL-based" else "N/A"
            st.markdown(f"""
            <div class="metric-card">
                <h3>Licensed Resources</h3>
                <h2>{licensed_cores}</h2>
                <p>Cores | {cal_count} CALs</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Database Environment</h3>
                <h2>{sql_config.database_count}</h2>
                <p>DBs | {sql_config.database_size_gb:.0f} GB</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>User Load</h3>
                <h2>{sql_config.concurrent_users}</h2>
                <p>Avg | {sql_config.peak_concurrent_users} Peak</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Display SQL optimization results if available
        if hasattr(st.session_state, 'sql_optimization') and st.session_state.sql_optimization:
            self.display_sql_optimization_results()
    
    def display_sql_optimization_results(self):
        """Display SQL Server licensing optimization results"""
        sql_opt = st.session_state.sql_optimization
        
        # Main optimization recommendation
        st.markdown(f"""
        <div class="sql-optimization">
            <h3>💡 SQL Licensing Optimization Recommendations</h3>
            <p><strong>Recommended Edition:</strong> {sql_opt.recommended_edition}</p>
            <p><strong>Recommended Licensing:</strong> {sql_opt.recommended_licensing_model}</p>
            <p><strong>Estimated Annual Savings:</strong> ${sql_opt.estimated_annual_savings:,.0f}</p>
            <p><strong>Optimization Confidence:</strong> {sql_opt.optimization_confidence:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Savings breakdown
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="cost-savings">
                <h3>Total Annual Savings</h3>
                <h2>${sql_opt.estimated_annual_savings:,.0f}</h2>
                <p>Combined optimization opportunities</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Azure Hybrid Benefit</h3>
                <h2>${sql_opt.hybrid_benefit_savings:,.0f}</h2>
                <p>Annual savings potential</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Right-sizing Savings</h3>
                <h2>${sql_opt.rightsizing_savings:,.0f}</h2>
                <p>Performance-based optimization</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed recommendations
        st.markdown("**📋 Detailed Optimization Recommendations**")
        
        for i, recommendation in enumerate(sql_opt.recommendations, 1):
            st.markdown(f"""
            <div class="optimization-insight">
                <strong>{i}.</strong> {recommendation}
            </div>
            """, unsafe_allow_html=True)
        
        # SQL licensing cost comparison chart
        self.render_sql_cost_comparison()
    
    def render_sql_cost_comparison(self):
        """Render SQL licensing cost comparison chart"""
        sql_config = st.session_state.sql_config
        sql_opt = st.session_state.sql_optimization
        
        # Calculate cost scenarios
        scenarios = {
            'Current Configuration': sql_config.current_annual_license_cost,
            'Optimized Configuration': sql_config.current_annual_license_cost - sql_opt.estimated_annual_savings,
            'With Azure Hybrid Benefit': sql_config.current_annual_license_cost - sql_opt.hybrid_benefit_savings,
            'Fully Optimized': sql_config.current_annual_license_cost - sql_opt.estimated_annual_savings - sql_opt.hybrid_benefit_savings
        }
        
        # Create comparison chart
        fig = go.Figure()
        
        scenario_names = list(scenarios.keys())
        scenario_costs = list(scenarios.values())
        colors = ['#dc3545', '#ffc107', '#17a2b8', '#28a745']
        
        fig.add_trace(go.Bar(
            x=scenario_names,
            y=scenario_costs,
            marker_color=colors,
            text=[f'${cost:,.0f}' for cost in scenario_costs],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="SQL Server Licensing Cost Optimization Scenarios",
            xaxis_title="Configuration Scenario",
            yaxis_title="Annual Licensing Cost ($)",
            height=400,
            font_color='#343a40'
        )
        
        st.plotly_chart(fig, use_container_width=True)

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
    
    def get_enhanced_ec2_pricing(self, instance_type: str, region: str, vrops_data: VRopsMetrics = None) -> Optional[PricingData]:
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
            
            # Calculate SQL licensing cost based on instance specs and usage
            sql_licensing_cost = self._calculate_sql_licensing_cost(specs, vrops_data)
            
            base_monthly_cost = on_demand_price * 730
            total_monthly_cost = base_monthly_cost + sql_licensing_cost
            
            return PricingData(
                service="EC2",
                instance_type=instance_type,
                region=region,
                price_per_hour=on_demand_price,
                price_per_month=base_monthly_cost,
                currency="USD",
                last_updated=datetime.now(),
                specifications=specs,
                reserved_pricing=reserved_pricing,
                spot_pricing=spot_price,
                sql_licensing_cost=sql_licensing_cost,
                total_monthly_cost=total_monthly_cost
            )
                        
        except Exception as e:
            logger.error(f"Error fetching enhanced EC2 pricing for {instance_type}: {e}")
            return None
    
    def _calculate_sql_licensing_cost(self, specs: Dict, vrops_data: VRopsMetrics = None) -> float:
        """Calculate SQL Server licensing cost based on instance specs"""
        if not specs:
            return 0.0
        
        vcpus = specs.get('vcpus', 4)
        
        # Base SQL Server Standard licensing cost per core per month
        sql_standard_core_monthly = 3717 / 12  # Annual to monthly
        
        # Minimum 4 cores required for SQL Server licensing
        license_cores = max(4, vcpus)
        
        # Adjust for actual utilization if vROps data available
        if vrops_data and vrops_data.cpu_usage_avg < 50:
            # Potential to reduce core count licensing for underutilized systems
            utilization_factor = max(0.5, vrops_data.cpu_usage_avg / 100)
            license_cores = max(4, int(license_cores * utilization_factor))
        
        monthly_sql_cost = license_cores * sql_standard_core_monthly
        return monthly_sql_cost
    
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
    def get_enhanced_sample_pricing_data(region: str, vrops_data: VRopsMetrics = None) -> List[PricingData]:
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
            
            # Calculate SQL licensing cost
            sql_licensing_cost = EnhancedMockPricingData._calculate_mock_sql_cost(vcpus, vrops_data)
            
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
            
            base_monthly_cost = adjusted_price * 730
            total_monthly_cost = base_monthly_cost + sql_licensing_cost
            
            pricing_data.append(PricingData(
                service="EC2",
                instance_type=instance_type,
                region=region,
                price_per_hour=adjusted_price,
                price_per_month=base_monthly_cost,
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
                spot_pricing=spot_price,
                sql_licensing_cost=sql_licensing_cost,
                total_monthly_cost=total_monthly_cost
            ))
        
        return sorted(pricing_data, key=lambda x: x.total_monthly_cost)
    
    @staticmethod
    def _calculate_mock_sql_cost(vcpus: int, vrops_data: VRopsMetrics = None) -> float:
        """Calculate mock SQL Server licensing cost"""
        sql_standard_core_monthly = 3717 / 12  # Annual to monthly
        license_cores = max(4, vcpus)
        
        # Adjust for actual utilization if vROps data available
        if vrops_data and vrops_data.cpu_usage_avg < 50:
            utilization_factor = max(0.5, vrops_data.cpu_usage_avg / 100)
            license_cores = max(4, int(license_cores * utilization_factor))
        
        return license_cores * sql_standard_core_monthly

class EnhancedClaudeAIIntegration:
    """Enhanced Claude AI integration with vROps and SQL optimization analysis"""
    
    def __init__(self):
        self.api_key = st.secrets.get("CLAUDE_API_KEY") if hasattr(st, 'secrets') else None
        self.base_url = "https://api.anthropic.com/v1/messages"
        
    async def get_comprehensive_analysis(self, 
                                       workload_data: Dict, 
                                       pricing_data: List[PricingData],
                                       vrops_data: VRopsMetrics = None,
                                       sql_config: SQLServerConfig = None) -> Tuple:
        """Get comprehensive AI analysis including vROps and SQL optimization"""
        try:
            if not self.api_key:
                return self._fallback_comprehensive_analysis(workload_data, pricing_data, vrops_data, sql_config)
                
            prompt = self._build_enhanced_comprehensive_prompt(workload_data, pricing_data, vrops_data, sql_config)
            
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            }
            
            payload = {
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 3000,  # Increased for comprehensive analysis
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=headers, json=payload, timeout=60) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result['content'][0]['text']
                        return self._parse_comprehensive_response(content)
                    else:
                        logger.error(f"Claude API error: {response.status}")
                        return self._fallback_comprehensive_analysis(workload_data, pricing_data, vrops_data, sql_config)
                        
        except Exception as e:
            logger.error(f"Error getting comprehensive AI analysis: {e}")
            return self._fallback_comprehensive_analysis(workload_data, pricing_data, vrops_data, sql_config)
    
    def _build_enhanced_comprehensive_prompt(self, workload_data: Dict, pricing_data: List[PricingData], 
                                           vrops_data: VRopsMetrics = None, sql_config: SQLServerConfig = None) -> str:
        """Build comprehensive prompt including vROps and SQL data"""
        vrops_section = ""
        if vrops_data:
            vrops_section = f"""
        vROPS PERFORMANCE METRICS:
        - CPU: Avg {vrops_data.cpu_usage_avg:.1f}%, Peak {vrops_data.cpu_usage_peak:.1f}%, 95th {vrops_data.cpu_usage_95th:.1f}%
        - CPU Ready: {vrops_data.cpu_ready_avg:.1f}% (Critical if >10%)
        - Memory: Avg {vrops_data.memory_usage_avg:.1f}%, Peak {vrops_data.memory_usage_peak:.1f}%, 95th {vrops_data.memory_usage_95th:.1f}%
        - Memory Balloon: {vrops_data.memory_balloon_avg:.1f}% (Concerning if >1%)
        - Memory Swapped: {vrops_data.memory_swapped_avg:.1f} MB
        - Disk Latency: Avg {vrops_data.disk_latency_avg:.1f}ms, Peak {vrops_data.disk_latency_peak:.1f}ms
        - IOPS: Avg {vrops_data.disk_iops_avg:.0f}, Peak {vrops_data.disk_iops_peak:.0f}
        - Network Usage: Avg {vrops_data.network_usage_avg:.1f}%, Peak {vrops_data.network_usage_peak:.1f}%
        - Data Collection: {vrops_data.collection_period_days} days, {vrops_data.data_completeness:.1f}% complete
            """
        
        sql_section = ""
        if sql_config:
            sql_section = f"""
        SQL SERVER CONFIGURATION:
        - Current Edition: {sql_config.current_edition}
        - Licensing Model: {sql_config.current_licensing_model}
        - Licensed Cores: {sql_config.current_cores_licensed}
        - Database Count: {sql_config.database_count}
        - Workload Type: {sql_config.workload_type}
        - Concurrent Users: {sql_config.concurrent_users} (Peak: {sql_config.peak_concurrent_users})
        - Has Software Assurance: {sql_config.has_software_assurance}
        - Azure Hybrid Benefit Eligible: {sql_config.eligible_for_ahb}
        - Requires Enterprise Features: {sql_config.requires_enterprise_features}
            """
        
        return f"""
        As a senior cloud architect with expertise in vROps performance analysis and SQL Server optimization, 
        provide a comprehensive migration analysis.

        WORKLOAD PROFILE:
        - Region: {workload_data.get('region', 'N/A')}
        - Environment: {workload_data.get('workload_type', 'N/A')}
        - Current CPU Cores: {workload_data.get('cpu_cores', 'N/A')}
        - Current RAM: {workload_data.get('ram_gb', 'N/A')} GB

        {vrops_section}

        {sql_section}

        TOP AWS PRICING OPTIONS (including SQL licensing):
        {self._format_enhanced_pricing_for_prompt(pricing_data[:5])}

        Provide comprehensive analysis in this JSON format:
        {{
            "recommendation": {{
                "recommendation": "Detailed recommendation with vROps-informed sizing",
                "confidence_score": 85,
                "cost_impact": "High/Medium/Low",
                "reasoning": "Technical justification including performance analysis",
                "risk_assessment": "Performance and migration risks",
                "implementation_timeline": "Phased implementation approach",
                "expected_savings": 75000
            }},
            "vrops_insights": [
                "Performance bottleneck identified: High CPU ready time indicates oversized vCPU allocation",
                "Memory optimization: Low utilization suggests right-sizing opportunity"
            ],
            "sql_optimization": [
                "SQL licensing recommendation: Downgrade to Standard edition",
                "Azure Hybrid Benefit: Enable for 55% licensing cost reduction"
            ],
            "risk_assessments": [
                {{
                    "category": "Performance Risk",
                    "risk_level": "Medium",
                    "description": "vROps data shows performance characteristics",
                    "mitigation_strategy": "Performance-based mitigation",
                    "impact": "Business impact assessment"
                }}
            ],
            "implementation_phases": [
                {{
                    "phase": "Phase 1: Performance Validation",
                    "duration": "2 weeks",
                    "activities": ["Validate vROps baselines", "Performance testing"],
                    "dependencies": ["vROps data validation"],
                    "deliverables": ["Performance baseline report"]
                }}
            ]
        }}
        """
    
    def _format_enhanced_pricing_for_prompt(self, pricing_data: List[PricingData]) -> str:
        """Format enhanced pricing data including SQL costs for AI prompt"""
        formatted = []
        for i, pricing_obj in enumerate(pricing_data, 1):
            specs = pricing_obj.specifications or {}
            reserved_1yr = pricing_obj.reserved_pricing.get('1_year_all_upfront', 0) if pricing_obj.reserved_pricing else 0
            reserved_3yr = pricing_obj.reserved_pricing.get('3_year_all_upfront', 0) if pricing_obj.reserved_pricing else 0
            
            formatted.append(
                f"{i}. {pricing_obj.instance_type} ({specs.get('family', 'Unknown')}) - "
                f"{specs.get('vcpus', '?')} vCPUs, {specs.get('ram', '?')} GB RAM\n"
                f"   Infrastructure: ${pricing_obj.price_per_month:.0f}/month\n"
                f"   SQL Licensing: ${pricing_obj.sql_licensing_cost:.0f}/month\n"
                f"   Total Cost: ${pricing_obj.total_monthly_cost:.0f}/month\n"
                f"   Reserved 3-Year Total: ${(reserved_3yr * 730 + pricing_obj.sql_licensing_cost):.0f}/month"
            )
        return "\n\n".join(formatted)
    
    def _parse_comprehensive_response(self, content: str) -> Tuple:
        """Parse comprehensive AI response with vROps and SQL insights"""
        try:
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                # Parse recommendation
                rec_data = data.get('recommendation', {})
                from dataclasses import dataclass
                
                @dataclass
                class AIRecommendation:
                    recommendation: str
                    confidence_score: float
                    cost_impact: str
                    reasoning: str
                    risk_assessment: str = ""
                    implementation_timeline: str = ""
                    expected_savings: float = 0.0
                
                @dataclass
                class RiskAssessment:
                    category: str
                    risk_level: str
                    description: str
                    mitigation_strategy: str
                    impact: str
                
                @dataclass
                class ImplementationPhase:
                    phase: str
                    duration: str
                    activities: List[str]
                    dependencies: List[str]
                    deliverables: List[str]
                
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
                
                # Extract vROps and SQL insights
                vrops_insights = data.get('vrops_insights', [])
                sql_optimization = data.get('sql_optimization', [])
                
                return recommendation, risks, phases, vrops_insights, sql_optimization
                
        except Exception as e:
            logger.warning(f"Failed to parse comprehensive AI response: {e}")
        
        return self._fallback_comprehensive_analysis({}, [], None, None)
    
    def _fallback_comprehensive_analysis(self, workload_data: Dict, pricing_data: List[PricingData],
                                       vrops_data: VRopsMetrics = None, sql_config: SQLServerConfig = None) -> Tuple:
        """Enhanced fallback analysis with vROps and SQL considerations"""
        from dataclasses import dataclass
        
        @dataclass
        class AIRecommendation:
            recommendation: str
            confidence_score: float
            cost_impact: str
            reasoning: str
            risk_assessment: str = ""
            implementation_timeline: str = ""
            expected_savings: float = 0.0
        
        @dataclass
        class RiskAssessment:
            category: str
            risk_level: str
            description: str
            mitigation_strategy: str
            impact: str
        
        @dataclass
        class ImplementationPhase:
            phase: str
            duration: str
            activities: List[str]
            dependencies: List[str]
            deliverables: List[str]
        
        # Enhanced recommendation based on available data
        recommendation_text = "Implement comprehensive migration strategy with performance-based right-sizing"
        if vrops_data:
            if vrops_data.cpu_usage_avg < 50:
                recommendation_text += " and CPU optimization based on vROps utilization data"
            if vrops_data.memory_balloon_avg > 1:
                recommendation_text += " with memory pressure mitigation"
        
        recommendation = AIRecommendation(
            recommendation=recommendation_text,
            confidence_score=75,
            cost_impact="Medium to High",
            reasoning="Analysis based on vROps performance data and SQL optimization opportunities",
            risk_assessment="Performance risks mitigated through data-driven sizing",
            implementation_timeline="4-8 months for comprehensive migration",
            expected_savings=50000
        )
        
        # Enhanced risk assessments
        risks = [
            RiskAssessment(
                category="Performance Risk",
                risk_level="Medium",
                description="Application performance during migration with vROps validation",
                mitigation_strategy="Use vROps baselines for performance validation",
                impact="Potential performance degradation"
            ),
            RiskAssessment(
                category="SQL Licensing Risk",
                risk_level="Low",
                description="SQL Server licensing compliance and optimization",
                mitigation_strategy="Validate licensing requirements and Azure Hybrid Benefit eligibility",
                impact="Licensing cost overruns or compliance issues"
            )
        ]
        
        # Enhanced implementation phases
        phases = [
            ImplementationPhase(
                phase="Phase 1: Performance Assessment & Validation",
                duration="3-4 weeks",
                activities=["vROps data validation", "Performance baseline establishment", "SQL licensing audit"],
                dependencies=["vROps data access", "Current environment documentation"],
                deliverables=["Performance assessment report", "Licensing optimization plan"]
            ),
            ImplementationPhase(
                phase="Phase 2: Environment Preparation",
                duration="4-6 weeks",
                activities=["AWS account setup", "Right-sized instance provisioning", "SQL licensing optimization"],
                dependencies=["Phase 1 completion", "Budget approval"],
                deliverables=["Optimized AWS environment", "SQL licensing configuration"]
            )
        ]
        
        # Generate vROps and SQL insights
        vrops_insights = []
        if vrops_data:
            if vrops_data.cpu_usage_avg < 40:
                vrops_insights.append("CPU over-provisioning detected - significant right-sizing opportunity")
            if vrops_data.memory_balloon_avg > 1:
                vrops_insights.append("Memory pressure indicators require memory allocation optimization")
            if vrops_data.disk_latency_avg > 20:
                vrops_insights.append("Storage performance concerns - consider provisioned IOPS EBS")
        
        sql_optimization = []
        if sql_config:
            if sql_config.has_software_assurance and sql_config.eligible_for_ahb:
                sql_optimization.append("Azure Hybrid Benefit eligible - up to 55% licensing cost reduction")
            if sql_config.current_edition == "Enterprise" and not sql_config.requires_enterprise_features:
                sql_optimization.append("SQL Standard edition sufficient - potential licensing cost reduction")
        
        return recommendation, risks, phases, vrops_insights, sql_optimization

class PDFReportGenerator:
    """Professional PDF report generator with vROps and SQL optimization"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
        
    def setup_custom_styles(self):
        """Setup custom styles for the PDF"""
        # Custom styles for enhanced reporting
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            fontName='Helvetica-Bold',
            textColor=HexColor('#1f4e79'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        # Additional styles would be defined here...
    
    def create_comprehensive_report(self, config: Dict, pricing_data: List[PricingData],
                                  recommendation, risk_assessments, implementation_phases,
                                  vrops_data: VRopsMetrics = None, sql_config: SQLServerConfig = None) -> BytesIO:
        """Create comprehensive PDF report with vROps and SQL analysis"""
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        
        story = []
        
        # Enhanced title page with vROps and SQL sections
        story.extend(self._create_enhanced_title_page(config, vrops_data, sql_config))
        story.append(PageBreak())
        
        # Executive summary with performance insights
        story.extend(self._create_enhanced_executive_summary(config, recommendation, pricing_data, vrops_data))
        story.append(PageBreak())
        
        # vROps performance analysis section
        if vrops_data:
            story.extend(self._create_vrops_analysis_section(vrops_data))
            story.append(PageBreak())
        
        # SQL optimization section
        if sql_config:
            story.extend(self._create_sql_optimization_section(sql_config))
            story.append(PageBreak())
        
        # Continue with existing sections...
        
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    def _create_enhanced_title_page(self, config: Dict, vrops_data: VRopsMetrics = None, sql_config: SQLServerConfig = None) -> List:
        """Create enhanced title page with vROps and SQL information"""
        elements = []
        
        elements.append(Spacer(1, 2*inch))
        elements.append(Paragraph("AWS Cloud Migration", self.styles['CustomTitle']))
        elements.append(Paragraph("Comprehensive Optimization Report", self.styles['CustomTitle']))
        elements.append(Paragraph("with vROps Performance Analysis & SQL Optimization", self.styles['Normal']))
        
        # Enhanced report details with performance and SQL data
        report_data = [
            ['Report Generated:', datetime.now().strftime('%B %d, %Y at %I:%M %p')],
            ['Analysis Region:', config.get('region', 'N/A')],
            ['Workload Type:', config.get('workload_type', 'N/A')],
        ]
        
        if vrops_data:
            report_data.extend([
                ['vROps Data Period:', f"{vrops_data.collection_period_days} days"],
                ['Data Completeness:', f"{vrops_data.data_completeness:.1f}%"],
            ])
        
        if sql_config:
            report_data.extend([
                ['SQL Server Edition:', sql_config.current_edition],
                ['Licensing Model:', sql_config.current_licensing_model],
            ])
        
        # Continue with table creation...
        
        return elements
    
    def _create_enhanced_executive_summary(self, config: Dict, recommendation, pricing_data: List[PricingData], vrops_data: VRopsMetrics = None) -> List:
        """Create enhanced executive summary with performance insights"""
        elements = []
        
        elements.append(Paragraph("Executive Summary", self.styles['CustomTitle']))
        
        # Enhanced summary with vROps insights
        summary_text = f"""
        This comprehensive analysis incorporates {len(pricing_data)} AWS instance configurations with 
        performance-based sizing recommendations derived from vRealize Operations data.
        """
        
        if vrops_data and vrops_data.cpu_usage_avg < 50:
            summary_text += f" Performance analysis reveals significant right-sizing opportunities with "
            summary_text += f"average CPU utilization of {vrops_data.cpu_usage_avg:.1f}%."
        
        elements.append(Paragraph(summary_text, self.styles['Normal']))
        
        return elements
    
    def _create_vrops_analysis_section(self, vrops_data: VRopsMetrics) -> List:
        """Create vROps performance analysis section"""
        elements = []
        
        elements.append(Paragraph("vRealize Operations Performance Analysis", self.styles['CustomTitle']))
        
        # Performance metrics table
        vrops_table_data = [
            ['Metric', 'Average', 'Peak', '95th Percentile', 'Analysis'],
            ['CPU Utilization (%)', f"{vrops_data.cpu_usage_avg:.1f}", f"{vrops_data.cpu_usage_peak:.1f}", 
             f"{vrops_data.cpu_usage_95th:.1f}", "Right-sizing opportunity" if vrops_data.cpu_usage_avg < 50 else "Appropriate"],
            ['Memory Utilization (%)', f"{vrops_data.memory_usage_avg:.1f}", f"{vrops_data.memory_usage_peak:.1f}",
             f"{vrops_data.memory_usage_95th:.1f}", "Memory pressure" if vrops_data.memory_balloon_avg > 1 else "Normal"],
            ['Disk Latency (ms)', f"{vrops_data.disk_latency_avg:.1f}", f"{vrops_data.disk_latency_peak:.1f}",
             "N/A", "Storage optimization needed" if vrops_data.disk_latency_avg > 20 else "Acceptable"],
        ]
        
        # Create and style the table...
        
        return elements
    
    def _create_sql_optimization_section(self, sql_config: SQLServerConfig) -> List:
        """Create SQL Server optimization analysis section"""
        elements = []
        
        elements.append(Paragraph("SQL Server Licensing Optimization", self.styles['CustomTitle']))
        
        # SQL configuration summary
        sql_summary = f"""
        Current Configuration: {sql_config.current_edition} edition with {sql_config.current_licensing_model} licensing.
        Licensed for {sql_config.current_cores_licensed} cores serving {sql_config.database_count} databases
        supporting {sql_config.concurrent_users} concurrent users.
        """
        
        elements.append(Paragraph(sql_summary, self.styles['Normal']))
        
        # Optimization recommendations
        if sql_config.has_software_assurance and sql_config.eligible_for_ahb:
            elements.append(Paragraph("Azure Hybrid Benefit Opportunity: Eligible for up to 55% licensing cost reduction", 
                                    self.styles['Normal']))
        
        return elements

# Continue with additional methods for pricing analysis, enhanced to include vROps and SQL data...

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