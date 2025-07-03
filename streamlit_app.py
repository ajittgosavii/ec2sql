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

# Simplified application class with essential methods
class EnhancedCloudPricingOptimizer:
    """Enhanced main application class with vROps integration and SQL optimization"""
    
    def __init__(self):
        self._initialize_session_state()
        
    def _initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'demo_mode' not in st.session_state:
            st.session_state.demo_mode = True
        if 'vrops_metrics' not in st.session_state:
            st.session_state.vrops_metrics = None
        if 'sql_config' not in st.session_state:
            st.session_state.sql_config = None
        if 'latest_pricing' not in st.session_state:
            st.session_state.latest_pricing = []
        if 'comprehensive_analysis' not in st.session_state:
            st.session_state.comprehensive_analysis = None
    
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
            
            with st.expander("🔍 vROps Data Input"):
                col1, col2 = st.columns(2)
                with col1:
                    cpu_avg = st.number_input("CPU Usage Avg (%)", 0.0, 100.0, 45.0)
                    cpu_peak = st.number_input("CPU Usage Peak (%)", 0.0, 100.0, 75.0)
                    mem_avg = st.number_input("Memory Usage Avg (%)", 0.0, 100.0, 60.0)
                    mem_peak = st.number_input("Memory Usage Peak (%)", 0.0, 100.0, 85.0)
                with col2:
                    cpu_ready = st.number_input("CPU Ready (%)", 0.0, 50.0, 2.0)
                    mem_balloon = st.number_input("Memory Balloon (%)", 0.0, 20.0, 0.0)
                    disk_latency = st.number_input("Disk Latency (ms)", 0.0, 200.0, 15.0)
                    collection_days = st.number_input("Collection Days", 1, 365, 30)
                
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
            
            with st.expander("⚙️ SQL Server Settings"):
                current_edition = st.selectbox("Current SQL Edition", 
                    ["Standard", "Enterprise", "Developer"])
                current_licensing = st.selectbox("Current Licensing Model", 
                    ["Core-based", "CAL-based"])
                current_cores = st.number_input("Licensed Cores", 1, 128, 8)
                concurrent_users = st.number_input("Concurrent Users", 1, 1000, 50)
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
            st.markdown(f"""
            <div class="metric-card">
                <h3>CPU Ready Time</h3>
                <h2>{vrops_metrics.cpu_ready_avg:.1f}%</h2>
                <p>{"⚠️ High" if vrops_metrics.cpu_ready_avg > 5 else "✅ Normal"}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Memory Balloon</h3>
                <h2>{vrops_metrics.memory_balloon_avg:.1f}%</h2>
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
    
    def render_pricing_analysis(self):
        """Render pricing analysis section"""
        st.markdown('<div class="section-header">💰 AWS Pricing Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write("Get comprehensive AWS EC2 pricing with SQL licensing costs included.")
        
        with col2:
            if st.button("🔄 Generate Pricing", type="primary", use_container_width=True):
                with st.spinner("Generating pricing data..."):
                    pricing_data = self.generate_sample_pricing()
                    st.session_state.latest_pricing = pricing_data
                    st.success(f"✅ Generated {len(pricing_data)} pricing options")
        
        # Display pricing results
        if st.session_state.latest_pricing:
            self.display_pricing_results(st.session_state.latest_pricing)
    
    def generate_sample_pricing(self):
        """Generate sample pricing data"""
        instances = [
            ('m5.large', 2, 8, 0.192),
            ('m5.xlarge', 4, 16, 0.384),
            ('m5.2xlarge', 8, 32, 0.768),
            ('r5.large', 2, 16, 0.252),
            ('r5.xlarge', 4, 32, 0.504),
            ('r5.2xlarge', 8, 64, 1.008),
            ('m6a.large', 2, 8, 0.173),
            ('m6a.xlarge', 4, 16, 0.346),
        ]
        
        pricing_data = []
        for instance_type, vcpus, ram, base_price in instances:
            # Add SQL Server licensing cost
            sql_licensing_cost = max(4, vcpus) * (3717 / 12)  # Monthly SQL Standard cost
            infrastructure_monthly = base_price * 730 * 4  # Windows multiplier
            total_monthly = infrastructure_monthly + sql_licensing_cost
            
            pricing_data.append(PricingData(
                service="EC2",
                instance_type=instance_type,
                region=st.session_state.config.get('region', 'us-east-1'),
                price_per_hour=base_price * 4,
                price_per_month=infrastructure_monthly,
                currency="USD",
                last_updated=datetime.now(),
                specifications={'vcpus': vcpus, 'ram': ram, 'family': 'General Purpose'},
                reserved_pricing={
                    "1_year_all_upfront": base_price * 4 * 0.6,
                    "3_year_all_upfront": base_price * 4 * 0.4
                },
                spot_pricing=base_price * 4 * 0.2,
                sql_licensing_cost=sql_licensing_cost,
                total_monthly_cost=total_monthly
            ))
        
        return sorted(pricing_data, key=lambda x: x.total_monthly_cost)
    
    def display_pricing_results(self, pricing_data):
        """Display pricing results"""
        st.markdown("**💸 Pricing Comparison (Infrastructure + SQL Licensing)**")
        
        # Create display data
        df_data = []
        for pricing_obj in pricing_data:
            specs = pricing_obj.specifications or {}
            df_data.append({
                'Instance Type': pricing_obj.instance_type,
                'vCPUs': specs.get('vcpus', 'N/A'),
                'RAM (GB)': specs.get('ram', 'N/A'),
                'Infrastructure (Monthly)': f"${pricing_obj.price_per_month:.2f}",
                'SQL Licensing (Monthly)': f"${pricing_obj.sql_licensing_cost:.2f}",
                'Total Monthly Cost': f"${pricing_obj.total_monthly_cost:.2f}",
                '3-Year RI Savings': f"{((pricing_obj.price_per_month - pricing_obj.reserved_pricing.get('3_year_all_upfront', 0) * 730) / pricing_obj.price_per_month * 100):.0f}%" if pricing_obj.reserved_pricing else "N/A"
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Cost breakdown chart
        fig = go.Figure()
        
        instance_names = [p.instance_type for p in pricing_data[:6]]
        infrastructure_costs = [p.price_per_month for p in pricing_data[:6]]
        sql_costs = [p.sql_licensing_cost for p in pricing_data[:6]]
        
        fig.add_trace(go.Bar(name='Infrastructure', x=instance_names, y=infrastructure_costs))
        fig.add_trace(go.Bar(name='SQL Licensing', x=instance_names, y=sql_costs))
        
        fig.update_layout(
            title="Monthly Cost Breakdown: Infrastructure vs SQL Licensing",
            xaxis_title="Instance Type",
            yaxis_title="Monthly Cost ($)",
            barmode='stack',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_ai_recommendations(self):
        """Render AI recommendations section"""
        st.markdown('<div class="section-header">🤖 AI-Powered Recommendations</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write("Get intelligent recommendations based on your vROps data and SQL configuration.")
        
        with col2:
            if st.button("🧠 Generate Recommendations", type="primary", use_container_width=True):
                with st.spinner("Generating AI recommendations..."):
                    recommendations = self.generate_recommendations()
                    st.session_state.comprehensive_analysis = recommendations
                    st.success("✅ Recommendations generated!")
        
        # Display recommendations
        if st.session_state.comprehensive_analysis:
            self.display_recommendations(st.session_state.comprehensive_analysis)
    
    def generate_recommendations(self):
        """Generate AI recommendations based on configuration"""
        vrops = st.session_state.vrops_metrics
        sql_config = st.session_state.sql_config
        
        # Generate recommendations based on data
        recommendation_text = "Implement comprehensive migration strategy"
        confidence_score = 80
        expected_savings = 50000
        
        if vrops and vrops.cpu_usage_avg < 50:
            recommendation_text += " with CPU right-sizing optimization"
            expected_savings += 20000
        
        if sql_config and sql_config.has_software_assurance:
            recommendation_text += " and Azure Hybrid Benefit activation"
            expected_savings += 25000
        
        recommendation = AIRecommendation(
            recommendation=recommendation_text,
            confidence_score=confidence_score,
            cost_impact="High",
            reasoning="Based on vROps performance data and SQL licensing analysis",
            expected_savings=expected_savings
        )
        
        risks = [
            RiskAssessment(
                category="Performance Risk",
                risk_level="Medium",
                description="Application performance validation needed",
                mitigation_strategy="Comprehensive testing in staging environment",
                impact="Potential performance degradation"
            )
        ]
        
        phases = [
            ImplementationPhase(
                phase="Phase 1: Assessment",
                duration="2-4 weeks",
                activities=["Performance baseline", "Cost analysis", "Risk assessment"],
                dependencies=["Stakeholder approval"],
                deliverables=["Migration plan", "Cost projections"]
            )
        ]
        
        return {
            'recommendation': recommendation,
            'risks': risks,
            'phases': phases
        }
    
    def display_recommendations(self, analysis):
        """Display AI recommendations"""
        recommendation = analysis['recommendation']
        
        st.markdown(f"""
        <div class="ai-recommendation">
            <h3>🎯 Strategic Recommendation</h3>
            <p><strong>Confidence Score:</strong> {recommendation.confidence_score}%</p>
            <p><strong>Expected Annual Savings:</strong> ${recommendation.expected_savings:,.0f}</p>
            <p><strong>Recommendation:</strong> {recommendation.recommendation}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk assessment
        if analysis['risks']:
            st.markdown("**⚠️ Risk Assessment**")
            for risk in analysis['risks']:
                st.markdown(f"""
                <div class="performance-warning">
                    <strong>{risk.risk_level} Risk - {risk.category}:</strong> {risk.description}
                    <br><strong>Mitigation:</strong> {risk.mitigation_strategy}
                </div>
                """, unsafe_allow_html=True)
        
        # Implementation phases
        if analysis['phases']:
            st.markdown("**🗓️ Implementation Timeline**")
            for phase in analysis['phases']:
                with st.expander(f"{phase.phase} ({phase.duration})"):
                    st.markdown("**Activities:**")
                    for activity in phase.activities:
                        st.markdown(f"• {activity}")
    
    def render_cost_comparison(self):
        """Render cost comparison section"""
        st.markdown('<div class="section-header">📈 Cost Comparison & ROI Analysis</div>', unsafe_allow_html=True)
        
        if not st.session_state.latest_pricing:
            st.warning("⚠️ Please generate pricing data first.")
            return
        
        # Get the most cost-effective option
        pricing_data = st.session_state.latest_pricing
        recommended_instance = pricing_data[0] if pricing_data else None
        
        if not recommended_instance:
            return
        
        # 5-year cost scenarios
        scenarios = {
            'On-Demand + SQL': recommended_instance.total_monthly_cost,
            '3-Year RI + SQL': (recommended_instance.reserved_pricing.get('3_year_all_upfront', 0) * 730 + 
                               recommended_instance.sql_licensing_cost),
            'Optimized SQL': recommended_instance.total_monthly_cost * 0.7  # 30% SQL optimization
        }
        
        # Calculate 5-year costs
        st.markdown("**💰 5-Year Total Cost of Ownership**")
        
        col1, col2, col3 = st.columns(3)
        
        for i, (scenario, monthly_cost) in enumerate(scenarios.items()):
            annual_cost = monthly_cost * 12
            five_year_cost = annual_cost * 5
            
            with [col1, col2, col3][i]:
                st.markdown(f"""
                <div class="cost-savings">
                    <h3>{scenario}</h3>
                    <h2>${five_year_cost:,.0f}</h2>
                    <p>5-Year Total</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Savings visualization
        base_cost = scenarios['On-Demand + SQL'] * 60  # 5 years
        optimized_cost = scenarios['Optimized SQL'] * 60
        savings = base_cost - optimized_cost
        
        st.markdown(f"""
        <div class="optimization-card">
            <h3>💡 Optimization Opportunity</h3>
            <h2>${savings:,.0f} Total Savings</h2>
            <p>Over 5 years through RI and SQL optimization</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sql_optimization(self):
        """Render SQL optimization section"""
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
                <p>Core-based licensing</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Concurrent Users</h3>
                <h2>{sql_config.concurrent_users}</h2>
                <p>Active users</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Optimization opportunities
        st.markdown("**💡 Optimization Opportunities**")
        
        opportunities = []
        
        if sql_config.has_software_assurance and sql_config.eligible_for_ahb:
            savings = sql_config.current_annual_license_cost * 0.55
            opportunities.append(f"Azure Hybrid Benefit: Save ${savings:,.0f} annually (55% reduction)")
        
        if sql_config.current_edition == "Enterprise":
            opportunities.append("Consider downgrading to Standard edition if Enterprise features aren't required")
        
        if not opportunities:
            opportunities.append("Current configuration appears optimized")
        
        for opportunity in opportunities:
            st.markdown(f"""
            <div class="sql-optimization">
                <h3>💰 Cost Optimization</h3>
                <p>{opportunity}</p>
            </div>
            """, unsafe_allow_html=True)
    
    def render_reports(self):
        """Render reports section"""
        st.markdown('<div class="section-header">📄 Professional Reports</div>', unsafe_allow_html=True)
        
        st.markdown("""
        Generate comprehensive PDF reports including:
        - Executive Summary
        - vROps Performance Analysis
        - SQL Licensing Optimization
        - Cost Analysis & Recommendations
        - Implementation Roadmap
        """)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("📊 Generate Executive Summary", use_container_width=True):
                st.success("✅ Executive summary would be generated here")
        
        with col2:
            if st.button("📋 Generate Technical Report", use_container_width=True):
                st.success("✅ Technical report would be generated here")
        
        # Export options
        st.markdown("**📊 Data Export Options**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export Pricing (CSV)", use_container_width=True):
                if st.session_state.latest_pricing:
                    csv_data = self.export_pricing_csv()
                    st.download_button(
                        "📥 Download CSV",
                        csv_data,
                        "aws_pricing_analysis.csv",
                        "text/csv"
                    )
        
        with col2:
            if st.button("📊 Export vROps (JSON)", use_container_width=True):
                if st.session_state.vrops_metrics:
                    json_data = json.dumps(asdict(st.session_state.vrops_metrics), indent=2)
                    st.download_button(
                        "📥 Download JSON",
                        json_data,
                        "vrops_metrics.json",
                        "application/json"
                    )
        
        with col3:
            if st.button("📊 Export SQL Config (JSON)", use_container_width=True):
                if st.session_state.sql_config:
                    json_data = json.dumps(asdict(st.session_state.sql_config), indent=2)
                    st.download_button(
                        "📥 Download JSON",
                        json_data,
                        "sql_configuration.json",
                        "application/json"
                    )
    # Add these methods to the EnhancedCloudPricingOptimizer class

    def render_connection_status(self):
        """Render connection status warnings and information"""
        if not self.aws_pricing.connection_status["connected"]:
            error_msg = self.aws_pricing.connection_status.get("error", "Unknown connection error")
            
            st.markdown(f"""
            <div style="background: #fff3cd; padding: 1rem; border-radius: 8px; margin: 1rem 0; border-left: 4px solid #ffc107;">
                <strong>⚠️ AWS Connection Status</strong><br>
                {error_msg}<br>
                <small>Using demo data for demonstration purposes.</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #d4edda; padding: 1rem; border-radius: 8px; margin: 1rem 0; border-left: 4px solid #28a745;">
                <strong>✅ AWS Connected</strong><br>
                Live pricing data available.
            </div>
            """, unsafe_allow_html=True)

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
            st.write("💡 Analyze AWS pricing options based on your workload requirements.")
        with col2:
            if st.button("🔍 Analyze Pricing", type="primary", use_container_width=True):
                with st.spinner("Fetching AWS pricing data..."):
                    self.fetch_and_display_pricing()

    def fetch_and_display_pricing(self):
        """Fetch and display pricing data"""
        config = st.session_state.config
        vrops_data = st.session_state.vrops_metrics
        
        try:
            if st.session_state.demo_mode:
                pricing_data = self.mock_data.get_enhanced_sample_pricing_data(config['region'], vrops_data)
            else:
                # Use real AWS pricing
                pricing_data = []
                instance_types = ['m5.large', 'm5.xlarge', 'm5.2xlarge', 'r5.large', 'r5.xlarge', 'r5.2xlarge']
                
                for instance_type in instance_types:
                    pricing = self.aws_pricing.get_enhanced_ec2_pricing(instance_type, config['region'], vrops_data)
                    if pricing:
                        pricing_data.append(pricing)
            
            if pricing_data:
                st.session_state.pricing_cache[config['region']] = pricing_data
                self.display_pricing_results(pricing_data)
                st.success("✅ Pricing analysis complete!")
            else:
                st.error("❌ No pricing data available.")
                
        except Exception as e:
            st.error(f"❌ Error fetching pricing: {str(e)}")

    def display_pricing_results(self, pricing_data: List):
        """Display pricing analysis results"""
        st.markdown("**💰 Pricing Analysis Results**")
        
        # Create pricing comparison table
        table_data = []
        for pricing in pricing_data[:10]:  # Show top 10
            specs = pricing.specifications or {}
            table_data.append({
                'Instance Type': pricing.instance_type,
                'vCPUs': specs.get('vcpus', 'N/A'),
                'RAM (GB)': specs.get('ram', 'N/A'),
                'Infrastructure ($/month)': f"${pricing.price_per_month:,.0f}",
                'SQL Licensing ($/month)': f"${pricing.sql_licensing_cost:,.0f}",
                'Total Cost ($/month)': f"${pricing.total_monthly_cost:,.0f}",
                'Family': specs.get('family', 'Unknown')
            })
        
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)
        
        # Cost visualization
        self.render_pricing_chart(pricing_data[:8])

    def render_pricing_chart(self, pricing_data: List):
        """Render pricing comparison chart"""
        # Create stacked bar chart for infrastructure vs SQL costs
        instance_types = [p.instance_type for p in pricing_data]
        infrastructure_costs = [p.price_per_month for p in pricing_data]
        sql_costs = [p.sql_licensing_cost for p in pricing_data]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Infrastructure',
            x=instance_types,
            y=infrastructure_costs,
            marker_color='#4a90e2'
        ))
        
        fig.add_trace(go.Bar(
            name='SQL Licensing',
            x=instance_types,
            y=sql_costs,
            marker_color='#20c997'
        ))
        
        fig.update_layout(
            title='Monthly Cost Breakdown: Infrastructure vs SQL Licensing',
            xaxis_title='Instance Type',
            yaxis_title='Monthly Cost ($)',
            barmode='stack',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def render_ai_recommendations(self):
        """Render AI recommendations section"""
        st.markdown('<div class="section-header">🤖 AI-Powered Recommendations</div>', unsafe_allow_html=True)
        
        if not hasattr(st.session_state, 'config') or not hasattr(st.session_state, 'pricing_cache'):
            st.info("⚠️ Please complete pricing analysis first to get AI recommendations.")
            return
        
        # AI Analysis Button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("🧠 Get intelligent migration recommendations based on your data.")
        with col2:
            if st.button("🚀 Get AI Analysis", type="primary", use_container_width=True):
                with st.spinner("Generating AI recommendations..."):
                    self.generate_ai_recommendations()

    def generate_ai_recommendations(self):
        """Generate and display AI recommendations"""
        config = st.session_state.config
        pricing_data = st.session_state.pricing_cache.get(config['region'], [])
        vrops_data = st.session_state.vrops_metrics
        sql_config = st.session_state.sql_config
        
        try:
            # Get AI analysis
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

    def display_ai_recommendations(self):
        """Display AI recommendations"""
        if not st.session_state.comprehensive_analysis:
            return
        
        analysis = st.session_state.comprehensive_analysis
        recommendation = analysis['recommendation']
        
        # Main AI recommendation
        st.markdown(f"""
        <div class="ai-recommendation">
            <h3>🤖 AI Recommendation</h3>
            <p><strong>Confidence Score:</strong> {recommendation.confidence_score:.0f}%</p>
            <p><strong>Expected Savings:</strong> ${recommendation.expected_savings:,.0f}</p>
            <p><strong>Cost Impact:</strong> {recommendation.cost_impact}</p>
            <p>{recommendation.recommendation}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # vROps insights
        if analysis.get('vrops_insights'):
            st.markdown("**📊 vROps Performance Insights**")
            for insight in analysis['vrops_insights']:
                st.markdown(f"""
                <div class="optimization-insight">
                    <strong>💡</strong> {insight}
                </div>
                """, unsafe_allow_html=True)
        
        # SQL optimization insights
        if analysis.get('sql_optimization'):
            st.markdown("**🗃️ SQL Optimization Opportunities**")
            for optimization in analysis['sql_optimization']:
                st.markdown(f"""
                <div class="optimization-insight">
                    <strong>💰</strong> {optimization}
                </div>
                """, unsafe_allow_html=True)

    def render_cost_comparison(self):
        """Render cost comparison section"""
        st.markdown('<div class="section-header">📈 Cost Comparison Analysis</div>', unsafe_allow_html=True)
        
        if not hasattr(st.session_state, 'pricing_cache') or not st.session_state.pricing_cache:
            st.info("⚠️ Please complete pricing analysis first.")
            return
        
        config = st.session_state.config
        pricing_data = st.session_state.pricing_cache.get(config['region'], [])
        
        if not pricing_data:
            st.warning("No pricing data available for comparison.")
            return
        
        # Cost comparison options
        st.markdown("**💰 Cost Scenarios Comparison**")
        
        # Select instances for comparison
        selected_instances = st.multiselect(
            "Select instances to compare:",
            [p.instance_type for p in pricing_data],
            default=[p.instance_type for p in pricing_data[:3]]
        )
        
        if selected_instances:
            self.render_cost_comparison_chart(pricing_data, selected_instances)

    def render_cost_comparison_chart(self, pricing_data: List, selected_instances: List[str]):
        """Render detailed cost comparison chart"""
        # Filter data for selected instances
        filtered_data = [p for p in pricing_data if p.instance_type in selected_instances]
        
        # Create comparison chart with different pricing models
        instance_types = [p.instance_type for p in filtered_data]
        on_demand_costs = [p.total_monthly_cost for p in filtered_data]
        
        # Calculate RI costs (1-year all upfront)
        ri_1yr_costs = []
        spot_costs = []
        
        for p in filtered_data:
            if p.reserved_pricing:
                ri_cost = (p.reserved_pricing.get('1_year_all_upfront', 0) * 730) + p.sql_licensing_cost
                ri_1yr_costs.append(ri_cost)
            else:
                ri_1yr_costs.append(p.total_monthly_cost * 0.6)  # Fallback estimate
            
            spot_cost = (p.spot_pricing * 730) + p.sql_licensing_cost if p.spot_pricing else p.total_monthly_cost * 0.3
            spot_costs.append(spot_cost)
        
        # Create grouped bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='On-Demand',
            x=instance_types,
            y=on_demand_costs,
            marker_color='#dc3545'
        ))
        
        fig.add_trace(go.Bar(
            name='Reserved Instance (1yr)',
            x=instance_types,
            y=ri_1yr_costs,
            marker_color='#28a745'
        ))
        
        fig.add_trace(go.Bar(
            name='Spot Instance',
            x=instance_types,
            y=spot_costs,
            marker_color='#ffc107'
        ))
        
        fig.update_layout(
            title='Total Monthly Cost Comparison (Infrastructure + SQL Licensing)',
            xaxis_title='Instance Type',
            yaxis_title='Total Monthly Cost ($)',
            barmode='group',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Savings calculation
        if len(filtered_data) > 0:
            cheapest = min(ri_1yr_costs)
            most_expensive = max(on_demand_costs)
            potential_savings = most_expensive - cheapest
            
            st.markdown(f"""
            <div class="cost-savings">
                <h3>Potential Monthly Savings</h3>
                <h2>${potential_savings:,.0f}</h2>
                <p>By choosing optimal pricing model</p>
            </div>
            """, unsafe_allow_html=True)

    def render_risk_assessment(self):
        """Render risk assessment section"""
        st.markdown('<div class="section-header">⚠️ Migration Risk Assessment</div>', unsafe_allow_html=True)
        
        if not hasattr(st.session_state, 'comprehensive_analysis') or not st.session_state.comprehensive_analysis:
            st.info("⚠️ Please complete AI analysis first to see risk assessment.")
            return
        
        analysis = st.session_state.comprehensive_analysis
        risks = analysis.get('risks', [])
        
        if not risks:
            st.success("✅ No significant risks identified in the analysis.")
            return
        
        st.markdown("**⚠️ Identified Risks and Mitigation Strategies**")
        
        for i, risk in enumerate(risks, 1):
            risk_color = {
                'Low': '#28a745',
                'Medium': '#ffc107', 
                'High': '#dc3545'
            }.get(risk.risk_level, '#6c757d')
            
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: {risk_color};">
                <h3>{risk.category}</h3>
                <p><strong>Risk Level:</strong> <span style="color: {risk_color}; font-weight: bold;">{risk.risk_level}</span></p>
                <p><strong>Description:</strong> {risk.description}</p>
                <p><strong>Impact:</strong> {risk.impact}</p>
                <p><strong>Mitigation:</strong> {risk.mitigation_strategy}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Implementation phases
        phases = analysis.get('phases', [])
        if phases:
            st.markdown("**📋 Implementation Phases**")
            
            for phase in phases:
                st.markdown(f"""
                <div class="optimization-insight">
                    <h4>{phase.phase}</h4>
                    <p><strong>Duration:</strong> {phase.duration}</p>
                    <p><strong>Key Activities:</strong> {', '.join(phase.activities)}</p>
                    <p><strong>Dependencies:</strong> {', '.join(phase.dependencies)}</p>
                    <p><strong>Deliverables:</strong> {', '.join(phase.deliverables)}</p>
                </div>
                """, unsafe_allow_html=True)

    def render_professional_reports(self):
        """Render professional reports section"""
        st.markdown('<div class="section-header">📄 Professional Reports</div>', unsafe_allow_html=True)
        
        if not hasattr(st.session_state, 'comprehensive_analysis') or not st.session_state.comprehensive_analysis:
            st.info("⚠️ Please complete the analysis first to generate reports.")
            return
        
        st.markdown("**📊 Generate comprehensive PDF reports for stakeholders**")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("Generate professional PDF reports including executive summary, technical analysis, and implementation roadmap.")
        
        with col2:
            if st.button("📄 Generate PDF Report", type="primary", use_container_width=True):
                self.generate_pdf_report()

    def generate_pdf_report(self):
        """Generate and offer PDF report for download"""
        try:
            with st.spinner("Generating professional PDF report..."):
                config = st.session_state.config
                pricing_data = st.session_state.pricing_cache.get(config['region'], [])
                analysis = st.session_state.comprehensive_analysis
                
                # Generate PDF
                pdf_buffer = self.pdf_generator.create_comprehensive_report(
                    config,
                    pricing_data,
                    analysis['recommendation'],
                    analysis.get('risks', []),
                    analysis.get('phases', []),
                    st.session_state.vrops_metrics,
                    st.session_state.sql_config
                )
                
                # Offer download
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_buffer.getvalue(),
                    file_name=f"aws_optimization_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                
                st.success("✅ PDF report generated successfully!")
                
        except Exception as e:
            st.error(f"❌ Error generating PDF report: {str(e)}")
            st.info("💡 PDF generation requires additional dependencies. Report data is available in the interface.")
    
    def export_pricing_csv(self):
        """Export pricing data as CSV"""
        if not st.session_state.latest_pricing:
            return ""
        
        data = []
        for pricing_obj in st.session_state.latest_pricing:
            specs = pricing_obj.specifications or {}
            data.append({
                'Instance Type': pricing_obj.instance_type,
                'vCPUs': specs.get('vcpus', 'N/A'),
                'RAM (GB)': specs.get('ram', 'N/A'),
                'Infrastructure Monthly': pricing_obj.price_per_month,
                'SQL Licensing Monthly': pricing_obj.sql_licensing_cost,
                'Total Monthly': pricing_obj.total_monthly_cost,
                'Region': pricing_obj.region,
                'Last Updated': pricing_obj.last_updated.strftime('%Y-%m-%d %H:%M:%S')
            })
        
        df = pd.DataFrame(data)
        return df.to_csv(index=False)

def main():
    """Main application entry point"""
    try:
        optimizer = EnhancedCloudPricingOptimizer()
        optimizer.render_main_interface()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #6c757d; font-size: 0.9rem;">
            <strong>Enhanced AWS Optimizer v3.0</strong> - Professional AWS Migration Analysis with vROps & SQL Optimization
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        logger.error(f"Application error: {e}")
        
        if st.checkbox("🔍 Show detailed error information"):
            st.exception(e)

if __name__ == "__main__":
    main()