import streamlit as st
import pandas as pd
from io import BytesIO
import io
from datetime import datetime
import os
import time
from dotenv import load_dotenv

# Import reportlab components for PDF generation
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    st.warning("⚠️ ReportLab not installed. PDF generation will be disabled. Install with: pip install reportlab")

# Import the calculator class
from ec2_sql_sizing import EC2DatabaseSizingCalculator

# Load environment variables from .env only if needed
if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
    load_dotenv()

# Configure page
st.set_page_config(
    page_title="Enterprise AWS EC2 SQL Sizing", 
    layout="wide",
    page_icon="📊"
)

# Add custom CSS for professional styling
st.markdown("""
<style>
    /* General container and font settings */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
        color: #333;
        line-height: 1.6;
    }
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 0.8em 1.5em;
        transition: all 0.2s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    .stDownloadButton>button {
        background-color: #2196F3;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 0.8em 1.5em;
        transition: all 0.2s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stDownloadButton>button:hover {
        background-color: #1976d2;
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }

    /* Alerts and Messages */
    .stAlert {
        border-radius: 8px;
        font-size: 0.95rem;
    }

    /* Metric Boxes */
    .metric-box {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
        background-color: #ffffff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-box:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .metric-title {
        font-weight: bold;
        margin-bottom: 5px;
        color: #2c3e50;
        font-size: 1.1em;
    }
    .metric-value {
        font-size: 1.5em;
        font-weight: 700;
        color: #3f51b5;
        margin-top: 5px;
    }

    /* Expander styling */
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
        margin-bottom: 10px;
    }

    /* Input fields */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #ccc;
        padding: 0.6em;
        font-size: 1em;
    }
    .stSelectbox>div>div>div {
        border-radius: 8px;
        border: 1px solid #ccc;
        padding: 0.4em;
        font-size: 1em;
    }

    /* Dataframes */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8f9fa;
        padding: 8px;
        border-radius: 8px;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 6px;
        padding: 10px 15px;
        border: 1px solid #e0e0e0;
        font-weight: 600;
        color: #555;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3f51b5;
        color: white;
        border-color: #3f51b5;
        box-shadow: 0 2px 8px rgba(63, 81, 181, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# --- PDF Report Generator Class ---
class PDFReportGenerator:
    """Generates PDF reports from analysis results."""

    def __init__(self):
        if not REPORTLAB_AVAILABLE:
            raise ImportError("ReportLab library not found. Please install with: pip install reportlab")
        
        self.styles = getSampleStyleSheet()
        self.styles.add(ParagraphStyle(name='H1_Custom', fontSize=24, leading=28, alignment=TA_CENTER, spaceAfter=20, fontName='Helvetica-Bold'))
        self.styles.add(ParagraphStyle(name='H2_Custom', fontSize=18, leading=22, spaceBefore=10, spaceAfter=10, fontName='Helvetica-Bold'))
        self.styles.add(ParagraphStyle(name='H3_Custom', fontSize=14, leading=18, spaceBefore=8, spaceAfter=8, fontName='Helvetica-Bold'))
        self.styles.add(ParagraphStyle(name='Normal_Custom', fontSize=10, leading=12, spaceAfter=6, alignment=TA_LEFT))
        self.styles.add(ParagraphStyle(name='Bullet_Custom', fontSize=10, leading=12, leftIndent=20, spaceAfter=6, bulletText='•', alignment=TA_LEFT))
        self.styles.add(ParagraphStyle(name='Table_Header', fontSize=10, leading=12, alignment=TA_CENTER, fontName='Helvetica-Bold', textColor=colors.whitesmoke))
        self.styles.add(ParagraphStyle(name='Table_Cell', fontSize=10, leading=12, alignment=TA_CENTER))

    def generate_report(self, all_results: list | dict):
        """Generates a PDF report based on the analysis results."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        story = []

        # Title Page
        story.append(Paragraph("AWS EC2 SQL Server Sizing Report", self.styles['H1_Custom']))
        story.append(Paragraph(f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", self.styles['Normal_Custom']))
        story.append(Spacer(1, 0.4 * inch))
        
        # Add a summary based on whether it's a single or bulk analysis
        if isinstance(all_results, dict): # Single analysis result
            display_results = [all_results]
        else: # Bulk analysis result (list of dicts)
            display_results = all_results

        if not display_results:
            story.append(Paragraph("No analysis results available to generate a report.", self.styles['Normal_Custom']))
            doc.build(story)
            buffer.seek(0)
            return buffer.getvalue()

        # Executive Summary
        story.append(Paragraph("1. Executive Summary", self.styles['H2_Custom']))
        
        summary_data = [
            [
                Paragraph("Database", self.styles['Table_Header']), 
                Paragraph("Region", self.styles['Table_Header']), 
                Paragraph("PROD Instance", self.styles['Table_Header']), 
                Paragraph("PROD vCPUs", self.styles['Table_Header']),
                Paragraph("PROD RAM (GB)", self.styles['Table_Header']), 
                Paragraph("Monthly Cost ($)", self.styles['Table_Header'])
            ]
        ]
        total_monthly_cost_prod = 0
        
        for result in display_results:
            inputs = result.get('inputs', {})
            prod_rec = result['recommendations'].get('PROD', {})
            
            db_name = inputs.get('db_name', 'N/A')
            region = inputs.get('region', 'N/A')
            instance_type = prod_rec.get('instance_type', 'N/A')
            vcpus = prod_rec.get('vCPUs', 'N/A')
            ram_gb = prod_rec.get('RAM_GB', 'N/A')
            monthly_cost = prod_rec.get('total_cost', 0)
            total_monthly_cost_prod += monthly_cost

            summary_data.append([
                Paragraph(str(db_name), self.styles['Table_Cell']),
                Paragraph(str(region), self.styles['Table_Cell']),
                Paragraph(str(instance_type), self.styles['Table_Cell']),
                Paragraph(str(vcpus), self.styles['Table_Cell']),
                Paragraph(str(ram_gb), self.styles['Table_Cell']),
                Paragraph(f"${monthly_cost:,.2f}", self.styles['Table_Cell'])
            ])

        table = Table(summary_data, colWidths=[1.2*inch, 0.8*inch, 1.2*inch, 0.7*inch, 0.9*inch, 1.2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3f51b5')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e0e0e0')),
            ('LEFTPADDING', (0,0), (-1,-1), 6),
            ('RIGHTPADDING', (0,0), (-1,-1), 6),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph(f"Total Monthly Cost (Production): ${total_monthly_cost_prod:,.2f}", self.styles['Normal_Custom']))
        story.append(Paragraph(f"Total Annual Cost (Production): ${total_monthly_cost_prod * 12:,.2f}", self.styles['Normal_Custom']))
        story.append(Spacer(1, 0.2 * inch))

        # Detailed Analysis for Each Database
        for i, result in enumerate(display_results):
            inputs = result.get('inputs', {})
            recommendations = result.get('recommendations', {})
            db_name = inputs.get('db_name', f'Database {i+1}')

            story.append(Paragraph(f"2. Detailed Analysis: {db_name}", self.styles['H2_Custom']))
            story.append(Paragraph("2.1. Current Configuration", self.styles['H3_Custom']))
            story.append(Paragraph(f"• Region: {inputs.get('region', 'N/A')}", self.styles['Bullet_Custom']))
            story.append(Paragraph(f"• CPU Cores (On-Prem): {inputs.get('on_prem_cores', 'N/A')} ({inputs.get('peak_cpu_percent', 'N/A')}% util)", self.styles['Bullet_Custom']))
            story.append(Paragraph(f"• RAM (On-Prem): {inputs.get('on_prem_ram_gb', 'N/A')} GB ({inputs.get('peak_ram_percent', 'N/A')}% util)", self.styles['Bullet_Custom']))
            story.append(Paragraph(f"• Storage (On-Prem): {inputs.get('storage_current_gb', 'N/A'):,} GB", self.styles['Bullet_Custom']))
            story.append(Paragraph(f"• Peak IOPS: {inputs.get('peak_iops', 'N/A'):,}", self.styles['Bullet_Custom']))
            story.append(Paragraph(f"• Peak Throughput: {inputs.get('peak_throughput_mbps', 'N/A')} MB/s", self.styles['Bullet_Custom']))
            story.append(Paragraph(f"• Annual Growth Rate: {inputs.get('storage_growth_rate', 'N/A')*100:.1f}%", self.styles['Bullet_Custom']))
            story.append(Paragraph(f"• Growth Projection Years: {inputs.get('years', 'N/A')}", self.styles['Bullet_Custom']))
            story.append(Spacer(1, 0.1 * inch))

            story.append(Paragraph("2.2. Recommended AWS Configurations", self.styles['H3_Custom']))
            rec_table_data = [
                [
                    Paragraph("Environment", self.styles['Table_Header']), 
                    Paragraph("Instance Type", self.styles['Table_Header']), 
                    Paragraph("vCPUs", self.styles['Table_Header']), 
                    Paragraph("RAM (GB)", self.styles['Table_Header']), 
                    Paragraph("Storage (GB)", self.styles['Table_Header']),
                    Paragraph("EBS Type", self.styles['Table_Header']),
                    Paragraph("Monthly Cost ($)", self.styles['Table_Header'])
                ]
            ]
            for env, rec in recommendations.items():
                rec_table_data.append([
                    Paragraph(str(env), self.styles['Table_Cell']), 
                    Paragraph(str(rec.get('instance_type', 'N/A')), self.styles['Table_Cell']), 
                    Paragraph(str(rec.get('vCPUs', 'N/A')), self.styles['Table_Cell']), 
                    Paragraph(str(rec.get('RAM_GB', 'N/A')), self.styles['Table_Cell']), 
                    Paragraph(f"{rec.get('storage_GB', 'N/A'):,}", self.styles['Table_Cell']),
                    Paragraph(str(rec.get('ebs_type', 'N/A')), self.styles['Table_Cell']),
                    Paragraph(f"${rec.get('total_cost', 0):,.2f}", self.styles['Table_Cell'])
                ])
            
            rec_table = Table(rec_table_data, colWidths=[1*inch, 1.1*inch, 0.7*inch, 0.7*inch, 1*inch, 0.8*inch, 1*inch])
            rec_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#764ba2')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e0e0e0')),
                ('LEFTPADDING', (0,0), (-1,-1), 4),
                ('RIGHTPADDING', (0,0), (-1,-1), 4),
                ('TOPPADDING', (0,0), (-1,-1), 4),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ]))
            story.append(rec_table)
            story.append(Spacer(1, 0.2 * inch))

            story.append(Paragraph("2.3. Key Considerations", self.styles['H3_Custom']))
            considerations = [
                f"Workload Profile: {inputs.get('workload_profile', 'General').title()}",
                f"Preferred Processor: {'AMD' if inputs.get('prefer_amd', False) else 'Intel/Mixed'}",
                "Automated backups and point-in-time recovery are recommended.",
                "EBS encryption at rest should be enabled for all environments.",
                "Implement robust monitoring with Amazon CloudWatch.",
                "Consider Reserved Instances for production environments for cost savings."
            ]
            for c in considerations:
                story.append(Paragraph(f"• {c}", self.styles['Bullet_Custom']))
            story.append(Spacer(1, 0.2 * inch))
        
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

# --- Initialize Session State ---
def initialize_session_state():
    """Initialize all session state variables."""
    if 'calculator' not in st.session_state:
        st.session_state.calculator = EC2DatabaseSizingCalculator()
    if 'pdf_generator' not in st.session_state and REPORTLAB_AVAILABLE:
        try:
            st.session_state.pdf_generator = PDFReportGenerator()
        except Exception as e:
            st.session_state.pdf_generator = None
            st.warning(f"PDF generator initialization failed: {str(e)}")
    if 'last_analysis_results' not in st.session_state:
        st.session_state.last_analysis_results = None

# --- Helper function to parse uploaded EC2 sizing files ---
def parse_uploaded_file_ec2(uploaded_file):
    """
    Parses an uploaded CSV/Excel file containing EC2 database configurations.
    Returns a list of dictionaries (valid_inputs) and a list of error strings.
    """
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Define expected columns and their types/defaults
        expected_columns_info = [
            ('db_name', str, 'Database Name', 'DB_1'),
            ('region', str, 'AWS Region', 'us-east-1'),
            ('on_prem_cores', int, 'On-Prem CPU Cores', 8),
            ('peak_cpu_percent', int, 'Peak CPU Utilization (%)', 65),
            ('on_prem_ram_gb', int, 'On-Prem RAM (GB)', 32),
            ('peak_ram_percent', int, 'Peak RAM Utilization (%)', 75),
            ('storage_current_gb', int, 'Current Storage (GB)', 500),
            ('storage_growth_rate', float, 'Annual Growth Rate (e.g., 0.15)', 0.15),
            ('peak_iops', int, 'Peak IOPS', 5000),
            ('peak_throughput_mbps', int, 'Peak Throughput (MB/s)', 200),
            ('years', int, 'Growth Projection Years', 3),
            ('workload_profile', str, 'Workload Profile (general, memory, compute)', 'general'),
            ('prefer_amd', bool, 'Prefer AMD (True/False)', True)
        ]

        # Create a mapping for user-friendly names to internal keys
        column_mapping_user_to_internal = {info[2]: info[0] for info in expected_columns_info}
        internal_to_user_mapping = {info[0]: info[2] for info in expected_columns_info}
        
        # Rename columns to internal keys for processing
        df_processed = pd.DataFrame()
        found_columns = []
        for user_col_name in df.columns:
            if user_col_name in column_mapping_user_to_internal:
                internal_name = column_mapping_user_to_internal[user_col_name]
                df_processed[internal_name] = df[user_col_name]
                found_columns.append(internal_name)
        
        # Check for mandatory columns
        required_internal_columns = [info[0] for info in expected_columns_info]
        missing_columns = [col for col in required_internal_columns if col not in found_columns]
        if missing_columns:
            missing_user_names = [internal_to_user_mapping[col] for col in missing_columns]
            return [], [f"Missing required columns in the uploaded file: {', '.join(missing_user_names)}"]

        valid_inputs = []
        errors = []
        
        for index, row in df_processed.iterrows():
            row_data = {}
            row_errors = []
            for col_key, col_type, _, default_value in expected_columns_info:
                try:
                    raw_value = row.get(col_key)
                    if pd.isna(raw_value) or raw_value is None:
                        if col_key in ['db_name', 'region', 'on_prem_cores', 'on_prem_ram_gb', 'storage_current_gb']:
                             row_errors.append(f"'{internal_to_user_mapping[col_key]}' cannot be empty.")
                        else:
                            row_data[col_key] = default_value
                    else:
                        if col_type == bool:
                            if isinstance(raw_value, str):
                                row_data[col_key] = raw_value.strip().lower() == 'true'
                            else:
                                row_data[col_key] = bool(raw_value)
                        elif col_type == str:
                            row_data[col_key] = str(raw_value).strip()
                        else:
                            row_data[col_key] = col_type(raw_value)
                except ValueError:
                    row_errors.append(f"Invalid value for '{internal_to_user_mapping[col_key]}': '{raw_value}'. Expected {col_type.__name__}.")
                except TypeError:
                    row_errors.append(f"Invalid data type for '{internal_to_user_mapping[col_key]}': '{raw_value}'. Expected {col_type.__name__}.")
            
            if row_errors:
                errors.append(f"Row {index + 2} (Excel row): " + "; ".join(row_errors))
            else:
                if 'db_name' not in row_data or not row_data['db_name']:
                    row_data['db_name'] = f"Database {index + 1}"
                valid_inputs.append(row_data)
        
        return valid_inputs, errors
        
    except Exception as e:
        return [], [f"File parsing error: {str(e)}. Please ensure it's a valid CSV/Excel file with correct headers and data types."]

# --- Helper function for Excel report export ---
def export_full_report_excel(all_results):
    """
    Exports comprehensive Excel report for EC2 SQL Sizing results.
    """
    try:
        output = io.BytesIO()
        
        if isinstance(all_results, dict):
            results_to_process = [all_results]
        else:
            results_to_process = all_results

        if not results_to_process:
            raise ValueError("No analysis results to export.")

        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Executive Summary sheet
            summary_data = []
            for result in results_to_process:
                inputs = result.get('inputs', {})
                prod_rec = result['recommendations'].get('PROD', {})
                
                summary_data.append({
                    "Database Name": inputs.get('db_name', 'N/A'),
                    "AWS Region": inputs.get('region', 'N/A'),
                    "PROD Instance Type": prod_rec.get('instance_type', 'N/A'),
                    "PROD vCPUs": prod_rec.get('vCPUs', 'N/A'),
                    "PROD RAM (GB)": prod_rec.get('RAM_GB', 'N/A'),
                    "PROD Storage (GB)": prod_rec.get('storage_GB', 'N/A'),
                    "PROD Monthly Cost": prod_rec.get('total_cost', 0),
                    "PROD Annual Cost": prod_rec.get('total_cost', 0) * 12,
                    "Optimization Score (%)": prod_rec.get('optimization_score', 'N/A')
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df['PROD Monthly Cost'] = summary_df['PROD Monthly Cost'].apply(lambda x: f"${x:,.2f}")
            summary_df['PROD Annual Cost'] = summary_df['PROD Annual Cost'].apply(lambda x: f"${x:,.2f}")

            summary_df.to_excel(writer, sheet_name='Executive Summary', index=False)
            
            # Detailed breakdown sheets for each database
            for i, result in enumerate(results_to_process):
                db_name = result['inputs'].get('db_name', f'Database_{i+1}')
                sheet_name = db_name[:31].replace('[', '').replace(']', '').replace(':', '').replace('*', '').replace('?', '').replace('/', '').replace('\\', '')
                
                detail_rows = []
                detail_rows.append({"Category": "--- Input Parameters ---", "Value": ""})
                for k, v in result['inputs'].items():
                    if k == 'storage_growth_rate':
                        detail_rows.append({"Category": k.replace('_', ' ').title(), "Value": f"{v*100:.1f}%"})
                    elif isinstance(v, (int, float)) and k not in ['on_prem_cores', 'on_prem_ram_gb', 'peak_cpu_percent', 'peak_ram_percent', 'peak_iops', 'peak_throughput_mbps', 'years']:
                         detail_rows.append({"Category": k.replace('_', ' ').title(), "Value": f"{v:,.0f}"})
                    else:
                        detail_rows.append({"Category": k.replace('_', ' ').title(), "Value": str(v)})
                
                detail_rows.append({"Category": "--- Sizing Recommendations ---", "Value": ""})
                
                for env, rec in result['recommendations'].items():
                    detail_rows.append({"Category": f"Environment: {env.upper()}", "Value": ""})
                    detail_rows.append({"Category": "  Instance Type", "Value": rec.get('instance_type', 'N/A')})
                    detail_rows.append({"Category": "  vCPUs", "Value": rec.get('vCPUs', 'N/A')})
                    detail_rows.append({"Category": "  RAM (GB)", "Value": rec.get('RAM_GB', 'N/A')})
                    detail_rows.append({"Category": "  Storage (GB)", "Value": f"{rec.get('storage_GB', 'N/A'):,}"})
                    detail_rows.append({"Category": "  EBS Type", "Value": rec.get('ebs_type', 'N/A')})
                    detail_rows.append({"Category": "  IOPS Required", "Value": f"{rec.get('iops_required', 'N/A'):,}"})
                    detail_rows.append({"Category": "  Throughput", "Value": rec.get('throughput_required', 'N/A')})
                    detail_rows.append({"Category": "  Instance Cost (Monthly)", "Value": f"${rec.get('instance_cost', 0):,.2f}"})
                    detail_rows.append({"Category": "  EBS Cost (Monthly)", "Value": f"${rec.get('ebs_cost', 0):,.2f}"})
                    detail_rows.append({"Category": "  Total Monthly Cost", "Value": f"${rec.get('total_cost', 0):,.2f}"})
                    detail_rows.append({"Category": "  Optimization Score", "Value": f"{rec.get('optimization_score', 'N/A')}%"})
                    detail_rows.append({"Category": "", "Value": ""})

                detail_df = pd.DataFrame(detail_rows)
                detail_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        output.seek(0)
        return output.getvalue()
        
    except Exception as e:
        st.error(f"Error generating Excel report: {str(e)}")
        raise

# --- Main App Functions ---

def render_manual_config_tab(calculator_instance: EC2DatabaseSizingCalculator):
    """Renders the manual configuration tab."""
    st.subheader("Manual EC2 Sizing")
    st.markdown("Enter your current on-premise SQL Server metrics for a single database:")

    inputs = calculator_instance.inputs

    with st.expander("Compute Resources", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            inputs["on_prem_cores"] = st.number_input("CPU Cores", min_value=1, value=inputs["on_prem_cores"],
                                                    help="Total number of CPU cores in your on-premise server", key="manual_cores")
            inputs["peak_cpu_percent"] = st.slider("Peak CPU Utilization (%)", 0, 100, inputs["peak_cpu_percent"],
                                                  help="Highest observed CPU utilization percentage", key="manual_cpu_util")
        with col2:
            inputs["on_prem_ram_gb"] = st.number_input("RAM (GB)", min_value=1, value=inputs["on_prem_ram_gb"],
                                                      help="Total physical RAM in the server", key="manual_ram")
            inputs["peak_ram_percent"] = st.slider("Peak RAM Utilization (%)", 0, 100, inputs["peak_ram_percent"],
                                                  help="Highest observed RAM utilization percentage", key="manual_ram_util")

    with st.expander("Storage & I/O", expanded=True):
        inputs["storage_current_gb"] = st.number_input("Current Storage (GB)", min_value=1, value=inputs["storage_current_gb"],
                                                       help="Current database storage size", key="manual_storage")
        inputs["storage_growth_rate"] = st.number_input("Annual Growth Rate", min_value=0.0, max_value=1.0, value=inputs["storage_growth_rate"], step=0.01,
                                                      format="%.2f", help="Expected annual storage growth (e.g., 0.15 for 15%)", key="manual_growth")
        inputs["peak_iops"] = st.number_input("Peak IOPS", min_value=1, value=inputs["peak_iops"],
                                               help="Highest observed Input/Output Operations Per Second", key="manual_iops")
        inputs["peak_throughput_mbps"] = st.number_input("Peak Throughput (MB/s)", min_value=1, value=inputs["peak_throughput_mbps"],
                                                         help="Highest observed data transfer rate", key="manual_throughput")

    with st.expander("Deployment Settings", expanded=True):
        inputs["years"] = st.slider("Growth Projection (Years)", 1, 10, inputs["years"],
                                   help="Number of years to plan for future growth", key="manual_years")
        inputs["workload_profile"] = st.selectbox("Workload Profile",
                                                  ["general", "memory", "compute"], index=["general", "memory", "compute"].index(inputs["workload_profile"]),
                                                  help="General: Balanced workloads, Memory: Data warehouses/analytics, Compute: OLTP/CPU-bound", key="manual_workload")
        inputs["prefer_amd"] = st.checkbox("Include AMD Instances (Cost Optimized)", value=inputs["prefer_amd"],
                                          help="AMD instances are typically 10-20% cheaper than comparable Intel instances", key="manual_amd")

    calculator_instance.inputs.update(inputs)

    if st.button("Generate Recommendations", key="generate_manual_btn"):
        generate_and_display_recommendations(calculator_instance, {"db_name": "Manual Input", **inputs})

def render_bulk_upload_tab(calculator_instance: EC2DatabaseSizingCalculator):
    """Renders the bulk upload tab."""
    st.subheader("Bulk EC2 Sizing from File")
    st.markdown("Upload a CSV or Excel file containing multiple database configurations for batch analysis.")

    uploaded_file = st.file_uploader(
        "Upload CSV/Excel file",
        type=["csv", "xlsx"],
        help="Download the template below for the required format."
    )

    if st.button("⬇️ Download Template (CSV)", key="download_template_btn_bulk"):
        template_data = {
            'Database Name': ['ProdDB_1', 'StagingDB_2', 'DevDB_3'],
            'AWS Region': ['us-east-1', 'us-west-2', 'us-east-1'],
            'On-Prem CPU Cores': [16, 8, 4],
            'Peak CPU Utilization (%)': [70, 60, 45],
            'On-Prem RAM (GB)': [64, 32, 16],
            'Peak RAM Utilization (%)': [80, 70, 55],
            'Current Storage (GB)': [1000, 500, 200],
            'Annual Growth Rate (e.g., 0.15)': [0.15, 0.10, 0.05],
            'Peak IOPS': [10000, 5000, 2000],
            'Peak Throughput (MB/s)': [500, 250, 100],
            'Growth Projection Years': [3, 2, 1],
            'Workload Profile (general, memory, compute)': ['general', 'memory', 'compute'],
            'Prefer AMD (True/False)': [True, False, True]
        }
        template_df = pd.DataFrame(template_data)
        csv_buffer = BytesIO()
        template_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        st.download_button(
            label="Download CSV Template",
            data=csv_buffer.getvalue(),
            file_name="ec2_sizing_template.csv",
            mime="text/csv",
            key="download_csv_template_actual"
        )

    if uploaded_file:
        process_bulk_upload_ec2(uploaded_file, calculator_instance)

def process_bulk_upload_ec2(uploaded_file, calculator_instance: EC2DatabaseSizingCalculator):
    """Processes the uploaded file for bulk EC2 sizing."""
    st.info("Processing uploaded file. This might take a moment for large files...")
    
    valid_inputs, errors = parse_uploaded_file_ec2(uploaded_file)
    
    if errors:
        st.error("❌ **Errors found in uploaded file:**")
        for err in errors:
            st.write(err)
        st.warning("Please correct the errors and re-upload the file.")
        return

    if not valid_inputs:
        st.warning("No valid configurations found in the uploaded file.")
        return

    st.success(f"✅ Successfully parsed {len(valid_inputs)} configurations.")
    st.write("---")
    st.subheader("Preview of Uploaded Data (First 5 Rows)")
    preview_df = pd.DataFrame(valid_inputs)
    st.dataframe(preview_df.head(), use_container_width=True)
    st.write("---")

    if st.button(f"🚀 Analyze {len(valid_inputs)} Databases", key="analyze_bulk_btn"):
        analyze_bulk_workload_ec2(valid_inputs, calculator_instance)

def analyze_bulk_workload_ec2(valid_inputs: list[dict], calculator_instance: EC2DatabaseSizingCalculator):
    """Analyzes multiple database configurations from bulk upload."""
    all_results = []
    progress_text = st.empty()
    progress_bar = st.progress(0)

    try:
        calculator_instance.fetch_current_prices(force_refresh=True)

        for i, inputs in enumerate(valid_inputs):
            db_name = inputs.get('db_name', f"Database {i+1}")
            progress_text.text(f"Analyzing {db_name} ({i+1}/{len(valid_inputs)})...")
            progress_bar.progress((i + 1) / len(valid_inputs))

            calculator_instance.inputs.update(inputs)
            recommendations = calculator_instance.generate_all_recommendations()
            
            all_results.append({
                'inputs': inputs,
                'recommendations': recommendations
            })
            time.sleep(0.1)

        st.session_state.last_analysis_results = all_results
        st.success("✅ Bulk analysis complete!")
        display_bulk_results_ec2(all_results)

    except Exception as e:
        st.error(f"An error occurred during bulk analysis: {e}")
        st.exception(e)
    finally:
        progress_text.empty()
        progress_bar.empty()

def display_bulk_results_ec2(all_results: list[dict]):
    """Displays aggregated results for bulk EC2 sizing analysis."""
    st.subheader("Bulk Analysis Summary")

    summary_data = []
    total_monthly_cost = 0
    total_on_prem_cost_estimate = 0

    for i, result in enumerate(all_results):
        inputs = result['inputs']
        prod_rec = result['recommendations'].get('PROD', {})

        db_name = inputs.get('db_name', f'Database {i+1}')
        region = inputs.get('region', 'N/A')
        instance_type = prod_rec.get('instance_type', 'N/A')
        total_cost = prod_rec.get('total_cost', 0)
        
        on_prem_cores = inputs.get('on_prem_cores', 0)
        on_prem_ram_gb = inputs.get('on_prem_ram_gb', 0)
        estimated_on_prem_monthly_cost = (on_prem_cores * 150) + (on_prem_ram_gb * 10)
        
        total_monthly_cost += total_cost
        total_on_prem_cost_estimate += estimated_on_prem_monthly_cost

        summary_data.append({
            "Database Name": db_name,
            "Region": region,
            "PROD Instance": instance_type,
            "PROD Monthly Cost": total_cost,
            "On-Prem Est. Monthly Cost": estimated_on_prem_monthly_cost,
            "Monthly Savings": estimated_on_prem_monthly_cost - total_cost
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    st.dataframe(
        summary_df.style.format({
            "PROD Monthly Cost": "${:,.2f}",
            "On-Prem Est. Monthly Cost": "${:,.2f}",
            "Monthly Savings": "${:,.2f}"
        }),
        use_container_width=True
    )

    st.write("---")
    st.subheader("Overall Portfolio Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f'<div class="metric-box">'
                    '<div class="metric-title">Total Databases Analyzed</div>'
                    f'<div class="metric-value">{len(all_results)}</div>'
                    '</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-box">'
                    '<div class="metric-title">Total AWS Monthly Cost (PROD)</div>'
                    f'<div class="metric-value">${total_monthly_cost:,.2f}</div>'
                    '</div>', unsafe_allow_html=True)
    with col3:
        total_savings = total_on_prem_cost_estimate - total_monthly_cost
        savings_percentage = (total_savings / total_on_prem_cost_estimate) * 100 if total_on_prem_cost_estimate > 0 else 0
        st.markdown(f'<div class="metric-box">'
                    '<div class="metric-title">Est. Total Monthly Savings</div>'
                    f'<div class="metric-value">${total_savings:,.2f} ({savings_percentage:,.1f}%)</div>'
                    '</div>', unsafe_allow_html=True)
    
    st.write("---")
    st.info("Go to the 'Reports & Export' tab to download detailed reports.")

def generate_and_display_recommendations(calculator_instance: EC2DatabaseSizingCalculator, inputs: dict):
    """Generates and displays recommendations for a single set of inputs."""
    start_time = time.time()
    
    with st.spinner("Calculating EC2 sizing recommendations..."):
        try:
            calculator_instance.inputs.update(inputs) 
            calculator_instance.fetch_current_prices()
            results = calculator_instance.generate_all_recommendations()
            
            st.session_state.last_analysis_results = {'inputs': inputs, 'recommendations': results}

            df = pd.DataFrame.from_dict(results, orient='index').reset_index()
            df.rename(columns={"index": "Environment"}, inplace=True)
            
            # Display input summary
            st.subheader("Input Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="metric-box">'
                            '<div class="metric-title">Compute</div>'
                            f'Cores: {inputs["on_prem_cores"]}<br>'
                            f'Peak CPU: {inputs["peak_cpu_percent"]}%<br>'
                            f'RAM: {inputs["on_prem_ram_gb"]} GB<br>'
                            f'Peak RAM: {inputs["peak_ram_percent"]}%'
                            '</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-box">'
                            '<div class="metric-title">Storage</div>'
                            f'Current: {inputs["storage_current_gb"]} GB<br>'
                            f'Growth: {inputs["storage_growth_rate"]*100:.1f}%<br>'
                            f'Projection: {inputs["years"]} years<br>'
                            f'Peak IOPS: {inputs["peak_iops"]:,}'
                            '</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="metric-box">'
                            '<div class="metric-title">Configuration</div>'
                            f'Region: {inputs["region"]}<br>'
                            f'Workload: {inputs["workload_profile"].title()}<br>'
                            f'AMD Instances: {"Yes" if inputs["prefer_amd"] else "No"}<br>'
                            '</div>', unsafe_allow_html=True)

            # Display results
            st.subheader("Sizing Recommendations")
            st.success(f"✅ EC2 Sizing Recommendations Generated for {inputs['region']}")
            
            # Format costs
            formatted_df = df.copy()
            cost_columns = ["instance_cost", "ebs_cost", "total_cost"]
            for col in cost_columns:
                if col in formatted_df.columns:
                    formatted_df[col] = formatted_df[col].apply(lambda x: f"${x:,.2f}")
            
            # Display table
            st.dataframe(
                formatted_df[[
                    "Environment", "instance_type", "vCPUs", "RAM_GB", 
                    "storage_GB", "ebs_type", "total_cost"
                ]],
                use_container_width=True
            )
            
            # Show detailed view
            with st.expander("Detailed View"):
                detail_columns = [
                    "Environment", "instance_type", "vCPUs", "RAM_GB", 
                    "storage_GB", "ebs_type", "iops_required", 
                    "throughput_required", "family", "processor", 
                    "instance_cost", "ebs_cost", "total_cost"
                ]
                available_columns = [col for col in detail_columns if col in formatted_df.columns]
                st.dataframe(formatted_df[available_columns], use_container_width=True)
            
            # Execution time
            exec_time = time.time() - start_time
            st.caption(f"Execution time: {exec_time:.2f} seconds")
            
            # Cost optimization note
            st.info(f"💲 **Cost Estimates**: Monthly costs for {inputs['region']} region include EC2 instance (Windows) and EBS storage. " 
                    "Actual costs may vary based on usage patterns and discounts.")
            
            if inputs["prefer_amd"]:
                st.info("💡 **Cost Optimization Tip**: AMD-based instances (m6a, r6a, c6a) typically offer 10-20% better price/performance than comparable Intel instances.")

        except Exception as e:
            st.error(f"Error generating recommendations: {str(e)}")
            st.exception(e)

def render_reports_tab():
    """Renders the reports and export tab."""
    st.subheader("Reports & Export Center")
    
    st.markdown("#### Generate Comprehensive Reports")
    
    report_col1, report_col2 = st.columns(2)
    
    with report_col1:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-title">📊 Executive Excel Report</div>
            <p style="font-size:0.9em; color:#555;">Download a comprehensive Excel report summarizing all analysis results, including detailed breakdowns per database and environment.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📈 Download Executive Excel Report", use_container_width=True, key="download_excel_report_btn"):
            if st.session_state.last_analysis_results:
                try:
                    excel_data = export_full_report_excel(st.session_state.last_analysis_results)
                    st.download_button(
                        label="Download Excel Report",
                        data=excel_data,
                        file_name=f"ec2_sizing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        key="actual_download_excel_btn"
                    )
                    st.success("✅ Excel report ready for download!")
                except Exception as e:
                    st.error(f"Failed to generate Excel report: {e}")
            else:
                st.info("💡 Please run a 'Manual Configuration' or 'Bulk Upload' analysis first to generate results.")

    with report_col2:
        st.markdown("""
        <div class="metric-box">
            <div class="metric-title">📄 Executive PDF Report</div>
            <p style="font-size:0.9em; color:#555;">Get a professional PDF summary of your database sizing analysis, perfect for executive presentations.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.last_analysis_results and REPORTLAB_AVAILABLE and st.session_state.get('pdf_generator'):
            try:
                with st.spinner("🔄 Preparing PDF report..."):
                    pdf_data = st.session_state.pdf_generator.generate_report(st.session_state.last_analysis_results)
                
                st.download_button(
                    label="📄 Download Executive PDF Report",
                    data=pdf_data,
                    file_name=f"ec2_sizing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="actual_download_pdf_btn",
                    help="Click to download the comprehensive PDF report"
                )
            except Exception as e:
                st.error(f"❌ PDF generation failed: {str(e)}")
        else:
            if not st.session_state.last_analysis_results:
                st.info("💡 Please run a 'Manual Configuration' or 'Bulk Upload' analysis first to generate results.")
            elif not REPORTLAB_AVAILABLE:
                st.error("❌ PDF generation unavailable: ReportLab not installed")
                st.info("💡 Install ReportLab: pip install reportlab")
            else:
                st.error("❌ PDF generator initialization failed")
    
    st.markdown("---")
    st.markdown("#### Quick Export of Raw Data")
    
    # Raw data export
    if st.session_state.last_analysis_results:
        if isinstance(st.session_state.last_analysis_results, dict):
            results_for_csv = [st.session_state.last_analysis_results]
        else:
            results_for_csv = st.session_state.last_analysis_results

        summary_csv_data = []
        for result in results_for_csv:
            inputs = result.get('inputs', {})
            prod_rec = result['recommendations'].get('PROD', {})
            summary_csv_data.append({
                "Database Name": inputs.get('db_name', 'N/A'),
                "Region": inputs.get('region', 'N/A'),
                "On-Prem Cores": inputs.get('on_prem_cores', 'N/A'),
                "On-Prem RAM (GB)": inputs.get('on_prem_ram_gb', 'N/A'),
                "PROD Instance Type": prod_rec.get('instance_type', 'N/A'),
                "PROD vCPUs": prod_rec.get('vCPUs', 'N/A'),
                "PROD RAM (GB)": prod_rec.get('RAM_GB', 'N/A'),
                "PROD Storage (GB)": prod_rec.get('storage_GB', 'N/A'),
                "PROD EBS Type": prod_rec.get('ebs_type', 'N/A'),
                "PROD Monthly Cost": prod_rec.get('total_cost', 'N/A')
            })
        
        summary_csv_df = pd.DataFrame(summary_csv_data)
        csv_data = summary_csv_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Summary CSV", csv_data, "ec2_sizing_summary.csv", "text/csv")
    else:
        st.info("Run an analysis to enable raw data export.")

# --- Main Application Layout ---
def main():
    # App header
    st.title("AWS EC2 SQL Server Sizing Calculator")
    st.markdown("""
    This enterprise-grade tool provides EC2 sizing recommendations for SQL Server workloads based on your on-premise infrastructure metrics.
    Recommendations include development, QA, staging, and production environments with detailed cost estimates.
    """)

    initialize_session_state()
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("AWS Configuration")
        region = st.selectbox(
            "AWS Region", 
            ["us-east-1", "us-west-1", "us-west-2"],
            index=["us-east-1", "us-west-1", "us-west-2"].index(st.session_state.calculator.inputs.get('region', 'us-east-1')),
            help="Select the AWS region for pricing and deployment",
            key="sidebar_region"
        )
        st.session_state.calculator.inputs['region'] = region

        st.markdown("---")
        st.markdown("""
        **Enterprise Features:**
        - Multi-region cost estimates
        - Environment-specific sizing (DEV, QA, STAGING, PROD)
        - AMD instance optimization for cost savings
        - Storage growth projections
        - I/O requirements calculation
        - Professional reporting (Excel, PDF)
        - Pricing validation and caching
        """)
        st.markdown("---")
        st.info("Input parameters in 'Manual Configuration' tab or via 'Bulk Upload'.")

    # Main Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Manual Configuration", "📁 Bulk Upload", "📋 Reports & Export"])

    with tab1:
        render_manual_config_tab(st.session_state.calculator)
    
    with tab2:
        render_bulk_upload_tab(st.session_state.calculator)

    with tab3:
        render_reports_tab()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; font-size: 0.9em;">
        AWS EC2 SQL Server Sizing Calculator v2.0
        <br>
        Powered by Streamlit, pandas, and ReportLab.
    </div>
    """, unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    main()