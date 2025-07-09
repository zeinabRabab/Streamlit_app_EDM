
import streamlit as st
from PIL import Image
import os

os.system("pip install plotly")

import shutil
import json
from urllib.request import urlopen
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Install required packages
os.system('pip install ultralytics')
from ultralytics import YOLO

# Model setup
model_url = "https://huggingface.co/spaces/Zeinab22/CracksSegmentation/resolve/main/best%20(2).pt"
local_model_path = "best.pt"
try:
    if not os.path.exists(local_model_path):
        with urlopen(model_url) as response, open(local_model_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        print("✅ Model downloaded successfully.")
    model = YOLO(local_model_path)
except Exception as e:
    print("❌ Model loading failed:", e)
    model = None

def segment_uploaded_image(image, conf_threshold):
    if model is None:
        return None, json.dumps({"error": "Model not loaded"}, indent=4)

    image = image.convert("RGB")
    results = model(image, conf=conf_threshold, task='segment')[0]

    import cv2
    annotated = results.plot()
    annotated = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

    preds = []
    if results.masks is not None and len(results.masks.data) > 0:
        for i, (cls_id, conf, polygon) in enumerate(zip(results.boxes.cls, results.boxes.conf, results.masks.xy)):
            label = results.names[int(cls_id.item())]
            coords = [(round(x, 2), round(y, 2)) for x, y in polygon.tolist()]
            preds.append({
                "mask_id": i,
                "confidence": round(float(conf), 3),
                "class": label,
                "polygon_coordinates": coords
            })
    else:
        preds = [{"note": "No masks detected"}]

    return annotated, json.dumps({"predictions": preds}, indent=4)

# Enhanced page configuration
st.set_page_config(
    page_title="🚧 Smart Road: AI-Powered Crack Detection",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(45deg, #ff9a00 0%, #ff3c00 50%, #ffd200 100%);;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(45deg, #ffc266 0%, #ffaa33 50%, #ff9933 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        /* Force equal height for all cards */
        min-height: 170px; /* Adjust this value based on actual content */
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #ffc266 0%, #ffaa33 50%, #ff9933 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .sidebar-content {
        background: linear-gradient(45deg, #ffc266 0%, #ffaa33 50%, #ff9933 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced sidebar
with st.sidebar:
    st.markdown("""
    <div class="sidebar-content">
        <h1 style="text-align: center; margin-bottom: 1rem;">🛣️ Smart Road</h1>
        <p style="text-align: center; font-size: 1.1rem;">AI-Powered Crack Detection & Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('---')
    
    # Enhanced radio buttons with emojis
    radio = st.radio(
        '🎯 Select Module:',
        options=(
            '🔍 Crack Detection',
            '📊 Prediction Analytics', 
            '📈 Model Performance',
            '🎛️ Advanced Settings'
        ),
        key="main_radio"
    )

# Initialize session state
if "predictions" not in st.session_state:
    st.session_state["predictions"] = []
if "training_data" not in st.session_state:
    st.session_state["training_data"] = None

# Main content based on selection
if radio == "🔍 Crack Detection":
    st.markdown('<h1 class="main-header">🔍 Automated Early Crack Detection System</h1>', unsafe_allow_html=True)
    
    # Enhanced metrics display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Model Status</h3>
            <p style="text-align: center; font-weight: bold;">Ready</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🔧 Version</h3>
            <p style="text-align: center; font-weight: bold;">YOLOv11</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Classes</h3>
            <p style="text-align: center; font-weight: bold;">6 Types</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>🎪 Predictions</h3>
            <p style="text-align: center; font-weight:bold;">{}</p>
        </div>
        """.format(len(st.session_state.predictions)), unsafe_allow_html=True)
    
    st.markdown('---')
    
    # Enhanced confidence slider
    st.markdown('<h4>🎚️ Detection Sensitivity</h4>', unsafe_allow_html=True)
    conf_slider = st.slider(
        "Confidence Threshold (Higher = More Precise)",
        0.0, 1.0, value=0.3, step=0.01,
        help="Lower values detect more cracks but may include false positives"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            st.subheader("📤 Upload Road Image")
            uploaded_image = st.file_uploader(
                'Select an image to analyze',
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=False,
                help="Supported formats: PNG, JPG, JPEG"
            )
            
            if uploaded_image is not None:
                st.image(uploaded_image, use_column_width=True, caption="Original Image")
                
                # Display image info
                img = Image.open(uploaded_image)
                st.info(f"📐 Size: {img.size[0]}x{img.size[1]} pixels | 📁 {uploaded_image.size/1024:.1f} KB")
    
    # Enhanced run button
    run_button = st.button("🚀 Analyze Image for Cracks", use_container_width=True)
    
    with col2:
        with st.container(border=True):
            st.subheader("🎯 Detection Results")
            
            if run_button and uploaded_image:
                with st.spinner("🔍 Analyzing image for cracks..."):
                    image = Image.open(uploaded_image)
                    segmented_image, prediction_json = segment_uploaded_image(image, conf_slider)
                    
                    if segmented_image:
                        st.image(segmented_image, caption="Detected Cracks", use_column_width=True)
                        
                        # Parse predictions to show summary
                        pred_data = json.loads(prediction_json)
                        if "predictions" in pred_data and pred_data["predictions"]:
                            num_cracks = len([p for p in pred_data["predictions"] if "confidence" in p])
                            st.success(f"✅ Found {num_cracks} crack(s) in the image!")
                            
                            # Show confidence levels
                            if num_cracks > 0:
                                confidences = [p["confidence"] for p in pred_data["predictions"] if "confidence" in p]
                                avg_conf = np.mean(confidences)
                                st.info(f"Average confidence: {avg_conf:.3f}")
                        else:
                            st.info("No cracks detected in this image.")
                        
                        # Save prediction
                        st.session_state.predictions.append({
                            "file": uploaded_image.name,
                            "json": prediction_json,
                            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        
                    else:
                        st.error("❌ Analysis failed. Please try again.")
            
            elif uploaded_image is None:
                st.info("🖼️ Upload an image to see detection results here.")

elif radio == "📊 Prediction Analytics":
    st.markdown('<h1 class="main-header">📊 Prediction Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    if st.session_state.predictions:
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        total_predictions = len(st.session_state.predictions)
        total_cracks = 0
        all_confidences = []
        
        for pred in st.session_state.predictions:
            pred_data = json.loads(pred["json"])
            if "predictions" in pred_data:
                cracks = [p for p in pred_data["predictions"] if "confidence" in p]
                total_cracks += len(cracks)
                all_confidences.extend([p["confidence"] for p in cracks])
        
        with col1:
            st.metric("🖼️ Images Analyzed", total_predictions)
        
        with col2:
            st.metric("🔍 Total Cracks Found", total_cracks)
        
        with col3:
            avg_conf = np.mean(all_confidences) if all_confidences else 0
            st.metric("🎯 Average Confidence", f"{avg_conf:.3f}")
        
        st.markdown('---')
        
        # Detailed predictions
        for i, pred in enumerate(st.session_state.predictions):
            with st.expander(f"📁 {pred['file']} - {pred.get('timestamp', 'Unknown time')}"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    pred_data = json.loads(pred["json"])
                    if "predictions" in pred_data:
                        cracks = [p for p in pred_data["predictions"] if "confidence" in p]
                        if cracks:
                            st.write(f"**Cracks detected:** {len(cracks)}")
                            for j, crack in enumerate(cracks):
                                st.write(f"- Crack {j+1}: {crack['confidence']:.3f} confidence")
                                st.write(f"  Class: {crack.get('class', 'Unknown')}")
                        else:
                            st.write("No cracks detected")
                
                with col2:
                    st.code(pred["json"], language="json")
        
        # Clear predictions button
        if st.button("🗑️ Clear All Predictions"):
            st.session_state.predictions = []
            st.rerun()
    
    else:
        st.info("ℹ️ No predictions found. Please analyze some images first in the Crack Detection module.")

elif radio == "📈 Model Performance":
    st.markdown('<h1 class="main-header">📈 Model Performance Dashboard</h1>', unsafe_allow_html=True)
    
    # File upload for training data
    st.subheader("📁 Training Data Analysis")
    uploaded_csv = st.file_uploader(
        "Upload your training log CSV file",
        type="csv",
        help="Upload the CSV file containing your model's training metrics"
    )
    
    if uploaded_csv is not None:
        try:
            df = pd.read_csv(uploaded_csv)
            st.session_state.training_data = df
            st.success("✅ Training data loaded successfully!")
            
            # Display basic info
            st.info(f"📊 Loaded {len(df)} training epochs")
            
        except Exception as e:
            st.error(f"❌ Error loading CSV: {e}")
    
    # Use loaded data or sample data
    if st.session_state.training_data is not None:
        df = st.session_state.training_data
    else:

        # df = pd.DataFrame({
        #     'epoch': list(range(1, 16)),
        #     'train/box_loss': [2.71777, 2.65947, 2.58678, 2.56408, 2.4518, 2.50178, 2.40292, 2.32179, 2.28197, 2.20664, 2.15811, 2.10343, 2.07932, 2.02662, 1.96249],
        #     'train/seg_loss': [3.03984, 2.71963, 2.67745, 2.73627, 2.6611, 2.18039, 2.15222, 2.10787, 2.06938, 2.02478, 2.02375, 1.99399, 1.97206, 1.9389, 1.92227],
        #     'train/cls_loss': [4.43229, 3.74369, 3.74961, 3.66991, 3.59897, 3.45587, 3.30206, 3.16778, 3.07025, 2.95183, 2.88948, 2.78278, 2.70469, 2.62929, 2.5796],
        #     'metrics/precision(B)': [0.03123, 0.36522, 0.3875, 0.27577, 0.5905, 0.40557, 0.15023, 0.14692, 0.22848, 0.19467, 0.19318, 0.20483, 0.17087, 0.2122, 0.1789],
        #     'metrics/recall(B)': [0.11078, 0.04783, 0.07778, 0.09575, 0.08194, 0.12528, 0.22913, 0.2176, 0.28313, 0.27418, 0.29682, 0.32069, 0.35704, 0.33741, 0.36242],
        #     'metrics/mAP50(B)': [0.01722, 0.01719, 0.031, 0.06411, 0.05853, 0.07574, 0.09097, 0.09742, 0.16462, 0.12564, 0.17055, 0.14382, 0.16736, 0.16752, 0.16394],
        #     'metrics/mAP50-95(B)': [0.00571, 0.00558, 0.01022, 0.01866, 0.02253, 0.02679, 0.03608, 0.03496, 0.06484, 0.05297, 0.07089, 0.06182, 0.07291, 0.07544, 0.07454],
        #     'metrics/precision(M)': [0.00284, 0.35797, 0.37392, 0.2364, 0.56468, 0.73932, 0.5685, 0.56049, 0.10827, 0.09764, 0.09143, 0.08497, 0.11506, 0.12854, 0.13129],
        #     'metrics/mAP50(M)': [0.00561, 0.02611, 0.02323, 0.04001, 0.03813, 0.04281, 0.06313, 0.05237, 0.09941, 0.11832, 0.11818, 0.12081, 0.12901, 0.14713, 0.14181]
        # })

        df= pd.read_excel(r'C:\Users\User\Documents\vscode_projects\Salem_grp_company\streamlit_dev\results.xlsx')
    
    if df is not None:
        # Performance metrics overview
        col1, col2, col3, col4 = st.columns(4)
        
        # Find columns dynamically
        map50_m_col = None
        precision_b_col = None
        recall_b_col = None
        precision_m_col=None
        recall_m_col=None
        
        for col in df.columns:
            if 'mAP50' in col and 'M' in col and '95' not in col:
                map50_m_col = col
            elif 'precision' in col and 'M' in col:
                precision_m_col = col
            elif 'recall' in col and 'M' in col:
                recall_m_col = col
            elif 'precision' in col and 'B' in col:
                precision_b_col= col
            elif 'recall' in col and 'B' in col:
                recall_b_col=col
        
        with col1:
            if map50_m_col:
                final_map = df[map50_m_col].iloc[-1]
                initial_map = df[map50_m_col].iloc[0]
                st.metric("🎯 Final mAP50 (Masks)", f"{final_map:.3f}", f"{final_map - initial_map:+.3f}")
        
        with col2:
            if precision_m_col:
                final_prec = df[precision_m_col].iloc[-1]
                initial_prec = df[precision_m_col].iloc[0]
                st.metric("🎯 Final Precision (masks)", f"{final_prec:.3f}", f"{final_prec - initial_prec:+.3f}")
        
        with col3:
            if recall_m_col:
                final_recall = df[recall_m_col].iloc[-1]
                initial_recall = df[recall_m_col].iloc[0]
                st.metric("🎯 Final Recall (masks)", f"{final_recall:.3f}", f"{final_recall - initial_recall:+.3f}")
        
        with col4:
            st.metric("📊 Training Epochs", len(df), "Complete")
        
        st.markdown('---')
        
        # Interactive plotting
        tab1, tab2, tab3 = st.tabs(["📉 Loss Metrics", "🎯 Precision & Recall", "🏆 mAP Analysis"])
        
        with tab1:
            # Loss metrics plot
            fig = go.Figure()
            
            loss_cols = [col for col in df.columns if 'loss' in col.lower()]
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            
            for i, col in enumerate(loss_cols):
                fig.add_trace(go.Scatter(
                    x=df['epoch'] if 'epoch' in df.columns else range(len(df)),
                    y=df[col],
                    mode='lines+markers',
                    name=col.replace('train/', '').replace('_', ' ').title(),
                    line=dict(color=colors[i % len(colors)])
                ))
            
            fig.update_layout(
                title="Training Loss Progression",
                xaxis_title="Epoch",
                yaxis_title="Loss Value",
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("ℹ️ Understanding Loss Metrics"):
                st.markdown("""
                - **Box Loss**: Measures bounding box coordinate prediction accuracy
                - **Segmentation Loss**: Measures pixel-level segmentation accuracy  
                - **Classification Loss**: Measures object class prediction accuracy
                - **Lower values indicate better performance**
                """)
        
        with tab2:
            # Precision & Recall plot
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Bounding box evaluation', 'Masks Evaluation')
            )
            
            
            # Bboxes metrics
            if precision_b_col and recall_b_col:
                fig.add_trace(go.Scatter(
                    x=df['epoch'] if 'epoch' in df.columns else range(len(df)),
                    y=df[precision_b_col],
                    mode='lines+markers',
                    name='Precision (B)',
                    line=dict(color='orange')
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=df['epoch'] if 'epoch' in df.columns else range(len(df)),
                    y=df[recall_b_col],
                    mode='lines+markers',
                    name='Recall (B)',
                    line=dict(color='purple')
                ), row=1, col=1)
            
            # Masks metrics
            # precision_m_col = None
            # for col in df.columns:
            #     if 'precision' in col and 'M' in col:
            #         precision_m_col = col
            #         break
            # recall_m_col = None
            # for col in df.columns:
            #   if 'recall' in col and 'M' in col:
            #     recall_m_col = col
            #     break

            if precision_m_col:
                fig.add_trace(go.Scatter(
                    x=df['epoch'] if 'epoch' in df.columns else range(len(df)),
                    y=df[precision_m_col],
                    mode='lines+markers',
                    name='Precision (M)',
                    line=dict(color='darkgreen')
                ), row=1, col=2)
                fig.add_trace(go.Scatter(
                    x=df['epoch'] if 'epoch' in df.columns else range(len(df)),
                    y=df[recall_m_col],
                    mode='lines+markers',
                    name='Recall (M)',
                    line=dict(color='blue')
                ), row=1, col=2)
            
            fig.update_layout(
                title="Precision & Recall Analysis",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # mAP analysis
            fig = go.Figure()
            
            map_cols = [col for col in df.columns if 'mAP' in col]
            colors = ['red', 'blue', 'green', 'orange']
            
            for i, col in enumerate(map_cols):
                fig.add_trace(go.Scatter(
                    x=df['epoch'] if 'epoch' in df.columns else range(len(df)),
                    y=df[col],
                    mode='lines+markers',
                    name=col.replace('metrics/', ''),
                    line=dict(color=colors[i % len(colors)])
                ))
            
            fig.update_layout(
                title="mAP (Mean Average Precision) Analysis",
                xaxis_title="Epoch",
                yaxis_title="mAP Score",
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("ℹ️ Understanding mAP"):
                st.markdown("""
                - **mAP50**: Mean Average Precision at IoU=0.5 (more lenient)
                - **mAP50-95**: Mean Average Precision across IoU 0.5-0.95 (stricter)
                - **B (Bounding Boxes)**: Object vs background detection
                - **M (Segmentation masks)**: Your 6 crack classes classification
                - **Higher values = Better performance**
                """)
        
       
        
     
elif radio == "🎛️ Advanced Settings":
    st.markdown('<h1 class="main-header">🎛️ Advanced Settings & Information</h1>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs([ "📋 System Info", "❓ Help & FAQ"])
    
   
    
    with tab1:
        st.subheader("System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🤖 Model Information:**")
            st.write("• Model: YOLOv11 Segmentation")
            st.write("• Classes: 6 crack types")
            st.write("• Task: Instance Segmentation")
            st.write("• Input: RGB Images")
            
            st.markdown("**📈 Performance Metrics:**")
            st.write("• mAP50: Mean Average Precision at IoU=0.5")
            st.write("• mAP50-95: Mean Average Precision (IoU 0.5-0.95)")
            st.write("• Precision: True Positives / (True Positives + False Positives)")
            st.write("• Recall: True Positives / (True Positives + False Negatives)")
        
        with col2:
            st.markdown("**🛠️ Technical Details:**")
            st.write("• Framework: Ultralytics YOLOv11")
            st.write("• Backend: PyTorch")
            st.write("• Interface: Streamlit")
            st.write("• Deployment: Cloud Ready")
            
            st.markdown("**📊 Data Processing:**")
            st.write("• Image preprocessing: Automatic")
            st.write("• Output format: JSON + Annotations")
            st.write("• Supported formats: PNG, JPG, JPEG")
    
    with tab2:
        st.subheader("Help & Frequently Asked Questions")
        
        with st.expander("❓ What do B and M mean in metrics?"):
            st.markdown("""
            - **B (Binary)**: Measures object vs background detection
            - **M (Multi-class)**: Measures classification among your 6 crack classes
            
            Your model first detects if there's a crack (binary), then classifies what type it is (multi-class).
            """)
        
        with st.expander("❓ How to interpret mAP scores?"):
            st.markdown("""
            - **mAP50**: More lenient, accepts predictions with >50% overlap
            - **mAP50-95**: Stricter, averages across multiple overlap thresholds
            - **0.0-0.3**: Poor performance, needs improvement
            - **0.3-0.5**: Moderate performance, acceptable for some use cases
            - **0.5-0.7**: Good performance, suitable for most applications
            - **0.7+**: Excellent performance, production ready
            """)
        
     
     
