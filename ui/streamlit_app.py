import streamlit as st
from PIL import Image as PILImage
from datetime import datetime
from loguru import logger
from agents.medical_agent import MedicalAgent


class StreamlitUI:
    """Streamlit user interface - Professional Edition"""
    
    def __init__(self, agent: MedicalAgent):
        """
        Initialize Streamlit UI
        
        Args:
            agent: Medical agent instance
        """
        self.agent = agent
        self._add_custom_css()
        self._initialize_session_state()
    
    def _add_custom_css(self):
        st.markdown("""
    <style>
        /* Full-width main container */
        .main .block-container {
            max-width: 100% !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
        
        /* Remove centered container */
        section.main > div {
            max-width: 100% !important;
            padding: 1rem 2rem !important;
        }
        
        /* Main background */
        .main {
            background: #0f172a;
        }
        
        /* Headers */
        h1 {
            color: #1e3a8a;
            font-weight: 700;
            font-size: 2.5rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        h2 {
            color: #3b82f6;
            font-weight: 600;
            font-size: 1.5rem !important;
            margin-top: 1rem !important;
        }
        
        h3 {
            color: #1e40af;
            font-size: 1.2rem !important;
        }
        
        /* Buttons */
        .stButton > button {
            width: 100%;
            border-radius: 12px;
            font-weight: 600;
            padding: 0.75rem 1.5rem;
            transition: all 0.3s ease;
            border: none;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        }
        
        /* Chat messages */
        .stChatMessage {
            padding: 1.5rem !important;
            border-radius: 16px !important;
            margin-bottom: 1rem !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            backdrop-filter: blur(10px);
        }
        
        /* Chat input */
        .stChatInput > div {
            border-radius: 25px !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
        
        /* File uploader */
        .uploadedFile {
            border: 2px dashed #3b82f6;
            border-radius: 16px;
            padding: 2rem;
            background: rgba(59, 130, 246, 0.05);
            transition: all 0.3s ease;
        }
        
        .uploadedFile:hover {
            border-color: #2563eb;
            background: rgba(59, 130, 246, 0.1);
        }
        
        /* Sidebar - full height */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
            border-right: 1px solid #334155;
            min-width: 350px !important;
        }
        
        section[data-testid="stSidebar"] > div {
            padding-top: 2rem;
        }
        
        /* Metrics */
        [data-testid="stMetricValue"] {
            font-size: 2rem !important;
            font-weight: 700 !important;
            color: #3b82f6 !important;
        }
        
        /* Image container */
        .image-container {
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
            margin: 1rem 0;
        }
        
        /* Success/Error boxes */
        .stSuccess {
            border-radius: 12px;
            padding: 1rem;
            background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        }
        
        .stError {
            border-radius: 12px;
            padding: 1rem;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }
        
        .stWarning {
            border-radius: 12px;
            padding: 1rem;
            background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        }
        
        /* Info boxes */
        .stInfo {
            border-radius: 12px;
            padding: 1.5rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        /* Dividers */
        hr {
            margin: 2rem 0;
            border: none;
            height: 2px;
            background: linear-gradient(90deg, transparent, #3b82f6, transparent);
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            border-radius: 8px;
            background: rgba(59, 130, 246, 0.1);
        }
    </style>
    """, unsafe_allow_html=True)
    
    def _initialize_session_state(self):
        """Initialize session state variables"""
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'uploaded_image' not in st.session_state:
            st.session_state.uploaded_image = None
        if 'image_filename' not in st.session_state:
            st.session_state.image_filename = None
    
    def _process_uploaded_image(self, uploaded_file) -> bool:
        """Process uploaded image file"""
        try:
            logger.info(f"Processing uploaded file: {uploaded_file.name}")
            
            image = PILImage.open(uploaded_file)
            logger.info(f"Image opened: {image.size}, {image.format}")
            
            metadata = self.agent.image_handler.store_image(image, uploaded_file.name)
            logger.info(f"✅ Image stored in handler: {uploaded_file.name}")
            
            has_pending = self.agent.image_handler.has_pending_image()
            num_uploaded = len(self.agent.image_handler.uploaded_images)
            logger.info(f"   Pending: {has_pending}, Total uploaded: {num_uploaded}")
            
            if not has_pending:
                logger.error("❌ Image NOT in handler after storage!")
                return False
            
            st.session_state.uploaded_image = image
            st.session_state.image_filename = uploaded_file.name
            
            logger.success(f"✅ Image processing complete: {uploaded_file.name}")
            return True
            
        except Exception as e:
            logger.error(f"Image upload failed: {e}")
            st.error(f"Failed to process image: {e}")
            return False
    
    def _send_message(self, user_message: str):
        """Send message to agent"""
        image_keywords = ['image', 'x-ray', 'scan', 'analyze', 'ct', 'mri', 'radiograph', 'xray', 'forearm', 'fracture']
        is_image_query = any(keyword in user_message.lower() for keyword in image_keywords)
        
        has_pending = self.agent.image_handler.has_pending_image()
        num_uploaded = len(self.agent.image_handler.uploaded_images)
        uploaded_keys = list(self.agent.image_handler.uploaded_images.keys())
        
        logger.info(f"=== SENDING MESSAGE ===")
        logger.info(f"Query: {user_message[:50]}...")
        logger.info(f"Is image query: {is_image_query}")
        logger.info(f"Has pending image: {has_pending}")
        logger.info(f"Uploaded count: {num_uploaded}")
        logger.info(f"Uploaded keys: {uploaded_keys}")
        logger.info(f"=====================")
        
        st.session_state.messages.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().strftime("%H:%M")
        })
        
        result = self.agent.query(user_message)
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": result.response,
            "tools": result.tools_used,
            "timestamp": datetime.now().strftime("%H:%M")
        })
        
        logger.info(f"Response received. Tools used: {result.tools_used}")
    
    def render(self):
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 20px; margin-bottom: 2rem; 
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);'>
             <h1 style='color: white; text-align: center; margin: 0;'>
                 Medical AI Assistant
             </h1>
             <p style='color: rgba(255,255,255,0.9); text-align: center; 
                  font-size: 1.1rem; margin-top: 0.5rem;'>
            AI-powered medical consultation and image analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Reset button
        col1, col2, col3 = st.columns([8, 1, 1])
        with col3:
            if st.button("🔄 Reset", use_container_width=True, type="secondary"):
                st.session_state.messages = []
                st.session_state.uploaded_image = None
                st.session_state.image_filename = None
                self.agent.clear_conversation()
                self.agent.image_handler.clear_all()
                st.rerun()
    
        chat_col, sidebar_col = st.columns([7, 3])
    
        with chat_col:
            self._render_chat_interface()
    
        with sidebar_col:
            self._render_sidebar()
    
        self._render_footer()
    
    def _render_chat_interface(self):
        """Render chat interface - IMPROVED"""
        
        st.markdown("### Conversation")
        
        # Larger chat container
        chat_container = st.container(height=750)
        
        with chat_container:
            if not st.session_state.messages:
                st.info("""
                 **Welcome to Medical AI Assistant!**
                
                **I can help you with:**
                -  Medical conditions and symptoms
                -  Treatment options and medications
                -  Medical image analysis (X-rays, CT, MRI)
                -  Clinical guidelines and protocols
                -  Health calculations (BMI, etc.)
                
                **To analyze an image:**
                1. Upload your medical image in the sidebar →
                2. Wait for confirmation 
                3. Click "Analyze" or ask your question
                
                **Example questions:**
                - "What are the symptoms of pneumonia?"
                - "Analyze this X-ray for fractures"
                - "What is the treatment for diabetes?"
                """)
            
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
                    st.markdown(msg["content"])
                    
                    info_parts = [f"{msg.get('timestamp', '')}"]
                    if "tools" in msg and msg["tools"]:
                        info_parts.append(f"🔧 {', '.join(msg['tools'])}")
                    
                    if info_parts:
                        st.caption(" • ".join(info_parts))
        
        st.divider()
        
        user_input = st.chat_input("Type your medical question here...", key="main_chat_input")
        
        if user_input:
            with st.spinner("Analyzing your question..."):
                self._send_message(user_input)
            st.rerun()
    
    def _render_sidebar(self):
        """Render sidebar - REDESIGNED"""
        
        st.markdown("### 📤 Upload Medical Image")
        
        if st.session_state.get('uploaded_image') is not None:
            st.success(f"✅ **{st.session_state.image_filename}**")
            
            if self.agent.image_handler.has_pending_image():
                st.success("🟢 **Ready for Analysis**")
            else:
                st.error("🔴 **System Error** - Please re-upload")
            
            # Display image in a nice container
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(
                st.session_state.uploaded_image,
                use_container_width=True,
                caption=f"📷 {st.session_state.image_filename}"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.divider()
            
            # Action buttons
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔬 **Analyze**", use_container_width=True, type="primary", key="analyze_btn"):
                    quick_query = "Provide a detailed radiological analysis of this medical image, identifying any abnormalities, fractures, or noteworthy features"
                    logger.info(f"Quick analyze clicked. Has pending: {self.agent.image_handler.has_pending_image()}")
                    with st.spinner("🔍 Analyzing image..."):
                        self._send_message(quick_query)
                    st.rerun()
            
            with col2:
                if st.button("🗑️ **Clear**", use_container_width=True, type="secondary", key="clear_btn"):
                    st.session_state.uploaded_image = None
                    st.session_state.image_filename = None
                    self.agent.image_handler.clear_all()
                    logger.info("Cleared all images")
                    st.rerun()
            
        else:
            # ============================================================
            # NO IMAGE STATE
            # ============================================================
            
            st.info("📸 No image uploaded yet")
            
            uploaded_file = st.file_uploader(
                "Choose a medical image",
                type=['png', 'jpg', 'jpeg', 'webp', 'bmp', 'dicom'],
                help="Upload X-ray, CT, MRI, or other medical images",
                label_visibility="collapsed",
                key="image_uploader"
            )
            
            if uploaded_file is not None:
                logger.info(f"File uploader detected file: {uploaded_file.name}")
                with st.spinner("⏳ Processing image..."):
                    success = self._process_uploaded_image(uploaded_file)
                    if success:
                        logger.info("Image processed successfully")
                        st.balloons()  # Fun animation!
                        st.rerun()
                    else:
                        st.error("❌ Failed to process. Please try again.")
            
            # Help section
            with st.expander("ℹ️ Supported Image Types"):
                st.markdown("""
                **Supported formats:**
                - PNG (.png)
                - JPEG (.jpg, .jpeg)
                - WebP (.webp)
                - BMP (.bmp)
                - DICOM (.dcm) *experimental*
                
                **Recommended:**
                - File size: < 10MB
                - Resolution: 512x512 or higher
                - Clear, high-contrast images
                """)
        
        st.divider()
        
        # ============================================================
        # SESSION STATISTICS
        # ============================================================
        
        st.markdown("### 📊 Session Statistics")
        
        num_messages = len(st.session_state.messages)
        num_questions = sum(1 for m in st.session_state.messages if m["role"] == "user")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="💬 Messages",
                value=num_messages,
                delta=None
            )
        
        with col2:
            st.metric(
                label="❓ Questions",
                value=num_questions,
                delta=None
            )
        
        try:
            stats = self.agent.get_statistics()
            st.caption(f"📝 {stats['memory_summary']}")
            
            if self.agent.image_handler.has_pending_image():
                st.success("🟢 Image Ready")
            else:
                st.info("⚪ No Image")
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
    
    def _render_footer(self):
        """Render professional footer"""
        
        st.divider() 
        current_date = datetime.now().strftime('%B %d, %Y')
        
        st.markdown("""
        <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                    border-radius: 16px; margin-top: 2rem;'>
            <p style='color: #475569; font-size: 0.9rem; margin: 0.5rem 0;'>
                <strong>⚕️ Medical Disclaimer:</strong> For educational purposes only
            </p>
            <p style='color: #475569; font-size: 0.9rem; margin: 0.5rem 0;'>
                <strong>🤖 Powered by:</strong> Google Gemini Vision AI
            </p>
            <p style='color: #64748b; font-size: 0.85rem; margin: 0.5rem 0;'>
                📅 Session: {current_date}
            </p>
        </div>
        """,unsafe_allow_html=True)
        
        with st.expander("⚠️ **Important Medical Disclaimer**"):
            st.warning("""
            **This AI assistant is for educational and informational purposes only.**
            
            - ❌ NOT a substitute for professional medical advice
            - ❌ NOT for emergency medical situations
            - ✅ Always consult qualified healthcare professionals
            - ✅ Do not use for diagnosis or treatment decisions
            - ⚠️ AI may make errors - verify all information
            
            **🚨 In case of emergency, call 911 or your local emergency number immediately.**
            """)


def run_app(agent: MedicalAgent):
    """Run Streamlit application"""
    ui = StreamlitUI(agent)
    ui.render()