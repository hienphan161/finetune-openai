"""
Streamlit UI for comparing base model vs fine-tuned model responses.
"""

import streamlit as st
from core import setup_openai_client, load_config, chat_with_model

# Page config
st.set_page_config(
    page_title="🎯 LLM Fine-tuning Comparison",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for fancy light theme styling
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f8f9fe 0%, #e8f4f8 50%, #fff5f5 100%);
    }
    
    /* Headers */
    h1, h2, h3 {
        background: linear-gradient(90deg, #e94560, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fe 100%);
        border-right: 2px solid #e94560;
    }
    
    /* Response cards */
    .response-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 24px;
        border: 2px solid #e94560;
        box-shadow: 0 4px 20px rgba(233, 69, 96, 0.1);
        margin: 10px 0;
    }
    
    .response-card-finetuned {
        background: #ffffff;
        border-radius: 16px;
        padding: 24px;
        border: 2px solid #10b981;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.1);
        margin: 10px 0;
    }
    
    .card-title {
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 12px;
    }
    
    .card-title-base {
        color: #e94560;
    }
    
    .card-title-finetuned {
        color: #10b981;
    }
    
    .response-text {
        color: #1f2937;
        font-size: 16px;
        line-height: 1.8;
    }
    
    /* Input styling */
    .stTextArea textarea {
        background-color: #ffffff !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 12px !important;
        color: #1f2937 !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #e94560 !important;
    }
    
    .stTextInput input {
        background-color: #ffffff !important;
        border: 2px solid #e2e8f0 !important;
        border-radius: 12px !important;
        color: #1f2937 !important;
    }
    
    .stTextInput input:focus {
        border-color: #e94560 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #e94560, #ff6b6b) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 32px !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(233, 69, 96, 0.3) !important;
    }
    
    /* Divider */
    .fancy-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #e94560, transparent);
        margin: 20px 0;
    }
    
    /* Info box */
    .info-box {
        background: #fff5f5;
        border-left: 4px solid #e94560;
        padding: 16px;
        border-radius: 0 12px 12px 0;
        margin: 16px 0;
        color: #1f2937;
    }
    
    .info-box code {
        background: #fef2f2;
        padding: 2px 6px;
        border-radius: 4px;
        color: #e94560;
    }
    
    /* Labels */
    .stSelectbox label, .stTextArea label, .stTextInput label {
        color: #1f2937 !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "messages_history" not in st.session_state:
        st.session_state.messages_history = []
    if "config" not in st.session_state:
        try:
            st.session_state.config = load_config()
        except FileNotFoundError:
            st.session_state.config = None
    if "client" not in st.session_state:
        st.session_state.client = None


def get_client():
    """Get or create OpenAI client."""
    if st.session_state.client is None and st.session_state.config:
        try:
            st.session_state.client = setup_openai_client(st.session_state.config)
        except ValueError as e:
            st.error(f"❌ {e}")
            return None
    return st.session_state.client


def render_sidebar():
    """Render sidebar with model configuration."""
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
        
        # Base model selection
        base_models = ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]
        default_base = st.session_state.config.get("fine_tuning", {}).get("model", "gpt-3.5-turbo") if st.session_state.config else "gpt-3.5-turbo"
        
        base_model = st.selectbox(
            "🤖 Base Model",
            base_models,
            index=base_models.index(default_base) if default_base in base_models else 0,
            help="The original model before fine-tuning"
        )
        
        # Fine-tuned model input (hidden like password)
        finetuned_model = st.text_input(
            "🎯 Fine-tuned Model ID",
            placeholder="ft:gpt-3.5-turbo:org::xxx",
            type="password",
            help="Your fine-tuned model ID from OpenAI"
        )
        
        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
        
        # System message
        system_message = st.text_area(
            "📝 System Message",
            value="You are a Singaporean chatbot.",
            height=100,
            help="The system prompt for both models"
        )
        
        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
        
        # Info
        st.markdown("""
        <div class="info-box">
            <strong>💡 Tip:</strong> Get your fine-tuned model ID using:<br>
            <code>python finetune_openai.py list-jobs</code>
        </div>
        """, unsafe_allow_html=True)
        
        return base_model, finetuned_model, system_message


def render_response_card(title: str, response: str, is_finetuned: bool = False):
    """Render a styled response card."""
    card_class = "response-card-finetuned" if is_finetuned else "response-card"
    title_class = "card-title-finetuned" if is_finetuned else "card-title-base"
    icon = "✨" if is_finetuned else "🤖"
    
    st.markdown(f"""
    <div class="{card_class}">
        <div class="card-title {title_class}">{icon} {title}</div>
        <div class="response-text">{response}</div>
    </div>
    """, unsafe_allow_html=True)


def main():
    init_session_state()
    
    # Header
    st.markdown("# 🎯 LLM Fine-tuning Comparison")
    st.markdown("Compare responses between **base model** and **fine-tuned model** side by side.")
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    
    # Check config
    if not st.session_state.config:
        st.error("❌ Configuration file not found. Please create `config.yaml` from `config.yaml.sample`.")
        st.code("cp config.yaml.sample config.yaml", language="bash")
        return
    
    # Sidebar
    base_model, finetuned_model, system_message = render_sidebar()
    
    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_message = st.text_area(
            "💬 Your Message",
            placeholder="Type your message here... (e.g., 'Why Singapore so hot?')",
            height=120,
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        compare_button = st.button("⚡ Compare", use_container_width=True)
    
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    
    # Compare responses
    if compare_button:
        if not user_message:
            st.warning("⚠️ Please enter a message to compare.")
            return
        
        if not finetuned_model:
            st.warning("⚠️ Please enter your fine-tuned model ID in the sidebar.")
            return
        
        client = get_client()
        if not client:
            return
        
        # Show prompt
        st.markdown("### 📤 Prompt")
        st.info(f"**System:** {system_message}\n\n**User:** {user_message}")
        
        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 📥 Responses")
        
        # Get responses
        col_base, col_ft = st.columns(2)
        
        with col_base:
            with st.spinner("🤖 Base model thinking..."):
                try:
                    base_response = chat_with_model(
                        client, base_model, user_message, system_message
                    )
                    render_response_card(f"Base Model ({base_model})", base_response)
                except Exception as e:
                    st.error(f"❌ Base model error: {e}")
        
        with col_ft:
            with st.spinner("✨ Fine-tuned model thinking..."):
                try:
                    ft_response = chat_with_model(
                        client, finetuned_model, user_message, system_message
                    )
                    render_response_card("Fine-tuned Model", ft_response, is_finetuned=True)
                except Exception as e:
                    st.error(f"❌ Fine-tuned model error: {e}")
        
        # Save to history
        st.session_state.messages_history.append({
            "user": user_message,
            "system": system_message,
            "base_model": base_model,
            "base_response": base_response if 'base_response' in dir() else "Error",
            "finetuned_model": finetuned_model,
            "finetuned_response": ft_response if 'ft_response' in dir() else "Error",
        })
    
    # History
    if st.session_state.messages_history:
        st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
        
        with st.expander("📜 Comparison History", expanded=False):
            for i, item in enumerate(reversed(st.session_state.messages_history[-5:])):
                st.markdown(f"**#{len(st.session_state.messages_history) - i}** - {item['user'][:50]}...")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"🤖 **Base:** {item['base_response'][:200]}...")
                with col2:
                    st.markdown(f"✨ **Fine-tuned:** {item['finetuned_response'][:200]}...")
                st.markdown("---")


if __name__ == "__main__":
    main()

