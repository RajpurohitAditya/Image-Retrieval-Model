import streamlit as st
import torch
import faiss
import numpy as np
import os
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

AUTHORIZED_USERS = {
    "admin": "kalamandir@2026"}

def check_password():
    """Returns True if the user had the correct password."""
    if "password_correct" not in st.session_state:
        # Show login form
        st.title("🔐 Company Login")
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if user in AUTHORIZED_USERS and pwd == AUTHORIZED_USERS[user]:
                st.session_state["password_correct"] = True
                st.session_state["current_user"] = user
                st.rerun() # Refresh to hide login and show app
            else:
                st.error("❌ Invalid credentials")
        return False
    else:
        return True

# --- 2. THE GATEKEEPER ---
if not check_password():
    st.stop() # 🛑 THIS STOPS THE SCRIPT HERE. Everything below is hidden.

# --- 3. MAIN APP (Only runs if password is correct) ---
st.title("💎 AI Visual Inventory Search")
st.sidebar.write(f"Logged in as: **{st.session_state.current_user}**")
if st.sidebar.button("Logout"):
    del st.session_state["password_correct"]
    st.rerun()

INVENTORY_DIR = "images" # Folder where your product images are
if not os.path.exists(INVENTORY_DIR):
    os.makedirs(INVENTORY_DIR)

# --- AI Model Loading (Cached) ---
@st.cache_resource
def load_model():
    # DINOv2 is currently the best 'out-of-the-box' feature extractor
    model_name = "facebook/dinov2-base"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to("cpu") # Use "cuda" if GPU available
    return processor, model

processor, model = load_model()

# --- Helper: Convert Image to Vector ---
def extract_features(img_path):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        # Using the [CLS] token as the global image embedding
        features = outputs.last_hidden_state[:, 0, :].numpy()
    faiss.normalize_L2(features) # Normalize for Cosine Similarity
    return features

# --- Step 1: Indexing the Inventory ---
@st.cache_data
def build_faiss_index(directory):
    valid_exts = ('.jpg', '.jpeg', '.png', '.webp')
    image_list = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(valid_exts)]
    
    if not image_list:
        return None, []

    embeddings = []
    progress_bar = st.progress(0)
    for i, img_path in enumerate(image_list):
        embeddings.append(extract_features(img_path))
        progress_bar.progress((i + 1) / len(image_list))
    
    embeddings = np.vstack(embeddings)
    dimension = embeddings.shape[1]
    
    # FlatIP index handles the Cosine Similarity after normalization
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    return index, image_list

# --- UI Sidebar: Inventory Status ---
with st.sidebar:
    st.header("Inventory Settings")
    if st.button("🔄 Refresh/Build Index"):
        st.cache_data.clear()
        st.success("Indexing started!")
    
    index, image_paths = build_faiss_index(INVENTORY_DIR)
    st.info(f"Currently indexing **{len(image_paths)}** items.")
    
    threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.75, 
                          help="Items below this score will be considered 'Not Found'.")

# --- Step 2: The Search Functionality ---
uploaded_file = st.file_uploader("Upload a photo to find similar items...", type=['jpg', 'png', 'jpeg'])

if uploaded_file and index:
    # Save uploaded file temporarily
    temp_path = "temp_query.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display Query
    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Your Search")
        st.image(uploaded_file, use_container_width=True)

    # Perform Search
    query_vector = extract_features(temp_path)
    D, I = index.search(query_vector, k=4) # Find top 4

    # Display Results
    with col2:
        st.subheader("Best Matches Found")
        res_cols = st.columns(4)
        
        found_any = False
        for i in range(4):
            score = D[0][i]
            img_path = image_paths[I[0][i]]
            
            if score >= threshold:
                found_any = True
                with res_cols[i]:
                    st.image(img_path, caption=f"Match: {score:.2%}")
            else:
                with res_cols[i]:
                    st.markdown("⚠️ *No high match*")
        
        if not found_any:
            st.error("Item not present in inventory. Similarity score too low.")

elif not index:
    st.warning("Please add images to the 'inventory' folder and refresh.")
