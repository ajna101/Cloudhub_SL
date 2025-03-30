import streamlit as st
import sqlite3
import bcrypt
import pandas as pd
import boto3
import numpy as np
import time
from openai import OpenAI
import json
from datetime import datetime
from typing import List, Tuple
from io import BytesIO
import io
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

st.set_page_config(page_title="EITA", layout="wide")

EMBEDDING_MODEL = "text-embedding-ada-002"
DB_NAME = "vector_chunks.db"
COMPUTED_DB = "data.db"
VALID_BUCKET = "etrm-etai-poc"
REJECTED_BUCKET = "etai-rejected-files"

aws_access_key = st.secrets["AWS_ACCESS_KEY"]
aws_secret_key = st.secrets["AWS_SECRET_KEY"]

client = OpenAI(api_key= st.secrets["OPENAI_API_KEY"])

computed_stats_json = []

# --- Initialize Session State ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "main_section" not in st.session_state:
    st.session_state.main_section = "Data Management AI Agent"
if "sub_section" not in st.session_state:
    st.session_state.sub_section = "Pipeline Dashboard"

# --- Database Functions ---
def init_sqlite_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password BLOB NOT NULL,
            role TEXT DEFAULT 'user'
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_text TEXT,
            embedding BLOB,
            source_file TEXT
        )
    """)
    cursor.execute('''
      CREATE TABLE IF NOT EXISTS agent_detail (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT NOT NULL,
          model TEXT NOT NULL,
          temperature REAL DEFAULT 0.7,
          prompt TEEXTT,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS feedback_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_query TEXT,
                        query_responce TEXT,
                        user_reaction BOOLEAN DEFAULT 0,
                        user_feedback TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                      )''')
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS file_tracking (
            upload_id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            date_processed TEXT,
            json_contents TEXT,
            is_processed INTEGER
        );
    """)
    conn.commit()
    conn.close()

def init_computed_db():
    conn = sqlite3.connect(COMPUTED_DB)
    conn.close()

init_sqlite_db()

def register_user(username, password):
    """Registers a new user with hashed password."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def authenticate_user(username, password):
    """Authenticates user with hashed password verification."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()

    if result:
        stored_password = result[0]  # Already in bytes format
        return bcrypt.checkpw(password.encode('utf-8'), stored_password)

    return False

# --- Sidebar Navigation ---
def sidebar_menu():
    """Sidebar with main and sub-sections."""
    with st.sidebar:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write(' ')

        with col2:
          st.image("https://img1.wsimg.com/isteam/ip/495c53b7-d765-476a-99bd-58ecde467494/blob-411e887.png/:/rs=w:127,h:95,cg:true,m/cr=w:127,h:95/qt=q:95")

        with col3:
            st.write(' ')

        st.markdown("<h2 style='text-align: center;'>ETAI Energey Trading AI Platform</h2>", unsafe_allow_html=True)
        st.divider()

        # Main Sections
        main_section = st.radio(
            "Select AI Agent",
            ["Data Management AI Agent", "RAG AI Agent", "Application AI Agent"],
            index=["Data Management AI Agent", "RAG AI Agent", "Application AI Agent"].index(st.session_state.main_section)
        )

        st.session_state.main_section = main_section

        # Sub-sections dictionary
        sub_sections = {
            "Data Management AI Agent": ["Pipeline Dashboard", "Data Pipeline", "Processed Data"],
            "RAG AI Agent": ["Dashboard", "Configure & Upload", "Fine Tuning", "Settings"],
            "Application AI Agent": ["Energy Tradeing Analysis", "Graph Query" ,"Deviation Analysis", "Root Cause Analysis", "Analysis History", "User Feedback"]
        }

        # Ensure sub_section is valid for the current main_section
        if st.session_state.sub_section not in sub_sections[main_section]:
            st.session_state.sub_section = sub_sections[main_section][0]

        sub_section = st.radio(
            f"Select {main_section} Section",
            sub_sections[main_section],
            index=sub_sections[main_section].index(st.session_state.sub_section)
        )

        st.session_state.sub_section = sub_section

        st.divider()


def logout():
    """Logs out the user."""
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()

def top_right_menu():
    """Displays the user profile and logout button."""
    username = st.session_state.get('username', 'Guest')

    col1, col2 = st.columns([8, 2])
    with col1:
        st.markdown(f"👤 **{username}**", unsafe_allow_html=True)
    with col2:
        if st.button("🔴 Logout", key="logout_btn", help="Logout from the platform"):
            logout()

# --- CSS Styling ---
st.markdown("""
    <style>
        /* Sidebar */
        [data-testid="stSidebar"] {
            box-shadow: 2px 0 10px rgba(0,0,0,0.1);
        }

        /* Buttons */
        button {
            border-radius: 10px !important;
        }
        button:hover {
            background: #007BFF !important;
            color: white !important;
            border: none !important;
        }

        /* Title & Text */
        h1, h2, h3 {
            color: #007BFF;
        }

        /* Custom Divider */
        hr {
            border: 1px solid #ccc;
        }
    .login-btn {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 12px 20px;
        margin: 10px 0;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-size: 16px;
        transition: 0.3s;
    }
    .login-btn:hover {
        background-color: #45a049;
    }
    .login-input {
        width: 100%;
        padding: 10px;
        margin: 10px 0;
        display: inline-block;
        border: 1px solid #ccc;
        border-radius: 5px;
        box-sizing: border-box;
    }
    </style>
""", unsafe_allow_html=True)

def process_metadata_alias(metadata_df: pd.DataFrame):
    global alias_mapping
    alias_mapping = {}
    # Assumes metadata CSV has columns named 'Column' and 'Description'
    for _, row in metadata_df.iterrows():
        col_name = str(row.get("Column", "")).strip()
        description = str(row.get("Description", "")).strip().lower()
        if col_name and description:
            alias_mapping[description] = col_name
    return alias_mapping

def upload_to_s3(file, filename, bucket):
    try:
        # Determine folder based on filename
        folder = ""
        if "NOP_CO2" in filename.upper():
            folder = "CO2/"
        elif "NOP_NG" in filename.upper():
            folder = "NG/"
        elif "NOP_PW" in filename.upper():
            folder = "PW/"
        else:
            folder = "misc/"  # fallback or unknown

        s3_key = folder + filename  # full path in S3

        # Upload to S3
        s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key
        )
        s3.upload_fileobj(file, bucket, s3_key)

        return True, f"✅ Uploaded {filename} to S3 bucket '{bucket}' in folder '{folder}'"
    except Exception as e:
        return False, f"❌ Upload failed: {e}"


def upload_to_s3_old(file, filename, bucket):
    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key
        )
        s3.upload_fileobj(file, bucket, filename)
        return True, f"✅ Uploaded {filename} to S3 bucket '{bucket}'"
    except Exception as e:
        return False, f"❌ Upload failed: {e}"

def validate_against_metadata(file_df: pd.DataFrame, metadata_df: pd.DataFrame, file_id: str) -> bool:
    file_cols = set(file_df.columns.str.strip().str.upper())
    metadata_cols = set(metadata_df["Field_Name"].str.strip().str.upper())

    st.subheader(f"🔍 Column Comparison for `{file_id}`")
    st.write("🗂 **Data File Columns**:")
    st.code("\n".join(sorted(file_cols)))
    st.write("📘 **Metadata Columns**:")
    st.code("\n".join(sorted(metadata_cols)))

    missing_in_file = metadata_cols - file_cols
    extra_in_file = file_cols - metadata_cols

    if missing_in_file:
        st.error(f"❌ Columns missing in data file: {', '.join(missing_in_file)}")
    if extra_in_file:
        st.warning(f"⚠️ Extra columns in data file: {', '.join(extra_in_file)}")

    if not missing_in_file:
        st.success("✅ Validation Passed.")
        return True
    else:
        st.error("❌ Validation Failed.")
        return False



def store_processed_data_in_sqlite(file_name, report_date, processed_data):
    """Stores processed statistics in SQLite."""
    conn = sqlite3.connect("data_store.db")
    cursor = conn.cursor()

    # Convert processed statistics to JSON
    processed_json = json.dumps(processed_data)

    # Insert processed data into SQLite
    cursor.execute('''
        INSERT INTO processed_data (file_name, report_date, processed_data, processed_flag)
        VALUES (?, ?, ?, ?)
    ''', (file_name, report_date, processed_json, 1))

    conn.commit()
    conn.close()
    print(f"✅ Processed data for {file_name} stored in SQLite")

def sanitize_for_json(obj):
    if isinstance(obj, list):
        return [sanitize_for_json(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d')
    return obj

def process_and_store_file(file_name: str, df: pd.DataFrame, cursor, folder_prefix):
    # Trim and convert column names to uppercase
    df.columns = df.columns.str.strip().str.upper()

    # Convert REPORT_DATE to datetime format
    def parse_dates(report_date):
        for fmt in ['%d%b%y', '%d-%m-%Y']:  # Try different formats
            try:
                return pd.to_datetime(report_date, format=fmt)
            except (ValueError, TypeError):
                continue
        return None  # Return None instead of NaT for JSON serialization

    df['REPORT_DATE'] = df['REPORT_DATE'].astype(str).str.strip()  # Ensure it's string
    df['REPORT_DATE'] = df['REPORT_DATE'].apply(parse_dates)  # Apply parsing function

    if folder_prefix == 'CO2':
        numeric_cols = ['VOLUME', 'MKTVAL', 'TRDVAL', 'TRDPRC']
    elif folder_prefix == 'NG':
        numeric_cols = ['VOLUME', 'VOLUME_TOTAL', 'QTY_PHY', 'MKT_VAL', 'QTY_FIN', 'TRD_VAL']
    elif folder_prefix == 'PW':
        numeric_cols = ['VOLUME_BL', 'VOLUME_PK', 'VOLUME_OFPK', 'MKT_VAL_BL', 'MKT_VAL_PK', 'MKT_VAL_OFPK', 'TRD_VAL_BL', 'TRD_VAL_PK', 'TRD_VAL_OFPK']
    else:
        raise ValueError(f"❌ Unknown folder prefix: {folder_prefix}")

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Grouping logic for each case
    if folder_prefix == 'CO2':
        df_daily_stats = df.groupby('REPORT_DATE').agg(
            TOTAL_VOLUME=('VOLUME', 'sum'), AVG_VOLUME=('VOLUME', 'mean'),
            MIN_VOLUME=('VOLUME', 'min'), MAX_VOLUME=('VOLUME', 'max'), STD_VOLUME=('VOLUME', 'std'),
            TOTAL_TRDVAL=('TRDVAL', 'sum'), AVG_TRDVAL=('TRDVAL', 'mean'),
            MIN_TRDVAL=('TRDVAL', 'min'), MAX_TRDVAL=('TRDVAL', 'max'), STD_TRDVAL=('TRDVAL', 'std'),
            TOTAL_MKTVAL=('MKTVAL', 'sum'), AVG_MKTVAL=('MKTVAL', 'mean'),
            MIN_MKTVAL=('MKTVAL', 'min'), MAX_MKTVAL=('MKTVAL', 'max'), STD_MKTVAL=('MKTVAL', 'std'),
            TOTAL_TRDPRC=('TRDPRC', 'sum'), AVG_TRDPRC=('TRDPRC', 'mean'),
            MIN_TRDPRC=('TRDPRC', 'min'), MAX_TRDPRC=('TRDPRC', 'max'), STD_TRDPRC=('TRDPRC', 'std')
        ).reset_index()

    elif folder_prefix == 'NG':
        df_daily_stats = df.groupby('REPORT_DATE').agg(
            TOTAL_VOLUME=('VOLUME', 'sum'), AVG_VOLUME=('VOLUME', 'mean'),
            MIN_VOLUME=('VOLUME', 'min'), MAX_VOLUME=('VOLUME', 'max'), STD_VOLUME=('VOLUME', 'std'),
            TOTAL_VOLUME_TOTAL=('VOLUME_TOTAL', 'sum'), AVG_VOLUME_TOTAL=('VOLUME_TOTAL', 'mean'),
            MIN_VOLUME_TOTAL=('VOLUME_TOTAL', 'min'), MAX_VOLUME_TOTAL=('VOLUME_TOTAL', 'max'), STD_VOLUME_TOTAL=('VOLUME_TOTAL', 'std'),
            TOTAL_QTY_PHY=('QTY_PHY', 'sum'), AVG_QTY_PHY=('QTY_PHY', 'mean'),
            MIN_QTY_PHY=('QTY_PHY', 'min'), MAX_QTY_PHY=('QTY_PHY', 'max'), STD_QTY_PHY=('QTY_PHY', 'std'),
            TOTAL_MKT_VAL=('MKT_VAL', 'sum'), AVG_MKT_VAL=('MKT_VAL', 'mean'),
            MIN_MKT_VAL=('MKT_VAL', 'min'), MAX_MKT_VAL=('MKT_VAL', 'max'), STD_MKT_VAL=('MKT_VAL', 'std'),
            TOTAL_QTY_FIN=('QTY_FIN', 'sum'), AVG_QTY_FIN=('QTY_FIN', 'mean'),
            MIN_QTY_FIN=('QTY_FIN', 'min'), MAX_QTY_FIN=('QTY_FIN', 'max'), STD_QTY_FIN=('QTY_FIN', 'std'),
            TOTAL_TRD_VAL=('TRD_VAL', 'sum'), AVG_TRD_VAL=('TRD_VAL', 'mean'),
            MIN_TRD_VAL=('TRD_VAL', 'min'), MAX_TRD_VAL=('TRD_VAL', 'max'), STD_TRD_VAL=('TRD_VAL', 'std')
        ).reset_index()

    elif folder_prefix == 'PW':
        df_daily_stats = df.groupby('REPORT_DATE').agg(
            TOTAL_VOLUME_BL=('VOLUME_BL', 'sum'), AVG_VOLUME_BL=('VOLUME_BL', 'mean'),
            MIN_VOLUME_BL=('VOLUME_BL', 'min'), MAX_VOLUME_BL=('VOLUME_BL', 'max'), STD_VOLUME_BL=('VOLUME_BL', 'std'),
            TOTAL_VOLUME_PK=('VOLUME_PK', 'sum'), AVG_VOLUME_PK=('VOLUME_PK', 'mean'),
            MIN_VOLUME_PK=('VOLUME_PK', 'min'), MAX_VOLUME_PK=('VOLUME_PK', 'max'), STD_VOLUME_PK=('VOLUME_PK', 'std'),
            TOTAL_VOLUME_OFPK=('VOLUME_OFPK', 'sum'), AVG_VOLUME_OFPK=('VOLUME_OFPK', 'mean'),
            MIN_VOLUME_OFPK=('VOLUME_OFPK', 'min'), MAX_VOLUME_OFPK=('VOLUME_OFPK', 'max'), STD_VOLUME_OFPK=('VOLUME_OFPK', 'std'),
            TOTAL_MKT_VAL_BL=('MKT_VAL_BL', 'sum'), AVG_MKT_VAL_BL=('MKT_VAL_BL', 'mean'),
            MIN_MKT_VAL_BL=('MKT_VAL_BL', 'min'), MAX_MKT_VAL_BL=('MKT_VAL_BL', 'max'), STD_MKT_VAL_BL=('MKT_VAL_BL', 'std'),
            TOTAL_MKT_VAL_PK=('MKT_VAL_PK', 'sum'), AVG_MKT_VAL_PK=('MKT_VAL_PK', 'mean'),
            MIN_MKT_VAL_PK=('MKT_VAL_PK', 'min'), MAX_MKT_VAL_PK=('MKT_VAL_PK', 'max'), STD_MKT_VAL_PK=('MKT_VAL_PK', 'std'),
            TOTAL_MKT_VAL_OFPK=('MKT_VAL_OFPK', 'sum'), AVG_MKT_VAL_OFPK=('MKT_VAL_OFPK', 'mean'),
            MIN_MKT_VAL_OFPK=('MKT_VAL_OFPK', 'min'), MAX_MKT_VAL_OFPK=('MKT_VAL_OFPK', 'max'), STD_MKT_VAL_OFPK=('MKT_VAL_OFPK', 'std'),
            TOTAL_TRD_VAL_BL=('TRD_VAL_BL', 'sum'), AVG_TRD_VAL_BL=('TRD_VAL_BL', 'mean'),
            MIN_TRD_VAL_BL=('TRD_VAL_BL', 'min'), MAX_TRD_VAL_BL=('TRD_VAL_BL', 'max'), STD_TRD_VAL_BL=('TRD_VAL_BL', 'std'),
            TOTAL_TRD_VAL_PK=('TRD_VAL_PK', 'sum'), AVG_TRD_VAL_PK=('TRD_VAL_PK', 'mean'),
            MIN_TRD_VAL_PK=('TRD_VAL_PK', 'min'), MAX_TRD_VAL_PK=('TRD_VAL_PK', 'max'), STD_TRD_VAL_PK=('TRD_VAL_PK', 'std'),
            TOTAL_TRD_VAL_OFPK=('TRD_VAL_OFPK', 'sum'), AVG_TRD_VAL_OFPK=('TRD_VAL_OFPK', 'mean'),
            MIN_TRD_VAL_OFPK=('TRD_VAL_OFPK', 'min'), MAX_TRD_VAL_OFPK=('TRD_VAL_OFPK', 'max'), STD_TRD_VAL_OFPK=('TRD_VAL_OFPK', 'std')
        ).reset_index()

    # Aggregation for BOOK, TGROUP1, SEGMENT
    if folder_prefix == 'PW':
        df_book_stats = df.groupby('BOOK').agg(
            TOTAL_VOLUME_BL=('VOLUME_BL', 'sum'),
            TOTAL_VOLUME_PK=('VOLUME_PK', 'sum'),
            TOTAL_VOLUME_OFPK=('VOLUME_OFPK', 'sum')
        ).reset_index() if 'BOOK' in df.columns else pd.DataFrame()

        df_tgroup1_stats = df.groupby('TGROUP1').agg(
            TOTAL_VOLUME_BL=('VOLUME_BL', 'sum'),
            TOTAL_VOLUME_PK=('VOLUME_PK', 'sum'),
            TOTAL_VOLUME_OFPK=('VOLUME_OFPK', 'sum')
        ).reset_index() if 'TGROUP1' in df.columns else pd.DataFrame()

        df_segment_stats = df.groupby('SEGMENT').agg(
            TOTAL_VOLUME_BL=('VOLUME_BL', 'sum'),
            TOTAL_VOLUME_PK=('VOLUME_PK', 'sum'),
            TOTAL_VOLUME_OFPK=('VOLUME_OFPK', 'sum')
        ).reset_index() if 'SEGMENT' in df.columns else pd.DataFrame()

        df_bucket_stats = df.groupby('BUCKET').agg(
            TOTAL_VOLUME_BL=('VOLUME_BL', 'sum'),
            TOTAL_VOLUME_PK=('VOLUME_PK', 'sum'),
            TOTAL_VOLUME_OFPK=('VOLUME_OFPK', 'sum')
        ).reset_index() if 'BUCKET' in df.columns else pd.DataFrame()

    else:  # Default case for CO2 and NG
        df_book_stats = df.groupby('BOOK').agg(TOTAL_VOLUME=('VOLUME', 'sum')).reset_index() if 'BOOK' in df.columns else pd.DataFrame()
        df_tgroup1_stats = df.groupby('TGROUP1').agg(TOTAL_VOLUME=('VOLUME', 'sum')).reset_index() if 'TGROUP1' in df.columns else pd.DataFrame()
        df_segment_stats = df.groupby('SEGMENT').agg(TOTAL_VOLUME=('VOLUME', 'sum')).reset_index() if 'SEGMENT' in df.columns else pd.DataFrame()
        df_bucket_stats = df.groupby('BUCKET').agg(TOTAL_VOLUME=('VOLUME', 'sum')).reset_index() if 'BUCKET' in df.columns else pd.DataFrame()

    combined_json = {
        "daily_totals": df_daily_stats.to_dict(orient='records'),
        "row_level_data": df.to_dict(orient='records'),
        "book_stats": df_book_stats.to_dict(orient='records'),
        "tgroup1_stats": df_tgroup1_stats.to_dict(orient='records'),
        "segment_stats": df_segment_stats.to_dict(orient='records'),
        "bucket_stats": df_bucket_stats.to_dict(orient='records'),
    }

    sanitized_json = sanitize_for_json(combined_json)
    cursor.execute(
        "INSERT INTO file_tracking (file_name, date_processed, json_contents, is_processed) VALUES (?, datetime('now'), ?, 1)",
        (file_name, json.dumps(sanitized_json))
    )

    return df

def process_files_from_s3_folder(bucket_name, folder_prefix):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key
    )
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    try:
        objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_prefix).get("Contents", [])
        print(f"🔍 Found {len(objects)} files in folder: {folder_prefix}")
        for obj in objects:
            file_key = obj["Key"]
            if not file_key.lower().endswith(".csv"):
                continue

            file_name = file_key.split("/")[-1]
            cursor.execute("SELECT 1 FROM file_tracking WHERE file_name = ? AND is_processed = 1", (file_name,))
            if cursor.fetchone():
                print(f"⏭️ Skipping already processed file: {file_name}")
                continue

            try:
                obj_data = s3.get_object(Bucket=bucket_name, Key=file_key)
                file_stream = io.BytesIO(obj_data["Body"].read())
                df = pd.read_csv(file_stream)
                df_processed = process_and_store_file(file_name, df, cursor,folder_prefix)
                conn.commit()
                print(f"✅ Processed and saved: {file_name}")
            except Exception as e:
                print(f"❌ Failed to process {file_name}: {e}")

    except Exception as e:
        print(f"❌ Error accessing S3: {e}")
    finally:
        conn.close()

def query_sqlite_json_with_openai(user_question):

    # Load all JSON contents from SQLite
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT file_name, json_contents FROM file_tracking WHERE is_processed = 1")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("⚠️ No processed JSON data found in the database.")
        return

    all_context = ""
    for file_name, json_text in rows:
        try:
            json_data = json.loads(json_text)
            # Truncate or format for prompt (you can adjust max length)
            summary = json.dumps(json_data, indent=2)[:3000]
            all_context += f"\n---\n📄 File: {file_name}\n{summary}"
        except Exception as e:
            all_context += f"\n---\n📄 File: {file_name}\n⚠️ Error reading JSON: {e}"

    # Final prompt with user question
    prompt = f"""
        Below is preprocessed trading statistics stored in SQLite from multiple files:

        {all_context}

        Now, answer the following question based on the above data:

        {user_question}
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    print("\n💬 GPT Answer:")
    print(response.choices[0].message.content.strip())
    return response.choices[0].message.content.strip()

def add_agent_detail(name, model, temperature, prompt):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO agent_detail (name, model, temperature, prompt)
    VALUES (?, ?, ?, ?)
    ''', (name, model, temperature, prompt))
    st.success(f"✅ Settings Saved!!")
    conn.commit()
    conn.close()

def save_computed_data_to_sqlite(json_data, table_name="computed_data"):
    if not json_data:
        st.warning("No computed data available to save.")
        return
    conn = sqlite3.connect(COMPUTED_DB)
    cursor = conn.cursor()
    cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
    original_columns = list(json_data[0].keys())
    sanitized_columns = []
    for col in original_columns:
        new_col = col.replace(":", "_")
        if new_col and new_col[0].isdigit():
            new_col = "_" + new_col
        sanitized_columns.append(new_col)
    def get_col_type(val):
        if isinstance(val, pd.Timestamp):
            return "TEXT"
        return "REAL" if isinstance(val, (int, float)) else "TEXT"
    col_defs = ", ".join([f'"{sanitized_columns[i]}" {get_col_type(json_data[0][original_columns[i]])}'
                          for i in range(len(original_columns))])
    create_table_query = f"CREATE TABLE {table_name} ({col_defs});"
    cursor.execute(create_table_query)
    placeholders = ", ".join(["?"] * len(sanitized_columns))
    cols = ", ".join('"' + col + '"' for col in sanitized_columns)
    insert_query = f"INSERT INTO {table_name} ({cols}) VALUES ({placeholders});"
    for row in json_data:
        values = [str(row[col]) if isinstance(row[col], pd.Timestamp) else row[col] for col in original_columns]
        cursor.execute(insert_query, values)
    conn.commit()
    conn.close()
    st.success(f"✅ Computed data saved to SQLite table '{table_name}'.")

def compute_statistics_and_save(df: pd.DataFrame, file_id: str, selected_fields: List[str] = None) -> List[dict]:
    if "REPORT_DATE" not in df.columns:
        st.error("REPORT_DATE column not found in the dataset!")
        return []
    df["REPORT_DATE"] = pd.to_datetime(df["REPORT_DATE"], errors='coerce')
    # Determine selected numeric fields: use provided list or alias mapping if available; else auto-detect numeric columns
    if selected_fields is None:
        selected_fields = list(alias_mapping.values()) if alias_mapping else list(df.select_dtypes(include=[np.number]).columns)
    # Clean the selected columns
    for col in selected_fields:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    agg_funcs = {col: ['sum', 'mean', 'min', 'max', 'std'] for col in selected_fields if col in df.columns}
    df_stats = df.groupby('REPORT_DATE').agg(agg_funcs)
    df_stats.columns = ['_'.join(col) for col in df_stats.columns]
    df_stats.reset_index(inplace=True)
    stats_json = df_stats.to_dict(orient="records")
    # Sanitize each record for JSON serialization
    stats_json = [sanitize_record(record) for record in stats_json]
    json_filename = f"aggregated_stats_{file_id}.json"
    with open(json_filename, "w") as f:
        json.dump(stats_json, f, default=str, indent=2)
    st.success(f"Aggregated statistics saved to {json_filename}")
    print(stats_json)
    return stats_json

selected_fields_ui = None



def sanitize_record(record: dict) -> dict:
    sanitized = {}
    for k, v in record.items():
        if isinstance(v, pd.Timestamp):
            sanitized[k] = str(v)
        else:
            sanitized[k] = v
    return sanitized

def store_computed_embeddings(json_data, table_name="computed_embeddings"):
    if not json_data:
        st.warning("No computed data available to store embeddings.")
        return
    conn = sqlite3.connect(COMPUTED_DB)
    cursor = conn.cursor()
    cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
    cursor.execute(f"""
        CREATE TABLE {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            json_data TEXT,
            embedding BLOB
        );
    """)
    texts = []
    debug_expander = st.expander("Debug: Raw computed records")
    for record in json_data:
        try:
            sanitized = sanitize_record(record)
            text = json.dumps(sanitized)
            texts.append(text)
        except Exception as e:
            debug_expander.error(f"Error serializing record: {record}\nError: {e}")
    if not texts:
        st.error("No records could be serialized for embedding.")
        return
    successful_texts = []
    successful_embeddings = []
    for text in texts:
        try:
            approx_tokens = len(text.split())
            if approx_tokens > 3000:
                st.warning(f"Record is very long (approx {approx_tokens} tokens). Attempting to embed. Preview: {text[:200]}...")
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
            embedding = np.array(response.data[0].embedding, dtype='float32')
            successful_texts.append(text)
            successful_embeddings.append(embedding)
        except Exception as e:
            st.error(f"Error embedding text (length {len(text)} chars): {text[:200]}...\nError: {e}")
    for text, emb in zip(successful_texts, successful_embeddings):
        blob = emb.tobytes()
        cursor.execute(f"INSERT INTO {table_name} (json_data, embedding) VALUES (?, ?)", (text, blob))
    conn.commit()
    conn.close()
    st.success(f"Computed embeddings stored in SQLite table '{table_name}'.")

def create_vector_embedings(bucket_name):
        try:
            s3 = boto3.client("s3", aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key)
            objects = s3.list_objects_v2(Bucket=bucket_name).get("Contents", [])

            computed_stats_json.clear()
            for obj in objects:
                filename = obj["Key"]
                if not filename.lower().endswith(".csv"):
                    continue
                try:
                    s3.download_file(bucket_name, filename, filename)
                    df = pd.read_csv(filename)
                    if df.empty or df.columns.size == 0:
                        continue
                    df.columns = df.columns.str.strip().str.upper()
                    if "REPORT_DATE" in df.columns:
                        df["REPORT_DATE"] = pd.to_datetime(df["REPORT_DATE"], errors='coerce').dt.strftime('%Y-%m-%d')
                    # Compute aggregated statistics using selected numeric fields
                    stats = compute_statistics_and_save(df, filename, selected_fields=selected_fields_ui)
                    if stats:
                        computed_stats_json.extend(stats)
                    save_computed_data_to_sqlite(stats, table_name="computed_data")
                    # Only store embeddings for computed aggregated statistics (no raw data embeddings)
                    store_computed_embeddings(stats, table_name="computed_embeddings")
                except Exception as e:
                    st.error(f"❌ Error processing {filename}: {e}")
        except Exception as e:
            st.error(f"❌ S3 Processing Error: {e}")

def search_computed_data(query: str, top_k=2) -> List[Tuple[str, float]]:
    q_embed = client.embeddings.create(model=EMBEDDING_MODEL, input=[query]).data[0].embedding
    q_vec = np.array(q_embed, dtype='float32').reshape(1, -1)
    conn = sqlite3.connect(COMPUTED_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT json_data, embedding FROM computed_embeddings")
    rows = cursor.fetchall()
    conn.close()
    computed_texts = []
    computed_embeddings = []
    for row in rows:
        text = row[0]
        emb_array = np.frombuffer(row[1], dtype='float32')
        computed_texts.append(text)
        computed_embeddings.append(emb_array)
    if not computed_embeddings:
        st.error("No computed embeddings available for search. Please process some data first.")
        return []
    sims = cosine_similarity(q_vec, np.vstack(computed_embeddings))[0]
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [(computed_texts[i], sims[i]) for i in top_indices]

def trim_context(text: str, max_chars: int = 3000) -> str:
    if len(text) > max_chars:
        return text[:max_chars] + "\n...[TRUNCATED]"
    return text


def add_feedback_log(query, responce):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    user_reaction = 1

    cursor.execute('''
        INSERT INTO feedback_log (user_query,query_responce,user_reaction)
        VALUES (?, ?, ?)
    ''', ( query, responce, user_reaction ))

    log_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return log_id

def update_feedback_log(reaction, user_feedback, log_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    user_reaction = 1 if reaction == "👍 Yes" else 0

    cursor.execute('''
    UPDATE feedback_log
    SET user_reaction = ?, user_feedback = ?
    WHERE id = ?
    ''', (user_reaction,user_feedback, log_id))

    conn.commit()
    conn.close()

#######Graph Query for Co2, NG, PW ###############

#######Start here ##################

def plot_graph_based_on_prompt_all(prompt, category_key):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT file_name, json_contents FROM file_tracking WHERE is_processed = 1")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        st.warning("⚠️ No processed JSON data found in the database.")
        return

    categories = {
        "CO2": ['VOLUME', 'TRDVAL', 'MKTVAL', 'TRDPRC'],
        "Natural Gas": ['VOLUME', 'VOLUME_TOTAL', 'QTY_PHY', 'MKT_VAL', 'QTY_FIN', 'TRD_VAL'],
        "Power": [
            'VOLUME_BL', 'VOLUME_PK', 'VOLUME_OFPK',
            'MKT_VAL_BL', 'MKT_VAL_PK', 'MKT_VAL_OFPK',
            'TRD_VAL_BL', 'TRD_VAL_PK', 'TRD_VAL_OFPK'
        ]
    }

    if category_key not in categories:
        st.error("❌ Invalid category selected.")
        return

    fields_to_plot = categories[category_key]

    data = []
    for file_name, json_text in rows:
        if category_key.replace(" ", "").lower() not in file_name.replace("_", "").lower():
            continue
        try:
            json_data = json.loads(json_text)
            if "daily_totals" in json_data:
                data.extend(json_data["daily_totals"])
        except Exception as e:
            st.error(f"⚠️ Error reading {file_name}: {e}")

    if not data:
        st.error(f"❌ No valid 'daily_totals' data found for {category_key}.")
        return

    df_graph = pd.DataFrame(data)
    if 'REPORT_DATE' not in df_graph.columns:
        st.error("❌ REPORT_DATE column missing in JSON data.")
        return

    df_graph['REPORT_DATE'] = pd.to_datetime(df_graph['REPORT_DATE'])

    plt.figure(figsize=(14, 7))
    found_any = False
    for field in fields_to_plot:
        if field.lower() in prompt.lower() and field in df_graph.columns:
            plt.bar(df_graph['REPORT_DATE'], df_graph[field], label=field, alpha=0.6)
            found_any = True

    if not found_any:
        for field in fields_to_plot:
            if field in df_graph.columns:
                plt.bar(df_graph['REPORT_DATE'], df_graph[field], label=field, alpha=0.6)
                found_any = True

    if not found_any:
        st.error(f"❌ None of the expected fields were found for {category_key}.")
        return

    plt.xlabel('Report Date')
    plt.ylabel('Values')
    plt.title(f"{category_key} - Daily Aggregated Values")
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(plt)

####End here ####

def plot_graph_based_on_prompt(prompt):
    # Connect to SQLite
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT file_name, json_contents FROM file_tracking WHERE is_processed = 1")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        st.warning("⚠️ No processed JSON data found in the database.")
        return

    # Extract JSON data into a list
    data = []
    for file_name, json_text in rows:
        try:
            json_data = json.loads(json_text)

            # Extract only the 'daily_totals' section
            if "daily_totals" in json_data:
                data.extend(json_data["daily_totals"])
            else:
                st.error(f"❌ 'daily_totals' not found in {file_name}.")
                return
        except Exception as e:
            st.error(f"⚠️ Error reading JSON from {file_name}: {e}")
            return

    # Convert the extracted data to a DataFrame
    if not data:
        st.error("❌ No valid JSON data found.")
        return

    df_graph = pd.DataFrame(data)

    # Ensure REPORT_DATE exists and convert to datetime
    if 'REPORT_DATE' in df_graph.columns:
        df_graph['REPORT_DATE'] = pd.to_datetime(df_graph['REPORT_DATE'])
    else:
        st.error("❌ REPORT_DATE column not found in the JSON data.")
        return

    # Define valid columns
    valid_columns = {
        "volume": "TOTAL_VOLUME",
        "trdval": "TOTAL_TRDVAL",
        "mktval": "TOTAL_MKTVAL",
        "trdprc": "TOTAL_TRDPRC"
    }

    # Plot the requested column
    for key, column in valid_columns.items():
        if key in prompt.lower():
            if column not in df_graph.columns:
                st.error(f"❌ Column '{column}' not found in the JSON data.")
                return

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(df_graph['REPORT_DATE'], df_graph[column], color='blue')
            ax.set_xlabel('Report Date')
            ax.set_ylabel(column)
            ax.set_title(f'{column} Over Time')
            plt.xticks(rotation=45)

            # Display the graph in Streamlit
            st.pyplot(fig)
            return

    st.error("❌ Could not detect a valid column in your query. Try mentioning Volume, TRDVAL, MKTVAL, or TRDPRC.")



# -------------------------------
# Single Combined Bar Chart CO2
# -------------------------------
def plot_combined_graph_CO2():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT file_name, json_contents FROM file_tracking WHERE is_processed = 1 AND file_name LIKE 'NOP_CO2%'")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("⚠️ No CO2 files found in the tracking database.")
        return

    data = []
    for file_name, json_text in rows:
        try:
            json_data = json.loads(json_text)
            if "daily_totals" in json_data:
                data.extend(json_data["daily_totals"])
        except Exception as e:
            print(f"⚠️ Error reading {file_name}: {e}")

    if not data:
        print("❌ No valid 'daily_totals' found in CO2 files.")
        return

    df_graph = pd.DataFrame(data)

    if 'REPORT_DATE' not in df_graph.columns:
        print("❌ REPORT_DATE column missing in CO2 JSON data.")
        return

    df_graph['REPORT_DATE'] = pd.to_datetime(df_graph['REPORT_DATE'])

    plt.figure(figsize=(12, 6))
    plt.bar(df_graph['REPORT_DATE'], df_graph['TOTAL_VOLUME'], label='Total Volume', alpha=0.6)
    plt.bar(df_graph['REPORT_DATE'], df_graph['TOTAL_TRDVAL'], label='Total TRDVAL', alpha=0.6)
    plt.bar(df_graph['REPORT_DATE'], df_graph['TOTAL_MKTVAL'], label='Total MKTVAL', alpha=0.6)
    plt.bar(df_graph['REPORT_DATE'], df_graph['TOTAL_TRDPRC'], label='Total TRDPRC', alpha=0.6)
    plt.xlabel('Report Date')
    plt.ylabel('Values')
    plt.title('CO2: Comparison of Volume, TRDVAL, MKTVAL, and TRDPRC')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

# -------------------------------
# Combined Graph for NG Files
# -------------------------------

def plot_combined_graph_NG():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT file_name, json_contents FROM file_tracking WHERE is_processed = 1 AND file_name LIKE 'NOP_NG%'")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("⚠️ No NG files found in the tracking database.")
        return

    data = []
    for file_name, json_text in rows:
        try:
            json_data = json.loads(json_text)
            if "daily_totals" in json_data:
                data.extend(json_data["daily_totals"])
        except Exception as e:
            print(f"⚠️ Error reading {file_name}: {e}")

    if not data:
        print("❌ No valid 'daily_totals' found in NG files.")
        return

    df_graph = pd.DataFrame(data)

    if 'REPORT_DATE' not in df_graph.columns:
        print("❌ REPORT_DATE column missing in NG JSON data.")
        return

    df_graph['REPORT_DATE'] = pd.to_datetime(df_graph['REPORT_DATE'])

    plt.figure(figsize=(12, 6))
    fields = ['VOLUME', 'VOLUME_TOTAL', 'QTY_PHY', 'MKT_VAL', 'QTY_FIN', 'TRD_VAL']
    for field in fields:
        if field in df_graph.columns:
            plt.bar(df_graph['REPORT_DATE'], df_graph[field], label=field, alpha=0.6)

    plt.xlabel('Report Date')
    plt.ylabel('Values')
    plt.title('NG: Comparison of Volume, Qty, TRD_VAL, MKT_VAL')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()

# -------------------------------
# Single Combined Bar Chart PW
# -------------------------------
def plot_combined_graph_PW():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT file_name, json_contents FROM file_tracking WHERE is_processed = 1 AND file_name LIKE 'NOP_PW%'")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("⚠️ No PW files found in the tracking database.")
        return

    data = []
    for file_name, json_text in rows:
        try:
            json_data = json.loads(json_text)
            if "daily_totals" in json_data:
                data.extend(json_data["daily_totals"])
        except Exception as e:
            print(f"⚠️ Error reading {file_name}: {e}")

    if not data:
        print("❌ No valid 'daily_totals' found in PW files.")
        return

    df_graph = pd.DataFrame(data)

    if 'REPORT_DATE' not in df_graph.columns:
        print("❌ REPORT_DATE column missing in PW JSON data.")
        return

    df_graph['REPORT_DATE'] = pd.to_datetime(df_graph['REPORT_DATE'])

    plt.figure(figsize=(14, 7))
    fields = [
        'VOLUME_BL', 'VOLUME_PK', 'VOLUME_OFPK',
        'MKT_VAL_BL', 'MKT_VAL_PK', 'MKT_VAL_OFPK',
        'TRD_VAL_BL', 'TRD_VAL_PK', 'TRD_VAL_OFPK'
    ]
    for field in fields:
        if field in df_graph.columns:
            plt.bar(df_graph['REPORT_DATE'], df_graph[field], label=field, alpha=0.6)

    plt.xlabel('Report Date')
    plt.ylabel('Values')
    plt.title('PW: Comparison of Volume, MKT_VAL, TRD_VAL across types')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()





def get_feedback_logs():
    conn = sqlite3.connect(DB_NAME)
    query = """
    SELECT
        id,
        user_query,
        query_responce,
        user_reaction,
        user_feedback,
        created_at
    FROM feedback_log
    """
    feedback_df = pd.read_sql_query(query, conn)
    conn.close()

    # Map user_reaction to readable feedback
    feedback_df["User Feedback"] = feedback_df["user_reaction"].apply(lambda x: "👍 Yes" if x == 1 else "👎 No")
    feedback_df["Timestamp"] = pd.to_datetime(feedback_df["created_at"])
    return feedback_df

# --- Main App ---

# --- User Authentication ---
if not st.session_state['logged_in']:
  col1, col2, col3 = st.columns([1, 2, 1])

  with col2:

      # --- Logo Section ---
      col1, col2, col3 = st.columns(3)

      with col1:
            st.write(' ')

      with col2:
          st.image("https://img1.wsimg.com/isteam/ip/495c53b7-d765-476a-99bd-58ecde467494/blob-411e887.png/:/rs=w:127,h:95,cg:true,m/cr=w:127,h:95/qt=q:95", width=150)

      with col3:
            st.write(' ')


      st.title("🔐 Welcome to ETAI Energey Trading AI Platform")

      tab1, tab2 = st.tabs(["Login", "Register"])

      with tab1:
              st.header("🔑 Login")
              username = st.text_input("Username", key="Username", placeholder="Enter your username")
              password = st.text_input("Password", type="password", key="password", placeholder="Enter your password")

              if st.button("Login", use_container_width=True,key="login", help="Click to login"):
                  if authenticate_user(username, password):
                      st.success("✅ Login successful!")
                      st.session_state['logged_in'] = True
                      st.session_state['username'] = username
                      st.rerun()
                  else:
                      st.error("❌ Invalid username or password.")

      with tab2:
              st.header("📝 Register")
              new_username = st.text_input("New Username")
              new_password = st.text_input("New Password", type="password")

              if st.button("Register", use_container_width=True):
                  if register_user(new_username, new_password):
                      st.success("✅ User registered successfully!")
                  else:
                      st.error("❌ Username already exists. Choose a different one.")

# --- Dashboard ---
else:
    sidebar_menu()
    top_right_menu()

    # --- Main Content Section ---
    st.title(f"📊 {st.session_state.main_section}")

    # Display content dynamically based on selected section
    if st.session_state.sub_section == "Pipeline Dashboard":
        st.subheader("📈 Pipeline Dashboard")

    elif st.session_state.sub_section == "Data Pipeline":
        st.subheader("🔧 Data Pipeline -> Energy Training")
        st.write("Upload and manage your data files efficiently.")

        st.markdown("---")

        # --- Section 1: Raw Data Upload ---
        st.markdown("### 🗂️ CO2 Data Upload")

        col1, col2 = st.columns(2)

        with col1:
            metadata_file1 = st.file_uploader("Upload Schema File (CSV)", type=["csv"], key="metadata_file1")

        with col2:
            raw_files1 = st.file_uploader("Upload Raw Data (CSV)", type=["csv"], key="raw_files1", accept_multiple_files=True)

        if metadata_file1:
            try:
                metadata_df = pd.read_csv(BytesIO(metadata_file1.getvalue()))
                # Process metadata to create alias mapping; assumes metadata CSV has columns 'Column' and 'Description'
                alias_mapping = process_metadata_alias(metadata_df)
                st.success("✅ Metadata processed successfully!")
                st.write("Alias Mapping:")
                st.code("\n".join([f"{desc} -> {col}" for desc, col in alias_mapping.items()]))
            except Exception as e:
                st.error(f"❌ Error processing metadata: {e}")

        # Submit button logic
        if st.button("Submit", key="co2"):
            if metadata_file1 and raw_files1:
                try:
                  if metadata_file1 and raw_files1:
                      metadata_df = pd.read_csv(BytesIO(metadata_file1.getvalue()))
                      for f in raw_files1:
                          f.seek(0)
                          df = pd.read_csv(f)
                          file_name = f.name

                          if validate_against_metadata(df, metadata_df, file_name):
                              f.seek(0)  # Reset pointer before re-using for upload
                              success, msg = upload_to_s3(f, file_name, VALID_BUCKET)
                              st.success(f"✅ VALID: {file_name} | {msg}")
                          else:
                              f.seek(0)
                              success, msg = upload_to_s3(f, file_name, REJECTED_BUCKET)
                              st.error(f"❌ INVALID: {file_name} | {msg}")

                except Exception as e:
                    st.error(f"❌ Error during submission: {e}")
            else:
                st.warning("⚠️ Please upload metadata and data files!")

        st.markdown("---")

        # --- Section 2: Processed Data Upload ---
        st.markdown("### 🔥 Natural Gas Data Upload")

        col3, col4 = st.columns(2)

        with col3:
            metadata_file2 = st.file_uploader("Upload Schema File (CSV)", type=["csv"], key="metadata_file2")

        with col4:
            raw_files2 = st.file_uploader("Upload Raw Data (CSV)", type=["csv"], key="raw_files2", accept_multiple_files=True)

        if metadata_file2:
            try:
                metadata_df = pd.read_csv(BytesIO(metadata_file2.getvalue()))
                # Process metadata to create alias mapping; assumes metadata CSV has columns 'Column' and 'Description'
                alias_mapping = process_metadata_alias(metadata_df)
                st.success("✅ Metadata processed successfully!")
                st.write("Alias Mapping:")
                st.code("\n".join([f"{desc} -> {col}" for desc, col in alias_mapping.items()]))
            except Exception as e:
                st.error(f"❌ Error processing metadata: {e}")

        # Submit button logic
        if st.button("Submit", key="ng"):
            if metadata_file2 and raw_files2:
                try:
                  if metadata_file2 and raw_files2:
                      metadata_df = pd.read_csv(BytesIO(metadata_file2.getvalue()))
                      for f in raw_files2:
                          f.seek(0)
                          df = pd.read_csv(f)
                          file_name = f.name

                          if validate_against_metadata(df, metadata_df, file_name):
                              f.seek(0)  # Reset pointer before re-using for upload
                              success, msg = upload_to_s3(f, file_name, VALID_BUCKET)
                              st.success(f"✅ VALID: {file_name} | {msg}")
                          else:
                              f.seek(0)
                              success, msg = upload_to_s3(f, file_name, REJECTED_BUCKET)
                              st.error(f"❌ INVALID: {file_name} | {msg}")

                except Exception as e:
                    st.error(f"❌ Error during submission: {e}")
            else:
                st.warning("⚠️ Please upload metadata and data files!")

        st.markdown("---")

        # --- Section 3: Model Output Upload ---
        st.markdown("### 🚀 Power Data Upload")

        col5, col6 = st.columns(2)

        with col5:
            metadata_file3 = st.file_uploader("Upload Schema File (CSV)", type=["csv"], key="metadata_file3")


        with col6:
            raw_files3 = st.file_uploader("Upload Raw Data (CSV)", type=["csv"], key="raw_files3", accept_multiple_files=True)

        if metadata_file3:
            try:
                metadata_df = pd.read_csv(BytesIO(metadata_file3.getvalue()))
                # Process metadata to create alias mapping; assumes metadata CSV has columns 'Column' and 'Description'
                alias_mapping = process_metadata_alias(metadata_df)
                st.success("✅ Metadata processed successfully!")
                st.write("Alias Mapping:")
                st.code("\n".join([f"{desc} -> {col}" for desc, col in alias_mapping.items()]))
            except Exception as e:
                st.error(f"❌ Error processing metadata: {e}")

        # Submit button logic
        if st.button("Submit", key="po"):
            if metadata_file3 and raw_files3:
                try:
                  if metadata_file3 and raw_files3:
                      metadata_df = pd.read_csv(BytesIO(metadata_file3.getvalue()))
                      for f in raw_files3:
                          f.seek(0)
                          df = pd.read_csv(f)
                          file_name = f.name

                          if validate_against_metadata(df, metadata_df, file_name):
                              f.seek(0)  # Reset pointer before re-using for upload
                              success, msg = upload_to_s3(f, file_name, VALID_BUCKET)
                              st.success(f"✅ VALID: {file_name} | {msg}")
                          else:
                              f.seek(0)
                              success, msg = upload_to_s3(f, file_name, REJECTED_BUCKET)
                              st.error(f"❌ INVALID: {file_name} | {msg}")

                except Exception as e:
                    st.error(f"❌ Error during submission: {e}")
            else:
                st.warning("⚠️ Please upload metadata and data files!")

        st.markdown("---")

    elif st.session_state.sub_section == "Processed Data":
        st.subheader("📊 Processed Data")
        st.write("Access and analyze the processed data records.")

    elif st.session_state.sub_section == "Dashboard":
        st.subheader("📚 Dashboard")
        st.write("Manage vector embeddings and storage.")

    elif st.session_state.sub_section == "Configure & Upload":
        st.subheader("🏗️ Configure & Upload Data")

        prefix_fields = {
            "CO2": ['VOLUME', 'TRDVAL', 'MKTVAL', 'TRDPRC'],
            "Natural Gas": ['VOLUME',	'VOLUME_TOTAL',	'QTY_PHY','MKT_VAL' ,'QTY_FIN','TRD_VAL'],
            "Power": ['VOLUME_BL',	'VOLUME_PK',	'VOLUME_OFPK',	'MKT_VAL_BL',	'MKT_VAL_PK',	'MKT_VAL_OFPK',	'TRD_VAL_BL',	'TRD_VAL_PK',	'TRD_VAL_OFPK']
        }

        name = st.text_input("RAG  Agent Name")
        col1, col2 = st.columns(2)
        with col1:
          bucket = st.selectbox("Select Bucket Name(S3)", ["etrm-etai-poc", "etrm-etai-poc-ng"])
        with col2:
          prefix = st.selectbox("Select Prefix", list(prefix_fields.keys()), index=0)

        col3, col4 = st.columns(2)
        with col3:
            model = st.selectbox("Model", ["OpenAI GPT-3.5", "OpenAI GPT-4", "Llama 2", "Claude 3.5", "Claude 4" ,"Custom Model"])
        with col4:
            temp = st.slider("Temperature (Creativity)", 0.0, 1.0, 0.7, 0.1)

        col5, col6 = st.columns(2)
        with col5:
             metadata_file = st.file_uploader("Upload Data Dictionary (CSV)", type=["csv"])

        with col6:
             uploaded_file = st.file_uploader("Upload Transaction Log (TXT, PDF, CSV, DOCX)", type=["txt", "pdf", "csv", "docx"])

        prompt = st.text_area("📝 Provide Prompt Instructions", key='prompt')

        if st.button("Submit & Process Data"):

            if prefix == 'CO2':
                prefix_value = "CO2"
            elif prefix == 'Natural Gas':
                prefix_value = "NG"
            elif prefix == 'Power':
                prefix_value = "PW"
            else:
                prefix_value = "misc"
            process_files_from_s3_folder(VALID_BUCKET, prefix_value)
            add_agent_detail(name, model, temp, prompt)
            #create_vector_embedings(bucket)


    elif st.session_state.sub_section == "Fine Tuning":
        st.subheader("📄 Fine Tuning")

    elif st.session_state.sub_section == "Settings":
        st.subheader("🚀 Settings")
        st.write("Get insights into your application's performance.")

    elif st.session_state.sub_section == "Energy Tradeing Analysis":

        if "query_answer" not in st.session_state:
            st.session_state["query_answer"] = None
        if "feedback_submitted" not in st.session_state:
            st.session_state["feedback_submitted"] = False

        st.subheader("📊 Energy Tradeing Analysis")
        st.write("💬 Ask Your Financial or Commodity Question")
        user_query = st.text_input("Example: What is the total Price Value on 13 Nov 2024?")
        query_answer= None
        if st.button("Submit Query") and user_query:
          with st.spinner("Thinking..."):
              query_answer = query_sqlite_json_with_openai(user_query)

              st.success(query_answer)
              log_id = add_feedback_log(user_query, query_answer)
              # Store the answer in session state to persist it
              st.session_state["query_answer"] = query_answer
              st.session_state["log_id"] = log_id
              st.session_state["feedback_submitted"] = False

      # Display feedback section only if query is answered
        if st.session_state["query_answer"]:
            feedback = st.radio("Did the AI answer your question correctly?:", ["👍 Yes", "👎 No"], key="feedback_radio", horizontal=True)
            feedback_comment = st.text_area("Additional comments or corrections:", height=100, key="feedback_comment")

            # Feedback submission
            if st.button("Submit Feedback"):
                if not feedback_comment.strip():
                    st.warning("Please add comments or corrections before submitting.")
                else:
                    # Avoid multiple submissions with session state
                    if not st.session_state["feedback_submitted"]:

                        update_feedback_log(feedback, feedback_comment, st.session_state["log_id"])
                        st.session_state["feedback_submitted"] = True
                        st.success("✅ Feedback submitted successfully!")
                    else:
                        st.info("Feedback already submitted.")

    elif st.session_state.sub_section == "Graph Query":
        st.subheader("📊 Graph Query")
        category_selected = st.selectbox("Select Category", ["CO2", "Natural Gas", "Power"])
        graph_query_input = st.text_input('Ask for a bar graph (e.g., "Show bar graph for TRDVAL")')
        if st.button('Generate Custom Graph'):
            plot_graph_based_on_prompt_all(graph_query_input, category_selected)

        if st.button('Combined CO2 Graph'):
            plot_combined_graph_CO2(graph_query_input)
        if st.button('Combined NG Graph'):
            plot_graph_based_on_prompt(graph_query_input)
        if st.button('Combined PW Graph'):
            plot_graph_based_on_prompt(graph_query_input)

    elif st.session_state.sub_section == "Deviation Analysis":
        st.subheader("📊 Deviation Analysis")

    elif st.session_state.sub_section == "Root Cause Analysis":
        st.subheader("⚠️Root Cause Analysis")
        st.write("Review and troubleshoot errors.")

    elif st.session_state.sub_section == "Root Cause Analysis":
          st.header("📊 Analysis History & Insights")

    elif st.session_state.sub_section == "User Feedback":
        st.subheader("📝 User Feedback Dashboard")
        feedback_df = get_feedback_logs()

        if not feedback_df.empty:
            st.subheader("📋 Collected Feedback")

            # Display relevant fields
            display_df = feedback_df[["id", "user_query", "query_responce", "User Feedback", "user_feedback", "Timestamp"]]
            st.dataframe(display_df, use_container_width=True)

            # Feedback summary
            st.subheader("📊 Feedback Summary")
            positive_feedback_count = (feedback_df["user_reaction"] == 1).sum()
            negative_feedback_count = (feedback_df["user_reaction"] == 0).sum()

            st.write(f"✅ **Positive Feedback:** {positive_feedback_count}")
            st.write(f"❌ **Negative Feedback:** {negative_feedback_count}")

            # Feedback trends over time
            st.subheader("📊 Feedback Trends Over Time")

            feedback_over_time = (
                feedback_df.groupby(feedback_df["Timestamp"].dt.date)["User Feedback"]
                .value_counts()
                .unstack()
                .fillna(0)
            )

            fig, ax = plt.subplots(figsize=(10, 6))
            feedback_over_time.plot(kind="bar", ax=ax, stacked=True)
            ax.set_xlabel("Date")
            ax.set_ylabel("Number of Feedbacks")
            ax.set_title("User Feedback Trends")
            plt.xticks(rotation=45)

            st.pyplot(fig)

            # Download CSV button
            csv_feedback = display_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Feedback as CSV", csv_feedback, "user_feedback.csv", "text/csv", key="download-feedback")

        else:
            st.warning("No feedback logs found in the database.")
