import re
import os
import ast
import json
import time
import pandas as pd
import streamlit as st
import warnings
import pdfplumber
from IPython.display import display, Markdown
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings("ignore")



#   Load environement variables and set up LLM
gemini_api_key = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=gemini_api_key)

#   Extracting text from pdf (without LLM Use)
def extract_text_and_tables_from_pdf(pdf_path):
    """Extract text and tables from a PDF file in a structured way."""
    text = ""
    extracted_tables = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            if '2' in str(page) or '3' in str(page) or '6' in str(page) or '7' in str(page):
                text += page.extract_text() + "\n"
                tables = page.extract_tables()
                for table in tables:
                    rows = []
                    for row in table:
                        for i in range(len(row)):
                            if row[i] == None:
                                row[i] = ""
                        rows.append(row)
                    structured_table = "\n".join([" | ".join(row) for row in rows])
                    extracted_tables.append(structured_table)
            else:
                continue
    return text + "\n\nTables:\n" + "\n\n".join(extracted_tables)


#   Extracting Rules from extracted text (Using LLM) – we can also change rules at this stage
def extract_rules_with_llm(document_text):
    """Use LLM to dynamically extract underwriting rules."""
    prompt = f"""
    Extract all underwriting rules from the following document:
    {document_text}
    Ensure tables and numeric values are preserved accurately.
    Provide structured eligibility rules for No Medical, Video Medical, and Physical Medical categories,
    along with calculations for approved sum assured.
    """
    return llm.invoke(prompt)


#   Saving Extracted Rules to local
def save_rules(rules):
    with open("Extracted_NML_Rules.json", "w") as file:
        json.dump({"Rules": rules}, file)


#   Loading saved rules

def load_rules():
    with open("Extracted_NML_Rules.json", "r") as file:
        extracted_rules = json.load(file)
        return extracted_rules

#   Loading customer data
def extract_user_data(excel_file):
    df = pd.read_excel(excel_file, sheet_name = "Data")
    return df

#   Defining functions to get the NML Decision and final category with limits
def get_nml_decision(customer_data, nml_rules):
    """Use LLM to determine NML category and sum assured."""
    prompt = f"""
    Given the underwriting rules:
    {nml_rules['Rules']}
    And the following customer details:
    {customer_data}
    Determine:
    1. Whether the customer falls under No Medical, Video Medical, or Physical Medical.
    2. The approved sum assured, no medical limit, vmer limit, medical tests (if requried) based on underwriting criteria.
    3. Short summary or conclusion
    """
    return llm.invoke(prompt)


def final_decision(underwriting_decision):
    """Use LLM to determine NML category and sum assured."""
    prompt = f"""
    Extract information from the given underwriting decision.

    Underwriting Decision: {underwriting_decision}

    Response Format:
    {{
        "Category": "No Medical, VMER, Physical Medical (return only 1 category that is applicable)"
        "MSUC": "extracted MSUC (return value only without any additional text or currency) else 0"
        "No_Medical_Limit": "extracted no medical limit (return value only without any additional text or currency) else 0"
        "VMER_Limit": "extracted vmer limit (return value only without any additional text or currency) if available else 0"
        "Medical_Grid": "extracted medical tests (return tests without any additional text) if available else na"
        "Conclusion": "extracted final conclusion or summary if available else na"
    }}
    
    Response:"""
    return llm.invoke(prompt)


#   Looping through the customer data and calling functions for NML Decision
def run_main(df, nml_rules):
    for i in df.index:
        customer_df = pd.DataFrame(df.loc[i, :])
        customer_data = {str(j) : str(customer_df.loc[j, i]).strip() for j in customer_df.index}
        nml_decision = get_nml_decision(customer_data, nml_rules)
        final_nml_decision = final_decision(nml_decision.content.strip())
        cleaned_response = re.sub(r"```[a-zA-Z]*\n|\n```", "", final_nml_decision.content.strip())
        
        try:
            try:
                response = json.loads(cleaned_response)
            
            except json.JSONDecodeError:
                response = ast.literal_eval(cleaned_response)

            category = response['Category']
            nml = response['No_Medical_Limit']
            vmer = response['VMER_Limit']
            MSUC = response['MSUC']
            medical_grid = response['Medical_Grid']
            conclusion = response['Conclusion']
            full_analysis = nml_decision.content.strip()
            cal_category = 'No Medical' if float(MSUC) <= float(nml) else 'VMER' if float(MSUC) <= float(vmer) else 'Physical Medical'
            df.loc[i, 'Medical_Cal_Category_genai'] = cal_category
            df.loc[i, 'Medical_Category_genai'] = category
            df.loc[i, 'MSUC'] = MSUC
            df.loc[i, 'No_Medical_Limit_genai'] = nml
            df.loc[i, 'VMER_Limit_genai'] = vmer
            df.loc[i, 'Medical_Grid_genai'] = medical_grid
            df.loc[i, 'Conclusion_genai'] = conclusion
            df.loc[i, 'Full_Analysis'] = full_analysis

        except Exception as e:
            raise e
    return df




# Adjust batch size here
batch_size = 1

# Load rules
nml_rules = load_rules()

st.title("NML Data Processing")

# Ensure session state variables exist
if 'uploaded_file' not in st.session_state:
    st.session_state['uploaded_file'] = None
if 'processed_data' not in st.session_state:
    st.session_state['processed_data'] = None
if 'processing_complete' not in st.session_state:
    st.session_state['processing_complete'] = False
if 'download_ready' not in st.session_state:
    st.session_state['download_ready'] = False  # To prevent reprocessing on download

# File uploader
uploaded_file = st.file_uploader("### Upload an Excel file", type=["xlsx", "xls", "pdf"])

def process_data():
    """Processes the uploaded file in batches and stores results in session state."""
    try:
        df = extract_user_data(st.session_state['uploaded_file'])

        # Display the dataframe
        st.write("### File Preview:")
        st.dataframe(df)

        total_rows = len(df)
        processed_data = pd.DataFrame()

        for start in range(0, total_rows, batch_size):
            end = min(start + batch_size, total_rows)
            batch_df = df.iloc[start:end]

            # Time the processing
            batch_start_time = time.time()
            processed_batch = run_main(batch_df, nml_rules)
            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time

            processed_data = pd.concat([processed_data, processed_batch], ignore_index=True)

            # Display processed batch and time taken
            st.write(f"### Processed Data (Rows {start + 1} to {end}):")
            st.dataframe(processed_batch)
            st.success(f"✅ Batch processed in {batch_duration:.2f} seconds.")

            # Countdown for the sleep time
            time.sleep(10)

        # Store processed data
        st.session_state['processed_data'] = processed_data
        st.session_state['processing_complete'] = True
        st.session_state['download_ready'] = True  # Mark as ready for download

    except Exception as e:
        st.error(f"Error processing file: {e}")

if uploaded_file is not None and uploaded_file != st.session_state['uploaded_file']:
    st.session_state['uploaded_file'] = uploaded_file
    st.session_state['processing_complete'] = False
    st.session_state['download_ready'] = False
    process_data()

if st.session_state['processing_complete'] and st.session_state['processed_data'] is not None:
    st.write("### Processed Data:")
    st.dataframe(st.session_state['processed_data'])

    if st.session_state['download_ready']:
        output_file = "NML_Output.xlsx"
        st.session_state['processed_data'].to_excel(output_file, index=False)

        with open(output_file, "rb") as f:
            st.download_button(
                label="Download Processed Excel File",
                data=f,
                file_name=output_file,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
