import pandas as pd
import streamlit as st
import time
import io


def determine_underwriting_requirements(customer_data):
    """ Determines the NML category, no medical limit, VMER limit, and required medical tests based on the provided underwriting rules and customer data.

    Args:
        customer_data (dict): A dictionary containing the customer's details, including:
            'Age' (int): Applicant's age.
            'Gender' (str): Applicant's gender ('M' or 'F').  (Not used in the current implementation)
            'Income' (float): Applicant's annual income.
            'MSUC' (float): Desired Sum Assured (MSUC).
            'Occupation' (str): Applicant's occupation.
            'Qualification' (str): Applicant's qualification.
            'Premium' (float): The premium amount.
            'SUC' (float): Sum Assured.

    Returns:
        dict: A dictionary containing the underwriting results:
            'category' (str): The applicant's NML category (Platinum, Gold, Silver, Bronze, or N/A).
            'nml_limit' (float): The Non-Medical Limit.
            'vmer_limit' (float): The Video Medical Examination Required Limit.
            'required_tests' (str or None): A comma-separated string of required medical tests, or None if no tests are required.
    """

    # Extract customer data from the input dictionary
    age = customer_data['Age']
    income = customer_data['Income']
    msuc = customer_data['MSUC']
    occupation = customer_data['Occupation']
    qualification = customer_data['Qualification']
    premium = customer_data['Premium']
    suc = customer_data['SUC']


    # Define NML/Video Limits Chart 1: This chart defines the base NML and VMER limits based on age.
    nml_limits_chart_1 = {
        (0, 17): {'nml': 5000000, 'vmer': 7500000},
        (18, 35): {'nml': 3500000, 'vmer': 5000000},
        (36, 40): {'nml': 2000000, 'vmer': 3500000},
        (41, 45): {'nml': 2000000, 'vmer': 3500000},
        (46, 50): {'nml': 1250000, 'vmer': 2500000},
        (51, 55): {'nml': 1250000, 'vmer': 1500000},
        (56, 60): {'nml': 500000, 'vmer': 0}
    }

    # Define Category Definitions: This function determines the customer's category (Platinum, Gold, etc.) based on income and qualification.
    def determine_category(income, qualification):
        if qualification == 'Graduate':
            if income >= 1500000:
                return 'Platinum'
            elif income >= 1000000:
                return 'Gold'
            elif income >= 800000:
                return 'Silver'
            elif income >= 600000:
                return 'Bronze'
        return 'N/A'

    # Define VMER Limits Chart 2 & 3: These charts define VMER limits based on age and category. Chart 2 is used when the SUC/Premium ratio is <= 20, Chart 3 when it's between 20 and 40
    vmer_limits_chart_2 = {
        (0, 17): {'Platinum': 20000000, 'Gold': 20000000, 'Silver': 20000000, 'Bronze': 20000000},
        (18, 35): {'Platinum': 40000000, 'Gold': 25000000, 'Silver': 12500000, 'Bronze': 7500000},
        (36, 40): {'Platinum': 30000000, 'Gold': 15000000, 'Silver': 7500000, 'Bronze': 5000000},
        (41, 45): {'Platinum': 30000000, 'Gold': 15000000, 'Silver': 7500000, 'Bronze': 5000000},
        (46, 50): {'Platinum': 20000000, 'Gold': 10000000, 'Silver': 6000000, 'Bronze': 3000000},
        (51, 55): {'Platinum': 5000000, 'Gold': 3000000, 'Silver': 2000000, 'Bronze': 2000000},
        (56, 60): {'Platinum': 2500000, 'Gold': 1500000, 'Silver': 500000, 'Bronze': 500000}
    }

    vmer_limits_chart_3 = {
        (0, 17): {'Platinum': 10000000, 'Gold': 10000000, 'Silver': 10000000, 'Bronze': 10000000},
        (18, 35): {'Platinum': 20000000, 'Gold': 12500000, 'Silver': 6000000, 'Bronze': 3750000},
        (36, 40): {'Platinum': 15000000, 'Gold': 7500000, 'Silver': 3750000, 'Bronze': 2500000},
        (41, 45): {'Platinum': 15000000, 'Gold': 7500000, 'Silver': 3750000, 'Bronze': 2500000},
        (46, 50): {'Platinum': 10000000, 'Gold': 5000000, 'Silver': 3000000, 'Bronze': 1500000},
        (51, 55): {'Platinum': 2500000, 'Gold': 1500000, 'Silver': 1250000, 'Bronze': 1250000},
        (56, 60): {'Platinum': 1250000, 'Gold': 750000, 'Silver': 500000, 'Bronze': 500000}
    }

    # Define Medical Underwriting Requirements: This chart defines the required medical tests based on age and MSUC.
    medical_tests_chart = {
        (0, 13): {
            (0, 100000): 'FMQ',
            (100000.01, 200000): 'FMQ',
            (200000.01, 300000): 'FMQ',
            (300000.01, 400000): 'FMQ',
            (400000.01, 500000): 'FMQ',
            (500000.01, 600000): 'FMQ',
            (600000.01, 1000000): 'FMQ',
            (1000000.01, 1500000): 'FMQ',
            (1500000.01, 2000000): 'FMQ',
            (2000000.01, 2500000): 'FMQ',
            (2500000.01, 3500000): 'FMQ',
            (3500000.01, 5000000): 'FMQ',
            (5000000.01, 7500000): 'VMER',
            (7500000.01, 10000000): 'JMER',
            (10000000.01, 20000000): 'JMER,RUA,BP,CBC',
            (20000000.01, 250000000): 'JMER,RUA,BP,CBC',
            (250000000.01, float('inf')): 'MER, RUA, BP, CBC, HbA1c, TMT, CXR, USG, 2D Echo, UMA,AHCV, PFT'
        },
        (14, 17): {
            (0, 100000): 'FMQ',
            (100000.01, 200000): 'FMQ',
            (200000.01, 300000): 'FMQ',
            (300000.01, 400000): 'FMQ',
            (400000.01, 500000): 'FMQ',
            (500000.01, 600000): 'FMQ',
            (600000.01, 1000000): 'FMQ',
            (1000000.01, 1500000): 'FMQ',
            (1500000.01, 2000000): 'FMQ',
            (2000000.01, 2500000): 'FMQ',
            (2500000.01, 3500000): 'FMQ',
            (3500000.01, 5000000): 'FMQ',
            (5000000.01, 7500000): 'VMER',
            (7500000.01, 10000000): 'JMER',
            (10000000.01, 20000000): 'JMER,RUA,BP,CBC',
            (20000000.01, 250000000): 'JMER,RUA,BP,CBC',
            (250000000.01, float('inf')): 'MER, RUA, BP, CBC, HbA1c, TMT, CXR, USG, 2D Echo, UMA,AHCV, PFT'
        },
        (18, 35): {
            (0, 100000): 'FMQ',
            (100000.01, 200000): 'FMQ',
            (200000.01, 300000): 'FMQ',
            (300000.01, 400000): 'FMQ',
            (400000.01, 500000): 'FMQ',
            (500000.01, 600000): 'FMQ',
            (600000.01, 1000000): 'FMQ',
            (1000000.01, 1500000): 'FMQ',
            (1500000.01, 2000000): 'FMQ',
            (2000000.01, 2500000): 'FMQ',
            (2500000.01, 3500000): 'FMQ',
            (3500000.01, 5000000): 'VMER',
            (5000000.01, 7500000): 'MER, RUA, BP, ECG',
            (7500000.01, 10000000): 'MER, RUA, BP, ECG',
            (10000000.01, 20000000): 'MER, RUA, BP, CBC, ECG',
            (20000000.01, 250000000): 'MER, RUA, BP, CBC, ECG',
            (250000000.01, float('inf')): 'MER, RUA, BP, CBC, HbA1c, TMT, CXR, USG, 2D Echo, UMA,AHCV, PFT'
        },
        (36, 45): {
            (0, 100000): 'FMQ',
            (100000.01, 200000): 'FMQ',
            (200000.01, 300000): 'FMQ',
            (300000.01, 400000): 'FMQ',
            (400000.01, 500000): 'FMQ',
            (500000.01, 600000): 'FMQ',
            (600000.01, 1000000): 'FMQ',
            (1000000.01, 1500000): 'FMQ',
            (1500000.01, 2000000): 'FMQ',
            (2000000.01, 2500000): 'VMER',
            (2500000.01, 3500000): 'VMER',
            (3500000.01, 5000000): 'MER, RUA, BP, ECG',
            (5000000.01, 7500000): 'MER, RUA, BP, CBC ECG, Hba1c',
            (7500000.01, 10000000): 'MER, RUA, BP, CBC ECG, Hba1c',
            (10000000.01, 20000000): 'MER, RUA, BP, CBC ECG, Hba1c',
            (20000000.01, 250000000): 'MER, RUA, BP, TMT, Hba1c, CBC',
            (250000000.01, float('inf')): 'MER, RUA, BP, CBC, HbA1c, TMT, CXR, USG, 2D Echo, UMA,AHCV, PFT'
        },
        (46, 50): {
            (0, 100000): 'FMQ',
            (100000.01, 200000): 'FMQ',
            (200000.01, 300000): 'FMQ',
            (300000.01, 400000): 'FMQ',
            (400000.01, 500000): 'FMQ',
            (500000.01, 600000): 'FMQ',
            (600000.01, 1000000): 'FMQ',
            (1000000.01, 1500000): 'VMER',
            (1500000.01, 2000000): 'VMER',
            (2000000.01, 2500000): 'VMER',
            (2500000.01, 3500000): 'MER, RUA, BP, CBC, ECG, Hba1c',
            (3500000.01, 5000000): 'MER, RUA, BP, CBC, ECG, Hba1c',
            (5000000.01, 7500000): 'MER, RUA, BP, CBC, TMT, Hba1c',
            (7500000.01, 10000000): 'MER, RUA, BP, CBC, TMT, Hba1c',
            (10000000.01, 20000000): 'MER, RUA, BP, CBC, TMT, HBA1c',
            (20000000.01, 250000000): 'MER, RUA, BP, CBC, TMT, HBA1c',
            (250000000.01, float('inf')): 'MER, RUA, BP, CBC, HbA1c, TMT, CXR, USG, 2D Echo, UMA,AHCV, PFT'
        },
        (51, 55): {
            (0, 100000): 'FMQ',
            (100000.01, 200000): 'FMQ',
            (200000.01, 300000): 'FMQ',
            (300000.01, 400000): 'FMQ',
            (400000.01, 500000): 'FMQ',
            (500000.01, 600000): 'FMQ',
            (600000.01, 1000000): 'VMER',
            (1000000.01, 1500000): 'VMER',
            (1500000.01, 2000000): 'MER, RUA, BP, CBC, ECG, Hba1c',
            (2000000.01, 2500000): 'MER, RUA, BP, CBC, ECG, Hba1c',
            (2500000.01, 3500000): 'MER, RUA, BP, CBC, ECG, Hba1c',
            (3500000.01, 5000000): 'MER, RUA, BP, CBC, ECG, Hba1c',
            (5000000.01, 7500000): 'MER, RUA, BP,CBC, TMT, Hba1c',
            (7500000.01, 10000000): 'MER, RUA, BP,CBC, TMT, Hba1c',
            (10000000.01, 20000000): 'MER, RUA, BP,CBC, TMT, Hba1c',
            (20000000.01, 250000000): 'MER, RUA, BP,CBC, TMT, Hba1c',
            (250000000.01, float('inf')): 'MER, RUA, BP, CBC, HbA1c, TMT, CXR, USG, 2D Echo, UMA,AHCV, PFT'
        },
        (56, 60): {
            (0, 100000): 'FMQ',
            (100000.01, 200000): 'FMQ',
            (200000.01, 300000): 'FMQ',
            (300000.01, 400000): 'FMQ',
            (400000.01, 500000): 'FMQ',
            (500000.01, 600000): 'MER, RUA, BP, ECG, CBC',
            (600000.01, 1000000): 'MER, RUA, BP, ECG, CBC',
            (1000000.01, 1500000): 'MER, RUA, BP, ECG, CBC',
            (1500000.01, 2000000): 'MER, RUA, BP, CBC, ECG, Hba1c',
            (2000000.01, 2500000): 'MER, RUA, BP, CBC, ECG, Hba1c',
            (2500000.01, 3500000): 'MER, RUA, BP, CBC, ECG, Hba1c',
            (3500000.01, 5000000): 'MER,RUA,BP, CBC,TMT,Hba 1c',
            (5000000.01, 7500000): 'MER,RUA,BP, CBC,TMT,Hba 1c',
            (7500000.01, 10000000): 'MER,RUA,BP, CBC,TMT,Hba 1c',
            (10000000.01, 20000000): 'MER,RUA,BP, CBC,TMT,Hba 1c',
            (20000000.01, 250000000): 'MER,RUA,BP, CBC,TMT,Hba 1c',
            (250000000.01, float('inf')): 'MER, RUA, BP, CBC, HbA1c, TMT, CXR, USG, 2D Echo, UMA,AHCV, PFT, PSA'
        },
        (61, 65): {
            (0, 100000): 'MER',
            (100000.01, 200000): 'MER, ECG, FBS',
            (200000.01, 300000): 'MER, ECG, FBS',
            (300000.01, 400000): 'MER, BP, ECG',
            (400000.01, 500000): 'MER, BP, ECG',
            (500000.01, 600000): 'MER, RUA, BP, ECG, CBC',
            (600000.01, 1000000): 'MER, RUA, BP, ECG, CBC',
            (1000000.01, 1500000): 'MER, RUA, BP, ECG, CBC',
            (1500000.01, 2000000): 'MER, RUA, BP, CBC, TMT, Hba1c',
            (2000000.01, 2500000): 'MER, RUA, BP, CBC, TMT, Hba1c',
            (2500000.01, 3500000): 'MER, RUA, BP, CBC, TMT, Hba1c',
            (3500000.01, 5000000): 'MER,RUA,BP, CBC,TMT,Hba1c',
            (5000000.01, 7500000): 'MER,RUA,BP, CBC,TMT,Hba1c',
            (7500000.01, 10000000): 'MER,RUA,BP, CBC,TMT,Hba1c',
            (10000000.01, 20000000): 'MER,RUA,BP, CBC,TMT,Hba1c',
            (20000000.01, 250000000): 'MER,RUA,BP, CBC,TMT,Hba1c',
            (250000000.01, float('inf')): 'MER, RUA, BP, CBC, HbA1c, TMT, CXR, USG, 2D Echo, UMA,AHCV, PFT, PSA'
        }
    }

    # 1. Determine Age Band: Find the age band that the customer falls into based on nml_limits_chart_1.
    age_band = None
    for (lower, upper) in nml_limits_chart_1:
        if lower <= age <= upper:
            age_band = (lower, upper)
            break

    # If the age is not within the defined ranges, return N/A values.
    if age_band is None:
        return {'category': 'N/A', 'nml_limit': None, 'vmer_limit': None, 'required_tests': None}

    # 2. Apply Non-Medical Limits from Chart 1: Get the NML and VMER limits based on the determined age band.
    nml_limit = nml_limits_chart_1[age_band]['nml']
    vmer_limit_chart_1 = nml_limits_chart_1[age_band]['vmer']

    #3. Check if MSUC falls within Chart 1 Limits: If MSUC is less than or equal to the NML limit, no medicals are required.
    if msuc <= nml_limit:
        return {'category': 'N/A', 'nml_limit': nml_limit, 'vmer_limit': vmer_limit_chart_1, 'required_tests': None}

    # 4. Determine Category: Determine the customer's category based on income and qualification.
    category = determine_category(income, qualification)

    # Calculate the ratio of Sum Assured to Premium
    suc_to_premium_ratio = suc / premium

    # 5. Check Exclusions: Determine the VMER limit based on occupation, category and SUC/Premium ratio.
    if occupation in ['Housewife', 'Agriculturist']:
        vmer_limit = vmer_limit_chart_1
    elif category == "N/A":
        vmer_limit = vmer_limit_chart_1
    elif suc_to_premium_ratio <= 20:
        vmer_limit = vmer_limits_chart_2[age_band][category]
    elif 20 < suc_to_premium_ratio <= 40:
        vmer_limit = vmer_limits_chart_3[age_band][category]
    else:
        vmer_limit = vmer_limit_chart_1

    # 8. Determine Required Medical Tests: Determine the required medical tests based on age and MSUC if MSUC exceeds VMER limit.
    if msuc > vmer_limit:
        # Determine Age bracket for medical tests
        age_bracket = None
        for (lower, upper) in medical_tests_chart:
            if lower <= age <= upper:
                age_bracket = (lower, upper)
                break

        if age_bracket:
            for (lower_msuc, upper_msuc), tests in medical_tests_chart[age_bracket].items():
                if lower_msuc <= msuc <= upper_msuc:
                    required_tests = tests
                    break
            else:
                required_tests = None  # Changed to None since it's possible to fall outside the MSUC ranges
        else:
            required_tests = "Age not within range"
    else:
        required_tests = None

    # Return the underwriting results. The vmer_limit is the max of the calculated vmer_limit and the base vmer_limit_chart_1.
    return {'category': category, 'nml_limit': nml_limit, 'vmer_limit': max(vmer_limit, vmer_limit_chart_1), 'required_tests': required_tests}      # max applied



def process_uploaded_file(uploaded_file, batch_size=5):
    df = pd.read_excel(uploaded_file, sheet_name="Sheet1")
    total_rows = len(df)
    processed_batches = []  # To store each processed batch for preview and download

    for start in range(0, total_rows, batch_size):
        end = min(start + batch_size, total_rows)
        batch_df = df.iloc[start:end].copy()  # Copy to avoid modifying original slice

        st.write(f"🔄 Processing rows {start + 1} to {end}...")

        for i in batch_df.index:
            try:
                customer_data = {
                    'Age': float(df.loc[i, 'LA Age at Entry']),
                    'Income': float(df.loc[i, 'LA ANNUAL INCOME']),
                    'MSUC': float(df.loc[i, 'MSUC']),
                    'Occupation': str(df.loc[i, 'LA Occupation']).strip(),
                    'Qualification': str(df.loc[i, 'LA Qualification']).strip(),
                    'Premium': float(df.loc[i, 'Premium']),
                    'SUC': float(df.loc[i, 'Sum Assured'])
                }
            except KeyError as e:
                st.warning(f"Row {i + 1}: Missing column {e}. Skipping.")
                continue

            try:
                nml_output = determine_underwriting_requirements(customer_data)

                nml_limit = float(nml_output['nml_limit']) if nml_output['nml_limit'] is not None else -1
                vmer_limit = float(nml_output['vmer_limit']) if nml_output['vmer_limit'] is not None else -1
                msuc = customer_data['MSUC']

                category = (
                    'No Medical' if nml_limit >= msuc else
                    'VMER' if vmer_limit >= msuc else
                    'Physical Medical'
                )

                df.loc[i, 'computer_category'] = category
                df.loc[i, 'computer_nml_limit'] = nml_limit
                df.loc[i, 'computer_vmer_limit'] = vmer_limit
                df.loc[i, 'computer_required_tests'] = nml_output['required_tests']

                batch_df.loc[i, 'computer_category'] = category
                batch_df.loc[i, 'computer_nml_limit'] = nml_limit
                batch_df.loc[i, 'computer_vmer_limit'] = vmer_limit
                batch_df.loc[i, 'computer_required_tests'] = nml_output['required_tests']

            except Exception as e:
                st.error(f"❌ Error processing row {i + 1}: {e}")
                df.loc[i, 'computer_category'] = 'Error'
                df.loc[i, 'computer_nml_limit'] = -1
                df.loc[i, 'computer_vmer_limit'] = -1
                df.loc[i, 'computer_required_tests'] = 'Error'

                batch_df.loc[i, 'computer_category'] = 'Error'
                batch_df.loc[i, 'computer_nml_limit'] = -1
                batch_df.loc[i, 'computer_vmer_limit'] = -1
                batch_df.loc[i, 'computer_required_tests'] = 'Error'

        st.success(f"✅ Batch {start + 1} to {end} processed.")

        # Show batch preview
        st.write(f"### Preview: Batch {start + 1} to {end}")
        st.dataframe(batch_df)

        # Save individual batch to memory
        batch_output = io.BytesIO()
        batch_df.to_excel(batch_output, index=False, engine='openpyxl')
        batch_output.seek(0)

        st.download_button(
            label=f"📥 Download Batch {start + 1} to {end}",
            data=batch_output,
            file_name=f"batch_{start+1}_to_{end}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        processed_batches.append(batch_df)

        if end < total_rows:
            with st.spinner("Waiting 10 seconds before next batch..."):
                time.sleep(10)

    # Combine all batches into final DataFrame
    final_df = pd.concat(processed_batches, ignore_index=True)
    return final_df


# --- Streamlit UI ---
st.title("Underwriting Excel Processor")

uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    with st.spinner("Processing file in batches..."):
        result_df = process_uploaded_file(uploaded_file)

    st.success("🎉 All batches processed!")
    st.write("### Preview: All Data")
    st.dataframe(result_df)

    # Full file download
    full_output = io.BytesIO()
    result_df.to_excel(full_output, index=False, engine='openpyxl')
    full_output.seek(0)

    st.download_button(
        label="📥 Download Complete Processed File",
        data=full_output,
        file_name="processed_output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
