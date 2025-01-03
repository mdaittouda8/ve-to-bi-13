import streamlit as st
import openai
import pandas as pd
import azure.cognitiveservices.speech as speechsdk
import pyodbc
import plotly.express as px
import os
#from dotenv import load_dotenv
from st_audiorec import st_audiorec
from pathlib import Path

# Load environment variables from .env file
#load_dotenv()

# Access secrets
openai_key = st.secrets["OPENAI_API_KEY"]
speech_key = st.secrets["AZURE_SPEECH_KEY"]

# Azure SQL Database Parameters
server = st.secrets["SQL_SERVER"]
database = st.secrets["SQL_DATABASE"]
username = st.secrets["SQL_USERNAME"]
password = st.secrets["SQL_PASSWORD"]
driver = st.secrets["SQL_DRIVER"]


# Azure Cognitive Services configuration
service_region = "northeurope"  # Azure region


def audio_to_text(filename):
    speech_config = speechsdk.SpeechConfig(
        subscription=speech_key, region=service_region
    )
    audio_config = speechsdk.audio.AudioConfig(filename=filename)
    speech_recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, audio_config=audio_config
    )

    # result = speech_recognizer.recognize_once()
    result = speech_recognizer.recognize_once_async().get()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        return "No speech recognized. Please try again."
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        return f"Speech recognition canceled: {cancellation_details.reason}"
    else:
        return "An unexpected error occurred."


# ------------------


# Configure OpenAI API Key
openai.api_key = openai_key

# Prompt for OpenAI
prompt = """
I have an Azure SQL database, and I want to generate queries based on natural language input. The table I want to query is called ServiceRecords with the following schema:

ServiceDate (DATE): The date of the job.
JobNumber (INT): A unique identifier for each job.
CustomerName (NVARCHAR): The name of the customer.
VehicleType (NVARCHAR): The type of vehicle serviced.
PartsCost (DECIMAL): The cost of parts for the job.
LaborCost (DECIMAL): The labor cost for the job.
BillableHours (DECIMAL): The hours billed for the job.
TechnicianName (NVARCHAR): The name of the technician who performed the job.
CustomerSatisfactionRating (INT): The customer’s satisfaction rating (1–5 stars).
Every query should only be compatible with Azure SQL. Your response must only include the SQL query and nothing else. For example, if I ask, 'What is the average labor cost for jobs completed in 2024?' you should respond with:

sql
Copy code
SELECT AVG(LaborCost) AS AverageLaborCost  
FROM ServiceRecords  
WHERE YEAR(ServiceDate) = 2024;
Let’s begin.

"""


def get_openai_response(question, prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Replace with the model you want to use
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question},
            ],
            max_tokens=150,
            temperature=0.7,
        )
        # Extract the content from the response
        sql_query = response["choices"][0]["message"]["content"].strip()
        return sql_query
    except Exception as e:
        st.error(f"Error retrieving OpenAI response: {e}")
        return None


def read_sql_query(sql):
    conn = pyodbc.connect(
        f"DRIVER={driver};SERVER={server};PORT=1433;DATABASE={database};UID={username};PWD={password}"
    )
    df = pd.read_sql_query(sql, conn)
    conn.close()
    return df


def get_sql_query_from_response(response):
    try:
        # Find the starting keyword: either "WITH" or "SELECT"
        if "WITH" in response:
            query_start = response.index("WITH")
        elif "SELECT" in response:
            query_start = response.index("SELECT")
        else:
            raise ValueError("No SQL query keywords found in the response.")

        # Find the end of the query (first semicolon)
        query_end = response.index(";") + 1

        # Extract and return the SQL query
        sql_query = response[query_start:query_end]
        return sql_query
    except ValueError as e:
        # Handle error with Streamlit (or alternative logging)
        st.error(f"Error extracting SQL query: {e}")
        return None


def isthercharttype(text):
    """
    Searches for specific chart type keywords in the given text.

    Args:
        text (str): The input string to search.

    Returns:
        str: The chart type ('bar', 'line', or 'pie') if found, otherwise an empty string.
    """
    text = text.lower()  # Convert text to lowercase for case-insensitive matching
    if "bar" in text:
        return "bar"
    elif "line" in text:
        return "line"
    elif "pie" in text:
        return "pie"
    return ""  # Return empty string if no match is found


def determine_chart_type(df, text):
    # Call isthercharttype() to check for specific chart types in the text
    chart_type = isthercharttype(text)
    if chart_type:  # If a chart type is found in the text
        return chart_type

    # Existing logic
    if len(df.columns) == 2:
        if df.dtypes[1] in ["int64", "float64"] and len(df) > 8:
            return "bar"
        elif df.dtypes[1] in ["int64", "float64"] and len(df) <= 10:
            return "pie"
    elif len(df.columns) >= 3 and df.dtypes[1] in ["int64", "float64"]:
        return "line"
    return None


def generate_chart(df, chart_type):
    if chart_type == "bar":
        fig = px.bar(
            df,
            x=df.columns[0],
            y=df.columns[1],
            title=f"{df.columns[0]} vs. {df.columns[1]}",
            template="plotly_white",
            color=df.columns[0],
            labels={df.columns[0]: "Category", df.columns[1]: "Count"},
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="Count"),
            xaxis=dict(title="Category"),
        )
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "pie":
        fig = px.pie(
            df,
            names=df.columns[0],
            values=df.columns[1],
            title=f"Distribution of {df.columns[0]}",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == "line":
        # Check if all the values in the first column are the same
        if df[df.columns[0]].nunique() == 1:  # Check for unique values in column 0
            x_col = df.columns[1]  # Use the second column as x-axis
            y_col = df.columns[2]  # Use the fourth column as y-axis
        else:
            x_col = df.columns[0]  # Use the first column as x-axis
            y_col = df.columns[1]  # Use the second column as y-axis

        # Create the line chart
        fig = px.line(
            df,
            x=x_col,
            y=y_col,
            title=f"{y_col} Over {x_col}",
            template="plotly_white",
            markers=True,
        )

        # Display the chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No suitable chart type determined for this data.")


# Streamlit interface
st.title("Speech to SQL Query & Visualization")

wav_audio_data = None
while wav_audio_data is None:
    wav_audio_data = st_audiorec()
    if wav_audio_data is None:
        st.info("Please Start Recording")
        st.stop()

# st.audio(wav_audio_data, format="audio/wav")
save_path = Path("recorded_audio.wav")
save_path.write_bytes(wav_audio_data)

# Get text from speech
speech_text = audio_to_text("recorded_audio.wav")
charttype = isthercharttype(speech_text)
st.write("Extracted Text: ", speech_text)
os.remove(save_path)
# Get SQL query from OpenAI based on speech input

response = get_openai_response(speech_text, prompt)
sql_query = get_sql_query_from_response(response)

if sql_query:
    st.code(sql_query, language="sql")

    # -------------------
    df = read_sql_query(sql_query)
    if not df.empty:
        col_data, col_chart = st.columns(2)
        with col_data:
            st.subheader("Query Results:")
            st.table(df.reset_index(drop=True))
        # st.dataframe(df.reset_index(drop=True)) #or is df not df.reset_index(drop=True)

        chart_type = determine_chart_type(df, charttype)

        if chart_type:
            with col_chart:
                st.subheader("Visualization:")
                generate_chart(df, chart_type)
    else:
        st.write("No results found for the query.")
