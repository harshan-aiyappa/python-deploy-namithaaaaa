import os
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Force non-GUI backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sqlalchemy import create_engine
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv # Make sure to call load_dotenv() if you use .env for DB_CREDENTIALS
from urllib.parse import urlparse, quote_plus # <<<--- ENSURE quote_plus IS IMPORTED
import re # For parsing summary and code

app = Flask(__name__)
CORS(app)

# Base directory for storing project-specific files
BASE_DATA_DIR = 'project_data'
if not os.path.exists(BASE_DATA_DIR):
    os.makedirs(BASE_DATA_DIR)

# Default directory for project files
DEFAULT_PROJECT_DIR = 'default_data'
default_dir_path = os.path.join(BASE_DATA_DIR, DEFAULT_PROJECT_DIR)
if not os.path.exists(default_dir_path):
    os.makedirs(default_dir_path)

# File names for project
FILE_NAMES = {
    'questions': 'user_questions.txt',
    'analysis': 'analysis_code.txt',
}

# Database Credentials
DB_CREDENTIALS = {
    "user": "postgres",
    "password": "Lingotran@123", # Password with special character '@'
    "database": "Lingotran_AnalyticsDB",
    "host": "10.10.20.73",
    "port": "5432"
}

# --- VVV MODIFIED/ADDED SECTION FOR SQLALCHEMY WITH PASSWORD ENCODING VVV ---
# URL-encode the password to handle special characters like '@'
encoded_password = quote_plus(DB_CREDENTIALS['password']) # <<<--- THIS IS THE FIX

# Construct the SQLAlchemy URL for PostgreSQL
SQLALCHEMY_DATABASE_URL = (
    f"postgresql+psycopg2://{DB_CREDENTIALS['user']}:{encoded_password}" # <<<--- USE encoded_password
    f"@{DB_CREDENTIALS['host']}:{DB_CREDENTIALS['port']}/{DB_CREDENTIALS['database']}"
)

try:
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
    # Optional: Test connection during startup if needed
    # with engine.connect() as connection:
    #     print("Successfully connected to the database using SQLAlchemy engine!")
except Exception as e:
    print(f"Error creating SQLAlchemy engine: {e}")
    # Consider how your app should behave if the DB connection fails at startup
    engine = None
# --- ^^^ END OF MODIFIED/ADDED SECTION ^^^ ---

def get_file_paths(project_dir):
    """
    Get the full file paths for questions and analysis files based on the project directory.
    """
    return {
        'questions': os.path.join(project_dir, FILE_NAMES['questions']),
        'analysis': os.path.join(project_dir, FILE_NAMES['analysis']),
    }

def ensure_files_exist(file_paths):
    """
    Make sure all required files exist, creating them if necessary.
    """
    for file_path in file_paths.values():
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                if os.path.basename(file_path) == FILE_NAMES['questions']:
                    f.write("UQ6=Other Category\n") # Example initial content
                # analysis_code.txt will be written to by append_analysis_code
                pass

# Global variables to store code and summaries
analysis_code = {} # Stores executable visualization code
saved_summaries = {} # Stores saved summary texts
current_project_dir = None
current_file_paths = None

def load_file_content(filename):
    """Load content from a file, return empty string if file doesn't exist"""
    try:
        with open(filename, 'r', encoding='utf-8') as file: # Added encoding
            return file.read()
    except FileNotFoundError:
        return ""

def load_project_code(project_dir):
    global analysis_code, saved_summaries, current_project_dir, current_file_paths
    
    current_project_dir = project_dir
    current_file_paths = get_file_paths(project_dir)
    
    print(f"Loading project from: {project_dir}")
    print(f"Analysis file path: {current_file_paths['analysis']}")
    
    ensure_files_exist(current_file_paths)
    
    analysis_code.clear()
    saved_summaries.clear()
    
    analysis_content = load_file_content(current_file_paths['analysis'])
    
    # Regex to find Analysis_key blocks and then parse summary and code within them
    # Using re.DOTALL so '.' matches newlines within summary/code
    pattern = re.compile(r"Analysis_key=(.*?)\n# --- SUMMARY START ---\n(.*?)\n# --- SUMMARY END ---\n# --- CODE START ---\n(.*?)\n# --- CODE END ---", re.DOTALL)
    
    for match in pattern.finditer(analysis_content):
        qkey = match.group(1).strip()
        summary = match.group(2).strip()
        code_block = match.group(3).strip()
        
        if qkey:
            analysis_code[qkey] = code_block
            saved_summaries[qkey] = summary
            
    print(f"Loaded analysis keys: {list(analysis_code.keys())}")
    print(f"Loaded summary keys: {list(saved_summaries.keys())}")


def fetch_data_from_db():
    # df1: map to tracker table
    # df2: map to toc
    """Fetch data from the PostgreSQL database for both tables using SQLAlchemy."""
    global engine # Access the global engine

    if engine is None:
        # This means engine creation failed at startup
        raise Exception("Database engine is not initialized. Check connection details and server status.")

    try:
        # The psycopg2.connect(...) and conn.close() calls are removed.
        # SQLAlchemy engine manages connections.

        query1 = '''
        SELECT 
        "LrnrPromptProgress_UserID",
        "LrnrPromptProgress_UserFullName" AS "User_FullName",
        "LrnrPromptProgress_OrganizationID",
        "LrnrPromptProgress_OrganizationName" AS "Organization_Name",
        "LrnrPromptProgress_ClassID",
        "LrnrPromptProgress_ClassName" AS "Grade",
        "LrnrPromptProgress_SectionID",
        "LrnrPromptProgress_SectionName" AS "Section_Name",
        "LrnrPromptProgress_ActivityTypeID",
        "LrnrPromptProgress_QuestionType" AS "Question_Type",
        "LrnrPromptProgress_CourseID",
        "LrnrPromptProgress_Course" AS "Course",
        "LrnrPromptProgress_ChapterID",
        "LrnrPromptProgress_Chapter",
        "LrnrPromptProgress_ChapterName" AS "Chapter",
        "LrnrPromptProgress_UnitID",
        "LrnrPromptProgress_Unit",
        "LrnrPromptProgress_UnitName" AS "Unit",
        "LrnrPromptProgress_QuestionID",
        "LrnrPromptProgress_Question" AS "Question",
        "LrnrPromptProgress_ActualAnswer" AS "ActualAnswer",
        "LrnrPromptProgress_PartsOfSpeech" AS "PartsOfSpeech",
        "LrnrPromptProgress_ModeofLearning" As "ModeofLearning",
        "LrnrPromptProgress_SequenceBuilderID",
        "LrnrPromptProgress_Attempts" AS "Attempts",
        "LrnrPromptProgress_IsUserResponseCorrect" As "IsUserResponseCorrect", 
        "LrnrPromptProgress_UserAnswer" As "UserAnswer",
        "LrnrPromptProgress_UserResponsePoS" As "UserResponsePoS",
        "LrnrPromptProgress_SourceLanguage" AS "SourceLanguage",
        "LrnrPromptProgress_TargetLanguage" AS "TargetLanguage"
        FROM public."LrnrPromptProgress"
        ORDER BY  "LrnrPromptProgress_UserID", 
        "LrnrPromptProgress_OrganizationID", 
        "LrnrPromptProgress_ClassName",
        "LrnrPromptProgress_SectionName",
        "LrnrPromptProgress_CourseID",
        "LrnrPromptProgress_Chapter",
        "LrnrPromptProgress_Unit"
        ''' 
        # Use the SQLAlchemy engine with pandas
        df1 = pd.read_sql_query(query1, engine)

        query2 = '''
        WITH "Chapter" AS (
        SELECT 
            lp."LearnersProgress_UserID",
            lp."LearnersProgress_ClassID",
            lp."LearnersProgress_SectionID",
            lp."LearnersProgress_CourseID",
            lp."LearnersProgress_ChapterID",
            AVG(lp."LearnersProgress_UnitCompletionPerc") AS "Chapter_CompletionPerc"
        FROM "LearnersProgress" lp
        GROUP BY 
            lp."LearnersProgress_UserID",
            lp."LearnersProgress_ClassID",
            lp."LearnersProgress_SectionID",
            lp."LearnersProgress_CourseID",
            lp."LearnersProgress_ChapterID")
        SELECT 
        lp."LearnersProgress_UserID" AS "User_ID",
        lp."LearnersProgress_LearnerName" AS "User_FullName",
        lp."LearnersProgress_ClassID" AS "Class_ID",
        ow."Overview_ClassName" AS "Class_Name",
        lp."LearnersProgress_SectionID" AS "Section_ID",
        CASE 
            WHEN lp."LearnersProgress_SectionID"=1 THEN 'Section A'
            WHEN lp."LearnersProgress_SectionID"=2 THEN 'Section B'
            WHEN lp."LearnersProgress_SectionID"=3 THEN 'Section C'
            WHEN lp."LearnersProgress_SectionID"=4 THEN 'Section D'
        END AS "Section_Name",
        lp."LearnersProgress_OrgID" AS "Organization_ID",
        lp."LearnersProgress_CourseID" AS "LessonJourney_ID",
        lp."LearnersProgress_Course" AS "LessonJourney_Name",
        lp."LearnersProgress_ChapterID" AS "Chapter_ID",
        lp."LearnersProgress_ChapterNo" As "Chapter_No",
        lp."LearnersProgress_Chapter" AS "Chapter",
        lp."LearnersProgress_UnitID" AS "Unit_ID",
        lp."LearnersProgress_UnitNo" AS "Unit_No",
        lp."LearnersProgress_Unit" AS "Unit",
        CASE
            WHEN ch."Chapter_CompletionPerc"=100 THEN 'Completed'
            WHEN ch."Chapter_CompletionPerc"<100 AND ch."Chapter_CompletionPerc">0 THEN 'In Progress'
            WHEN ch."Chapter_CompletionPerc"=0 OR ch."Chapter_CompletionPerc" IS NULL THEN 'Yet to Start'
        END "Chapter_Status",
        lp."LearnersProgress_UnitStatus" AS "Unit_Status_New"
        FROM "LearnersProgress" lp
        JOIN "Chapter" ch 
        ON lp."LearnersProgress_UserID"=ch."LearnersProgress_UserID"
        AND    lp."LearnersProgress_ClassID"=ch."LearnersProgress_ClassID"
        AND    lp."LearnersProgress_SectionID"=ch."LearnersProgress_SectionID"
        AND    lp."LearnersProgress_CourseID"=ch."LearnersProgress_CourseID"
        AND    lp."LearnersProgress_ChapterID"=ch."LearnersProgress_ChapterID"
        JOIN "Overview" ow
        ON lp."LearnersProgress_ClassID"=ow."Overview_ClassID"
        AND lp."LearnersProgress_OrgID"=ow."Overview_Organization_ID"
        GROUP BY
            lp."LearnersProgress_UserID",
        lp."LearnersProgress_LearnerName",
        lp."LearnersProgress_ClassID",
        lp."LearnersProgress_SectionID",
        lp."LearnersProgress_OrgID",
        lp."LearnersProgress_CourseID",
        lp."LearnersProgress_Course",
        lp."LearnersProgress_ChapterID",
        lp."LearnersProgress_ChapterNo",
        lp."LearnersProgress_Chapter",
        lp."LearnersProgress_UnitID",
        lp."LearnersProgress_UnitNo",
        lp."LearnersProgress_Unit",
        lp."LearnersProgress_UnitStatus",
        ch."Chapter_CompletionPerc",
        ow."Overview_ClassName"
        ORDER BY lp."LearnersProgress_UserID",ow."Overview_ClassName", lp."LearnersProgress_CourseID", lp."LearnersProgress_ChapterNo",lp."LearnersProgress_UnitNo"
        ''' 
        # Use the SQLAlchemy engine with pandas
        df2 = pd.read_sql_query(query2, engine)

        return df1, df2
    except Exception as e:
        # Log the error for debugging or re-raise a more specific one
        # The original error from psycopg2 (or other DB driver) will be part of 'e'
        print(f"Error during database operation with SQLAlchemy: {e}")
        # You might want to raise a custom error or just re-raise the original
        raise Exception(f"Database query failed: {str(e)}")

@app.route('/questions')
def get_questions():
    if not current_project_dir or not current_file_paths:
        return jsonify({'error': 'No project selected. Initialize project first.'}), 400
    questions = {}
    try:
        with open(current_file_paths['questions'], "r", encoding='utf-8') as file: # Added encoding
            for line in file:
                line = line.strip()
                if not line:
                    continue
                if '=' in line:
                    key, category = line.split('=', 1)
                    questions[key] = {'category': category, 'questions': []}
                elif '|' in line:
                    qkey, question = line.split('|', 1)
                    category_key = qkey.split('.')[0]
                    if category_key in questions:
                        questions[category_key]['questions'].append({'key': qkey, 'text': question})
        return jsonify(questions)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def append_question(question_text):
    try:
        if not current_file_paths:
            return None
            
        uq6_questions = []
        # Use encoding for reading and writing
        with open(current_file_paths['questions'], "r", encoding='utf-8') as file:
            found_uq6 = False
            for line in file:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("UQ6="):
                    found_uq6 = True
                elif found_uq6 and line.startswith("UQ6."):
                    uq6_questions.append(line.split('|')[0])
                elif found_uq6 and '=' in line and not line.startswith("UQ6.") : # ensure it's a new category
                    break 
        
        if not found_uq6:
            with open(current_file_paths['questions'], "a", encoding='utf-8') as file:
                file.write("\nUQ6=Other Category\n")
        
        next_number = 1
        if uq6_questions:
            # Ensure robust parsing of numbers after "UQ6."
            numbers = []
            for q in uq6_questions:
                match = re.match(r"UQ6\.(\d+)", q)
                if match:
                    numbers.append(int(match.group(1)))
            if numbers:
                next_number = max(numbers) + 1

        question_key = f"UQ6.{next_number}"
        
        with open(current_file_paths['questions'], "a", encoding='utf-8') as file:
            file.write(f"{question_key}|{question_text}\n")
        
        return question_key
    except Exception as e:
        print(f"Error appending question: {str(e)}")
        return None

def append_analysis_code(question_key, viz_code, summary_text):
    try:
        if not current_file_paths:
            return
        # Use encoding for writing
        with open(current_file_paths['analysis'], "a", encoding='utf-8') as file:
            file.write(f"Analysis_key={question_key}\n")
            file.write(f"# --- SUMMARY START ---\n{summary_text}\n# --- SUMMARY END ---\n")
            file.write(f"# --- CODE START ---\n{viz_code}\n# --- CODE END ---\n\n")
    except Exception as e:
        print(f"Error appending analysis code: {str(e)}")

def execute_analysis_code(code_to_execute, df1, df2, params=None, sheet_index=None):
    local_vars = {'df1': df1, 'df2': df2, 'plt': plt, 'sns': sns, 'io': io, 'base64': base64, 'pd': pd}
    # The 'df' variable will be set based on sheet_index for the executed code to use
    selected_df = df1 if sheet_index == 0 else df2 # Assuming sheet_index 0 for df1, 1 for df2

    if params:
        local_vars['params'] = params
    
    local_vars['df'] = selected_df # Make the selected DataFrame available as 'df'
                                   # for the create_visualization function

    try:
        plt.clf() 
        print(f"Executing analysis definition with params: {params} for sheet_index: {sheet_index}")
        
        # Execute the string, which should define a function like 'create_visualization'
        exec(code_to_execute, globals(), local_vars)
        
        # Now, try to call the function that was defined by the exec'd code.
        # We assume the function is named 'create_visualization' and takes 'df' as an argument.
        # If your saved code defines a different function name or signature, adjust this.
        if 'create_visualization' in local_vars and callable(local_vars['create_visualization']):
            print("Calling the defined 'create_visualization' function.")
            # Pass the selected_df (which is 'df' in local_vars) to the function
            analysis_result = local_vars['create_visualization'](local_vars['df'])
        else:
            # Fallback or error if the expected function isn't defined
            error_message = "Analysis Error: 'create_visualization' function not found after executing saved code."
            print(error_message)
            # Generate an error image
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "Execution Error:\n'create_visualization' not defined by code.",
                     ha='center', va='center', color='red', wrap=True, fontsize=10)
            plt.title('Code Execution Issue')
            plt.axis('off')
            img_err = io.BytesIO()
            plt.savefig(img_err, format='png', bbox_inches='tight')
            img_err.seek(0)
            plt.close()
            return base64.b64encode(img_err.getvalue()).decode()

        # analysis_result should now be the base64 string from the create_visualization function
        if not isinstance(analysis_result, str) or not analysis_result.startswith(('iVBOR', '/9j/')): # Basic check for base64 PNG/JPG
            # This means create_visualization might have returned something else or an error string itself
            print(f"Warning: 'create_visualization' did not return an expected base64 image string. Result: {str(analysis_result)[:100]}")
            if isinstance(analysis_result, str) and "Error generating visualization" in analysis_result: # If it returned an error string from its own try-except
                 return analysis_result # Pass through the error string from the viz code

            # If it's not an error string from the viz code but also not a base64 image, treat as an execution issue
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "Execution Error:\nUnexpected output from visualization code.",
                     ha='center', va='center', color='red', wrap=True, fontsize=10)
            plt.title('Code Execution Issue')
            plt.axis('off')
            img_err = io.BytesIO()
            plt.savefig(img_err, format='png', bbox_inches='tight')
            img_err.seek(0)
            plt.close()
            return base64.b64encode(img_err.getvalue()).decode()

        return analysis_result # This should be the raw base64 image string
    
    except Exception as e:
        error_message = f"Outer Analysis Execution Error: {str(e)}"
        print(error_message)
        # import traceback
        # traceback.print_exc() # For more detailed server-side logs
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f'Outer Analysis Execution Error:\n{str(e)}', 
                 ha='center', va='center', color='red', wrap=True, fontsize=8)
        plt.title('Code Execution Error')
        plt.axis('off')
        img_err = io.BytesIO()
        plt.savefig(img_err, format='png', bbox_inches='tight')
        img_err.seek(0)
        plt.close()
        return base64.b64encode(img_err.getvalue()).decode()

from gemini import main as call_gemini_main

@app.route('/sendCustomQuestion', methods=['POST'])
def fetch_analysis():
    data = request.json
    question_text = data.get('question')

    if not question_text:
        return jsonify({'error': 'Missing question'}), 400

    if not current_project_dir: # Ensure project context is loaded
        print("No current project dir, loading default...")
        load_project_code(default_dir_path)

    try:
        df1, df2 = fetch_data_from_db()
        
        file_path_placeholder = "database_data" 
        print(f"Calling Gemini for question: '{question_text}'")
        gemini_response, status_code = call_gemini_main(file_path_placeholder, question_text, df1, df2)
        
        # --- DETAILED LOGGING OF GEMINI RESPONSE ---
        print(f"--- Gemini Raw Response for '{question_text}' ---")
        print(f"Status Code from Gemini Main: {status_code}")
        import pprint
        pprint.pprint(gemini_response) # Pretty print the whole dictionary
        print(f"--- End Gemini Raw Response ---")
        # --- END DETAILED LOGGING ---

        if status_code == 200 and gemini_response:
            question_key = append_question(question_text)
            print(f"Appended question, got key: {question_key}")
            
            # Initialize with defaults
            visualization_details = {}
            summary_details = {}
            
            # Check top level first
            if isinstance(gemini_response.get('visualization'), dict):
                visualization_details = gemini_response.get('visualization', {})
            if isinstance(gemini_response.get('summary'), dict):
                summary_details = gemini_response.get('summary', {})

            # Handle potentially nested 'result' key
            if 'result' in gemini_response and isinstance(gemini_response.get('result'), dict):
                nested_result = gemini_response['result']
                # If nested result has these keys, they take precedence or fill if top-level was empty
                if isinstance(nested_result.get('visualization'), dict):
                     visualization_details = nested_result.get('visualization', visualization_details)
                if isinstance(nested_result.get('summary'), dict):
                    summary_details = nested_result.get('summary', summary_details)
            
            viz_code_from_gemini = visualization_details.get('visualization_code')
            summary_text_from_gemini = summary_details.get('text', "No summary provided by AI.")

            print(f"Extracted viz_code_from_gemini (type: {type(viz_code_from_gemini)}): '{str(viz_code_from_gemini)[:100]}...'")
            print(f"Extracted summary_text_from_gemini (type: {type(summary_text_from_gemini)}): '{str(summary_text_from_gemini)[:100]}...'")

            if question_key and viz_code_from_gemini: # Check if viz_code is not None and not empty
                print(f"Proceeding to append analysis code for key: {question_key}")
                lines = viz_code_from_gemini.split('\n')
                clean_lines = []
                for line in lines:
                    stripped_line = line.strip()
                    if not (stripped_line.startswith('import ') or stripped_line.startswith('from ')):
                        clean_lines.append(line)
                
                cleaned_viz_code = '\n'.join(clean_lines).strip()

                append_analysis_code(question_key, cleaned_viz_code, summary_text_from_gemini)
                analysis_code[question_key] = cleaned_viz_code # Update in-memory
                saved_summaries[question_key] = summary_text_from_gemini # Update in-memory
                print(f"Successfully saved code and summary to file and memory for new key: {question_key}")
            else:
                print(f"Skipping append_analysis_code. question_key: {question_key}, viz_code_from_gemini is None or empty: {not viz_code_from_gemini}")
                if not question_key:
                    print("Reason: question_key is None (failed to append question).")
                if not viz_code_from_gemini:
                     print("Reason: viz_code_from_gemini is None or empty.")
        
        # Return the full response from Gemini to the frontend
        return jsonify(gemini_response), status_code
    
    except Exception as e:
        import traceback
        print("--- ERROR in /sendCustomQuestion ---")
        traceback.print_exc()
        print("--- END ERROR ---")
        return jsonify({'error': f"Error processing custom question: {str(e)}", 'status': 500, 'ui_hint': 'SHOW_RETRY_POPUP', 'popup_message': f'An internal server error occurred: {str(e)}. Please try again.'}), 500

@app.route('/get_parameter_options', methods=['POST'])
def get_parameter_options():
    data = request.json
    column_name = data.get('column_name')
    sheet_index = data.get('sheet_index', 0) # Default to 0 (df1) if not provided
    
    if not column_name:
        return jsonify({'error': 'Missing column_name'}), 400
        
    try:
        df1, df2 = fetch_data_from_db()
        df_selected = df1 if sheet_index == 0 else df2
        
        if column_name not in df_selected.columns:
            return jsonify({'error': f'Column "{column_name}" not found in the selected data table (index {sheet_index}).'}), 400
            
        options = df_selected[column_name].dropna().unique().tolist()
        # Attempt to sort if numeric, otherwise sort as string
        try:
            # Check if all options can be converted to float for numeric sort
            numeric_options = [float(opt) for opt in options]
            options = sorted(numeric_options)
            options = [str(int(opt)) if float(opt).is_integer() else str(opt) for opt in options] # Convert back to string, int if whole
        except ValueError:
            options = sorted(options, key=str) # Sort as string if not all numeric

        print(f"Options for {column_name} in table {sheet_index}: {options}")
        return jsonify({'options': options}), 200
    except Exception as e:
        return jsonify({'error': f'Error retrieving options: {str(e)}'}), 500

@app.route('/process_question', methods=['POST'])
def process_question():
    data = request.json
    question_key = data.get('question')
    params = data.get('params', {})
    sheet_index_from_request = data.get('sheet_index') # Could be None, 0, or 1

    # Determine sheet_index for data fetching and code execution.
    # Default to 0 (df1) if not specified or if None.
    # This aligns with how your Gemini code might expect 'df' if not parameterized.
    effective_sheet_index = 0 if sheet_index_from_request is None else int(sheet_index_from_request)

    if not question_key:
        return jsonify({'error': 'Missing question_key'}), 400

    if not current_project_dir:
        load_project_code(default_dir_path)
        
    print(f"Processing question key: {question_key} with params: {params} for effective_sheet_index: {effective_sheet_index}")

    try:
        df1, df2 = fetch_data_from_db()
    except Exception as e:
        return jsonify({'error': f'Error loading data from database: {str(e)}', 'ui_hint': 'SHOW_RETRY_POPUP'}), 500

    # Prepare the response structure that the frontend's displayResults expects
    response_payload = {
        "message": "Analysis processed", # Will be updated
        "result": { # This 'result' key is important for the frontend
            "analysis": { # This sub-object is for consistency, might not be fully used by frontend here
                "analysis_code": None, 
                "error": None, 
                "result": None, # This will hold the base64 image string if successful
                "warning": None
            },
            "data_info": {"message": "Data used from database"},
            "preprocessing": {"status": "Not applicable for pre-defined questions"},
            "summary": {
                "text": "Summary for this question." # Will be updated
            },
            "visualization": { # This sub-object is key for the frontend
                "error": None, 
                "graph_url": None, # This will hold the base64 image string if successful
                "needed": False, 
                "visualization_code": None # Can store the code that was run
            }
        },
        "status": 200 # Will be updated
    }

    if question_key in analysis_code:
        executable_code_str = analysis_code[question_key]
        saved_summary_text = saved_summaries.get(question_key, "Summary not found for this question.")
        
        print(f"Found saved code for key: {question_key}")
        
        # Execute the saved code definition (which defines a function) and call it
        visualization_base64_output = execute_analysis_code(executable_code_str, df1, df2, params, effective_sheet_index)
        
        response_payload["message"] = "Analysis retrieved from saved definition."
        response_payload["result"]["summary"]["text"] = saved_summary_text
        response_payload["result"]["visualization"]["visualization_code"] = executable_code_str # Store the code used

        # Check if visualization_base64_output is a valid base64 image string
        # A simple check: not None, is a string, and doesn't start with typical error messages.
        is_successful_viz = (visualization_base64_output and 
                             isinstance(visualization_base64_output, str) and 
                             not visualization_base64_output.startswith("Analysis Error:") and
                             not visualization_base64_output.startswith("Execution Error:") and
                             not visualization_base64_output.startswith("Outer Analysis Execution Error:"))


        if is_successful_viz:
            # IMPORTANT: The frontend expects the RAW base64 string here.
            # The 'data:image/png;base64,' prefix is added by the frontend's displayResults.
            response_payload["result"]["visualization"]["graph_url"] = visualization_base64_output
            # For compatibility with one of the paths displayResults checks for image:
            response_payload["result"]["analysis"]["result"] = visualization_base64_output 
            response_payload["result"]["visualization"]["needed"] = True
            response_payload["status"] = 200
        else:
            # The visualization_base64_output contains an error message (or is None)
            error_msg = visualization_base64_output if visualization_base64_output else "Unknown error during visualization."
            response_payload["result"]["visualization"]["error"] = error_msg
            response_payload["result"]["analysis"]["error"] = error_msg # Propagate error
            response_payload["result"]["visualization"]["needed"] = False # No valid graph
            response_payload["status"] = 500
            response_payload["message"] = "Error executing saved analysis."
            response_payload['ui_hint'] = 'SHOW_RETRY_POPUP'
            response_payload['popup_message'] = f"Error during saved analysis: {error_msg[:100]}..."
            # Optionally, you can still try to return the summary text
            # response_payload["result"]["summary"]["text"] = saved_summary_text 
            # But if viz fails, usually the whole result is problematic.
   
        return jsonify(response_payload), response_payload["status"]
    else:
        print(f"No saved code found for key: {question_key}")
        response_payload["message"] = f"Analysis definition for question '{question_key}' not found."
        response_payload["result"]["summary"]["text"] = "No analysis found for this question."
        response_payload["result"]["analysis"]["error"] = f"Analysis for '{question_key}' not found."
        response_payload["result"]["visualization"]["needed"] = False
        response_payload["status"] = 404
        response_payload['ui_hint'] = 'SHOW_RETRY_POPUP'
        response_payload['popup_message'] = f"I couldn't find a pre-defined analysis for '{question_key}'. Try asking in your own words or check the FAQs."
        return jsonify(response_payload), response_payload["status"]

@app.route('/get_column_names', methods=['POST'])
def get_column_names():
    try:
        df1, df2 = fetch_data_from_db()
        # Get columns from both, remove duplicates, and sort
        columns_df1 = df1.columns.tolist() if df1 is not None else []
        columns_df2 = df2.columns.tolist() if df2 is not None else []
        all_columns = sorted(list(set(columns_df1 + columns_df2)))
        return jsonify({'columns': all_columns}), 200
    except Exception as e:
        return jsonify({'error': f'Error retrieving columns: {str(e)}'}), 500

@app.route('/get_current_project', methods=['GET'])
def get_current_project():
    if not current_project_dir:
        return jsonify({'error': 'No project currently loaded'}), 400
    
    project_name = os.path.basename(current_project_dir)
    return jsonify({
        'project_dir': current_project_dir,
        'project_name': project_name,
        'file_paths': current_file_paths
    }), 200
    
@app.route('/get_project_files', methods=['POST'])
def get_project_files():
    data = request.get_json()
    project_dir = data['project_dir']
    
    file_paths = get_file_paths(project_dir)
    
    try:
        files = {
            "user_questions": open(file_paths['questions'], encoding='utf-8').read(),
            "analysis_code": open(file_paths['analysis'], encoding='utf-8').read(),
        }
        return jsonify(files)
    except FileNotFoundError as e:
        return jsonify({"error": f"File not found: {str(e)}"}), 404
    except Exception as e: # Catch other potential errors like permission issues
        return jsonify({"error": f"Error reading project files: {str(e)}"}), 500


@app.route('/')
def index():
    return render_template('index.html')

load_dotenv()
# # BASE_URL = os.getenv('BASE_URL', 'http://127.0.0.1:5001') # Default to 127.0.0.1 for local
# BASE_URL = os.getenv('BASE_URL')
# print(BASE_URL)
# parsed_url = urlparse(BASE_URL)
# HOST = parsed_url.hostname if parsed_url.hostname else '127.0.0.1'
# PORT = int(parsed_url.port) if parsed_url.port else int(os.getenv('PORT', 5001))

@app.route('/config')
def get_config():
    return jsonify({'base_url': BASE_URL})

# if __name__ == '__main__':
#     load_project_code(default_dir_path) # Load default project on startup
#     app.run(debug=True, use_reloader=True) 
# else:
#     # For Gunicorn or other WSGI servers, load project code when module is loaded
#     load_project_code(default_dir_path)

# Load environment variables from .env file
load_dotenv()

# Get the BASE_URL from environment variables for frontend use
# BASE_URL = os.getenv('BASE_URL', 'http://0.0.0.0:5001')  # Default value if not set in .env
# BASE_URL = 'https://learningreports.azurewebsites.net'
BASE_URL = 'http://10.10.20.122:5001'

# Parse BASE_URL to get HOST and PORT
parsed_url = urlparse(BASE_URL)
HOST = '0.0.0.0'  # Always bind to all interfaces locally
PORT = int(parsed_url.port) if parsed_url.port else int(os.getenv('PORT', 5001))  # Use BASE_URL port, else PORT, else 5001

if __name__ == '__main__':
    # Local development: Use Flask's built-in server
    default_project_dir = os.path.join(BASE_DATA_DIR, DEFAULT_PROJECT_DIR)
    load_project_code(default_project_dir)
    app.run(host=HOST, port=PORT, debug=True)
else:
    # Production (Azure): Gunicorn will handle serving, no app.run() needed
    load_project_code(default_dir_path)
    
