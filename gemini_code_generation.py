import os
import io
import base64
import time
import logging
import numpy as np
import json
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config
from statistics import mean, median, mode, stdev
from scipy import stats
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Gemini API key from config
GOOGLE_API_KEY = config.API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

# Global variables
df = None
visualization_errors = 0
MAX_VISUALIZATION_ERRORS = 5
MAX_SAMPLE_ROWS = 100  # Maximum number of rows to send to Gemini

# Helper Functions
def convert_numpy_types(obj):
    """
    Recursively convert numpy types and pandas objects to native Python types.
    """
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.astype(object).to_dict()
    elif isinstance(obj, pd.DataFrame):
        return obj.astype(object).to_dict(orient='records')
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def extract_code(text):
    """Enhanced code extraction with robust error handling"""
    lines = text.strip().split('\n')
    cleaned_lines = []
    
    for line in lines:
        stripped_line = line.strip()
        
        # Skip empty lines, comment lines, markdown code block markers, and print statements
        if (not stripped_line or 
            stripped_line.startswith('#') or 
            stripped_line.startswith('```') or 
            stripped_line.endswith('```') or
            stripped_line.startswith('print(')):
            continue
        
        cleaned_lines.append(line)
    
    # Join the lines
    code = '\n'.join(cleaned_lines)
    
    # Ensure the code has a valid structure
    if not code.strip():
        return """
def analyze_data(df):
    result = "No analysis could be performed"
    return result
"""
    
    # Add error handling to prevent empty blocks
    lines = code.split('\n')
    processed_lines = []
    for i, line in enumerate(lines):
        if line.strip().endswith(':'):
            # If the next line is empty or just whitespace, add a pass statement
            if i + 1 >= len(lines) or not lines[i+1].strip():
                processed_lines.append(line)
                processed_lines.append('    pass')
            else:
                processed_lines.append(line)
        else:
            processed_lines.append(line)
    
    return '\n'.join(processed_lines)

def extract_python_code(response):
    """Extract Python code from Gemini response with better error handling"""
    if not response or not isinstance(response, str):
        return """
import matplotlib.pyplot as plt
import io
import base64

def create_visualization(df):
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, "No visualization available", 
             ha='center', va='center', fontsize=14)
    plt.title("Error: Empty response from AI")
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf8')
"""
    
    # Check if the response contains Python code blocks
    if "```python" in response:
        # Extract code from Python code block
        code_parts = response.split("```python")
        if len(code_parts) > 1:
            code = code_parts[1].split("```")[0].strip()
            return code
    elif "```" in response:
        # Try generic code blocks
        code_parts = response.split("```")
        if len(code_parts) > 1:
            code = code_parts[1].strip()
            return code
    
    # If no code blocks, assume the entire response is code
    # (removing any markdown-style headings)
    code = "\n".join([line for line in response.split("\n") 
                    if not line.strip().startswith("#")])
    return code

def safe_exec_analysis(code, df):
    """
    Safely execute the generated analysis code with robust result extraction.
    """
    # Prepare a comprehensive execution environment
    exec_globals = {
        'df': df,
        'pd': pd,
        'np': np,
        'mean': mean,
        'median': median,
        'mode': mode,
        'stdev': stdev,
        'stats': stats
    }
    exec_locals = {}
    
    try:
        # Execute the code
        exec(code, exec_globals, exec_locals)
        
        # Extract the result
        result = exec_locals.get('result', None)
        
        # If no result is found, look for the first meaningful variable
        if result is None:
            result = next(
                (value for key, value in exec_locals.items() 
                 if not key.startswith('__') and 
                 key != 'result' and 
                 not isinstance(value, (type, type(pd), type(np))) and
                 not callable(value)),
                None
            )
        
        # Convert numpy and pandas types in the result to native Python types
        result = convert_numpy_types(result)
        
        return result
    
    except Exception as e:
        # Comprehensive error logging
        logger.error(f"Execution error: {str(e)}")
        logger.error(f"Problematic code:\n{code}")
        logger.error(f"Local variables: {list(exec_locals.keys())}")
        
        # Return a meaningful error message
        return f"Analysis could not be completed due to an error: {str(e)}"

def safe_llm_invoke(prompt, max_retries=3):
    """Safely invoke the LLM with retries and error handling"""
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt).content
            return response
        except Exception as e:
            logger.error(f"LLM invoke error (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                raise
            time.sleep(1)  # Wait before retrying

def create_error_visualization(error_message="An error occurred during visualization generation"):
    """Create a visualization showing an error message"""
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, error_message, 
            ha='center', va='center', fontsize=14, color='red')
    plt.text(0.5, 0.4, "Please try again with a different query or dataset", 
            ha='center', va='center', fontsize=12)
    
    # Add a sad face emoji or icon
    plt.text(0.5, 0.7, "😔", ha='center', va='center', fontsize=50)
    
    plt.title("Visualization Error")
    plt.axis('off')  # Hide axes
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf8')

def create_fallback_visualization(df, question):
    """Create a more sophisticated fallback visualization when the generated code fails"""
    try:
        # Create a figure with subplots for multiple visualizations
        fig = plt.figure(figsize=(15, 12))
        
        # Get numeric and categorical columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Track subplot positions
        subplot_count = 0
        max_subplots = min(4, 1 + len(numeric_cols))  # Limit to 4 subplots
        
        # Add a title for the overall figure
        fig.suptitle(f"Data Overview for: {question}", fontsize=16, y=0.98)
        
        # 1. Start with data summary text
        if subplot_count < max_subplots:
            subplot_count += 1
            ax_summary = fig.add_subplot(2, 2, subplot_count)
            
            summary_text = "Dataset Summary:\n"
            summary_text += f"- {df.shape[0]} rows, {df.shape[1]} columns\n"
            summary_text += f"- {len(numeric_cols)} numeric columns\n"
            summary_text += f"- {len(cat_cols)} categorical columns\n"
            
            if len(numeric_cols) > 0:
                summary_text += f"\nNumeric Columns Statistics:\n"
                for col in numeric_cols[:3]:  # Limit to first 3
                    summary_text += f"- {col}: mean={df[col].mean():.2f}, median={df[col].median():.2f}\n"
            
            ax_summary.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=10)
            ax_summary.set_title("Dataset Information")
            ax_summary.axis('off')
        
        # 2. Create a correlation heatmap if at least 2 numeric columns
        if len(numeric_cols) >= 2 and subplot_count < max_subplots:
            subplot_count += 1
            ax_corr = fig.add_subplot(2, 2, subplot_count)
            
            corr_data = df[numeric_cols].corr()
            sns.heatmap(corr_data, annot=True, cmap='coolwarm', ax=ax_corr, fmt='.2f', 
                        linewidths=0.5, cbar=False, annot_kws={"size": 8})
            
            ax_corr.set_title("Correlation Between Numeric Variables")
            plt.setp(ax_corr.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            plt.setp(ax_corr.get_yticklabels(), fontsize=8)
        
        # 3. Create a histogram/distribution plot for numeric data
        if len(numeric_cols) >= 1 and subplot_count < max_subplots:
            subplot_count += 1
            ax_hist = fig.add_subplot(2, 2, subplot_count)
            
            # Plot histograms for up to 3 numeric columns
            for i, col in enumerate(numeric_cols[:3]):
                sns.kdeplot(df[col], ax=ax_hist, label=col)
                
            ax_hist.set_title("Distribution of Numeric Variables")
            ax_hist.legend(fontsize=8)
        
        # 4. Create a bar chart for categorical data if available
        if len(cat_cols) >= 1 and subplot_count < max_subplots:
            subplot_count += 1
            ax_bar = fig.add_subplot(2, 2, subplot_count)
            
            # Select first categorical column with fewer than 10 unique values
            for cat_col in cat_cols:
                if df[cat_col].nunique() < 10:
                    counts = df[cat_col].value_counts().head(7)  # Top 7 categories
                    counts.plot(kind='bar', ax=ax_bar)
                    ax_bar.set_title(f"Top Categories in {cat_col}")
                    plt.setp(ax_bar.get_xticklabels(), rotation=45, ha='right', fontsize=8)
                    break
            else:
                # If no suitable categorical column found
                ax_bar.text(0.5, 0.5, "No suitable categorical data for visualization", 
                          ha='center', va='center', fontsize=10)
                ax_bar.set_title("Categorical Data")
                ax_bar.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the title
        
        # Save the figure to bytes
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode('utf8')
    
    except Exception as e:
        logger.error(f"Error in fallback visualization: {str(e)}")
        # If the fallback itself fails, return a very simple error image
        return create_error_visualization("Unable to create visualization")

def get_data_sample_and_metadata(df, max_rows=MAX_SAMPLE_ROWS):
    """
    Get a representative sample of the dataframe along with metadata
    that will help Gemini understand the full dataset structure
    """
    # Strip whitespace from column names to avoid mismatches
    df.columns = df.columns.str.strip()
    
    # Get basic metadata
    total_rows = len(df)
    column_info = []
    
    for col in df.columns:
        # Determine column type
        if pd.api.types.is_numeric_dtype(df[col]):
            col_type = "numeric"
            # Add statistics for numeric columns
            col_stats = {
                "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                "median": float(df[col].median()) if not pd.isna(df[col].median()) else None,
                "std": float(df[col].std()) if not pd.isna(df[col].std()) else None,
                "null_count": int(df[col].isna().sum())
            }
        elif pd.api.types.is_datetime64_dtype(df[col]):
            col_type = "datetime"
            # Add date range for datetime columns
            col_stats = {
                "min_date": str(df[col].min()) if not pd.isna(df[col].min()) else None,
                "max_date": str(df[col].max()) if not pd.isna(df[col].max()) else None,
                "null_count": int(df[col].isna().sum())
            }
        else:
            col_type = "categorical"
            # Add category information
            unique_values = df[col].nunique()
            # Only include sample values if there aren't too many
            sample_values = df[col].dropna().unique()[:5].tolist() if unique_values < 20 else []
            col_stats = {
                "unique_values": int(unique_values),
                "top_value": str(df[col].value_counts().index[0]) if not df[col].value_counts().empty else None,
                "top_count": int(df[col].value_counts().iloc[0]) if not df[col].value_counts().empty else None,
                "sample_values": sample_values,
                "null_count": int(df[col].isna().sum())
            }
        
        column_info.append({
            "name": col,
            "type": col_type,
            "stats": col_stats
        })
    
    # Get a strategic sample of data rows
    # First, get some rows from the beginning
    sample_head = df.head(min(5, total_rows))
    
    # Then, get some random rows for better representation
    if total_rows > 10:
        random_sample_size = min(max_rows - len(sample_head), total_rows - len(sample_head))
        if random_sample_size > 0:
            random_sample = df.iloc[5:].sample(random_sample_size)
            sample_df = pd.concat([sample_head, random_sample])
        else:
            sample_df = sample_head
    else:
        sample_df = df
    
    # Convert sample to string representation
    sample_data = sample_df.to_string()
    
    # Create a comprehensive metadata dictionary
    metadata = {
        "total_rows": total_rows,
        "total_columns": len(df.columns),
        "column_info": column_info,
        "sample_size": len(sample_df),
        "sample_representation": f"This is a sample of {len(sample_df)} rows from a total of {total_rows} rows"
    }
    
    return sample_data, metadata

def load_data(file_path):
    """Load data from file"""
    global df
    
    try:
        logger.info(f"Loading file: {file_path}")
        
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            return {"error": "Unsupported file type. Please use CSV or Excel files only."}, 400
        
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"Columns after loading: {list(df.columns)}")
        
        # Better date parsing
        for col in df.columns:
            if any(date_term in col.lower() for date_term in ['date', 'time', 'day', 'month', 'year']):
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass  # Keep as is if conversion fails
        
        logger.info(f"Columns after date parsing: {list(df.columns)}")
        logger.info(f"Sample data:\n{df.head(5).to_string()}")
        
        # Get sample data and metadata instead of passing the full dataset
        sample_data, metadata = get_data_sample_and_metadata(df)
        
        # Return more information about the loaded data
        return {
            "message": "File loaded successfully", 
            "columns": list(df.columns),
            "rows": len(df),
            "summary": df.describe().to_dict(),
            "sample": df.head(3).to_dict(orient='records'),
            "metadata": metadata  # Include metadata for more comprehensive information
        }, 200
    except Exception as e:
        logger.error(f"Error loading file: {str(e)}")
        return {"error": f"Error processing file: {str(e)}"}, 500

def perform_analysis(question):
    """Perform data analysis based on user question"""
    global df
    if df is None:
        return {"error": "No data loaded yet"}, 400
    
    try:
        if not question:
            return {"error": "No question provided"}, 400
        
        logger.info(f"Analysis question: {question}")
        
        # Strip whitespace from column names to avoid mismatches
        df.columns = df.columns.str.strip()
        logger.info(f"Columns after stripping whitespace: {list(df.columns)}")

        # Get sample data and metadata instead of passing the full dataset
        sample_data, metadata = get_data_sample_and_metadata(df)
        
        prompt = f"""
        You are an expert data analyst. I will provide a sample and metadata from a larger dataset.
        
        Dataset Sample:
        {sample_data}
        
        Dataset Metadata:
        {json.dumps(metadata, indent=2)}
        
        User question: "{question}"
        
        Instructions:
        Write Python code to analyze the FULL dataset and answer the user's question using packages like 'pandas', 'numpy', 'statistics', or 'scipy' as needed. Follow these steps:
        1. Identify relevant columns based on the question's context (e.g., IDs, names, or numeric values).
        2. Dynamically group data by appropriate columns if required, inferred from the question.
        3. Generate lists of unique values if the question asks for them.
        4. Perform calculations (e.g., count, mean, median, mode, standard deviation) using exact values when specified in the question.
        5. Use appropriate Python packages:
           - 'pandas' for data manipulation and grouping
           - 'numpy' for numerical operations
           - 'statistics' for basic stats like mean, median, mode
           - 'scipy' for advanced statistical analysis
        6. Return the Python code as a single, valid, executable code block.
        7. Double-check the logic of the code for accuracy.
        
        IMPORTANT:
        - Base analysis strictly on the provided data sample and metadata. The actual analysis will run on the complete dataset.
        - Do not approximate or round unless asked.
        - Assume 'df' is the pandas DataFrame containing the full dataset.
        - Ensure the code is complete, properly indented, and can be copied and pasted directly into a Python environment to run (assuming 'df' is defined).
        - Include necessary imports at the top of the code block.
        - Ensure the code is syntactically correct, properly indented Python with no extra text, comments, or print statements.
        - Assign the final result to a variable named 'result' instead of printing it.
        - Only use column names that exist in the dataset provided in the metadata.
        
        Response:
        Provide the Python code as plain text, ready to be copied and pasted.
        """
        
        try:
            raw_code = safe_llm_invoke(prompt).strip()
            code = extract_code(raw_code)
            logger.info(f"Generated analysis code after cleaning:\n{code}")
            
            # Use the new safe execution method
            result = safe_exec_analysis(code, df)
            
            # Convert result to a JSON-serializable format
            def convert_to_jsonable(value):
                if isinstance(value, (pd.Series, pd.DataFrame)):
                    return value.to_dict()
                elif isinstance(value, np.ndarray):
                    return value.tolist()
                elif isinstance(value, (np.int64, np.float64)):
                    return float(value)  # Convert numpy numeric types to Python float
                elif isinstance(value, (list, dict, str, int, float)):
                    return value
                else:
                    return str(value)
            
            # Convert the result, handling nested structures
            if isinstance(result, (list, dict)):
                serializable_result = result
            else:
                serializable_result = convert_to_jsonable(result)
            
            return {
                "analysis_code": code,
                "result": serializable_result,
            }, 200
                     
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            return {
                "error": f"Analysis failed: {str(e)}",
                "analysis_code": code if 'code' in locals() else None
            }, 500
            
    except Exception as e:
        logger.error(f"Error in perform_analysis: {str(e)}")
        return {"error": f"Analysis failed: {str(e)}"}, 500

def create_visualization(question, analysis, table_instructions, result):
    """Generate visualization code and execute it"""
    global df, visualization_errors
    if df is None:
        return {"error": "No data loaded yet"}, 400
    
    # Check if too many errors have occurred
    if visualization_errors >= MAX_VISUALIZATION_ERRORS:
        # Reset after a while to try again
        if visualization_errors == MAX_VISUALIZATION_ERRORS:
            visualization_errors += 1
            time.sleep(30)  # Wait 30 seconds before trying again
            visualization_errors = 0
        else:
            # Return a simple fallback visualization
            return {
                "error": "Too many visualization errors. Using fallback visualization.",
                "graph_url": create_fallback_visualization(df, question)
            }, 200
    
    # Outer try-except block to catch any unexpected errors
    try:
        if not question or not result:
            return {"error": "Missing question or result"}, 400
        
        logger.info(f"Visualization request for question: {question}")
        
        # Get sample data and metadata
        sample_data, metadata = get_data_sample_and_metadata(df)
        
        prompt = f"""
            You are an expert data scientist creating a SINGLE, high-quality visualization with matplotlib and seaborn.

            Data sample: {sample_data}
            
            Dataset metadata: {json.dumps(metadata, indent=2)}

            User question: "{question}"

            Analysis provided: "{analysis}" and the corresponding results {result}

            Table instructions: "{table_instructions}"

            VISUALIZATION REQUIREMENTS:

            1. Create a SINGLE, comprehensive visualization that best answers the user's question:
            - Choose only ONE chart type that is most appropriate for the data and question
            - The visualization should clearly communicate the key insight from the analysis
            
            2. Visualization Strategy:
            - Select the MOST APPROPRIATE graph type based on data and question:
                * Time series → Line chart with trend lines
                * Categorical comparisons → Bar/violin plots
                * Distribution → Box plots, kernel density plots
                * Relationships → Scatter plots with regression lines
                * Composition → Pie charts, treemaps
                * Simple results → Text box with formatted results (when a graph isn't necessary)

            3. Design Excellence:
            - Use professional color palette (seaborn, viridis, coolwarm)
            - Use appropriate figure size (10, 6) for a single visualization
            - Add a clear, descriptive title that summarizes the key insight
            - Ensure axes are properly labeled
            - Include a legend if multiple data series are shown

            4. Technical Graph Generation Instructions:
            Write a Python function `create_visualization(df)` that:
            - Takes pandas DataFrame as input
            - Creates a SINGLE figure with ONE visualization (or a text box if no graph is needed)
            - Returns base64 encoded PNG image
            - Handles errors gracefully with try/except blocks

            Visualization Function Template:
        ```python
            import matplotlib.pyplot as plt
            import seaborn as sns
            import io
            import base64
            import numpy as np

            def create_visualization(df):
                try:
                    # Create a single figure
                    plt.figure(figsize=(10, 6))
                    
                    # YOUR VISUALIZATION CODE HERE
                    # [Code for the single most appropriate visualization]
                    
                    # For text-only results when a graph isn't necessary
                    # plt.text(0.5, 0.5, "Key Result: [Insert result here]", ha='center', va='center', fontsize=14)
                    # plt.axis('off')
                    
                    plt.title('Clear Descriptive Title', fontsize=14)
                    plt.tight_layout()
                    
                    # Convert to base64
                    img = io.BytesIO()
                    plt.savefig(img, format='png', dpi=300, bbox_inches='tight')
                    img.seek(0)
                    return base64.b64encode(img.getvalue()).decode('utf8')
                """
        
        # LLM Try-Except Block
        try:
            response = safe_llm_invoke(prompt)
            code = extract_python_code(response)
            
            namespace = {
                'df': df,
                'plt': plt,
                'sns': sns,
                'io': io,
                'base64': base64,
                'pd': pd,
                'np': np
            }
            
            # Code Execution Try-Except Block
            try:
                exec(code, namespace)
                if 'create_visualization' in namespace:
                    # Function Execution Try-Except Block
                    try:
                        graph_url = namespace['create_visualization'](df)
                        # Reset error counter on success
                        visualization_errors = 0
                        return {
                            "visualization_code": code,
                            "graph_url": graph_url
                        }, 200
                    except Exception as func_error:
                        logger.error(f"Function execution error: {str(func_error)}")
                        visualization_errors += 1
                        return {
                            "error": f"Visualization function execution failed: {str(func_error)}",
                            "graph_url": create_fallback_visualization(df, question)
                        }, 200
                else:
                    visualization_errors += 1
                    logger.error("No create_visualization function found")
                    return {
                        "error": "No create_visualization function found",
                        "graph_url": create_fallback_visualization(df, question)
                    }, 200
            except Exception as outer_error:
                visualization_errors += 1
                logger.error(f"Code execution error: {str(outer_error)}")
                # Ensure a fallback is returned even if the outer try fails
                try:
                    fallback_graph = create_fallback_visualization(df, question)
                except Exception as fallback_error:
                    logger.error(f"Fallback visualization failed: {str(fallback_error)}")
                    fallback_graph = create_error_visualization("Complete visualization failure")
                return {
                    "error": f"Code execution failed: {str(outer_error)}",
                    "graph_url": fallback_graph
                }, 200
                
        except Exception as llm_error:
            visualization_errors += 1
            logger.error(f"LLM error in visualization: {str(llm_error)}")
            return {
                "error": f"AI code generation failed: {str(llm_error)}",
                "graph_url": create_fallback_visualization(df, question)
            }, 200
    
    # This is the main error handler for the entire function
    except Exception as outer_most_error:
        visualization_errors += 1
        logger.error(f"General error in visualization: {str(outer_most_error)}")
        return {
            "error": f"Visualization process failed: {str(outer_most_error)}",
            "graph_url": create_fallback_visualization(df, question)
        }, 200

def preprocess_data(operations):
    """Preprocess the data with specified operations"""
    global df
    if df is None:
        return {"error": "No data loaded yet"}, 400
    
    results = {}
    
    for op in operations:
        op_type = op.get('type')
        try:
            if op_type == 'fillna':
                column = op.get('column')
                value = op.get('value')
                if column and column in df.columns:
                    df[column] = df[column].fillna(value)
                    results[f"fillna_{column}"] = "success"
            
            elif op_type == 'drop_duplicates':
                columns = op.get('columns', [])
                if columns:
                    df = df.drop_duplicates(subset=columns)
                else:
                    df = df.drop_duplicates()
                results["drop_duplicates"] = "success"
            
        except Exception as e:
            results[f"{op_type}_error"] = str(e)
    
    return {
        "message": "Preprocessing completed",
        "operations_results": results,
        "new_shape": df.shape
    }, 200

def main(file_path, question, operations=None, table_instructions=None):
    """
    Main function to orchestrate the entire data analysis and visualization process.
    
    Args:
        file_path (str): Path to the data file
        question (str): User's analysis question
        operations (list, optional): List of preprocessing operations
        table_instructions (str, optional): Instructions for table formatting
        
    Returns:
        dict: Results of the analysis and visualization
    """
    # Step 1: Load data
    load_result, status_code = load_data(file_path)
    if status_code != 200:
        return load_result, status_code
    
    # Step 2: Preprocess data if operations are provided
    if operations:
        preprocess_result, status_code = preprocess_data(operations)
        if status_code != 200:
            return preprocess_result, status_code
    
    # Step 3: Perform analysis
    analysis_result, status_code = perform_analysis(question)
    if status_code != 200:
        return analysis_result, status_code
    
    # Step 4: Create visualization
    visualization_result, status_code = create_visualization(
        question, 
        analysis_result.get("analysis_code", ""), 
        table_instructions or "", 
        analysis_result.get("result", {})
    )
    
    # Step 5: Combine results
    return {
        "data_info": load_result,
        "analysis": {
            "result": analysis_result.get("result", {}),
            "analysis_code": analysis_result.get("analysis_code", "")
        },
        "visualization": {
            "graph_url": visualization_result.get("graph_url", ""),
            "visualization_code": visualization_result.get("visualization_code", "")
        }
    }, 200
