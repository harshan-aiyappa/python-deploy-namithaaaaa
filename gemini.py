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
import config # Assuming you have a config.py with API_KEY
from statistics import mean, median, mode, stdev
from scipy import stats
import seaborn as sns
import re # Import regex for parsing the check response

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Gemini API key from config
try:
    GOOGLE_API_KEY = config.API_KEY
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
except AttributeError:
    logger.error("API_KEY not found in config.py. Please ensure it's defined.")
    raise AttributeError("API_KEY not found in config.py")


# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)

# Global variables
df = None
visualization_errors = 0
MAX_VISUALIZATION_ERRORS = 5
MAX_SAMPLE_ROWS = 100

# Helper Functions (convert_numpy_types, extract_code, extract_python_code, safe_exec_analysis, safe_llm_invoke)
# ... (These functions remain as you provided, no changes needed for this specific fix) ...
# ... (Assume they are present here) ...

def convert_numpy_types(obj):
    """
    Recursively convert numpy types and pandas objects to native Python types.
    Handles potential non-serializable types more robustly.
    """
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        if np.isnan(obj):
            return None
        elif np.isinf(obj):
            return 'Infinity' if obj > 0 else '-Infinity'
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, pd.Series):
        return convert_numpy_types(obj.to_list())
    elif isinstance(obj, pd.DataFrame):
        return convert_numpy_types(obj.to_dict(orient='records'))
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        try:
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)


def extract_code(text):
    """Enhanced code extraction with robust error handling"""
    if not isinstance(text, str):
        text = "" 

    lines = text.strip().split('\n')
    cleaned_lines = []
    in_code_block = False

    if "```" not in text and ('def ' in text or 'import ' in text):
        for line in lines:
             stripped_line = line.strip()
             if (not stripped_line or
                 stripped_line.startswith('#') or
                 stripped_line.startswith('print(')):
                 continue
             cleaned_lines.append(line) 

    else: 
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith('```python'):
                in_code_block = True
                continue 
            elif stripped_line.startswith('```') and in_code_block:
                in_code_block = False
                continue 
            elif stripped_line.startswith('```'):
                continue

            if in_code_block:
                if stripped_line.startswith('print('):
                    continue
                cleaned_lines.append(line) 
        
        if not cleaned_lines and "```" in text:
            code_parts = text.split("```")
            if len(code_parts) > 1:
                 potential_code = code_parts[1]
                 potential_lines = potential_code.strip().split('\n')
                 if potential_lines and potential_lines[0].strip().isalpha():
                     potential_lines = potential_lines[1:]
                 cleaned_lines = [line for line in potential_lines if not line.strip().startswith('print(')]


    code = '\n'.join(cleaned_lines).strip()

    if not code:
        logger.warning("Code extraction resulted in empty string. Using fallback.")
        return """
import pandas as pd
def analyze_data(df):
    result = "No analysis could be performed due to code extraction issues."
    return result
"""
    lines = code.split('\n')
    processed_lines = []
    indent_level = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        current_indent = len(line) - len(line.lstrip(' '))
        if stripped.endswith(':'):
            processed_lines.append(line)
            if i + 1 >= len(lines) or \
               (len(lines[i+1]) - len(lines[i+1].lstrip(' ')) <= current_indent) or \
               not lines[i+1].strip() or \
               lines[i+1].strip().startswith('#'):
                processed_lines.append(' ' * (current_indent + 4) + 'pass')
        else:
            processed_lines.append(line)

    final_code = '\n'.join(processed_lines)

    try:
        compile(final_code, '<string>', 'exec')
    except SyntaxError as e:
        logger.error(f"Generated code has syntax errors: {e}\nCode:\n{final_code}")
        return f"""
import pandas as pd
def analyze_data(df):
    result = "Generated code had syntax errors: {e}"
    return result
"""
    return final_code


def extract_python_code(response):
    if not response or not isinstance(response, str):
        logger.warning("Received empty or non-string response for visualization code.")
        return """
import matplotlib.pyplot as plt
import io
import base64

def create_visualization(df):
    try:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No visualization available (AI response error)",
                 ha='center', va='center', fontsize=14, color='red')
        plt.title("Error: Invalid AI Response")
        plt.axis('off')

        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode('utf8')
    except Exception as e_viz_fallback: # Renamed 'e' to avoid conflict if outer scope has 'e'
        print(f"Error in error visualization: {e_viz_fallback}")
        return None
"""

    lines = response.strip().split('\n')
    cleaned_lines = []
    in_code_block = False

    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith('```python'):
            in_code_block = True
            continue
        elif stripped_line.startswith('```') and in_code_block:
            in_code_block = False
            break 
        if in_code_block:
            cleaned_lines.append(line)

    if not cleaned_lines and "```" in response:
         code_parts = response.split("```")
         if len(code_parts) > 1:
             potential_code = code_parts[1]
             potential_lines = potential_code.strip().split('\n')
             if potential_lines and potential_lines[0].strip().isalpha():
                 cleaned_lines = potential_lines[1:]
             else:
                 cleaned_lines = potential_lines

    if not cleaned_lines:
        cleaned_lines = [line for line in lines if 'import ' in line or 'plt.' in line or 'sns.' in line or 'def ' in line or line.strip().startswith(' ')]
        if not cleaned_lines:
             logger.warning("Could not find code block in visualization response.")
             return extract_python_code(None) 


    code = '\n'.join(cleaned_lines).strip()

    if "def create_visualization(df):" not in code:
        logger.warning("Extracted code does not contain 'def create_visualization(df):'")
        if 'plt.' in code or 'sns.' in code:
            code = f"""
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import numpy as np
import pandas as pd

def create_visualization(df):
    try:
        plt.figure(figsize=(10, 6))
{code}
        plt.tight_layout()
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=300, bbox_inches='tight')
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode('utf8')
    except Exception as e_viz_generated: # Renamed 'e'
        print(f"Error during visualization execution: {{e_viz_generated}}")
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"Error generating plot: {{e_viz_generated}}", ha='center', va='center', fontsize=12, color='red')
        plt.title("Visualization Execution Error")
        plt.axis('off')
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode('utf8')
"""
        else:
            return extract_python_code(None)
    return code

def safe_exec_analysis(code, df):
    exec_globals = {
        'df': df.copy(), 
        'pd': pd,
        'np': np,
        'mean': mean,
        'median': median,
        'mode': mode,
        'stdev': stdev,
        'stats': stats,
        'result': None 
    }
    exec_locals = {}

    try:
        exec(code, exec_globals, exec_locals)
        final_result = exec_locals.get('result', exec_globals.get('result', None))

        if final_result is None:
            potential_results = {k: v for k, v in exec_locals.items() if not k.startswith('__') and not callable(v)}
            if potential_results:
                if len(potential_results) == 1:
                    final_result = list(potential_results.values())[0]
                else:
                     logger.warning(f"No explicit 'result' variable found. Available locals: {list(potential_results.keys())}")
                     final_result = "Analysis complete, but 'result' variable not found."
        
        logger.info(f"Raw analysis result type: {type(final_result)}")
        serializable_result = convert_numpy_types(final_result)
        logger.info("Analysis result converted for JSON serialization.")
        return serializable_result

    except Exception as e:
        logger.error(f"Execution error during analysis: {str(e)}", exc_info=True)
        logger.error(f"Problematic analysis code:\n{code}")
        return {"error": f"Analysis execution failed: {str(e)}", "code_executed": code}


def safe_llm_invoke(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            if llm is None:
                raise ValueError("LLM object is not initialized.")

            logger.info(f"Invoking LLM (Attempt {attempt+1}/{max_retries})...")
            response = llm.invoke(prompt)

            if hasattr(response, 'content'):
                content = response.content
                if content:
                    logger.info("LLM invocation successful.")
                    return content
                else:
                    logger.warning("LLM invocation successful but returned empty content.")
                    if attempt == max_retries - 1:
                        return "" 
                    else:
                        raise ValueError("LLM returned empty content")
            else:
                 logger.error(f"LLM response object does not have 'content' attribute. Response: {response}")
                 raise TypeError("Invalid LLM response structure")

        except Exception as e:
            logger.error(f"LLM invoke error (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                logger.error("Max retries reached for LLM invocation.")
                return f"Error: LLM invocation failed after {max_retries} attempts: {str(e)}"
            time.sleep(2 ** attempt) 
    return "Error: LLM invocation failed unexpectedly."


def create_error_visualization(error_message="An error occurred during visualization generation"):
    logger.info(f"Creating error visualization: {error_message}")
    try:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.6, "😔", ha='center', va='center', fontsize=60, color='gray')
        plt.text(0.5, 0.45, f"Error: {error_message}",
                ha='center', va='center', fontsize=14, color='red', wrap=True)
        plt.text(0.5, 0.3, "Could not generate the requested visualization.",
                ha='center', va='center', fontsize=10, color='black')
        plt.title("Visualization Error", fontsize=16)
        plt.axis('off') 

        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        plt.close() 
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode('utf8')
    except Exception as e_err_viz: # Renamed 'e'
         logger.error(f"Failed even to create error visualization: {e_err_viz}")
         return None

def create_fallback_visualization(df, question):
    """Create a more sophisticated fallback visualization when the generated code fails"""
    logger.info("Creating fallback visualization.")
    try:
        fig = plt.figure(figsize=(15, 12))
        fig.suptitle(f"Fallback Data Overview for: '{question}'", fontsize=16, y=0.99)

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        subplot_count = 0
        max_subplots = 4 
        grid_size = (2, 2)

        if subplot_count < max_subplots:
            subplot_count += 1
            ax_summary = fig.add_subplot(grid_size[0], grid_size[1], subplot_count)
            summary_text = f"Dataset Summary:\n"
            summary_text += f"- {df.shape[0]} rows, {df.shape[1]} columns\n"
            summary_text += f"- {len(numeric_cols)} numeric, {len(cat_cols)} categorical columns\n\n"

            if numeric_cols:
                summary_text += f"Numeric Columns (Top 3):\n"
                stats_df = df[numeric_cols].describe().loc[['mean', '50%', 'std']].T
                stats_df.columns = ['Mean', 'Median', 'Std Dev']
                summary_text += stats_df.head(3).to_string(float_format='%.2f') + "\n\n"

            # --- MODIFIED BLOCK TO HANDLE UNHASHABLE TYPES IN FALLBACK ---
            if cat_cols:
                summary_text += f"Categorical Columns (Top 3 by unique values):\n"
                cat_unique_counts = {}
                cat_top_values = {}
                processed_cat_cols_for_summary = []

                for col in cat_cols:
                    try:
                        col_series = df[col]
                        if col_series.dtype == 'object' and col_series.dropna().apply(lambda x: isinstance(x, (list, dict))).any():
                            s_transformed = col_series.astype(str)
                        else:
                            s_transformed = col_series
                        
                        cat_unique_counts[col] = s_transformed.nunique()
                        if not s_transformed.empty:
                            mode_val = s_transformed.mode()
                            cat_top_values[col] = mode_val.iloc[0] if not mode_val.empty else 'N/A'
                        else:
                            cat_top_values[col] = 'N/A'
                        processed_cat_cols_for_summary.append(col)
                    except TypeError as te: # Catching specific TypeError for unhashable
                        logger.warning(f"TypeError (likely unhashable) for cat_col '{col}' in fallback summary: {te}")
                        # Not adding to processed_cat_cols_for_summary
                    except Exception as e_cat_stats:
                        logger.warning(f"Could not get stats for cat_col '{col}' in fallback summary: {e_cat_stats}")
                
                if processed_cat_cols_for_summary:
                    valid_unique_counts = {k: cat_unique_counts[k] for k in processed_cat_cols_for_summary if k in cat_unique_counts}
                    valid_top_values = {k: cat_top_values[k] for k in processed_cat_cols_for_summary if k in cat_top_values}

                    if valid_unique_counts:
                        cat_summary_df = pd.DataFrame({
                            'Unique': pd.Series(valid_unique_counts),
                            'Top Value': pd.Series(valid_top_values)
                        })
                        if not cat_summary_df.empty:
                            cat_summary_df['sort_col'] = pd.to_numeric(cat_summary_df['Unique'], errors='coerce') # For robust sorting
                            cat_summary_df = cat_summary_df.sort_values('sort_col', na_position='last').drop(columns=['sort_col']).head(3)
                            summary_text += cat_summary_df.to_string() + "\n\n"
                        else:
                             summary_text += "  No categorical column stats could be displayed in summary.\n\n"
                    else:
                        summary_text += "  No categorical column stats could be generated for summary.\n\n"
                else:
                    summary_text += "  No categorical columns were suitable for summary statistics.\n\n"
            # --- END OF MODIFIED BLOCK ---

            ax_summary.text(0.05, 0.95, summary_text, ha='left', va='top', fontsize=9, family='monospace')
            ax_summary.set_title("Dataset Information", fontsize=12)
            ax_summary.axis('off')

        if len(numeric_cols) > 1 and subplot_count < max_subplots:
            subplot_count += 1
            ax_corr = fig.add_subplot(grid_size[0], grid_size[1], subplot_count)
            try:
                corr_data = df[numeric_cols].corr()
                mask = np.triu(np.ones_like(corr_data, dtype=bool))
                sns.heatmap(corr_data, annot=True, cmap='coolwarm', ax=ax_corr, fmt='.2f',
                            linewidths=0.5, cbar=True, annot_kws={"size": 8}, mask=mask)
                ax_corr.set_title("Numeric Correlation (Lower Triangle)", fontsize=12)
                plt.setp(ax_corr.get_xticklabels(), rotation=45, ha='right', fontsize=8)
                plt.setp(ax_corr.get_yticklabels(), rotation=0, ha='right', fontsize=8)
            except Exception as corr_err:
                 logger.warning(f"Could not generate correlation heatmap: {corr_err}")
                 ax_corr.text(0.5, 0.5, "Could not generate correlation plot", ha='center', va='center')
                 ax_corr.set_title("Numeric Correlation (Error)", fontsize=12)
                 ax_corr.axis('off')

        if numeric_cols and subplot_count < max_subplots:
            subplot_count += 1
            ax_hist = fig.add_subplot(grid_size[0], grid_size[1], subplot_count)
            try:
                cols_to_plot = numeric_cols[:min(len(numeric_cols), 4)]
                for col in cols_to_plot:
                     try:
                         sns.kdeplot(df[col].dropna(), ax=ax_hist, label=col, fill=True, alpha=0.3)
                     except Exception as kde_err:
                         logger.warning(f"KDE plot failed for {col}: {kde_err}")
                if cols_to_plot: 
                    ax_hist.set_title("Numeric Variable Distributions (KDE)", fontsize=12)
                    ax_hist.legend(fontsize=8)
                    ax_hist.set_xlabel("Value")
                    ax_hist.set_ylabel("Density")
                else:
                    ax_hist.text(0.5, 0.5, "No numeric data suitable for KDE", ha='center', va='center')
                    ax_hist.set_title("Numeric Distributions", fontsize=12)
                    ax_hist.axis('off')

            except Exception as hist_err:
                logger.warning(f"Could not generate distribution plot: {hist_err}")
                ax_hist.text(0.5, 0.5, "Could not generate distribution plot", ha='center', va='center')
                ax_hist.set_title("Numeric Distributions (Error)", fontsize=12)
                ax_hist.axis('off')

        if cat_cols and subplot_count < max_subplots:
            subplot_count += 1
            ax_bar = fig.add_subplot(grid_size[0], grid_size[1], subplot_count)
            suitable_cat_col = None
            for cat_col in cat_cols: # Find first categorical column with a reasonable number of unique values
                try:
                    # Check if column can be processed for nunique
                    if df[cat_col].dtype == 'object' and df[cat_col].dropna().apply(lambda x: isinstance(x, (list, dict))).any():
                        num_unique = df[cat_col].astype(str).nunique()
                    else:
                        num_unique = df[cat_col].nunique()
                    
                    if 1 < num_unique < 20:
                        suitable_cat_col = cat_col
                        break
                except TypeError: # Skip unhashable columns for this fallback plot selection
                    logger.warning(f"Skipping unhashable cat_col '{cat_col}' for fallback bar chart selection.")
                    continue
            
            if suitable_cat_col:
                try:
                    # For plotting, convert to string if it was complex
                    if df[suitable_cat_col].dtype == 'object' and df[suitable_cat_col].dropna().apply(lambda x: isinstance(x, (list, dict))).any():
                        plot_series = df[suitable_cat_col].astype(str)
                    else:
                        plot_series = df[suitable_cat_col]

                    counts = plot_series.value_counts().head(10) 
                    counts.plot(kind='bar', ax=ax_bar, color=sns.color_palette('viridis', len(counts)))
                    ax_bar.set_title(f"Top 10 Categories in '{suitable_cat_col}'", fontsize=12)
                    ax_bar.set_ylabel("Count")
                    plt.setp(ax_bar.get_xticklabels(), rotation=45, ha='right', fontsize=8)
                    ax_bar.tick_params(axis='x', labelsize=8) 
                    ax_bar.tick_params(axis='y', labelsize=8)
                except Exception as bar_err:
                    logger.warning(f"Could not generate bar chart for {suitable_cat_col}: {bar_err}")
                    ax_bar.text(0.5, 0.5, f"Error plotting '{suitable_cat_col}'", ha='center', va='center')
                    ax_bar.set_title("Categorical Data (Error)", fontsize=12)
                    ax_bar.axis('off')
            else:
                ax_bar.text(0.5, 0.5, "No suitable categorical column\n(with 2-19 unique values) found.",
                          ha='center', va='center', fontsize=10)
                ax_bar.set_title("Categorical Data", fontsize=12)
                ax_bar.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.97]) 
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=150) 
        plt.close(fig) 
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode('utf8') # Returns RAW base64

    except Exception as e_fallback: # Renamed 'e'
        logger.error(f"Error during fallback visualization generation: {str(e_fallback)}", exc_info=True)
        plt.close('all') 
        return create_error_visualization(f"Fallback visualization failed: {str(e_fallback)}")


def get_data_sample_and_metadata(df, max_rows=MAX_SAMPLE_ROWS):
    if df is None:
        return "No data loaded", {"error": "DataFrame is None"}

    try:
        df.columns = df.columns.astype(str).str.strip()
        sample_df_base = df.copy()

        for col in sample_df_base.columns:
            if sample_df_base[col].dtype == 'object':
                 needs_conversion = sample_df_base[col].apply(lambda x: isinstance(x, (list, dict))).any()
                 if needs_conversion:
                     sample_df_base[col] = sample_df_base[col].astype(str)


        total_rows = len(df)
        column_info = []

        for col in df.columns:
            col_data = df[col]
            col_info = {"name": col, "type": str(col_data.dtype)} 
            col_stats = {"null_count": int(col_data.isna().sum())}

            try:
                if pd.api.types.is_numeric_dtype(col_data):
                    col_info["type"] = "numeric"
                    stats_to_calc = {"min": col_data.min, "max": col_data.max, "mean": col_data.mean, "median": col_data.median, "std": col_data.std}
                    for stat_name, stat_func in stats_to_calc.items():
                        try:
                            value = stat_func()
                            if pd.notna(value):
                                col_stats[stat_name] = convert_numpy_types(value) 
                            else:
                                col_stats[stat_name] = None
                        except (TypeError, ValueError): 
                            col_stats[stat_name] = "Calculation Error"

                elif pd.api.types.is_datetime64_any_dtype(col_data) or pd.api.types.is_timedelta64_dtype(col_data):
                     col_info["type"] = "datetime" if pd.api.types.is_datetime64_any_dtype(col_data) else "timedelta"
                     try: col_stats["min"] = str(col_data.min()) if pd.notna(col_data.min()) else None
                     except: pass
                     try: col_stats["max"] = str(col_data.max()) if pd.notna(col_data.max()) else None
                     except: pass

                elif pd.api.types.is_categorical_dtype(col_data) or pd.api.types.is_object_dtype(col_data) or pd.api.types.is_string_dtype(col_data):
                     col_info["type"] = "categorical/text" 
                     
                     # Handle potential unhashable types for nunique/mode in metadata
                     try:
                         if col_data.dtype == 'object' and col_data.dropna().apply(lambda x: isinstance(x, (list, dict))).any():
                             s_transformed_meta = col_data.astype(str)
                         else:
                             s_transformed_meta = col_data
                         
                         col_stats["unique_values"] = int(s_transformed_meta.nunique())
                         if not s_transformed_meta.empty:
                             mode_val_meta = s_transformed_meta.mode()
                             if not mode_val_meta.empty:
                                 top_val_meta = mode_val_meta.iloc[0]
                                 col_stats["top_value"] = str(top_val_meta)
                                 col_stats["top_count"] = int((s_transformed_meta == top_val_meta).sum())
                             else:
                                  col_stats["top_value"] = None
                                  col_stats["top_count"] = 0
                         else:
                             col_stats["top_value"] = None
                             col_stats["top_count"] = 0
                     except TypeError:
                         logger.warning(f"Metadata: TypeError for nunique/mode on column {col}, likely unhashable. Reporting as error.")
                         col_stats["unique_values"] = "Error (unhashable)"
                         col_stats["top_value"] = "Error (unhashable)"
                         col_stats["top_count"] = "Error (unhashable)"
                     except Exception as e_meta_cat:
                         logger.warning(f"Could not get mode/top value for column {col} in metadata: {e_meta_cat}")
                         col_stats["top_value"] = "Error"
                         col_stats["top_count"] = "Error"
                else:
                     col_info["type"] = "other"

                col_info["stats"] = col_stats
                column_info.append(col_info)

            except Exception as e:
                logger.warning(f"Error processing metadata for column {col}: {str(e)}")
                column_info.append({
                    "name": col,
                    "type": "unknown",
                    "stats": {"error": str(e), "null_count": int(df[col].isna().sum())}
                })

        n_head = min(5, total_rows)
        n_tail = min(5, total_rows - n_head)
        n_random = min(max(0, max_rows - n_head - n_tail), total_rows - n_head - n_tail)

        sample_parts = []
        if n_head > 0:
            sample_parts.append(sample_df_base.head(n_head))
        if n_random > 0:
            middle_indices = sample_df_base.index[n_head:total_rows - n_tail]
            if len(middle_indices) >= n_random:
                random_indices = np.random.choice(middle_indices, n_random, replace=False)
                sample_parts.append(sample_df_base.loc[random_indices])
            elif len(middle_indices) > 0: 
                 sample_parts.append(sample_df_base.loc[middle_indices])
        if n_tail > 0:
            sample_parts.append(sample_df_base.tail(n_tail))

        if sample_parts:
            sample_df_final = pd.concat(sample_parts)
        else: 
            sample_df_final = sample_df_base
        
        try:
            sample_data_string = sample_df_final.to_markdown(index=False)
        except ImportError: 
             sample_data_string = sample_df_final.to_string(index=False)

        metadata = {
            "total_rows": total_rows,
            "total_columns": len(df.columns),
            "column_info": column_info,
            "sample_rows_provided": len(sample_df_final),
            "sample_description": f"Showing head ({n_head}), random ({len(sample_df_final)-n_head-n_tail}), and tail ({n_tail}) rows." if total_rows > max_rows else f"Showing all {total_rows} rows."
        }
        
        try:
            json_metadata = json.dumps(metadata)
            if len(json_metadata) > 50000: 
                logger.warning("Metadata size is large, potentially truncating column stats.")
                metadata['column_info'] = metadata['column_info'][:50] 
                metadata['warning'] = "Metadata truncated due to size limits."
        except (TypeError, OverflowError) as e_json: # Renamed 'e'
            logger.error(f"Could not serialize metadata: {e_json}")
            metadata = {"error": "Failed to serialize metadata", "total_rows": total_rows, "total_columns": len(df.columns)}

        return sample_data_string, metadata

    except Exception as e_sample_meta: # Renamed 'e'
        logger.error(f"Error in get_data_sample_and_metadata: {str(e_sample_meta)}", exc_info=True)
        return "Error retrieving sample", {
            "total_rows": len(df) if df is not None else 0,
            "total_columns": len(df.columns) if df is not None and hasattr(df, 'columns') else 0,
            "error": f"Failed to generate sample/metadata: {str(e_sample_meta)}"
        }


def load_data(df1, df2):
    global df
    try:
        logger.info("Loading data...")
        if df1 is None or not isinstance(df1, pd.DataFrame):
             logger.error("df1 is None or not a DataFrame.")
             return {"error": "Invalid input DataFrame (df1)"}, 400
        if df2 is not None and not isinstance(df2, pd.DataFrame):
            logger.warning("df2 is provided but not a DataFrame. It will be ignored.")

        df = df1.copy(deep=True)
        logger.info(f"Loaded DataFrame shape: {df.shape}")

        original_columns = df.columns.tolist()
        df.columns = df.columns.astype(str).str.strip()
        renamed_columns = {orig: new for orig, new in zip(original_columns, df.columns) if orig != new}
        if renamed_columns:
             logger.info(f"Renamed columns by stripping whitespace: {renamed_columns}")
        logger.info(f"Columns after stripping: {list(df.columns)}")
        
        df = df.infer_objects() 

        for col in df.columns:
             is_potential_date = False
             if any(date_term in str(col).lower() for date_term in ['date', 'time', 'day', 'month', 'year', '_dt', '_ts']):
                 is_potential_date = True
             
             if is_potential_date and df[col].dtype == 'object': 
                 try:
                     logger.info(f"Attempting datetime conversion for column: {col}")
                     parsed_dates = pd.to_datetime(df[col], errors='coerce')
                     success_rate = parsed_dates.notna().sum() / len(df[col]) if len(df[col]) > 0 else 0
                     if success_rate > 0.8: 
                          df[col] = parsed_dates
                          logger.info(f"Successfully converted '{col}' to datetime (success rate: {success_rate:.1%}).")
                     else:
                          logger.warning(f"Low success rate ({success_rate:.1%}) for datetime conversion in '{col}'. Keeping as object.")
                 except Exception as e_date_parse: # Renamed 'e'
                     logger.warning(f"Could not parse dates in column '{col}': {str(e_date_parse)}. Keeping as object.")
        
        logger.info(f"Data types after initial processing:\n{df.dtypes}")
        logger.info(f"Sample data (head):\n{df.head(3).to_string()}")
        
        sample_data, metadata = get_data_sample_and_metadata(df)

        if isinstance(metadata, dict) and metadata.get("error"):
            logger.error(f"Metadata generation failed: {metadata['error']}")
            return {"error": f"Failed to process data metadata: {metadata['error']}"}, 500

        return {
            "message": "Data loaded and processed successfully",
            "columns": list(df.columns),
            "rows": len(df),
            "dtypes": {k: str(v) for k, v in df.dtypes.items()}, 
            "metadata_preview": { 
                 "total_rows": metadata.get("total_rows"),
                 "total_columns": metadata.get("total_columns"),
                 "column_names": [c['name'] for c in metadata.get("column_info", [])[:10]] 
            },
            "sample_head": convert_numpy_types(df.head(3).to_dict(orient='records')) 
        }, 200

    except Exception as e_load_data: # Renamed 'e'
        logger.error(f"Error loading or processing data: {str(e_load_data)}", exc_info=True)
        df = None 
        return {"error": f"Error loading/processing data: {str(e_load_data)}"}, 500


def generate_summary(question, analysis_code, result, metadata):
    logger.info("Generating natural language summary...")
    try:
        result_str = json.dumps(convert_numpy_types(result), indent=2)
        if len(result_str) > 2000: 
            result_str = result_str[:2000] + "\n... (result truncated)"
    except (TypeError, OverflowError) as e_json_sum: # Renamed 'e'
        logger.error(f"Could not serialize result for summary prompt: {e_json_sum}")
        result_str = f"Error: Could not display result ({e_json_sum})"

    prompt = f"""
    You are a data analyst explaining results to a non-technical user.
    Dataset Metadata:
    {json.dumps(metadata, indent=2)}
    User's Question:
    "{question}"
    Python code used for analysis:
    ```python
    {analysis_code}
    ```
    Analysis Result:
    ```json
    {result_str}
    ```
    Task:
    Based ONLY on the provided information (question, code, result, metadata), write a concise and clear natural language summary of the findings that directly answers the user's question.
    - Start directly with the answer.
    - Focus on the key information from the result.
    - Do not add interpretations beyond the data shown.
    - Keep it brief (2-4 sentences).
    - Do not mention the python code itself in the summary.
    - If the result indicates an error or no data, state that clearly.
    
    IMPORTANT: The data is shown to the end user, so it should not include IDs. Insted, names and the descriptions are to be used. While a Student's data is shown please include the organization name and class name for him.
    Do not mention the exact column names, instead use the descriptions provided in the metadata.
    
    Example Summary for "What is the average age?":
    "The average age in the dataset is 35.6 years."
    Example Summary for "List unique cities":
    "The unique cities found are London, Paris, and Tokyo."
    Example Summary for error:
    "The analysis could not be completed due to an error: [Error message from result]."
    Your Summary:
    """   
    try:
        summary = safe_llm_invoke(prompt)
        if "Error: LLM invocation failed" in summary:
             logger.error(f"Summary generation failed: {summary}")
             return "Could not generate summary due to LLM error."
        logger.info(f"Generated Summary: {summary.strip()}")
        return summary.strip()
    except Exception as e_gen_sum: # Renamed 'e'
        logger.error(f"Error generating summary: {str(e_gen_sum)}", exc_info=True)
        return "An error occurred while generating the summary."


def check_visualization_needed(question, summary, result, metadata):
    logger.info("Checking if visualization is needed...")
    try:
        result_str = json.dumps(convert_numpy_types(result), indent=2)
        if len(result_str) > 1000: 
            result_str = result_str[:1000] + "\n... (result truncated)"
    except (TypeError, OverflowError) as e_json_viz_check: # Renamed 'e'
        result_str = f"Error: Could not display result ({e_json_viz_check})"

    prompt = f"""
    You are an assistant deciding if a data visualization is helpful.
    User's Question: "{question}"
    Analysis Summary: "{summary}"
    Analysis Result (raw data):
    ```json
    {result_str}
    ```
    Dataset Metadata Summary:
    - Total Rows: {metadata.get('total_rows', 'N/A')}
    - Total Columns: {metadata.get('total_columns', 'N/A')}
    - Column Names Sample: {metadata.get('column_names', 'N/A')} 
    Task:
    Based on the question, the summary, and the nature of the result (e.g., single number, list, table, statistics), would a standard chart (like a bar chart, line graph, pie chart, scatter plot, histogram, or even a formatted table visualization) significantly improve the understanding of the answer compared to the text summary alone?
    Consider:
    - Is the result complex (e.g., trends, distributions, comparisons across many categories)? -> YES
    - Is the result a simple value or short list that the summary already covers well? -> NO
    - Does the question imply comparison, trend, or distribution? -> YES
    - Is the result primarily text-based or qualitative? -> Maybe NO (unless comparing frequencies)
    - Is the result an error message? -> NO
    Answer ONLY with the word "YES" or "NO". Do not provide any explanation.
    Decision:
    """
    try:
        response = safe_llm_invoke(prompt)
        if "Error: LLM invocation failed" in response:
            logger.error(f"Visualization check failed: {response}")
            return False 

        decision = response.strip().upper()
        logger.info(f"LLM decision on visualization: '{decision}'")
        if re.search(r"^\s*YES\s*$", decision):
             return True
        else:
             if not re.search(r"^\s*NO\s*$", decision):
                 logger.warning(f"Visualization check returned unexpected response: '{response}'. Defaulting to NO.")
             return False
    except Exception as e_check_viz: # Renamed 'e'
        logger.error(f"Error checking visualization need: {str(e_check_viz)}", exc_info=True)
        return False 

def perform_analysis(question):
    global df
    if df is None:
        logger.error("perform_analysis called with no data loaded.")
        return {"error": "No data loaded yet. Please load data first."}, 400

    try:
        if not question or not isinstance(question, str) or not question.strip():
             logger.error("perform_analysis called with empty or invalid question.")
             return {"error": "No valid question provided for analysis."}, 400

        question = question.strip()
        logger.info(f"Starting analysis for question: {question}")
        
        df.columns = df.columns.astype(str).str.strip()
        logger.debug(f"Columns available for analysis: {list(df.columns)}")
        
        sample_data, metadata = get_data_sample_and_metadata(df)
        if isinstance(metadata, dict) and metadata.get("error"):
             logger.error(f"Failed to get metadata for analysis prompt: {metadata['error']}")
             return {"error": f"Internal error preparing analysis: {metadata['error']}"}, 500
        
        prompt = f"""
        You are an expert data analyst Python coder. Your task is to write Python code to answer a specific question based on a pandas DataFrame named 'df'.
        Dataset Sample (for context, use metadata for column details):
        ```markdown
        {sample_data}
        ```
        Dataset Metadata (Use this for available columns, types, and basic stats):
        {json.dumps(metadata, indent=2)}
        User question: "{question}"
        Instructions:
        1. Write Python code using pandas, numpy, statistics, or scipy to analyze the FULL dataset ('df') and answer the user's question.
        2. Base your analysis ONLY on columns listed in the metadata. Do NOT invent columns.
        3. Perform necessary data manipulations (grouping, filtering, calculations like count, mean, median, mode, stddev, unique values, correlations etc.) as required by the question.
        4. Ensure the code imports necessary libraries within the generated script block (e.g., `import pandas as pd`, `import numpy as np`).
        5. The final result of the analysis MUST be assigned to a variable named 'result'. Do NOT use print().
        6. The code must be a single, complete, executable Python script block.
        7. Ensure correct indentation and syntax.
        8. Handle potential errors gracefully if possible (e.g., check if columns exist before using them, handle division by zero).
        9. If the question cannot be answered with the available data (e.g., requires non-existent columns), the 'result' variable should contain a clear message stating why (e.g., "Column 'XYZ' not found in the dataset.").
        Example Code Structure:
        ```python
        import pandas as pd
        import numpy as np
        # Add other imports like statistics if needed

        def analyze_data(df):
            try:
                # Your analysis code here
                # Example: Calculate mean of 'Age' column
                # if 'Age' in df.columns and pd.api.types.is_numeric_dtype(df['Age']):
                #     calculated_mean = df['Age'].mean()
                #     result = f"The average age is {{calculated_mean:.2f}}"
                # else:
                #     result = "Could not calculate average age. 'Age' column missing or not numeric."
                result = "Replace this with your calculated result or final data structure (e.g., DataFrame, Series, list, dict, string)"
            except Exception as e_analyze_example: # Renamed e
                result = f"An error occurred during analysis: {{str(e_analyze_example)}}"
            return result
        ```
        Your Python Code (define the function `analyze_data(df)`):
        """

        logger.info("Requesting analysis code from LLM...")
        raw_code_response = safe_llm_invoke(prompt)

        if "Error: LLM invocation failed" in raw_code_response or not raw_code_response:
             logger.error(f"LLM failed to generate analysis code: {raw_code_response}")
             return {"error": f"AI failed to generate analysis code: {raw_code_response}"}, 500
        
        code_to_execute = extract_code(raw_code_response)
        if "def analyze_data(df):" not in code_to_execute:
            logger.warning("LLM analysis response missing 'def analyze_data(df):'. Wrapping the code.")
            code_to_execute = f"""
import pandas as pd
import numpy as np
import statistics
from scipy import stats

def analyze_data(df):
    try:
{code_to_execute}
        if 'result' not in locals():
             result = "Analysis code executed, but 'result' variable was not explicitly assigned."
    except Exception as e_analyze_wrap: # Renamed e
        result = f"An error occurred wrapping or executing the analysis code: {{str(e_analyze_wrap)}}"
    return result
"""
        logger.info(f"Generated analysis code (after cleaning):\n{code_to_execute}")
        
        exec_globals = {
            'pd': pd, 'np': np, 'mean': mean, 'median': median, 'mode': mode, 'stdev': stdev, 'stats': stats,
            'analyze_data': None 
        }
        exec(code_to_execute, exec_globals)
        
        analysis_func = exec_globals.get('analyze_data')
        if not callable(analysis_func):
             logger.error("Failed to define 'analyze_data' function from generated code.")
             return {"error": "Internal error: Failed to create analysis function from AI code.", "analysis_code": code_to_execute}, 500

        logger.info("Executing generated analysis function...")
        analysis_result = analysis_func(df.copy()) 

        if isinstance(analysis_result, dict) and 'error' in analysis_result:
            logger.error(f"Analysis code execution resulted in an error: {analysis_result['error']}")
            return {
                "error": f"Analysis failed: {analysis_result['error']}",
                "analysis_code": code_to_execute
            }, 500 

        if isinstance(analysis_result, str) and ("error occurred" in analysis_result.lower() or "could not" in analysis_result.lower()):
             logger.warning(f"Analysis function reported an issue: {analysis_result}")
             return {
                 "analysis_code": code_to_execute,
                 "result": analysis_result, 
                 "warning": "Analysis completed, but reported issues."
                 }, 200

        serializable_result = convert_numpy_types(analysis_result)
        logger.info("Analysis execution successful.")

        return {
            "analysis_code": code_to_execute,
            "result": serializable_result,
        }, 200

    except Exception as e_perform_analysis: # Renamed e
        logger.error(f"Unexpected error in perform_analysis function: {str(e_perform_analysis)}", exc_info=True)
        return {"error": f"Unexpected analysis system error: {str(e_perform_analysis)}"}, 500


def create_visualization(question, analysis_code, table_instructions, result):
    """Generate visualization code and execute it"""
    global df, visualization_errors
    if df is None:
        logger.error("create_visualization called with no data loaded.")
        return {"error": "No data loaded yet"}, 400

    if visualization_errors >= MAX_VISUALIZATION_ERRORS:
        logger.warning(f"Max visualization errors ({MAX_VISUALIZATION_ERRORS}) reached. Skipping generation.")
        return {
            "error": f"Skipping visualization due to previous errors ({visualization_errors}).",
            "graph_url": create_error_visualization(f"Skipped due to {visualization_errors} prior errors") # Returns RAW base64
        }, 200

    try:
        if not question or result is None:
            logger.error("create_visualization called with missing question or result.")
            return {"error": "Missing required arguments (question, result) for visualization"}, 400

        logger.info(f"Starting visualization generation for question: {question}")
        sample_data, metadata = get_data_sample_and_metadata(df)
        if isinstance(metadata, dict) and metadata.get("error"):
             logger.error(f"Failed to get metadata for visualization prompt: {metadata['error']}")
             return {"error": f"Internal error preparing visualization: {metadata['error']}"}, 500

        prompt = f"""
            You are an expert data scientist specializing in Python visualization using matplotlib and seaborn.
            Dataset Sample (for context):
            ```markdown
            {sample_data}
            ```
            Dataset Metadata (for column details):
            {json.dumps(metadata, indent=2)}
            User question: "{question}"
            Analysis Code Used (for context):
            ```python
            {analysis_code if analysis_code else '# No analysis code provided #'}
            ```
            Analysis Result (to be visualized):
            ```json
            {json.dumps(convert_numpy_types(result), indent=2, default=str)}
            ```
            Table/Formatting Instructions (Optional): "{table_instructions}"
            Task:
            Write a Python function `create_visualization(df)` that generates the SINGLE most appropriate and informative visualization based on the user's question and the provided analysis result.
            Requirements:
            1.  **Function Definition:** Create a Python function `create_visualization(df)` that takes the *full* pandas DataFrame `df` as input.
            2.  **Appropriate Chart:** Choose the *best* chart type.
            3.  **Use Data:** The function should use the input DataFrame `df` and potentially the `result` data.
            4.  **Clarity:** Ensure the plot is clear, with a descriptive title, labeled axes, and a legend if necessary. Use `plt.tight_layout()`.
            5.  **Single Plot:** Generate only ONE plot.
            6.  **Imports:** Include necessary imports INSIDE the function or ensure they are standard.
            7.  **Output:** The function must return a base64 encoded PNG image string. Do NOT `plt.show()`.
            8.  **Error Handling:** Include basic `try...except Exception as e_generated_viz_code:` within the function to catch plotting errors and return an error visualization if necessary, using `e_generated_viz_code` in the message.
            9.  **Code Only:** Provide *only* the Python code for the `create_visualization(df)` function, enclosed in ```python ... ``` markers.

            Example (Bar Chart):
            ```python
            import matplotlib.pyplot as plt
            import seaborn as sns
            import io
            import base64
            import pandas as pd

            def create_visualization(df):
                try:
                    plt.figure(figsize=(10, 6))
                    # Example: data_to_plot = pd.Series(result) # Access 'result' if needed from outer scope
                    # sns.barplot(x=data_to_plot.index, y=data_to_plot.values)
                    # plt.title('Example Bar Chart from Result')
                    # plt.xlabel('Categories'); plt.ylabel('Values')
                    # plt.xticks(rotation=45, ha='right')
                    # Dummy plot for example:
                    df.head(5).plot(kind='bar', ax=plt.gca()) # Use plt.gca() to get current axes
                    plt.title('Example Plot of DataFrame Head')

                    plt.tight_layout()
                    img = io.BytesIO()
                    plt.savefig(img, format='png', dpi=150, bbox_inches='tight')
                    img.seek(0)
                    plt.close() 
                    return base64.b64encode(img.getvalue()).decode('utf8')
                except Exception as e_generated_viz_code: # Using specific exception variable name
                    plt.close() # Ensure plot is closed on error too
                    plt.figure(figsize=(10, 6))
                    plt.text(0.5, 0.5, f'Error in generated plot code: {{str(e_generated_viz_code)}}', ha='center', va='center', color='red', wrap=True)
                    plt.title('Visualization Generation Error (in AI code)')
                    plt.axis('off')
                    img_err = io.BytesIO()
                    plt.savefig(img_err, format='png', bbox_inches='tight')
                    img_err.seek(0)
                    plt.close()
                    return base64.b64encode(img_err.getvalue()).decode('utf8')
            ```
            Your Python Code:
            """

        logger.info("Requesting visualization code from LLM...")
        response = safe_llm_invoke(prompt)

        if "Error: LLM invocation failed" in response or not response:
             logger.error(f"LLM failed to generate visualization code: {response}")
             visualization_errors += 1
             return {
                 "error": f"AI failed to generate visualization code: {response}",
                 "graph_url": create_fallback_visualization(df, question) # Returns RAW base64
             }, 200

        logger.info("Extracting visualization code...")
        vis_code = extract_python_code(response) # This ensures 'def create_visualization(df):' structure

        namespace = {
            'df': df.copy(), 'plt': plt, 'sns': sns, 'io': io, 'base64': base64,
            'pd': pd, 'np': np, 'result': result, # Pass analysis result
            'create_visualization': None
        }

        logger.info("Executing visualization code...")
        try:
            exec(vis_code, namespace)
            vis_function = namespace.get('create_visualization')

            if callable(vis_function):
                 logger.info("Executing 'create_visualization' function...")
                 try:
                     # Execute the LLM-generated function, it should return RAW base64
                     raw_image_base64 = vis_function(namespace['df']) # Pass the df copy

                     if raw_image_base64 and isinstance(raw_image_base64, str):
                         logger.info("Raw base64 for visualization generated successfully by LLM code.")
                         visualization_errors = 0
                         plt.close('all')
                         return {
                             "visualization_code": vis_code,
                             "graph_url": raw_image_base64 # <<< CRITICAL: Return RAW base64
                         }, 200
                     else:
                         logger.error("Visualization function did not return a valid raw base64 string.")
                         visualization_errors += 1
                         raise ValueError("Generated visualization function returned invalid or empty output.")
                 except Exception as func_error: # Catches errors from *inside* the LLM's vis_function
                     logger.error(f"Error executing the generated visualization function: {str(func_error)}", exc_info=True)
                     visualization_errors += 1
                     plt.close('all')
                     # The LLM's function should ideally have its own try-except and return an error image (raw base64)
                     # If it crashes before that, or returns None, we use fallback.
                     fallback_raw_base64 = create_fallback_visualization(df, question) # Returns RAW base64
                     return {
                         "error": f"Generated visualization code failed during execution: {str(func_error)}",
                         "visualization_code": vis_code,
                         "graph_url": fallback_raw_base64 # <<< CRITICAL: Return RAW base64 from fallback
                     }, 200
            else:
                 logger.error("Generated code did not define a callable 'create_visualization' function.")
                 visualization_errors += 1
                 raise ValueError("Function 'create_visualization' not found in generated code")
        except Exception as exec_error: # Catches syntax errors in vis_code or the ValueError above
            logger.error(f"Error executing the visualization code block or function not found: {str(exec_error)}", exc_info=True)
            visualization_errors += 1
            plt.close('all')
            fallback_raw_base64_exec = create_fallback_visualization(df, question) # Returns RAW base64
            return {
                "error": f"Syntax or execution error in generated visualization code: {str(exec_error)}",
                "visualization_code": vis_code, # Still return the problematic code
                "graph_url": fallback_raw_base64_exec # <<< CRITICAL: Return RAW base64 from fallback
            }, 200
    except Exception as outer_most_error: # Catches errors in this function's setup
        visualization_errors += 1
        logger.error(f"Unexpected error in create_visualization host function: {str(outer_most_error)}", exc_info=True)
        plt.close('all')
        fallback_raw_base64_outer = create_fallback_visualization(df, question) # Returns RAW base64
        return {
            "error": f"Unexpected visualization system error: {str(outer_most_error)}",
            "graph_url": fallback_raw_base64_outer # <<< CRITICAL: Return RAW base64 from fallback
        }, 200


def preprocess_data(operations):
    global df
    if df is None:
        logger.error("preprocess_data called with no data loaded.")
        return {"error": "No data loaded yet"}, 400
    if not operations or not isinstance(operations, list):
         logger.warning("No valid preprocessing operations provided.")
         return {"message": "No preprocessing operations performed.", "new_shape": df.shape}, 200

    logger.info(f"Starting preprocessing with operations: {operations}")
    results = {}
    original_shape = df.shape

    try:
        temp_df = df.copy() 

        for i, op in enumerate(operations):
            op_type = op.get('type', '').lower()
            op_key = f"op_{i}_{op_type}"
            logger.info(f"Applying operation: {op}")

            try:
                if op_type == 'fillna':
                    column = op.get('column')
                    value = op.get('value', '') 
                    if column and column in temp_df.columns:
                        original_nulls = temp_df[column].isna().sum()
                        temp_df[column] = temp_df[column].fillna(value)
                        filled_count = original_nulls - temp_df[column].isna().sum()
                        results[op_key] = f"success (filled {filled_count} nulls in '{column}')"
                        logger.info(f"fillna on '{column}' successful.")
                    elif not column:
                        results[op_key] = "error: 'column' parameter missing for fillna"
                        logger.warning("fillna skipped: column missing.")
                    else:
                        results[op_key] = f"error: column '{column}' not found"
                        logger.warning(f"fillna skipped: column '{column}' not found.")

                elif op_type == 'dropna':
                    columns = op.get('columns') 
                    how = op.get('how', 'any').lower() 
                    thresh = op.get('threshold') 
                    subset = [col for col in columns if col in temp_df.columns] if columns else None
                    if columns and not subset:
                         results[op_key] = f"error: none of the specified columns {columns} found"
                         logger.warning(f"dropna skipped: columns {columns} not found.")
                         continue

                    original_rows = temp_df.shape[0]
                    temp_df = temp_df.dropna(subset=subset, how=how, thresh=thresh)
                    rows_dropped = original_rows - temp_df.shape[0]
                    results[op_key] = f"success (dropped {rows_dropped} rows)"
                    logger.info(f"dropna successful (subset={subset}, how={how}, thresh={thresh}).")

                elif op_type == 'drop_duplicates':
                    columns = op.get('columns') 
                    keep = op.get('keep', 'first').lower() 
                    subset = [col for col in columns if col in temp_df.columns] if columns else None
                    if columns and not subset:
                         results[op_key] = f"error: none of the specified columns {columns} found"
                         logger.warning(f"drop_duplicates skipped: columns {columns} not found.")
                         continue

                    original_rows = temp_df.shape[0]
                    temp_df = temp_df.drop_duplicates(subset=subset, keep=keep)
                    rows_dropped = original_rows - temp_df.shape[0]
                    results[op_key] = f"success (dropped {rows_dropped} duplicate rows)"
                    logger.info(f"drop_duplicates successful (subset={subset}, keep={keep}).")

                elif op_type == 'rename_column':
                    old_name = op.get('old_name')
                    new_name = op.get('new_name')
                    if old_name and new_name and old_name in temp_df.columns:
                        temp_df = temp_df.rename(columns={old_name: new_name})
                        results[op_key] = f"success (renamed '{old_name}' to '{new_name}')"
                        logger.info(f"Renamed '{old_name}' to '{new_name}'.")
                    elif not old_name or not new_name:
                         results[op_key] = "error: 'old_name' or 'new_name' missing"
                         logger.warning("rename_column skipped: missing parameters.")
                    else:
                         results[op_key] = f"error: column '{old_name}' not found"
                         logger.warning(f"rename_column skipped: column '{old_name}' not found.")
                else:
                    results[op_key] = f"error: unknown operation type '{op_type}'"
                    logger.warning(f"Skipped unknown operation type: {op_type}")

            except Exception as e_op: # Renamed 'e'
                error_msg = f"error during operation {op}: {str(e_op)}"
                results[op_key] = error_msg
                logger.error(f"Error applying operation {op}: {str(e_op)}", exc_info=True)
        
        df = temp_df
        logger.info(f"Preprocessing completed. Original shape: {original_shape}, New shape: {df.shape}")
        return {
            "message": "Preprocessing completed.",
            "operations_results": results,
            "original_shape": original_shape,
            "new_shape": df.shape
        }, 200

    except Exception as e_preprocess: # Renamed 'e'
         logger.error(f"Unexpected error during preprocessing: {str(e_preprocess)}", exc_info=True)
         return {"error": f"Unexpected preprocessing system error: {str(e_preprocess)}"}, 500


def main(file_path, question, df1, df2, operations=None, table_instructions=None):
    global df, visualization_errors 
    start_time = time.time()
    logger.info("--- Starting Main Process ---")
    final_result = {}

    logger.info("Step 1: Loading Data...")
    load_result, status_code = load_data(df1, df2)
    final_result["data_info"] = load_result 
    if status_code != 200:
        logger.error(f"Data loading failed with status {status_code}: {load_result.get('error')}")
        return final_result, status_code 

    preprocess_info = {"status": "Not run"}
    if operations:
        logger.info("Step 2: Preprocessing Data...")
        preprocess_result, status_code = preprocess_data(operations)
        preprocess_info = preprocess_result 
        if status_code != 200:
            logger.error(f"Preprocessing failed: {preprocess_result.get('error')}")
            preprocess_info["error"] = preprocess_result.get('error', 'Unknown preprocessing error')
            logger.warning("Proceeding with original data after preprocessing error.")
        else:
             logger.info("Preprocessing successful.")
    final_result["preprocessing"] = preprocess_info

    logger.info("Fetching final metadata after potential preprocessing...")
    _, final_metadata = get_data_sample_and_metadata(df)
    if isinstance(final_metadata, dict) and final_metadata.get("error"):
         logger.error(f"Failed to get final metadata: {final_metadata['error']}")
         final_result["error"] = f"Internal error fetching final metadata: {final_metadata['error']}"
         return final_result, 500

    logger.info("Step 3: Performing Analysis...")
    analysis_output, analysis_status_code = perform_analysis(question) # Renamed variables for clarity
    analysis_outcome = {
        "result": analysis_output.get("result"),
        "analysis_code": analysis_output.get("analysis_code"),
        "error": analysis_output.get("error"),
        "warning": analysis_output.get("warning")
    }
    final_result["analysis"] = analysis_outcome
    if analysis_status_code != 200 or analysis_outcome.get("error"):
        logger.error(f"Analysis failed: Status {analysis_status_code}, Error: {analysis_outcome.get('error')}")
        final_result["summary"] = {"text": "Analysis could not be completed.", "error": analysis_outcome.get('error')}
        final_result["visualization"] = {"needed": False, "error": "Skipped due to analysis failure."}
        total_time = time.time() - start_time
        logger.info(f"--- Main Process Ended (Analysis Error) in {total_time:.2f} seconds ---")
        return final_result, 200 

    actual_analysis_result = analysis_outcome.get("result")
    analysis_code_for_summary_viz = analysis_outcome.get("analysis_code", "") # Use this name for clarity

    logger.info("Step 4: Generating Summary...")
    summary_text = generate_summary(question, analysis_code_for_summary_viz, actual_analysis_result, final_metadata)
    final_result["summary"] = {"text": summary_text}
    if "Could not generate summary" in summary_text:
         final_result["summary"]["error"] = "Failed to generate summary text."

    logger.info("Step 5: Checking if Visualization is Needed...")
    needs_visualization = check_visualization_needed(question, summary_text, actual_analysis_result, final_metadata)
    final_result["visualization"] = {"needed": needs_visualization} 

    if needs_visualization:
        logger.info("Step 6: Creating Visualization (Condition Met)...")
        if visualization_errors >= MAX_VISUALIZATION_ERRORS:
             logger.warning(f"Skipping visualization generation due to max errors ({visualization_errors}) reached.")
             final_result["visualization"]["graph_url"] = create_error_visualization(f"Skipped due to {visualization_errors} prior errors") # RAW
             final_result["visualization"]["error"] = f"Skipped due to {visualization_errors} prior errors."
             final_result["visualization"]["visualization_code"] = None
        else:
            visualization_output_dict, viz_status_code = create_visualization( # Renamed for clarity
                question,
                analysis_code_for_summary_viz,
                table_instructions or "",
                actual_analysis_result
            )
            final_result["visualization"].update({
                "graph_url": visualization_output_dict.get("graph_url"), # This is NOW RAW base64
                "visualization_code": visualization_output_dict.get("visualization_code"),
                "error": visualization_output_dict.get("error") 
            })
            if viz_status_code != 200 or visualization_output_dict.get("error"):
                 logger.warning(f"Visualization generation finished with status {viz_status_code} or error: {visualization_output_dict.get('error')}")
            else:
                 logger.info("Visualization generated successfully.")
    else:
        logger.info("Step 6: Skipping Visualization (Condition Not Met)...")
        final_result["visualization"]["message"] = "Visualization deemed not necessary or helpful for this question/result."
        final_result["visualization"]["graph_url"] = None
        final_result["visualization"]["visualization_code"] = None

    total_time = time.time() - start_time
    logger.info(f"--- Main Process Ended Successfully in {total_time:.2f} seconds ---")
    return final_result, 200

if __name__ == '__main__':
    try:
        data1 = {'colA': [1, 2, 3, 4, 5], 'colB': ['apple', 'banana', 'apple', 'orange', 'banana'], 'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-01', '2023-01-03', '2023-01-02'])}
        test_df1 = pd.DataFrame(data1)
        test_df2 = None 

        if not os.path.exists("config.py") or not hasattr(config, "API_KEY"):
             print("ERROR: config.py not found or API_KEY not set inside it.")
             print("Please create config.py with the line: API_KEY='YOUR_GOOGLE_API_KEY'")
             exit()
        
        print("\n--- Test Case 1: Count ---")
        question1 = "How many rows are there in the dataset?"
        result1, status1 = main(None, question1, test_df1, test_df2)
        print(f"Status Code: {status1}")
        print(f"Summary: {result1.get('summary', {}).get('text')}")
        print(f"Visualization Needed: {result1.get('visualization', {}).get('needed')}")
        print(f"Graph URL Present: {bool(result1.get('visualization', {}).get('graph_url'))}")

        print("\n--- Test Case 2: Value Counts ---")
        question2 = "Show the counts for each value in colB."
        result2, status2 = main(None, question2, test_df1, test_df2)
        print(f"Status Code: {status2}")
        print(f"Summary: {result2.get('summary', {}).get('text')}")
        print(f"Visualization Needed: {result2.get('visualization', {}).get('needed')}")
        print(f"Graph URL (first 50 chars if exists): {str(result2.get('visualization', {}).get('graph_url'))[:50] if result2.get('visualization', {}).get('graph_url') else 'None'}")

        print("\n--- Test Case 3: Analysis Error ---")
        question3 = "What is the average value of 'non_existent_column'?"
        result3, status3 = main(None, question3, test_df1, test_df2)
        print(f"Status Code: {status3}")
        print(f"Analysis Error: {result3.get('analysis', {}).get('error')}")
        print(f"Summary: {result3.get('summary', {}).get('text')}")
        print(f"Visualization Needed: {result3.get('visualization', {}).get('needed')}")

        print("\n--- Test Case 4: With Preprocessing ---")
        data_with_nulls = {'colA': [1, 2, None, 4, 5, 1], 'colB': ['x', 'y', 'x', 'z', 'y', 'x']}
        test_df_nulls = pd.DataFrame(data_with_nulls)
        question4 = "What is the most frequent value in colB after removing duplicates in colA?"
        preprocess_ops = [
            {'type': 'fillna', 'column': 'colA', 'value': 0},
            {'type': 'drop_duplicates', 'columns': ['colA'], 'keep': 'first'}
        ]
        result4, status4 = main(None, question4, test_df_nulls, None, operations=preprocess_ops)
        print(f"Status Code: {status4}")
        print(f"Preprocessing Results: {result4.get('preprocessing', {}).get('operations_results')}")
        print(f"Analysis Result: {result4.get('analysis', {}).get('result')}")
        print(f"Summary: {result4.get('summary', {}).get('text')}")
        print(f"Visualization Needed: {result4.get('visualization', {}).get('needed')}")

    except Exception as e_main_test:
        print(f"\nAn error occurred during the example run: {e_main_test}")
        import traceback
        traceback.print_exc()