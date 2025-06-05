# AI-Powered Data Analysis and Visualization API

This project is a Flask-based backend API that connects to a PostgreSQL database, processes user questions (both pre-defined and custom natural language queries via Google's Gemini AI), and generates data-driven insights, summaries, and visualizations.

## Features

*   **Database Interaction:** Connects to a PostgreSQL database using SQLAlchemy to fetch data for analysis.
*   **Pre-defined Questions:** Loads and processes analysis for a set of pre-defined questions stored in text files.
*   **Custom Question Processing:** Integrates with Google's Gemini AI to understand natural language questions, generate Python analysis/visualization code, and produce summaries.
*   **Dynamic Visualization:** Executes generated Python code (using Pandas, Matplotlib, Seaborn) to create visualizations and returns them as base64 encoded images.
*   **Persistent Storage:** Saves user questions, AI-generated summaries, and visualization code to local files within project-specific directories.
*   **Parameterization:** Supports dynamic parameter selection for pre-defined analyses.
*   **CORS Enabled:** Allows cross-origin requests, suitable for frontend integration.
*   **Environment Configuration:** Uses `.env` files for managing sensitive credentials and configurations.

## Prerequisites

Before you begin, ensure you have met the following requirements:

*   **Python:** Version 3.8+
*   **pip:** Python package installer
*   **PostgreSQL:** A running PostgreSQL server instance (version [Your PostgreSQL Version, e.g., 12+])
*   **Google Gemini API Key:** You'll need an API key from Google AI Studio or Google Cloud for the `gemini_new.py` functionality.
*   **Git:** For cloning the repository.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [Your Repository URL]
    cd [Your Repository Directory Name]
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    Create a `requirements.txt` file (if you don't have one already) by running:
    ```bash
    pip freeze > requirements.txt
    ```
    Then install the packages:
    ```bash
    pip install -r requirements.txt
    ```
    Key dependencies include: `Flask`, `Flask-CORS`, `SQLAlchemy`, `psycopg2-binary` (or `psycopg2`), `pandas`, `matplotlib`, `seaborn`, `python-dotenv`, `google-generativeai`, `urllib3`.

4.  **Set up the PostgreSQL Database:**
    *   Ensure your PostgreSQL server is running.
    *   Create the database specified in your `DB_CREDENTIALS` (e.g., `Lingotran_AnalyticsDB`).
    *   Ensure the user specified (e.g., `postgres`) has the necessary permissions on this database.
    *   Make sure the tables (`LrnrPromptProgress`, `LearnersProgress`, `Overview`) exist and are populated with data.

5.  **Configure Environment Variables:**
    Create a `.env` file in the root project directory with your specific configurations:
    ```env
    # Database Credentials
    DB_USER="postgres"
    DB_PASSWORD="Lingotran@123" # Ensure this is properly URL-encoded if needed elsewhere, but for this app, quote_plus handles it.
    DB_DATABASE="Lingotran_AnalyticsDB"
    DB_HOST="10.10.20.73"
    DB_PORT="5432"

    # Google Gemini API Key (used by gemini_new.py)
    GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"

    # Base URL for the application (optional, but good practice)
    # If deploying, change this to your production URL.
    BASE_URL="http://127.0.0.1:5001"
    # PORT (optional, if you want to override the default 5001)
    # PORT=5001
    ```
    **Important:** Add `.env` to your `.gitignore` file to avoid committing sensitive credentials.

## Running the Application

1.  Ensure your PostgreSQL server is running and accessible with the configured credentials.
2.  Ensure your `.env` file is correctly configured.
3.  From the project root directory (with the virtual environment activated), run:
    ```bash
    python main_new.py
    ```
    The application will start by default on `http://127.0.0.1:5001` (or as configured by `BASE_URL`/`PORT` in `.env`).

    For production deployment (e.g., on a cloud server), you would typically use a WSGI server like Gunicorn:
    ```bash
    gunicorn --bind 0.0.0.0:[PORT_NUMBER] main_new:app
    ```
    (Replace `[PORT_NUMBER]` with your desired port).

## Project Structure

*   `main_new.py`: The main Flask application file.
*   `gemini_new.py`: Handles interaction with the Google Gemini AI for custom question processing.
*   `templates/index_new1.html`: The basic HTML frontend served by the application.
*   `project_data/`: Base directory for storing project-specific files.
    *   `default_data/`: Default project directory created on startup.
        *   `user_questions.txt`: Stores user questions and categories.
        *   `analysis_code.txt`: Stores AI-generated summaries and Python visualization code.
*   `.env`: (To be created by user) Stores environment variables (DB credentials, API keys).
*   `requirements.txt`: Lists Python dependencies.

## API Endpoints

A brief overview of the main API endpoints:

*   `GET /`: Serves the `index_new1.html` frontend.
*   `GET /config`: Returns basic configuration like `BASE_URL`.
*   `GET /questions`: Fetches the list of pre-defined questions for the current project.
*   `POST /sendCustomQuestion`: Accepts a custom natural language question, processes it via Gemini AI, and stores the analysis.
    *   Payload: `{ "question": "Your natural language question" }`
*   `POST /process_question`: Processes a pre-defined question (by key) or a newly added custom question, executes its associated analysis code, and returns the summary and visualization.
    *   Payload: `{ "question": "UQ1.1", "params": { "param_name": "value" }, "sheet_index": 0 }`
*   `POST /get_parameter_options`: Fetches unique values for a given column to populate filter dropdowns.
    *   Payload: `{ "column_name": "Grade", "sheet_index": 0 }`
*   `POST /get_column_names`: Fetches all unique column names from the database tables.
*   `GET /get_current_project`: Returns details about the currently loaded project.
*   `POST /get_project_files`: Fetches the content of `user_questions.txt` and `analysis_code.txt` for a given project.
    *   Payload: `{ "project_dir": "project_data/default_data" }`

## Deployment to Cloud

When deploying to a cloud environment (e.g., AWS EC2, Google Cloud Run, Heroku, Azure App Service):

1.  **Environment Variables:** Ensure all necessary environment variables (`DB_USER`, `DB_PASSWORD`, `DB_DATABASE`, `DB_HOST`, `DB_PORT`, `GOOGLE_API_KEY`, `BASE_URL`) are securely configured in your cloud provider's environment settings. Do NOT hardcode them or commit the `.env` file.
2.  **Database Accessibility:** The cloud application instance must be able to connect to your PostgreSQL database. This might involve configuring firewall rules, VPC peering, or using a cloud-managed database service.
3.  **WSGI Server:** Use a production-grade WSGI server like Gunicorn or uWSGI instead of Flask's built-in development server.
    Example Gunicorn command: `gunicorn --workers 4 --bind 0.0.0.0:$PORT main_new:app` (Cloud providers often set the `$PORT` environment variable).
4.  **`requirements.txt`:** Ensure this file is up-to-date and used to install dependencies in the cloud environment.
5.  **`project_data` Directory:**
    *   This directory is created by the application if it doesn't exist.
    *   For cloud environments with ephemeral filesystems (like Cloud Run or serverless functions if you adapt this), you might need to use a persistent storage solution (like a cloud bucket or a mounted file system) for `project_data` if you need the generated question/analysis files to persist across deployments or instances.
    *   If persistence of `project_data` across restarts/deployments isn't critical or is handled by a backing data store eventually, the current setup might be acceptable for some cloud platforms.
6.  **Matplotlib Backend:** The line `matplotlib.use('Agg')` is important for non-GUI server environments, ensuring Matplotlib doesn't try to use an interactive backend.
7.  **Logging:** Configure proper logging to capture application output and errors in your cloud environment (e.g., CloudWatch, Stackdriver).
8.  **HTTPS:** Ensure your application is served over HTTPS in production, typically handled by a load balancer or reverse proxy in front of your application.

## Troubleshooting

*   **Database Connection Issues:**
    *   Verify `DB_CREDENTIALS` in `main_new.py` or environment variables in `.env` are correct.
    *   Check network connectivity and firewall rules between the application server and the database server.
    *   Ensure the PostgreSQL server is running and the specified database/user exists.
*   **Gemini API Issues:**
    *   Ensure your `GOOGLE_API_KEY` is valid and has the Gemini API enabled.
    *   Check for rate limits or billing issues with your Google Cloud/AI account.
*   **File Permissions:** If running in a restricted environment, ensure the application has write permissions for the `BASE_DATA_DIR` (`project_data/`).
*   **`No module named ...`:** Ensure all dependencies from `requirements.txt` are installed in your active virtual environment.