from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests
import json
import re
from dotenv import load_dotenv
import msal
import pyodbc
import platform

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],  
)

SERVER = os.getenv("SQL_SERVER", "kalpita.database.windows.net")
DATABASE = os.getenv("SQL_DATABASE", "KalpitaRecruit-Dev")
USERNAME = os.getenv("SQL_USERNAME")
TENANT_ID = os.getenv("TENANT_ID")
CLIENT_ID = os.getenv("CLIENT_ID")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api")

class LoginRequest(BaseModel):
    email: str

class AskLlamaRequest(BaseModel):
    question: str
    userEmail: str
    userRoles: list[dict]
    accessibleTables: list[str]
    format: str = "text"

class ConversationalRequest(BaseModel):
    prompt: str
    userEmail: str
    userRoles: list[dict]

class MessageClassificationRequest(BaseModel):
    message: str
    accessibleTables: list[str]
    userEmail: str
    userRoles: list[dict]

def get_access_token():
    authority = f"https://login.microsoftonline.com/{TENANT_ID}"
    msal_app = msal.PublicClientApplication(client_id=CLIENT_ID, authority=authority)
    accounts = msal_app.get_accounts(username=USERNAME)
    if accounts:
        result = msal_app.acquire_token_silent(scopes=["https://database.windows.net/.default"], account=accounts[0])
        if result:
            return result["access_token"]
    result = msal_app.acquire_token_interactive(scopes=["https://database.windows.net/.default"])
    if "access_token" in result:
        return result["access_token"]
    raise Exception(f"Failed to acquire token: {result.get('error')} - {result.get('error_description')}")

def establish_connection():
    try:
        driver = "{ODBC Driver 17 for SQL Server}" if platform.system() == 'Windows' else "{ODBC Driver 18 for SQL Server}"
        connection_string = f"Driver={driver};Server={SERVER};Database={DATABASE};Authentication=ActiveDirectoryInteractive;UID={USERNAME};"
        conn = pyodbc.connect(connection_string)
        return conn, None
    except Exception as e:
        return None, f"Connection error: {str(e)}"

def get_all_database_objects():
    """Fetch all tables and views from the connected database"""
    conn, error = establish_connection()
    if error:
        raise HTTPException(status_code=500, detail=error)
    
    cursor = conn.cursor()
    
    # Get all tables
    tables_query = """
    SELECT TABLE_NAME 
    FROM INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_SCHEMA = 'dbo' AND TABLE_TYPE = 'BASE TABLE'
    """
    cursor.execute(tables_query)
    tables = [row[0] for row in cursor.fetchall()]
    
    # Get all views
    views_query = """
    SELECT TABLE_NAME 
    FROM INFORMATION_SCHEMA.VIEWS 
    WHERE TABLE_SCHEMA = 'dbo'
    """
    cursor.execute(views_query)
    views = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    return tables + views

def get_role_definitions():
    """Get role definitions from the database"""
    conn, error = establish_connection()
    if error:
        raise HTTPException(status_code=500, detail=error)
    
    cursor = conn.cursor()
    
    # Check if the RoleTableMapping table exists
    check_query = """
    SELECT COUNT(*) 
    FROM INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = 'RoleTableMapping'
    """
    cursor.execute(check_query)
    table_exists = cursor.fetchone()[0] > 0
    
    # If RoleTableMapping exists, use it for dynamic RBAC
    if table_exists:
        roles_query = """
        SELECT r.RoleName, rtm.TableName
        FROM [dbo].[RoleTableMapping] rtm
        JOIN [dbo].[Roles] r ON rtm.RoleID = r.RoleID
        WHERE rtm.IsActive = 1
        """
        cursor.execute(roles_query)
        role_mappings = cursor.fetchall()
        
        role_definitions = {}
        for role_name, table_name in role_mappings:
            if role_name not in role_definitions:
                role_definitions[role_name] = []
            role_definitions[role_name].append(table_name)
        
        conn.close()
        return role_definitions
    else:
        conn.close()
        return {
            "Admin": [],
            "Recruiter": ["Sourcing", "Candidate", "Education", "PreferredLocation", "NoticePeriod"],
            "Requestor": ["Request", "Requisition", "Vacancy", "Position", "WorkLocation", "Employee"],
            "Interviewer": ["Feedback", "Interview", "Interviewer"]
        }

def get_view_related_tables():
    """Get mapping between views and the tables they reference"""
    conn, error = establish_connection()
    if error:
        raise HTTPException(status_code=500, detail=error)
    
    cursor = conn.cursor()
    
    # Query to get all views and their definitions
    views_query = """
    SELECT v.name AS view_name, OBJECT_DEFINITION(v.object_id) AS view_definition
    FROM sys.views v
    """
    cursor.execute(views_query)
    views = cursor.fetchall()
    
    # Get all tables
    tables_query = """
    SELECT name FROM sys.tables
    """
    cursor.execute(tables_query)
    tables = [row[0] for row in cursor.fetchall()]
    
    view_table_mapping = {}
    
    for view_name, view_definition in views:
        if view_definition:
            related_tables = []
            for table in tables:
                # Check if table name appears in view definition
                pattern = r'\[' + re.escape(table) + r'\]|\b' + re.escape(table) + r'\b'
                if re.search(pattern, view_definition, re.IGNORECASE):
                    related_tables.append(table)
            view_table_mapping[view_name] = related_tables
    
    conn.close()
    return view_table_mapping

def calculate_similarity(str1, str2):
    """Calculate string similarity for fuzzy matching"""
    # Simple implementation of Levenshtein distance ratio
    if not str1 or not str2:
        return 0
    
    # Calculate Levenshtein distance
    m, n = len(str1), len(str2)
    dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
    
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    # Convert to similarity ratio (0 to 1)
    max_length = max(m, n)
    if max_length == 0:
        return 1.0  # Both strings are empty
    
    return 1 - (dp[m][n] / max_length)

def define_table_access_by_role(role_name, all_tables):
    """Define table access based on role and filter from all available tables"""
    role_definitions = get_role_definitions()
    if role_name == "Admin":
        return all_tables
    
    view_table_mapping = get_view_related_tables()
    patterns = role_definitions.get(role_name, [])
    accessible = []
    
    if not patterns:
        patterns = [role_name]
    
    # First identify accessible base tables
    accessible_base_tables = []
    for table in all_tables:
        # Check if table name matches any pattern
        if any(pattern.lower() in table.lower() for pattern in patterns):
            accessible.append(table)
            accessible_base_tables.append(table)
    
    # Now check views based on related tables
    for view_name, related_tables in view_table_mapping.items():
        # If the view is directly accessible by pattern, it's already added
        if view_name in accessible:
            continue
            
        # Check if any of the related tables are accessible
        related_accessible = False
        for rel_table in related_tables:
            if rel_table in accessible_base_tables or any(pattern.lower() in rel_table.lower() for pattern in patterns):
                related_accessible = True
                break
                
        if related_accessible:
            accessible.append(view_name)
    
    return accessible

def get_user_role(email):
    conn, error = establish_connection()
    if error:
        raise HTTPException(status_code=500, detail=error)
    cursor = conn.cursor()
    query = """
    SELECT r.RoleID, r.RoleName 
    FROM [dbo].[UserRoleMapping] urm
    JOIN [dbo].[Roles] r ON urm.RoleID = r.RoleID
    WHERE urm.UserEmail = ? AND urm.IsActive = 1
    """
    cursor.execute(query, (email,))
    roles = [{"id": row[0], "name": row[1]} for row in cursor.fetchall()]
    conn.close()
    if not roles:
        raise HTTPException(status_code=401, detail="No active roles found for this email")
    return roles

def get_db_metadata():
    """Get all table and column metadata to help with query understanding"""
    conn, error = establish_connection()
    if error:
        raise HTTPException(status_code=500, detail=error)
    
    cursor = conn.cursor()
    
    # Get all tables/views and their columns
    query = """
    SELECT 
        t.TABLE_NAME,
        c.COLUMN_NAME,
        c.DATA_TYPE
    FROM INFORMATION_SCHEMA.TABLES t
    JOIN INFORMATION_SCHEMA.COLUMNS c ON t.TABLE_NAME = c.TABLE_NAME
    WHERE t.TABLE_SCHEMA = 'dbo'
    UNION
    SELECT 
        v.TABLE_NAME,
        c.COLUMN_NAME,
        c.DATA_TYPE
    FROM INFORMATION_SCHEMA.VIEWS v
    JOIN INFORMATION_SCHEMA.COLUMNS c ON v.TABLE_NAME = c.TABLE_NAME
    WHERE v.TABLE_SCHEMA = 'dbo'
    """
    
    cursor.execute(query)
    results = cursor.fetchall()
    
    metadata = {}
    for table_name, column_name, data_type in results:
        if table_name not in metadata:
            metadata[table_name] = []
        metadata[table_name].append({
            "column": column_name,
            "type": data_type
        })
    
    conn.close()
    return metadata

def get_table_schema(table_name):
    conn, error = establish_connection()
    if error:
        raise HTTPException(status_code=500, detail=error)
    cursor = conn.cursor()
    table_check_query = """
    SELECT COUNT(*) 
    FROM INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = ?
    """
    cursor.execute(table_check_query, (table_name,))
    table_exists = cursor.fetchone()[0] > 0
    if table_exists:
        schema_query = f"""
        SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, IS_NULLABLE, COLUMN_DEFAULT
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = '{table_name}'
        ORDER BY ORDINAL_POSITION
        """
        cursor.execute(schema_query)
        columns = cursor.fetchall()
        schema = f"CREATE TABLE [dbo].[{table_name}](\n"
        for i, col in enumerate(columns):
            column_name, data_type, max_length, is_nullable, default_value = col
            data_type_str = f"{data_type}({max_length})" if max_length and max_length != -1 and data_type in ('char', 'varchar', 'nchar', 'nvarchar') else data_type
            nullable = "NULL" if is_nullable == "YES" else "NOT NULL"
            default = f" DEFAULT {default_value}" if default_value else ""
            comma = "" if i == len(columns) - 1 else ","
            schema += f"    [{column_name}] [{data_type_str}] {nullable}{default}{comma}\n"
        schema += ")"
        conn.close()
        return schema
    else:
        view_check_query = """
        SELECT COUNT(*) 
        FROM INFORMATION_SCHEMA.VIEWS 
        WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = ?
        """
        cursor.execute(view_check_query, (table_name,))
        view_exists = cursor.fetchone()[0] > 0
        if view_exists:
            view_def_query = f"""
            SELECT VIEW_DEFINITION
            FROM INFORMATION_SCHEMA.VIEWS
            WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = '{table_name}'
            """
            cursor.execute(view_def_query)
            view_def = cursor.fetchone()[0]
            schema = f"CREATE VIEW [dbo].[{table_name}] AS\n{view_def}"
            conn.close()
            return schema
        conn.close()
        raise HTTPException(status_code=404, detail=f"The {table_name} table/view does not exist")

def determine_target_table(question, accessible_tables, metadata):
    """Determine which table to query based on the question content and metadata with fuzzy matching for spelling mistakes"""
    
    # Create a prompt for the LLM to identify the most relevant table
    tables_info = []
    for table in accessible_tables:
        if table in metadata:
            columns = ", ".join([col["column"] for col in metadata[table]])
            tables_info.append(f"- {table} (columns: {columns})")
    
    tables_text = "\n".join(tables_info)
    
    prompt = f"""Given a database query question and a list of available tables with their columns, 
determine which table is the most relevant to answer the question. 
Users may make spelling mistakes - be forgiving of minor errors in table or column names.
If multiple tables could potentially answer the question, choose the most specific one.

Available tables and their columns:
{tables_text}

User question:
"{question}"

Return only the table name that is most relevant to the question, without any explanation or additional text.
Match the exact table name from the list even if the user made a minor spelling mistake.
"""
    
    response = requests.post(
        f"{OLLAMA_API_URL}/generate",
        json={"model": "llama3", "prompt": prompt, "stream": False, "temperature": 0.1}
    )
    
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Ollama API error: {response.status_code}")
    
    result = response.json()
    suggested_table = result.get("response", "").strip()
    
    # Clean up the response - sometimes the LLM might add quotes or extra text
    suggested_table = re.sub(r'^[\"\']|[\"\']$', '', suggested_table)  # Remove quotes
    suggested_table = suggested_table.split("\n")[0].strip()  # Get only the first line
    
    # Verify that the suggested table is in the accessible tables
    if suggested_table in accessible_tables:
        return suggested_table
    
    # If not exact match, try to find a partial match with fuzzy matching
    best_match = None
    highest_similarity = 0
    
    for table in accessible_tables:
        # Simple similarity score based on character matching
        similarity = calculate_similarity(suggested_table.lower(), table.lower())
        if similarity > highest_similarity and similarity > 0.7:  # 70% similarity threshold
            highest_similarity = similarity
            best_match = table
    
    if best_match:
        return best_match
    
    # If no match found and we have accessible tables, return the first one
    if accessible_tables:
        return accessible_tables[0]
    
    return None

def get_nl2sql_response(question, table_name, schema, user_role=None):
    role_context = f"\nNote that this query is being made by a user with {user_role} role. " if user_role else ""
    if user_role and user_role != "Admin":
        if any(op in question.lower() for op in ["delete", "drop", "truncate", "update", "insert", "create"]):
            raise HTTPException(status_code=403, detail="You don't have permission to perform data modification operations")
    
    # Get column metadata to help with fuzzy matching
    conn, error = establish_connection()
    if error:
        raise HTTPException(status_code=500, detail=error)
    
    cursor = conn.cursor()
    columns_query = f"""
    SELECT COLUMN_NAME
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME = '{table_name}'
    """
    cursor.execute(columns_query)
    columns = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    columns_list = ", ".join(columns)
    
    prompt = f"""Given the following SQL Server database schema:
{schema}

The table/view has the following columns: {columns_list}

Convert this question into a SQL query to run against the {table_name} table/view:
{question}{role_context}

Important notes:
1. The user might make minor spelling mistakes in column names.
2. Try to match columns even with minor spelling variations.
3. Use exact column names from the schema in your final query.

Return only the SQL query without any explanation or additional text.
"""
    response = requests.post(
        f"{OLLAMA_API_URL}/generate",
        json={"model": "llama3", "prompt": prompt, "stream": False, "temperature": 0.1}
    )
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Ollama API returned status code {response.status_code}")
    result = response.json()
    sql_response = result.get("response", "").strip()
    if "```sql" in sql_response:
        sql_match = re.search(r'```sql(.+?)```', sql_response, re.DOTALL)
        if sql_match:
            sql_response = sql_match.group(1).strip()
    elif "```" in sql_response:
        sql_match = re.search(r'```(.+?)```', sql_response, re.DOTALL)
        if sql_match:
            sql_response = sql_match.group(1).strip()
    if "LIMIT" in sql_response:
        limit_match = re.search(r'LIMIT\s+(\d+)', sql_response, re.IGNORECASE)
        if limit_match:
            limit_num = limit_match.group(1)
            sql_response = re.sub(r'LIMIT\s+\d+', '', sql_response, flags=re.IGNORECASE)
            sql_response = sql_response.replace("SELECT", f"SELECT TOP {limit_num}", 1)
    return sql_response

def execute_sql_query(sql_query):
    conn, error = establish_connection()
    if error:
        raise HTTPException(status_code=500, detail=error)
    cursor = conn.cursor()
    cursor.execute(sql_query)
    columns = [column[0] for column in cursor.description] if cursor.description else []
    results = []
    if columns:
        rows = cursor.fetchall()
        for row in rows:
            result_row = {}
            for i, value in enumerate(row):
                if isinstance(value, (bytes, bytearray)):
                    value = "<binary data>"
                elif hasattr(value, 'isoformat'):
                    value = value.isoformat()
                result_row[columns[i]] = value
            results.append(result_row)
    conn.close()
    return results

def convert_results_to_natural_language(results, question, table_name):
    """Convert SQL results to natural language using LLM"""
    if not results:
        return "I didn't find any data matching your query."
    
    # Serialize the results to JSON for the prompt
    results_json = json.dumps(results, indent=2)
    
    prompt = f"""Here are the results of a query against the {table_name} table:
```json
{results_json}
```

The original question was: "{question}"

Please convert these SQL query results into a natural language response that directly answers the question.
Make your response conversational and friendly. Focus on the key information and insights. 
Only mention specific numbers if they're significant to the answer.
DO NOT mention SQL, queries, or tables in your response.
"""
    
    response = requests.post(
        f"{OLLAMA_API_URL}/generate",
        json={
            "model": "llama3",  # Changed from llama2:13b to llama3
            "prompt": prompt,
            "stream": False,
            "temperature": 0.1
        }
    )
    
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Ollama API returned status code {response.status_code}")
    
    result = response.json()
    natural_language = result.get("response", "").strip()
    
    return natural_language


@app.post("/api/conversational-response")
async def get_conversational_response(request: ConversationalRequest):
    """Endpoint to get conversational responses from the LLM"""
    try:
        # Generate response using Ollama API
        response = requests.post(
            f"{OLLAMA_API_URL}/generate",
            json={
                "model": "llama3",
                "prompt": request.prompt,
                "stream": False,
                "temperature": 0.7  # Slightly higher temperature for more varied responses
            }
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Ollama API returned status code {response.status_code}")
        
        result = response.json()
        message = result.get("response", "").strip()
        
        return {"success": True, "message": message}
    except Exception as e:
        return {"success": False, "error": str(e)}

# @app.post("/api/classify-message")
# async def classify_message(request: MessageClassificationRequest):
#     """Endpoint to classify if a message is conversational or a database query"""
#     try:
#         # Format a list of tables for context
#         tables_list = ", ".join(request.accessibleTables[:10])  # Limit to first 10 for brevity
#         additional_tables = f" and {len(request.accessibleTables) - 10} more" if len(request.accessibleTables) > 10 else ""
        
#         # Create classification prompt
#         prompt = f"""You are an AI assistant that helps classify user messages. 
# Determine if the following message is a casual conversation or a database query.

# User message: "{request.message}"

# Available database tables: {tables_list}{additional_tables}

# Classify the message as either:
# - "conversational": general conversation, greetings, small talk, questions about you, etc.
# - "database_query": any question that seems to be asking for information from a database

# Only respond with one word: either "conversational" or "database_query"."""

#         # Send to LLM for classification
#         response = requests.post(
#             f"{OLLAMA_API_URL}/generate",
#             json={
#                 "model": "llama3",
#                 "prompt": prompt,
#                 "stream": False,
#                 "temperature": 0.1  # Low temperature for more consistent classification
#             }
#         )
        
#         if response.status_code != 200:
#             raise HTTPException(status_code=500, detail=f"Ollama API returned status code {response.status_code}")
        
#         result = response.json()
#         response_text = result.get("response", "").strip().lower()
        
#         # Clean up the response to ensure we get exactly what we need
#         if "conversational" in response_text:
#             message_type = "conversational"
#         else:
#             message_type = "database_query"
        
#         return {"success": True, "type": message_type}
#     except Exception as e:
#         return {"success": False, "error": str(e), "type": "database_query"}
    

@app.post("/api/classify-message")
async def classify_message(request: MessageClassificationRequest):
    """Classify the message as conversational, database_query, or document_query"""
    try:
        # We use LLM to classify into 3 types
        prompt = f"""You are a classification assistant.

Your job is to classify the user's question into one of three categories:

- conversational: small talk, greetings, chit-chat, questions about the assistant
- database_query: business questions that require live data or tables
- document_query: questions based on company policies, guidelines, or internal documents

Here are some examples:
- "Hey, how are you today?" → conversational
- "How many candidates applied last week?" → database_query
- "What is our leave policy?" → document_query

Now classify this:

User question: "{request.message}"

Only respond with one of: conversational, database_query, document_query
"""

        response = requests.post(
            f"{OLLAMA_API_URL}/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False,
                "temperature": 0.1
            }
        )

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="LLM classification failed")

        result = response.json()
        label = result.get("response", "").strip().lower()

        # fallback in case of strange response
        if "document" in label:
            label = "document_query"
        elif "conversation" in label:
            label = "conversational"
        elif "database" in label:
            label = "database_query"

        return {"success": True, "type": label}

    except Exception as e:
        return {"success": False, "type": "database_query", "error": str(e)}

from document_qa.qa_engine import ask_documents  # at top of app.py

class DocSearchRequest(BaseModel):
    question: str

@app.post("/api/ask-docs")
async def ask_docs(request: DocSearchRequest):
    try:
        answer = ask_documents(request.question)
        return {
            "success": True,
            "message": answer,  # match conversational format
            "format": "text"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

    
@app.post("/api/login")
async def login(request: LoginRequest):
    roles = get_user_role(request.email)
    all_database_objects = get_all_database_objects()
    accessible_tables = []
    for role in roles:
        role_tables = define_table_access_by_role(role["name"], all_database_objects)
        accessible_tables.extend(role_tables)
    
    accessible_tables = list(set(accessible_tables))
    return {
        "success": True,
        "email": request.email,
        "roles": roles,
        "accessibleTables": accessible_tables
    }


@app.post("/api/ask-llama")
async def ask_llama(request: AskLlamaRequest):
    format_type = request.format if hasattr(request, "format") else "text"  
    
    # Check for format requests in the question
    if "tabular format" in request.question.lower() or "table format" in request.question.lower():
        format_type = "table"
    
    # Extract table name if explicitly provided in the question
    table_name = None
    original_table_name = None
    
    if request.question.startswith("[Table:"):
        match = re.match(r'\[Table: ([^\]]+)\] (.+)', request.question)
        if match:
            original_table_name = match.group(1)
            question = match.group(2)
            
            # First check for exact match
            if original_table_name in request.accessibleTables:
                table_name = original_table_name
            else:
                # Try fuzzy matching if the exact table name wasn't found
                best_match = None
                highest_similarity = 0
                
                for accessible_table in request.accessibleTables:
                    similarity = calculate_similarity(original_table_name.lower(), accessible_table.lower())
                    if similarity > highest_similarity and similarity > 0.7:  # 70% similarity threshold
                        highest_similarity = similarity
                        best_match = accessible_table
                
                if best_match:
                    table_name = best_match
    else:
        question = request.question
        
        # Get database metadata for table detection
        metadata = get_db_metadata()
        
        # Use LLM to determine most relevant table based on question
        try:
            table_name = determine_target_table(question, request.accessibleTables, metadata)
        except Exception as e:
            print(f"Error determining table: {str(e)}")
            table_name = None
    
    # Check if user has permission to access the table
    if table_name and table_name not in request.accessibleTables and "Admin" not in [role["name"] for role in request.userRoles]:
        return {"success": False, "error": f"You don't have permission to query the {table_name} table"}
    
    if not table_name:
        return {"success": False, "error": "Could not determine which table to query. Please specify a table in your question."}
    
    # Track if there was a table name correction
    corrected_table = None
    if original_table_name and original_table_name != table_name:
        corrected_table = table_name
    
    try:
        schema = get_table_schema(table_name)
        user_role = request.userRoles[0]["name"] if request.userRoles else None
        sql_query = get_nl2sql_response(question, table_name, schema, user_role)
        
        if sql_query.startswith("Error:"):
            return {"success": False, "error": sql_query}
        
        results = execute_sql_query(sql_query)
        
        if not results:
            return {"success": True, "results": [], "message": "No results found for your query.", "correctedTable": corrected_table}
        
        if format_type == "table":
            return {"success": True, "results": results, "format": "table", "correctedTable": corrected_table}
        
        natural_language_response = convert_results_to_natural_language(results, question, table_name)
        return {"success": True, "message": natural_language_response, "format": "text", "correctedTable": corrected_table}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/tables")
async def get_tables():
    """Endpoint to get all available tables and views"""
    try:
        all_database_objects = get_all_database_objects()
        return {"success": True, "tables": all_database_objects}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/table-schema/{table_name}")
async def get_schema(table_name: str):
    """Endpoint to get the schema for a specific table or view"""
    try:
        schema = get_table_schema(table_name)
        return {"success": True, "schema": schema}
    except Exception as e:
        return {"success": False, "error": str(e)}
