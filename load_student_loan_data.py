#!/usr/bin/env python3
"""
Student Loan Risk Demo - Data Warehouse Loader

This script loads all synthetic student loan data from CSV files into an Impala data warehouse.
It creates the LoanTechSolutions database and all required tables with proper schemas.

Author: Student Loan Risk Demo
Date: 2024
"""

import os
import pandas as pd
import cml.data_v1 as cmldata
from datetime import datetime
import sys

# Configuration
CONNECTION_NAME = "default-impala-aws"
DATABASE_NAME = "LoanTechSolutions"
DATA_PATH = "data/synthetic"

# Table schema definitions based on CSV analysis
TABLE_SCHEMAS = {
    "student_loan_borrowers": {
        "columns": [
            ("borrower_id", "STRING"),
            ("_risk_segment", "STRING"),
            ("age", "INT"),
            ("credit_score_at_origination", "INT"),
            ("annual_income", "DOUBLE"),
            ("employment_status", "STRING"),
            ("gender", "STRING"), 
            ("state", "STRING"),
            ("dependents", "INT"),
            ("housing_status", "STRING")
        ],
        "description": "Basic borrower demographic and financial information"
    },
    
    "student_loan_education": {
        "columns": [
            ("borrower_id", "STRING"),
            ("school_name", "STRING"),
            ("degree_type", "STRING"),
            ("major", "STRING"),
            ("graduation_year", "INT"),
            ("gpa", "DOUBLE"),
            ("school_type", "STRING"),
            ("completion_status", "STRING")
        ],
        "description": "Educational background and academic performance data"
    },
    
    "student_loan_loans": {
        "columns": [
            ("loan_id", "STRING"),
            ("borrower_id", "STRING"),
            ("loan_amount", "DOUBLE"),
            ("origination_date", "STRING"), # Using STRING for date to avoid parsing issues
            ("loan_term_months", "INT"),
            ("interest_rate", "DOUBLE"),
            ("loan_type", "STRING"),
            ("loan_status", "STRING"),
            ("current_balance", "DOUBLE"),
            ("monthly_payment", "DOUBLE")
        ],
        "description": "Individual loan details and current status"
    },
    
    "student_loan_payments": {
        "columns": [
            ("payment_id", "STRING"),
            ("loan_id", "STRING"),
            ("borrower_id", "STRING"),
            ("payment_date", "STRING"), # Using STRING for date to avoid parsing issues
            ("scheduled_amount", "DOUBLE"),
            ("actual_amount", "DOUBLE"),
            ("payment_status", "STRING"),
            ("days_late", "INT")
        ],
        "description": "Payment history and transaction records"
    },
    
    "student_loan_delinquency_features": {
        "columns": [
            ("borrower_id", "STRING"),
            ("avg_days_late", "DOUBLE"),
            ("max_days_late", "INT"),
            ("std_days_late", "DOUBLE"),
            ("total_payments_made", "DOUBLE"),
            ("total_payment_count", "INT"),
            ("total_scheduled", "DOUBLE"),
            ("missed_payment_count", "INT"),
            ("payment_ratio", "DOUBLE"),
            ("missed_payment_rate", "DOUBLE"),
            ("recent_avg_days_late", "DOUBLE"),
            ("recent_missed_payments", "DOUBLE"),
            ("is_delinquent", "INT"),
            ("risk_score", "DOUBLE")
        ],
        "description": "Calculated risk features and delinquency indicators"
    },
    
    "student_loan_master_dataset": {
        "columns": [
            ("borrower_id", "STRING"),
            ("age", "INT"),
            ("credit_score_at_origination", "INT"),
            ("annual_income", "DOUBLE"),
            ("employment_status", "STRING"),
            ("gender", "STRING"),
            ("state", "STRING"),
            ("dependents", "INT"),
            ("housing_status", "STRING"),
            ("school_name", "STRING"),
            ("degree_type", "STRING"),
            ("major", "STRING"),
            ("graduation_year", "INT"),
            ("gpa", "DOUBLE"),
            ("school_type", "STRING"),
            ("completion_status", "STRING"),
            ("avg_days_late", "DOUBLE"),
            ("max_days_late", "INT"),
            ("std_days_late", "DOUBLE"),
            ("total_payments_made", "DOUBLE"),
            ("total_payment_count", "INT"),
            ("total_scheduled", "DOUBLE"),
            ("missed_payment_count", "INT"),
            ("payment_ratio", "DOUBLE"),
            ("missed_payment_rate", "DOUBLE"),
            ("recent_avg_days_late", "DOUBLE"),
            ("recent_missed_payments", "DOUBLE"),
            ("is_delinquent", "INT"),
            ("risk_score", "DOUBLE"),
            ("total_loan_amount", "DOUBLE"),
            ("loan_count", "INT"),
            ("avg_loan_amount", "DOUBLE"),
            ("avg_interest_rate", "DOUBLE"),
            ("total_current_balance", "DOUBLE"),
            ("total_monthly_payment", "DOUBLE"),
            ("debt_to_income_ratio", "DOUBLE")
        ],
        "description": "Complete dataset with all features combined for ML model training"
    },
    
    "realistic_student_loan_dataset": {
        "columns": [
            ("borrower_id", "STRING"),
            ("age", "INT"),
            ("credit_score_at_origination", "INT"),
            ("annual_income", "DOUBLE"),
            ("employment_status", "STRING"),
            ("total_loan_amount", "DOUBLE"),
            ("loan_count", "INT"),
            ("total_monthly_payment", "DOUBLE"),
            ("debt_to_income_ratio", "DOUBLE"),
            ("gender", "STRING"),
            ("state", "STRING"),
            ("housing_status", "STRING"),
            ("gpa", "DOUBLE"),
            ("graduation_year", "INT"),
            ("degree_type", "STRING"),
            ("major", "STRING"),
            ("school_name", "STRING"),
            ("school_type", "STRING"),
            ("completion_status", "STRING"),
            ("avg_days_late", "DOUBLE"),
            ("max_days_late", "INT"),
            ("std_days_late", "DOUBLE"),
            ("total_payments_made", "DOUBLE"),
            ("total_payment_count", "INT"),
            ("total_scheduled", "DOUBLE"),
            ("missed_payment_count", "INT"),
            ("payment_ratio", "DOUBLE"),
            ("missed_payment_rate", "DOUBLE"),
            ("recent_avg_days_late", "DOUBLE"),
            ("recent_missed_payments", "DOUBLE"),
            ("total_current_balance", "DOUBLE"),
            ("risk_score", "DOUBLE"),
            ("is_delinquent", "INT"),
            ("avg_loan_amount", "DOUBLE"),
            ("avg_interest_rate", "DOUBLE"),
            ("dependents", "INT")
        ],
        "description": "Large-scale realistic dataset for comprehensive analysis"
    }
}

def connect_to_impala():
    """Establish connection to Impala data warehouse."""
    try:
        print("🔌 Connecting to Impala data warehouse...")
        conn = cmldata.get_connection(CONNECTION_NAME)
        print("✅ Successfully connected to Impala")
        return conn
    except Exception as e:
        print(f"❌ Error connecting to Impala: {str(e)}")
        raise

def create_database(conn):
    """Create the LoanTechSolutions database."""
    try:
        print(f"🏗️  Setting up database: {DATABASE_NAME}")
        
        # Get cursor for executing SQL
        cursor = conn.get_cursor()
        
        # Check if database exists first (case-insensitive check)
        database_exists = False
        try:
            databases_df = conn.get_pandas_dataframe("SHOW DATABASES")
            # Check both exact match and case-insensitive match
            database_names = [db.lower() for db in databases_df['name'].values]
            if DATABASE_NAME.lower() in database_names:
                database_exists = True
                print(f"🗑️  Dropping existing database {DATABASE_NAME} and all its tables")
                drop_db_sql = f"DROP DATABASE IF EXISTS {DATABASE_NAME} CASCADE"
                cursor.execute(drop_db_sql)
                print(f"✅ Dropped existing database {DATABASE_NAME}")
            else:
                print(f"ℹ️  Database {DATABASE_NAME} doesn't exist yet")
        except Exception as check_error:
            print(f"⚠️  Could not check existing databases: {str(check_error)}")
        
        # Create new database (use IF NOT EXISTS for safety)
        create_db_sql = f"CREATE DATABASE IF NOT EXISTS {DATABASE_NAME}"
        cursor.execute(create_db_sql)
        if database_exists:
            print(f"✅ Recreated database: {DATABASE_NAME}")
        else:
            print(f"✅ Created database: {DATABASE_NAME}")
        
        # Use the new database
        use_db_sql = f"USE {DATABASE_NAME}"
        cursor.execute(use_db_sql)
        print(f"📂 Now using database: {DATABASE_NAME}")
        
        cursor.close()
        
    except Exception as e:
        print(f"❌ Error creating database: {str(e)}")
        raise

def create_table(conn, table_name, schema_info):
    """Create a table with the specified schema."""
    try:
        print(f"📋 Creating table: {table_name}")
        
        # Get cursor for executing SQL
        cursor = conn.get_cursor()
        
        # Ensure we're in the correct database
        cursor.execute(f"USE {DATABASE_NAME}")
        
        # Build column definitions
        column_defs = []
        for col_name, col_type in schema_info["columns"]:
            column_defs.append(f"{col_name} {col_type}")
        
        columns_sql = ",\n    ".join(column_defs)
        
        # Create table SQL with fully qualified name
        create_table_sql = f"""
        CREATE TABLE {DATABASE_NAME}.{table_name} (
            {columns_sql}
        )
        STORED AS TEXTFILE
        """
        
        cursor.execute(create_table_sql)
        cursor.close()
        
        print(f"✅ Created table: {DATABASE_NAME}.{table_name}")
        print(f"   📝 Description: {schema_info['description']}")
        print(f"   📊 Columns: {len(schema_info['columns'])}")
        
    except Exception as e:
        print(f"❌ Error creating table {table_name}: {str(e)}")
        raise

def load_csv_to_table(conn, table_name, csv_file_path):
    """Load data from CSV file into Impala table."""
    try:
        print(f"📤 Loading data into {table_name} from {csv_file_path}")
        
        # Check if file exists
        if not os.path.exists(csv_file_path):
            print(f"⚠️  Warning: File {csv_file_path} not found, skipping...")
            return
        
        # Read CSV file - limit to first 100 rows for demo
        df = pd.read_csv(csv_file_path)
        original_rows = len(df)
        
        # Limit to first 100 rows for faster demo loading
        if original_rows > 100:
            df = df.head(100)
            print(f"   📊 Limited to first 100 rows (original: {original_rows:,} rows)")
        else:
            print(f"   📊 Using all {original_rows:,} rows (less than 100)")
        
        total_rows = len(df)
        
        if total_rows == 0:
            print(f"⚠️  Warning: No data in {csv_file_path}, skipping...")
            return
        
        # Get table schema to ensure column order matches
        schema = TABLE_SCHEMAS[table_name]["columns"]
        expected_columns = [col[0] for col in schema]
        
        # Reorder DataFrame columns to match table schema
        available_columns = [col for col in expected_columns if col in df.columns]
        if len(available_columns) != len(expected_columns):
            missing_cols = set(expected_columns) - set(available_columns)
            extra_cols = set(df.columns) - set(expected_columns)
            print(f"⚠️  Column mismatch for {table_name}:")
            if missing_cols:
                print(f"      Missing: {missing_cols}")
            if extra_cols:
                print(f"      Extra: {extra_cols}")
        
        # Use available columns in correct order
        df = df[available_columns]
        
        # Convert DataFrame to list of tuples for insertion
        data_tuples = []
        for _, row in df.iterrows():
            # Convert row to tuple, handling NaN values
            tuple_row = tuple(None if pd.isna(val) else val for val in row)
            data_tuples.append(tuple_row)
        
        # Batch insert - split into chunks for large datasets
        chunk_size = 1000
        chunks = [data_tuples[i:i + chunk_size] for i in range(0, len(data_tuples), chunk_size)]
        
        total_inserted = 0
        for i, chunk in enumerate(chunks):
            try:
                # Create placeholder string for values
                placeholders = ", ".join(["%s"] * len(available_columns))
                insert_sql = f"INSERT INTO {DATABASE_NAME}.{table_name} VALUES ({placeholders})"
                
                # Execute batch insert using cursor
                cursor = conn.get_cursor()
                # Ensure we're in the correct database
                cursor.execute(f"USE {DATABASE_NAME}")
                for row_data in chunk:
                    cursor.execute(insert_sql, row_data)
                cursor.close()
                total_inserted += len(chunk)
                
                if len(chunks) > 1:  # Only show progress for large tables
                    print(f"   📈 Progress: {total_inserted:,}/{total_rows:,} rows ({(total_inserted/total_rows*100):.1f}%)")
                    
            except Exception as e:
                print(f"❌ Error inserting chunk {i+1}: {str(e)}")
                print(f"   Sample row: {chunk[0] if chunk else 'No data'}")
                raise
        
        print(f"✅ Successfully loaded {total_inserted:,} rows into {table_name}")
        
        # Verify data was loaded
        count_sql = f"SELECT COUNT(*) as row_count FROM {DATABASE_NAME}.{table_name}"
        result = conn.get_pandas_dataframe(count_sql)
        actual_count = result['row_count'].iloc[0]
        print(f"   ✅ Verification: {actual_count:,} rows in table")
        
    except Exception as e:
        print(f"❌ Error loading data into {table_name}: {str(e)}")
        raise

def show_database_summary(conn):
    """Display summary of loaded data."""
    try:
        print("\n" + "="*60)
        print("📊 DATABASE LOADING SUMMARY")
        print("="*60)
        
        # Show database info
        print(f"🏛️  Database: {DATABASE_NAME}")
        
        # Show tables and row counts - use the specific database
        tables_sql = f"SHOW TABLES IN {DATABASE_NAME}"
        tables_df = conn.get_pandas_dataframe(tables_sql)
        
        print(f"📋 Tables created: {len(tables_df)}")
        print()
        
        total_rows = 0
        for table_name in tables_df['name']:
            try:
                count_sql = f"SELECT COUNT(*) as row_count FROM {DATABASE_NAME}.{table_name}"
                result = conn.get_pandas_dataframe(count_sql)
                row_count = result['row_count'].iloc[0]
                total_rows += row_count
                
                # Get table info
                desc = TABLE_SCHEMAS.get(table_name, {}).get('description', 'No description')
                print(f"   📁 {table_name}")
                print(f"      Rows: {row_count:,}")
                print(f"      Description: {desc}")
                print()
                
            except Exception as e:
                print(f"   ❌ Error getting count for {table_name}: {str(e)}")
        
        print(f"📈 Total rows across all tables: {total_rows:,}")
        print(f"✅ Data warehouse loading completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error generating summary: {str(e)}")

def main():
    """Main execution function."""
    print("🚀 STUDENT LOAN DATA WAREHOUSE LOADER")
    print("="*60)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Target Database: {DATABASE_NAME}")
    print(f"📂 Source Data Path: {DATA_PATH}")
    print()
    
    conn = None
    try:
        # Step 1: Connect to Impala
        conn = connect_to_impala()
        
        # Step 2: Create database
        create_database(conn)
        
        # Step 3: Create all tables
        print("\n📋 Creating tables...")
        for table_name, schema_info in TABLE_SCHEMAS.items():
            create_table(conn, table_name, schema_info)
        
        # Step 4: Load data into tables
        print("\n📤 Loading data...")
        for table_name in TABLE_SCHEMAS.keys():
            csv_filename = f"{table_name}.csv"
            csv_path = os.path.join(DATA_PATH, csv_filename)
            load_csv_to_table(conn, table_name, csv_path)
        
        # Step 5: Show summary
        show_database_summary(conn)
        
    except Exception as e:
        print(f"\n💥 FATAL ERROR: {str(e)}")
        print("❌ Data warehouse loading failed!")
        sys.exit(1)
        
    finally:
        # Always close connection
        if conn:
            try:
                conn.close()
                print("🔌 Connection closed")
            except:
                pass

if __name__ == "__main__":
    main()