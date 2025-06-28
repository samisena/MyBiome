import sqlite3

# Using 'with' statement ensures proper connection cleanup
# This automatically closes the connection even if an error occurs
with sqlite3.connect('learning.db') as conn:
    cursor = conn.cursor()
    
    # Create table only if it doesn't exist
    # This prevents errors on subsequent runs
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS students (
                       id INTEGER PRIMARY KEY,
                       name TEXT NOT NULL,
                       age INTEGER,
                       major TEXT
                   )
                   ''')
    
    # Insert data - this will add a new record each time you run the script
    # Consider checking for duplicates if you don't want multiple Alice records
    cursor.execute("INSERT INTO students (name, age, major) VALUES ('Alice', 21, 'Economics')")
    
    # Query the data
    cursor.execute("SELECT name, major FROM students WHERE age >= 20")
    results = cursor.fetchall()
    
    print(f"Query results: {results}")