import psycopg2
from psycopg2 import pool
import logging

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Initialize connection pool (configure with your database credentials)
try:
    db_pool = pool.SimpleConnectionPool(
        minconn=1,
        maxconn=10,
        dbname="casaos",   # Replace with your database name
        user="casaos",     # Replace with your username
        password="casaos", # Replace with your password
        host="security.local",         # Replace with your host
        port="5432"               # Replace with your port
    )
except psycopg2.Error as e:
    logger.error(f"Error initializing connection pool: {e}")
    db_pool = None

def get_all_devices():
    if not db_pool:
        logger.error("Connection pool not initialized")
        return []
    
    try:
        # Get a connection from the pool
        connection = db_pool.getconn()
        # Create a cursor to execute the query
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM devices ORDER BY name ASC")
            # Fetch all rows
            rows = cursor.fetchall()
        return rows
    except psycopg2.Error as e:
        logger.error(f"Error fetching devices: {e}")
        return []
    finally:
        # Release the connection back to the pool
        if connection:
            db_pool.putconn(connection)

# Optional: Close all connections in the pool when done
def close_pool():
    if db_pool:
        db_pool.closeall()