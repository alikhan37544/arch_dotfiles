import pandas as pd
import mysql.connector
import json
import glob
from tqdm import tqdm  # Progress bar

# MySQL connection details
db_config = {
    'host': 'localhost',
    'user': 'ali',
    'password': 'admin',
    'database': 'youtube_trending'
}

# Establish connection to MySQL
conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()

# Function to create a table for a specific category
def create_category_table(category_id):
    table_name = f"category_{category_id}_videos"
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            video_id VARCHAR(255),
            trending_date DATE,
            title TEXT,
            channel_title TEXT,
            publish_time DATETIME,
            tags TEXT,
            views BIGINT,
            likes BIGINT,
            dislikes BIGINT,
            comment_count BIGINT,
            thumbnail_link TEXT,
            comments_disabled BOOLEAN,
            ratings_disabled BOOLEAN,
            video_error_or_removed BOOLEAN,
            description TEXT,
            PRIMARY KEY (video_id, trending_date)
        );
    """)

# Function to load categories from JSON and create separate tables for each category
def load_category_data():
    json_files = glob.glob('archive/*_category_id.json')

    # Progress bar for loading category JSON files
    for file in tqdm(json_files, desc="Loading Categories", unit="file"):
        with open(file) as f:
            data = json.load(f)
            for item in data['items']:
                category_id = item['id']
                title = item['snippet']['title']

                # Create separate table for each category
                create_category_table(category_id)

                # Insert category data into video_categories table
                cursor.execute("""
                    INSERT IGNORE INTO video_categories (category_id, title)
                    VALUES (%s, %s)
                """, (category_id, title))
            conn.commit()

# Function to create a table for a specific country with a foreign key constraint on category_id
def create_country_table(country_code):
    table_name = f"{country_code}_videos"
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            video_id VARCHAR(255),
            trending_date DATE,
            title TEXT,
            channel_title TEXT,
            category_id INT,
            publish_time DATETIME,
            tags TEXT,
            views BIGINT,
            likes BIGINT,
            dislikes BIGINT,
            comment_count BIGINT,
            thumbnail_link TEXT,
            comments_disabled BOOLEAN,
            ratings_disabled BOOLEAN,
            video_error_or_removed BOOLEAN,
            description TEXT,
            PRIMARY KEY (video_id, trending_date),
            FOREIGN KEY (category_id) REFERENCES video_categories(category_id) ON DELETE CASCADE ON UPDATE CASCADE
        );
    """)

# Function to load data from CSV files and insert into respective country table
def load_csv_data():
    csv_files = glob.glob('archive/*.csv')

    # Progress bar for the overall file loading
    for file in tqdm(csv_files, desc="Loading Video Data", unit="file"):
        if 'category_id' not in file:
            country_code = file.split('/')[-1].split('videos')[0]  # Extract country code

            # Use ISO-8859-1 encoding to handle non-UTF-8 characters
            try:
                df = pd.read_csv(file, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(file, encoding='ISO-8859-1')
            
            # Format trending_date to YYYY-MM-DD and publish_time to YYYY-MM-DD HH:MM:SS
            df['trending_date'] = pd.to_datetime(df['trending_date'], format='%y.%d.%m').dt.strftime('%Y-%m-%d')
            df['publish_time'] = pd.to_datetime(df['publish_time']).dt.strftime('%Y-%m-%d %H:%M:%S')

            # Replace NaN values with None for compatibility with MySQL
            df = df.where(pd.notnull(df), None)

            # Create table for the country with foreign key
            create_country_table(country_code)

            # Insert data into the country-specific table with progress bar
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Inserting rows into {country_code}_videos", unit="row"):
                cursor.execute(f"""
                    INSERT IGNORE INTO {country_code}_videos (video_id, trending_date, title, channel_title, category_id, publish_time, tags, views, likes, dislikes, comment_count, thumbnail_link, comments_disabled, ratings_disabled, video_error_or_removed, description)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    row['video_id'], row['trending_date'], row['title'], row['channel_title'], row['category_id'], row['publish_time'], 
                    row['tags'], row['views'], row['likes'], row['dislikes'], row['comment_count'], row['thumbnail_link'], 
                    row['comments_disabled'], row['ratings_disabled'], row['video_error_or_removed'], row['description']
                ))
            conn.commit()

# Main execution
# Create category table for categories that apply to all countries
cursor.execute("""
    CREATE TABLE IF NOT EXISTS video_categories (
        category_id INT PRIMARY KEY,
        title VARCHAR(255)
    );
""")

# Load the categories first
load_category_data()

# Load the CSV data into country-specific tables
load_csv_data()

# Close the database connection
cursor.close()
conn.close()

print("Data has been successfully loaded into the MySQL database with category-specific and country-specific tables.")
