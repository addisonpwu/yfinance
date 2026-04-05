-- Database initialization script for PostgreSQL
-- Creates the stock_analysis database and tables

-- Note: Run this script as superuser (postgres)
-- psql -U postgres -f scripts/init_db.sql

-- Create database (if not exists)
-- This must be run separately from table creation
-- CREATE DATABASE stock_analysis;

-- Connect to database
-- \c stock_analysis

-- Create stocks table
CREATE TABLE IF NOT EXISTS stocks (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    market VARCHAR(10) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create news table with direct stock relationship (one-to-many)
CREATE TABLE IF NOT EXISTS news (
    id SERIAL PRIMARY KEY,
    stock_id INTEGER NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    content TEXT,
    publish_time TIMESTAMP WITH TIME ZONE NOT NULL,
    url VARCHAR(512) NOT NULL UNIQUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_stocks_symbol ON stocks(symbol);
CREATE INDEX IF NOT EXISTS idx_stocks_market ON stocks(market);
CREATE INDEX IF NOT EXISTS idx_news_stock_id ON news(stock_id);
CREATE INDEX IF NOT EXISTS idx_news_publish_time ON news(publish_time DESC);
CREATE INDEX IF NOT EXISTS idx_news_url ON news(url);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to stocks table
DROP TRIGGER IF EXISTS update_stocks_updated_at ON stocks;
CREATE TRIGGER update_stocks_updated_at
    BEFORE UPDATE ON stocks
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();