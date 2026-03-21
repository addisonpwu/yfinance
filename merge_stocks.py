#!/usr/bin/env python3
"""
Stock JSON Merge Script

Merges multiple stock_*.json files into a single stock.json file.
Supports news deduplication, backup management, and atomic writes.

Usage:
    python3 merge_stocks.py [--input-dir INPUT_DIR] [--output-file OUTPUT_FILE]
                            [--max-news MAX_NEWS] [--max-backups MAX_BACKUPS]
                            [--dry-run] [--verbose] [--keep-source] [--no-backup]
"""

import argparse
import glob
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Merge multiple stock_*.json files into a single stock.json file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python3 merge_stocks.py --verbose
  python3 merge_stocks.py --input-dir ./reports --output-file ./merged/stock.json
  python3 merge_stocks.py --dry-run --verbose
  python3 merge_stocks.py --keep-source --no-backup
        '''
    )
    
    parser.add_argument(
        '--input-dir',
        type=str,
        default='reports/',
        help='Directory containing stock_*.json files (default: reports/)'
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        default='reports/stock.json',
        help='Output merged JSON file path (default: reports/stock.json)'
    )
    
    parser.add_argument(
        '--max-news',
        type=int,
        default=20,
        help='Maximum number of news items per stock (default: 20)'
    )
    
    parser.add_argument(
        '--max-backups',
        type=int,
        default=5,
        help='Maximum number of backup files to keep (default: 5)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simulate operations without making changes'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--keep-source',
        action='store_true',
        help='Keep source files after merging (default: delete)'
    )
    
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backup of existing stock.json'
    )
    
    return parser.parse_args()


def discover_stock_files(input_dir: str) -> List[Path]:
    """
    Discover all stock_*.json files in the input directory.
    
    Args:
        input_dir: Directory path to search for stock files
        
    Returns:
        List of Path objects for discovered stock files, sorted by name
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return []
    
    if not input_path.is_dir():
        logger.error(f"Input path is not a directory: {input_dir}")
        return []
    
    pattern = str(input_path / 'stock_*.json')
    files = sorted([Path(f) for f in glob.glob(pattern)])
    
    logger.info(f"Discovered {len(files)} stock file(s) in {input_dir}")
    
    if logger.isEnabledFor(logging.DEBUG):
        for f in files:
            logger.debug(f"  - {f.name}")
    
    return files


def load_stock_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load and parse a stock JSON file.
    
    Supports both formats:
    - List format: [{stockCode, news, ...}, ...]
    - Dict format: {"stocks": [{stockCode, news, ...}, ...]}
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Parsed JSON data or None if loading fails
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle list format (direct array of stocks)
        if isinstance(data, list):
            logger.debug(f"Loaded {file_path.name}: {len(data)} stocks (list format)")
            return {'stocks': data}
        
        # Handle dict format (with 'stocks' key)
        if isinstance(data, dict):
            stocks = data.get('stocks', [])
            logger.debug(f"Loaded {file_path.name}: {len(stocks)} stocks (dict format)")
            return data
        
        logger.error(f"Unexpected data format in {file_path.name}: {type(data)}")
        return None
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path.name}: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to load {file_path.name}: {e}")
        return None


def create_news_key(news_item: Dict[str, Any]) -> str:
    """
    Create a unique key for news deduplication.
    
    Priority:
    1. URL (if available)
    2. Title + Agency combination
    
    Args:
        news_item: News item dictionary
        
    Returns:
        Unique key string for deduplication
    """
    # Priority 1: Use URL
    if 'url' in news_item and news_item['url']:
        return f"url:{news_item['url']}"
    
    # Priority 2: Use title + agency
    title = news_item.get('title', '')
    agency = news_item.get('agency', '')
    
    if title or agency:
        return f"title:{title}|agency:{agency}"
    
    # Fallback: Use hash of entire item
    return f"fallback:{hash(frozenset(news_item.items()))}"


def deduplicate_news(news_list: List[Dict[str, Any]], max_news: int) -> List[Dict[str, Any]]:
    """
    Deduplicate news items and sort by publish time.
    
    Args:
        news_list: List of news items (may contain duplicates)
        max_news: Maximum number of news items to keep
        
    Returns:
        Deduplicated and sorted news list
    """
    seen_keys: Set[str] = set()
    unique_news: List[Dict[str, Any]] = []
    
    for news in news_list:
        key = create_news_key(news)
        
        if key not in seen_keys:
            seen_keys.add(key)
            unique_news.append(news)
        else:
            logger.debug(f"Duplicate news filtered: {key[:50]}...")
    
    # Sort by publishTime (descending - newest first)
    unique_news.sort(
        key=lambda x: x.get('publishTime', 0),
        reverse=True
    )
    
    # Limit to max_news
    if len(unique_news) > max_news:
        logger.info(f"Trimmed news from {len(unique_news)} to {max_news} items")
        unique_news = unique_news[:max_news]
    
    return unique_news


def merge_stocks(
    stock_data_list: List[Dict[str, Any]],
    max_news: int,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Merge multiple stock data dictionaries into one.
    
    Args:
        stock_data_list: List of stock data dictionaries
        max_news: Maximum news items per stock
        verbose: Enable verbose logging
        
    Returns:
        Merged stock data dictionary
    """
    merged_stocks: Dict[str, Dict[str, Any]] = {}
    total_news_before = 0
    total_news_after = 0
    
    for stock_data in stock_data_list:
        stocks = stock_data.get('stocks', [])
        
        for stock in stocks:
            stock_code = stock.get('stockCode')
            
            if not stock_code:
                logger.warning("Skipping stock without stockCode")
                continue
            
            if stock_code in merged_stocks:
                # Merge existing stock
                existing = merged_stocks[stock_code]
                existing_news = existing.get('news', [])
                new_news = stock.get('news', [])
                
                total_news_before += len(existing_news) + len(new_news)
                
                # Merge and deduplicate news
                combined_news = existing_news + new_news
                deduped_news = deduplicate_news(combined_news, max_news)
                
                existing['news'] = deduped_news
                total_news_after += len(deduped_news)
                
                if verbose:
                    logger.info(f"Merged {stock_code}: {len(existing_news)} + {len(new_news)} -> {len(deduped_news)} news")
            else:
                # New stock
                stock_news = stock.get('news', [])
                total_news_before += len(stock_news)
                
                deduped_news = deduplicate_news(stock_news, max_news)
                stock['news'] = deduped_news
                total_news_after += len(deduped_news)
                
                merged_stocks[stock_code] = stock
                
                if verbose:
                    logger.info(f"Added {stock_code}: {len(stock_news)} -> {len(deduped_news)} news")
    
    # Convert back to list
    merged_list = list(merged_stocks.values())
    
    # Sort by stockCode for consistent output
    merged_list.sort(key=lambda x: x.get('stockCode', ''))
    
    logger.info(f"Merged {len(merged_list)} unique stocks")
    logger.info(f"News deduplication: {total_news_before} -> {total_news_after} items")
    
    return {
        'stocks': merged_list,
        'metadata': {
            'mergedAt': datetime.now().isoformat(),
            'sourceFiles': len(stock_data_list),
            'totalStocks': len(merged_list),
            'totalNewsBefore': total_news_before,
            'totalNewsAfter': total_news_after,
            'maxNewsPerStock': max_news
        }
    }


def create_backup(output_path: Path, max_backups: int, dry_run: bool = False) -> Optional[Path]:
    """
    Create backup of existing stock.json file.
    
    Args:
        output_path: Path to the output file
        max_backups: Maximum number of backups to keep
        dry_run: Simulate operation
        
    Returns:
        Path to backup file or None if no backup created
    """
    if not output_path.exists():
        logger.info("No existing stock.json to backup")
        return None
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    backup_path = output_path.with_suffix(f'.json.backup.{timestamp}')
    
    if dry_run:
        logger.info(f"[DRY-RUN] Would create backup: {backup_path.name}")
        return backup_path
    
    try:
        shutil.copy2(output_path, backup_path)
        logger.info(f"Created backup: {backup_path.name}")
        
        # Cleanup old backups
        cleanup_old_backups(output_path.parent, output_path.name, max_backups, dry_run)
        
        return backup_path
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        return None


def cleanup_old_backups(
    backup_dir: Path,
    base_filename: str,
    max_backups: int,
    dry_run: bool = False
) -> int:
    """
    Remove old backup files, keeping only the most recent ones.
    
    Args:
        backup_dir: Directory containing backup files
        base_filename: Base filename to match backups
        max_backups: Maximum number of backups to keep
        dry_run: Simulate operation
        
    Returns:
        Number of backups removed
    """
    # Find all backup files matching pattern
    pattern = f"{base_filename}.backup.*"
    backup_files = sorted([
        Path(f) for f in glob.glob(str(backup_dir / pattern))
    ])
    
    if len(backup_files) <= max_backups:
        logger.debug(f"Backup count ({len(backup_files)}) within limit ({max_backups})")
        return 0
    
    # Remove oldest backups
    to_remove = backup_files[:-max_backups]
    removed_count = 0
    
    for backup_path in to_remove:
        if dry_run:
            logger.info(f"[DRY-RUN] Would remove old backup: {backup_path.name}")
        else:
            try:
                backup_path.unlink()
                logger.info(f"Removed old backup: {backup_path.name}")
                removed_count += 1
            except Exception as e:
                logger.error(f"Failed to remove backup {backup_path.name}: {e}")
    
    if to_remove:
        logger.info(f"Cleaned up {removed_count} old backup(s), keeping {max_backups}")
    
    return removed_count


def write_atomic(data: Dict[str, Any], output_path: Path, dry_run: bool = False) -> bool:
    """
    Write data to file atomically using temp file + rename.
    
    Args:
        data: Data to write
        output_path: Target file path
        dry_run: Simulate operation
        
    Returns:
        True if successful, False otherwise
    """
    if dry_run:
        logger.info(f"[DRY-RUN] Would write to: {output_path}")
        return True
    
    # Create temp file in same directory
    temp_path = output_path.with_suffix(f'.tmp.{os.getpid()}')
    
    try:
        # Write to temp file
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # Atomic rename
        temp_path.rename(output_path)
        
        logger.info(f"Successfully wrote: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to write file: {e}")
        
        # Cleanup temp file if it exists
        if temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass
        
        return False


def delete_source_files(
    source_files: List[Path],
    dry_run: bool = False
) -> int:
    """
    Delete source stock files after successful merge.
    
    Args:
        source_files: List of source file paths
        dry_run: Simulate operation
        
    Returns:
        Number of files deleted
    """
    deleted_count = 0
    
    for file_path in source_files:
        if dry_run:
            logger.info(f"[DRY-RUN] Would delete: {file_path.name}")
            deleted_count += 1
        else:
            try:
                file_path.unlink()
                logger.info(f"Deleted source file: {file_path.name}")
                deleted_count += 1
            except Exception as e:
                logger.error(f"Failed to delete {file_path.name}: {e}")
    
    return deleted_count


def print_summary(
    source_files: List[Path],
    merged_data: Dict[str, Any],
    backup_path: Optional[Path],
    deleted_count: int,
    args: argparse.Namespace
) -> None:
    """
    Print summary of merge operation.
    
    Args:
        source_files: List of processed source files
        merged_data: Merged stock data
        backup_path: Path to backup file (if created)
        deleted_count: Number of deleted source files
        args: Command line arguments
    """
    metadata = merged_data.get('metadata', {})
    
    print("\n" + "=" * 60)
    print("📊 STOCK MERGE SUMMARY")
    print("=" * 60)
    print(f"Source files processed: {len(source_files)}")
    print(f"Unique stocks merged:   {metadata.get('totalStocks', 0)}")
    print(f"News items (before):    {metadata.get('totalNewsBefore', 0)}")
    print(f"News items (after):     {metadata.get('totalNewsAfter', 0)}")
    print(f"News deduplication:     {metadata.get('totalNewsBefore', 0) - metadata.get('totalNewsAfter', 0)} removed")
    print(f"Max news per stock:     {metadata.get('maxNewsPerStock', 20)}")
    print(f"Backup created:         {'Yes' if backup_path else 'No'}")
    print(f"Source files deleted:   {deleted_count}")
    print(f"Output file:            {args.output_file}")
    
    if args.dry_run:
        print("\n⚠️  DRY-RUN MODE - No changes were made")
    
    print("=" * 60 + "\n")


def main() -> int:
    """
    Main entry point for stock merge script.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting stock merge operation")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output file: {args.output_file}")
    
    if args.dry_run:
        logger.warning("DRY-RUN MODE ENABLED")
    
    # Discover source files
    source_files = discover_stock_files(args.input_dir)
    
    if not source_files:
        logger.info("No stock_*.json files found to merge")
        print("\nℹ️  No stock files found in", args.input_dir)
        print("    Expected pattern: stock_*.json\n")
        return 0
    
    # Load all stock data
    stock_data_list: List[Dict[str, Any]] = []
    
    for file_path in source_files:
        data = load_stock_file(file_path)
        if data is not None:
            stock_data_list.append(data)
    
    if not stock_data_list:
        logger.error("No valid stock data loaded")
        return 1
    
    # Merge stocks
    merged_data = merge_stocks(stock_data_list, args.max_news, args.verbose)
    
    # Create backup (if not disabled)
    backup_path = None
    output_path = Path(args.output_file)
    
    if not args.no_backup:
        backup_path = create_backup(output_path, args.max_backups, args.dry_run)
    
    # Ensure output directory exists
    if not args.dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write merged data atomically
    success = write_atomic(merged_data, output_path, args.dry_run)
    
    if not success:
        logger.error("Failed to write merged stock.json")
        return 1
    
    # Delete source files (if not keeping)
    deleted_count = 0
    
    if not args.keep_source and not args.dry_run:
        deleted_count = delete_source_files(source_files, args.dry_run)
    elif not args.keep_source and args.dry_run:
        deleted_count = len(source_files)
    
    # Print summary
    print_summary(source_files, merged_data, backup_path, deleted_count, args)
    
    logger.info("Stock merge operation completed successfully")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
