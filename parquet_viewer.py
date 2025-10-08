#!/usr/bin/env python3
"""
Simple Parquet File Viewer
Quick script to view sample content from earnings transcript parquet files
"""

import pandas as pd
import glob
import sys
from pathlib import Path


def view_parquet_sample(filename=None, num_rows=10):
    """
    View a sample of parquet file content
    
    Args:
        filename: Specific parquet file to view (optional)
        num_rows: Number of rows to display (default: 10)
    """
    
    # If no filename specified, find the most recent parquet file
    if not filename:
        parquet_files = glob.glob("Data/combined_transcripts.parquet")
        if not parquet_files:
            print("No parquet files found in current directory")
            return
        
        # Get the most recent file
        filename = max(parquet_files, key=lambda f: Path(f).stat().st_mtime)
        print(f"Using most recent parquet file: {filename}")
    
    try:
        # Read the parquet file
        df = pd.read_parquet(filename)
        
        print(f"\n{'='*60}")
        print(f"PARQUET FILE: {filename}")
        print(f"{'='*60}")
        
        # Display basic info
        print(f"Shape: {df.shape} (rows × columns)")
        print(f"Columns: {list(df.columns)}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        # Display data types
        print(f"\nData Types:")
        print(df.dtypes)
        
        # Display first few rows
        print(f"\nFirst {num_rows} rows:")
        print("-" * 60)
        
        # Show key columns with text truncated for readability
        display_columns = ['speaker', 'text', 'word_count', 'ticker', 'quarter']
        available_columns = [col for col in display_columns if col in df.columns]
        
        if available_columns:
            sample_df = df[available_columns].head(num_rows).copy()
            
            # Truncate text column if it exists
            if 'text' in sample_df.columns:
                sample_df['text'] = sample_df['text'].str[:100] + '...'
            
            print(sample_df.to_string(index=True, max_colwidth=50))
        
        # Display summary statistics
        print(f"\nSummary Statistics:")
        print("-" * 60)
        
        if 'word_count' in df.columns:
            print(f"Total segments: {len(df)}")
            print(f"Total words: {df['word_count'].sum():,}")
            print(f"Average words per segment: {df['word_count'].mean():.1f}")
            print(f"Longest segment: {df['word_count'].max()} words")
            print(f"Shortest segment: {df['word_count'].min()} words")
        
        # Speaker statistics
        if 'speaker' in df.columns:
            print(f"\nSpeaker Statistics:")
            print("-" * 60)
            speaker_stats = df.groupby('speaker')['word_count'].agg(['count', 'sum', 'mean']).round(1)
            speaker_stats.columns = ['Segments', 'Total Words', 'Avg Words/Segment']
            print(speaker_stats)
        
        # Display sample of full text (first segment)
        if 'text' in df.columns and len(df) > 0:
            print(f"\nSample Full Text (First Segment):")
            print("-" * 60)
            first_speaker = df.iloc[0]['speaker'] if 'speaker' in df.columns else 'Unknown'
            first_text = df.iloc[0]['text']
            print(f"Speaker: {first_speaker}")
            print(f"Text: {first_text[:300]}{'...' if len(first_text) > 300 else ''}")
        
    except Exception as e:
        print(f"Error reading parquet file: {e}")


def list_parquet_files():
    """List all parquet files in the current directory"""
    parquet_files = glob.glob("*.parquet")
    
    if not parquet_files:
        print("No parquet files found in current directory")
        return
    
    print("Available parquet files:")
    print("-" * 40)
    
    for i, filename in enumerate(parquet_files, 1):
        try:
            df = pd.read_parquet(filename)
            file_size = Path(filename).stat().st_size / 1024  # KB
            print(f"{i}. {filename}")
            print(f"   Shape: {df.shape}, Size: {file_size:.1f} KB")
            if 'ticker' in df.columns and 'quarter' in df.columns:
                ticker = df['ticker'].iloc[0] if len(df) > 0 else 'Unknown'
                quarter = df['quarter'].iloc[0] if len(df) > 0 else 'Unknown'
                print(f"   Company: {ticker} - {quarter}")
        except Exception as e:
            print(f"{i}. {filename} (Error: {e})")
        print()


def main():
    """Main function with command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='View parquet file content')
    parser.add_argument('file', nargs='?', help='Parquet file to view (optional)')
    parser.add_argument('--rows', '-r', type=int, default=10, 
                       help='Number of rows to display (default: 10)')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List all available parquet files')
    
    args = parser.parse_args()
    
    if args.list:
        list_parquet_files()
    else:
        view_parquet_sample(args.file, args.rows)


if __name__ == "__main__":
    main()
