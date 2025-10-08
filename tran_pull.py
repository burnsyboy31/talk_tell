#!/usr/bin/env python3
"""
Earnings Call Transcript Scraper for The Motley Fool
Scrapes earnings call transcripts from fool.com URLs
"""

import requests
from bs4 import BeautifulSoup
import re
import json
import sys
from urllib.parse import urlparse
import time
import argparse
import pandas as pd
from datetime import datetime
import os


class MotleyFoolTranscriptScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
    
    def extract_transcript_content(self, soup):
        """Extract the main transcript content from the parsed HTML"""
        
        transcript_text = ""
        company_info = {}
        
        # Look for company and earnings info in meta tags or headings
        title_tag = soup.find('title')
        if title_tag:
            title_text = title_tag.get_text().strip()
            company_info['title'] = title_text
            
            # Extract company ticker and quarter info from title
            ticker_match = re.search(r'([A-Z]{2,5})', title_text)
            if ticker_match:
                company_info['ticker'] = ticker_match.group(1)
            
            quarter_match = re.search(r'(Q[1-4]\s+\d{4})', title_text)
            if quarter_match:
                company_info['quarter'] = quarter_match.group(1)
        
        # Get all text content from the page
        page_text = soup.get_text()
        
        # Look for the "Full Conference Call Transcript" marker
        transcript_start_marker = "Full Conference Call Transcript"
        transcript_start_index = page_text.find(transcript_start_marker)
        
        if transcript_start_index != -1:
            # Extract text starting from after the marker
            transcript_text = page_text[transcript_start_index + len(transcript_start_marker):]
            
            # Find the end of the transcript (look for common ending patterns)
            end_markers = [
                "This concludes today's conference",
                "Thank you for your participation",
                "You may disconnect",
                "This concludes the conference call",
                "End of Q&A",
                "This article is a transcript"
            ]
            
            transcript_end_index = len(transcript_text)
            for marker in end_markers:
                end_index = transcript_text.find(marker)
                if end_index != -1:
                    transcript_end_index = min(transcript_end_index, end_index)
            
            # Cut off the transcript at the end marker
            transcript_text = transcript_text[:transcript_end_index].strip()
            
        else:
            # Fallback: try to find transcript content using traditional selectors
            print("Warning: 'Full Conference Call Transcript' marker not found. Using fallback extraction method.")
            
            # Try multiple selectors to find the transcript content
            selectors = [
                'article',
                '.article-content',
                '.transcript-content',
                '.content',
                'main',
                '.main-content'
            ]
            
            content_area = None
            for selector in selectors:
                content_area = soup.select_one(selector)
                if content_area:
                    break
            
            if not content_area:
                # Fallback: look for any div with substantial text content
                divs = soup.find_all('div')
                for div in divs:
                    text_content = div.get_text().strip()
                    if len(text_content) > 1000 and 'earnings' in text_content.lower():
                        content_area = div
                        break
            
            if content_area:
                transcript_text = content_area.get_text()
        
        if transcript_text:
            # Clean up extra whitespace and formatting
            transcript_text = re.sub(r'\n\s*\n', '\n\n', transcript_text)
            transcript_text = re.sub(r'[ \t]+', ' ', transcript_text)
            transcript_text = transcript_text.strip()
            
            # Try to extract speaker information
            speakers = self.extract_speakers(transcript_text)
            company_info['speakers'] = speakers
        
        return transcript_text, company_info
    
    def extract_speakers(self, text):
        """Extract speaker names from the transcript"""
        # Common patterns for speaker identification
        speaker_patterns = [
            r'^([A-Z][a-z]+\s+[A-Z][a-z]+):',
            r'^([A-Z][a-z]+):',
            r'^([A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+):',
        ]
        
        speakers = set()
        lines = text.split('\n')
        
        for line in lines[:50]:  # Check first 50 lines for speakers
            line = line.strip()
            for pattern in speaker_patterns:
                match = re.match(pattern, line)
                if match:
                    speaker = match.group(1).strip()
                    if len(speaker) > 2 and len(speaker) < 50:
                        speakers.add(speaker)
        
        return list(speakers)
    
    def parse_transcript_into_segments(self, transcript_text):
        """Parse transcript into structured segments (speaker, text, timestamp)"""
        segments = []
        lines = transcript_text.split('\n')
        
        current_speaker = None
        current_text = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line starts with a speaker name (common patterns)
            speaker_match = re.match(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*):\s*(.*)$', line)
            
            if speaker_match:
                # Save previous segment if exists
                if current_speaker and current_text:
                    segments.append({
                        'speaker': current_speaker,
                        'text': ' '.join(current_text).strip(),
                        'word_count': len(' '.join(current_text).split())
                    })
                
                # Start new segment
                current_speaker = speaker_match.group(1).strip()
                remaining_text = speaker_match.group(2).strip()
                current_text = [remaining_text] if remaining_text else []
            else:
                # Continuation of current speaker's text
                current_text.append(line)
        
        # Don't forget the last segment
        if current_speaker and current_text:
            segments.append({
                'speaker': current_speaker,
                'text': ' '.join(current_text).strip(),
                'word_count': len(' '.join(current_text).split())
            })
        
        return segments
    
    def scrape_transcript(self, url):
        """Scrape transcript from the given URL"""
        try:
            print(f"Fetching: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            print("Parsing HTML content...")
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract transcript content
            transcript_text, company_info = self.extract_transcript_content(soup)
            
            if not transcript_text:
                print("Warning: No transcript content found. The page structure might have changed.")
                return None, None
            
            print(f"Successfully extracted {len(transcript_text)} characters of transcript content")
            
            return transcript_text, company_info
            
        except requests.RequestException as e:
            print(f"Error fetching URL: {e}")
            return None, None
        except Exception as e:
            print(f"Error parsing content: {e}")
            return None, None
    
    def save_transcript(self, transcript_text, company_info, output_format='txt'):
        """Save the transcript to a file"""
        if not transcript_text:
            print("No transcript to save")
            return
        
        # Create Data directory if it doesn't exist
        data_dir = "Data"
        os.makedirs(data_dir, exist_ok=True)
        
        # Generate filename
        ticker = company_info.get('ticker', 'UNKNOWN')
        quarter = company_info.get('quarter', 'UNKNOWN').replace(' ', '_')
        
        if output_format == 'txt':
            filename = os.path.join(data_dir, f"{ticker}_{quarter}_transcript.txt")
            
            with open(filename, 'w', encoding='utf-8') as f:
                # Write header information
                if company_info.get('title'):
                    f.write(f"Title: {company_info['title']}\n")
                if company_info.get('ticker'):
                    f.write(f"Ticker: {company_info['ticker']}\n")
                if company_info.get('quarter'):
                    f.write(f"Quarter: {company_info['quarter']}\n")
                if company_info.get('speakers'):
                    f.write(f"Speakers: {', '.join(company_info['speakers'])}\n")
                f.write("\n" + "="*50 + "\n\n")
                
                # Write transcript content
                f.write(transcript_text)
            
            print(f"Transcript saved to: {filename}")
            
        elif output_format == 'json':
            filename = os.path.join(data_dir, f"{ticker}_{quarter}_transcript.json")
            
            data = {
                'metadata': company_info,
                'transcript': transcript_text,
                'scraped_at': time.strftime("%Y-%m-%d %H:%M:%S"),
                'source_url': getattr(self, 'last_url', 'Unknown')
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"Transcript saved to: {filename}")
            
        elif output_format == 'parquet':
            filename = os.path.join(data_dir, f"{ticker}_{quarter}_transcript.parquet")
            
            # Parse transcript into structured segments
            segments = self.parse_transcript_into_segments(transcript_text)
            
            # Create DataFrame with segments
            df_segments = pd.DataFrame(segments)
            
            # Add metadata columns to each segment
            df_segments['ticker'] = company_info.get('ticker', 'UNKNOWN')
            df_segments['quarter'] = company_info.get('quarter', 'UNKNOWN')
            df_segments['title'] = company_info.get('title', '')
            df_segments['scraped_at'] = datetime.now()
            df_segments['source_url'] = getattr(self, 'last_url', 'Unknown')
            df_segments['total_segments'] = len(segments)
            df_segments['total_speakers'] = len(company_info.get('speakers', []))
            
            # Save to Parquet
            df_segments.to_parquet(filename, index=False)
            
            print(f"Transcript saved to: {filename}")
            print(f"DataFrame shape: {df_segments.shape}")
            print(f"Columns: {list(df_segments.columns)}")
            
            # Also save a summary
            summary_filename = os.path.join(data_dir, f"{ticker}_{quarter}_summary.json")
            summary_data = {
                'metadata': company_info,
                'total_segments': len(segments),
                'total_words': sum(seg['word_count'] for seg in segments),
                'speakers_summary': df_segments.groupby('speaker')['word_count'].sum().to_dict(),
                'scraped_at': datetime.now().isoformat(),
                'source_url': getattr(self, 'last_url', 'Unknown')
            }
            
            with open(summary_filename, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
            print(f"Summary saved to: {summary_filename}")


def scrape_multiple_urls(scraper, urls, output_format='parquet'):
    """Scrape multiple URLs and combine into a single file"""
    all_segments = []
    all_metadata = []
    successful_scrapes = 0
    
    for i, url in enumerate(urls, 1):
        print(f"\n{'='*60}")
        print(f"Processing URL {i}/{len(urls)}: {url}")
        print(f"{'='*60}")
        
        # Scrape the transcript
        transcript_text, company_info = scraper.scrape_transcript(url)
        
        if transcript_text:
            print(f"✓ Successfully scraped transcript ({len(transcript_text)} characters)")
            
            # Parse into segments
            segments = scraper.parse_transcript_into_segments(transcript_text)
            
            # Add URL-specific metadata to each segment
            for segment in segments:
                segment['source_url'] = url
                segment['scrape_order'] = i
                segment['ticker'] = company_info.get('ticker', 'UNKNOWN')
                segment['quarter'] = company_info.get('quarter', 'UNKNOWN')
                segment['title'] = company_info.get('title', '')
                segment['scraped_at'] = datetime.now()
                segment['total_segments_in_transcript'] = len(segments)
                segment['total_speakers_in_transcript'] = len(company_info.get('speakers', []))
            
            all_segments.extend(segments)
            all_metadata.append({
                'url': url,
                'company_info': company_info,
                'segment_count': len(segments),
                'scrape_order': i
            })
            successful_scrapes += 1
            
            # Show preview
            print(f"\nPreview (first 200 chars):")
            print("-" * 40)
            print(transcript_text[:200] + "..." if len(transcript_text) > 200 else transcript_text)
            
        else:
            print(f"✗ Failed to scrape transcript from: {url}")
    
    if not all_segments:
        print("\nNo transcripts were successfully scraped!")
        return
    
    # Create combined DataFrame
    df_combined = pd.DataFrame(all_segments)
    
    # Create Data directory if it doesn't exist
    data_dir = "Data"
    os.makedirs(data_dir, exist_ok=True)
    
    if output_format == 'parquet':
        filename = os.path.join(data_dir, "combined_transcripts.parquet")
        df_combined.to_parquet(filename, index=False)
        
        print(f"\n{'='*60}")
        print(f"COMBINED RESULTS")
        print(f"{'='*60}")
        print(f"✓ Successfully scraped: {successful_scrapes}/{len(urls)} URLs")
        print(f"✓ Total segments: {len(all_segments):,}")
        print(f"✓ Total words: {df_combined['word_count'].sum():,}")
        print(f"✓ Unique companies: {df_combined['ticker'].nunique()}")
        print(f"✓ Unique speakers: {df_combined['speaker'].nunique()}")
        print(f"✓ File saved: {filename}")
        print(f"✓ DataFrame shape: {df_combined.shape}")
        
        # Save metadata summary
        metadata_filename = os.path.join(data_dir, "combined_metadata.json")
        metadata_summary = {
            'total_urls_processed': len(urls),
            'successful_scrapes': successful_scrapes,
            'failed_scrapes': len(urls) - successful_scrapes,
            'total_segments': len(all_segments),
            'total_words': int(df_combined['word_count'].sum()),
            'companies': df_combined['ticker'].value_counts().to_dict(),
            'speakers': df_combined['speaker'].value_counts().to_dict(),
            'scraped_at': datetime.now().isoformat(),
            'metadata': all_metadata
        }
        
        with open(metadata_filename, 'w', encoding='utf-8') as f:
            json.dump(metadata_summary, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Metadata saved: {metadata_filename}")
        
    elif output_format == 'json':
        filename = os.path.join(data_dir, "combined_transcripts.json")
        combined_data = {
            'metadata': {
                'total_urls_processed': len(urls),
                'successful_scrapes': successful_scrapes,
                'total_segments': len(all_segments),
                'scraped_at': datetime.now().isoformat()
            },
            'transcripts': all_metadata,
            'segments': all_segments
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Combined data saved: {filename}")
    
    return df_combined


def main():
    parser = argparse.ArgumentParser(description='Scrape earnings call transcripts from The Motley Fool')
    parser.add_argument('urls', nargs='*', help='URL(s) of the earnings call transcript(s)')
    parser.add_argument('--format', choices=['txt', 'json', 'parquet'], default='parquet', 
                       help='Output format (default: parquet)')
    parser.add_argument('--output', '-o', help='Output filename (optional)')
    parser.add_argument('--file', '-f', help='File containing URLs (one per line)')
    
    args = parser.parse_args()
    
    # Get URLs from command line or file
    urls = []
    
    if args.file:
        try:
            with open(args.file, 'r') as f:
                file_urls = [line.strip() for line in f if line.strip()]
                urls.extend(file_urls)
            print(f"Loaded {len(file_urls)} URLs from file: {args.file}")
        except FileNotFoundError:
            print(f"Error: File {args.file} not found")
            sys.exit(1)
    
    urls.extend(args.urls)
    
    if not urls:
        print("Error: No URLs provided")
        sys.exit(1)
    
    # Validate URLs
    valid_urls = []
    for url in urls:
        parsed_url = urlparse(url)
        if 'fool.com' not in parsed_url.netloc:
            print(f"Warning: URL {url} is not from fool.com")
        valid_urls.append(url)
    
    print(f"Processing {len(valid_urls)} URLs...")
    
    scraper = MotleyFoolTranscriptScraper()
    
    if len(valid_urls) == 1:
        # Single URL - use original behavior
        url = valid_urls[0]
        scraper.last_url = url
        
        transcript_text, company_info = scraper.scrape_transcript(url)
        
        if transcript_text:
            print("\n" + "="*50)
            print("TRANSCRIPT PREVIEW (first 500 characters):")
            print("="*50)
            print(transcript_text[:500] + "..." if len(transcript_text) > 500 else transcript_text)
            print("="*50)
            
            if args.output:
                # Save with custom filename
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(transcript_text)
                print(f"Transcript saved to: {args.output}")
            else:
                # Save with auto-generated filename
                scraper.save_transcript(transcript_text, company_info, args.format)
            
            print(f"\nCompany Info: {company_info}")
        else:
            print("Failed to extract transcript content")
            sys.exit(1)
    else:
        # Multiple URLs - combine into single file
        scrape_multiple_urls(scraper, valid_urls, args.format)


if __name__ == "__main__":
    # If no command line arguments, try to use sample_urls.txt file
    if len(sys.argv) == 1:
        # Check if sample_urls.txt exists
        if os.path.exists('sample_urls.txt'):
            print("No arguments provided. Using sample_urls.txt file...")
            # Simulate command line arguments to use the file
            sys.argv = ['tran_pull.py', '--file', 'sample_urls.txt']
            main()
        else:
            # Fallback to single URL if no file exists
            url = "https://www.fool.com/earnings/call-transcripts/2025/08/05/microsoft-msft-q4-2025-earnings-call-transcript/"
            print(f"No sample_urls.txt found. Using default single URL: {url}")
            scraper = MotleyFoolTranscriptScraper()
            scraper.last_url = url
            
            transcript_text, company_info = scraper.scrape_transcript(url)
            
            if transcript_text:
                print("\n" + "="*50)
                print("TRANSCRIPT PREVIEW (first 500 characters):")
                print("="*50)
                print(transcript_text[:500] + "..." if len(transcript_text) > 500 else transcript_text)
                print("="*50)
                
                scraper.save_transcript(transcript_text, company_info, 'parquet')
                print(f"\nCompany Info: {company_info}")
            else:
                print("Failed to extract transcript content")
    else:
        main()
