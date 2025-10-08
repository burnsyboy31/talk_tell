# talk_tell

Analysis Pipeline:

First decide which company to analyze (I chose MSFT for example)

1) Place desired earnings calls URLs from Motley fool (fool.com) into sample_urls.txt

2)run tran_pull.py to scrape motley fool for the earnings calls transcript
    -change this to include new company ticker
                    -filename = os.path.join(data_dir, "combined_transcripts.parquet") 
    -you can run parquet_viewer.py to check out your results 
        switch out parquet_files = glob.glob("Data/combined_transcripts.parquet") with new file 

3)run feature_prep.py to get data ready for analysis
            #Run command line (switch out output ticker)
    python3 feature_prep.py \
   --in /Users/zachburns/Desktop/talk_tell/Data/combined_transcripts.parquet \
   --out /Users/zachburns/Desktop/talk_tell/Data/msft_features.parquet

4)run train_msft_logit.py for logistic regression
    Run command line:
    python3 train_msft_logit.py \
  --in /Users/zachburns/Desktop/talk_tell/Data/msft_features.parquet \
  --out /Users/zachburns/Desktop/talk_tell/Data/msft_logit_oos.csv


