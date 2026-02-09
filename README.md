Steps: REMEMBER TO ADJUST THE PATHS
1. Prepare the corpora in the following formats (very important)
    1. .csv file, each line correspond to one sentence. The line should match between the token and the lemmatised version.
    2. 1 lemmatised and 1 raw file for each lemma
2.Parsed the files using FrameNet_parsed.ipynb or FrameNet_parsed.py
3. In FrameNet_parsed.ipynb, there are other codes for calculating skipped sentences
4. To calculate the number of fall-backs from BASE to SMALL, check the log file of FrameNet_parsed.py
5. Check the token-lemma alignement using Alignment_Check.ipynb
6. To test the Swedish model, first download from https://github.com/lucyYB/SweFN-SRL (releases section), then add the paths of the XML file and the model folder to the FrameNet_parsed.ipynb
7. After parsing, run FrameNet_lemma.ipynb notebook to get the quantitative results (i.e., JSD, ranking)
8. For qualitative inspection, go to /SynFlow/case_studies/SemEval_en_FrameFlow/Frame_Flow.ipynb
