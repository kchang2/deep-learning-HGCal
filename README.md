# Deep Learning for the CMS HGCal upgrade
We apply deep learning techniques to the backend of the CMS HGCal data analysis.

## Pre-process data
You can do this in two options

1. The modified [HGCal Reco N-tuple analyzer](https://github.com/kchang2/reco-ntuples)
2. The root_processor.py that comes installed.

The difference between the two options is that the first option has more features than the second option.

### Running the first option
Follow the instruction sfrom the reco-ntuple [README](https://github.com/kchang2/reco-ntuples).

### Running the second option
The program runs by python standards
``` python root_processor.py ```

If you want to run it with a script to loop through all the files at once, just specific the parameters within script.py and run that instead
``` python script.py ```

