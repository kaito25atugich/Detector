
This project is designed with reproducibility in mind and can be used for paper reproduction purposes. Below is a list of available detectors:

- DetectGPT
- FastDetectGPT
- NPR
- LRR
- FastNPR
- Binoculars
- Log-likelihood
- Rank
- Log-Rank
- Entropy
- Fine-tuned RoBERTa-large for GPT2
- RADAR
- Intrinsic-Dimension

### How to use
0. First, You need to create directories.
```
mkdir results plots txtdata
```

1. Run the following code to obtain results for the Zero-shot Detector without Binoculars:
```
python detection.py
```
2. If you specifically need results from the Binoculars detector, execute the following command:
```
python detection_binoculars_only.py
```
3. To enable the use of other detectors, set the is_{detector} flag in detection.py. This allows you to experiment with different methods.

For white-box settings, use the following code:

```
python detection.py --prompt
```


### Acknowledgement
Some code in this projects is derived from or refers to the work of the following repositories:
- https://github.com/eric-mitchell/detect-gpt
- https://github.com/baoguangsheng/fast-detect-gpt
- https://github.com/ArGintum/GPTID
- https://github.com/ahans30/Binoculars

Feel free to explore and enhance the functionality based on your specific needs. If you have any questions or suggestions, please don't hesitate to reach out.
