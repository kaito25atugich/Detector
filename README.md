
You can use detectors shown below.

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
If you execute following code, then you can get the result for Zero-shot Detector without Binoculars.

```
python detection.py
```

If you want to get the binoculars result, then run `detection_binoculars_only.py`.

When you set `is_{detector}` flag in `detection.py`, it enables the use of other methods.

And you can run white-box settings using the following code.

```
python detection.py --prompt
```


### Acknowledgement
Some code in this projects is derived from or refers to the work of the following repositories:
- https://github.com/eric-mitchell/detect-gpt
- https://github.com/baoguangsheng/fast-detect-gpt
- https://github.com/ArGintum/GPTID
- https://github.com/ahans30/Binoculars

