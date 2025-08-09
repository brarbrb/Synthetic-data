# Synthetic-data

Installation of BlenderProc: 

```conda create -n synth python=3.10

conda activate synth

pip install blenderproc

blenderproc quickstart
```


### EDA results:
tweezers have only one material to work cand change, whilst the needle holder has two ptoperties that can vary.


Note: change location, scale, and rotation!


### Saving the dataset - correct format
For 2D pose estimation with two object classes (needle holder and tweezers) and synthetic data you’re generating yourself, the trick is to save the dataset in one general, model-agnostic format so you can later convert it to whatever a specific model requires.
That way, you only do the annotation once and can reformat as needed.
