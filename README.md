# <div align='center'>Evaluate SPANet's performance</div>

## Reconstruct gluino masses and plot reconstruction efficiency vs gluino mass

Choose SPANet's version(s) in ```EvaluatePerformance_signal.py``` and run script:

```
python EvaluatePerformance_signal.py
```

One could compare the reconstruction efficiency from different SPANet networks/versions with the ```ComparePerformances.py``` script.

## Reconstruct gluino masses for signal and dijets and compare 

Use the ```Signal_vs_dijets_ReconstructedGluinoMasses.py``` script for preparing inputs and then ```MakePlots.py``` for making comparison plots (plots are made with the first script but luminosity is only considered in the second one).

**NOTE:** the above script also produces all inputs needed to make plots with the https://gitlab.cern.ch/atlas-phys-susy-wg/RPVLL/rpvmultijet/rpv-ml-jet-matching-common package (mass, ROC and significance curves for all networks, i.e. not only SPANet).
