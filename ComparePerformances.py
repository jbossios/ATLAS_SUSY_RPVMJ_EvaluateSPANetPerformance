import h5py,os,sys
import pandas as pd
import numpy as np

Versions = {
  'v11_reproduction' : 'FullStats',
  'v25'              : 'SameStatsPerMass',
}

############################################################################################################################################################################
# DO NOT MODIFY (below this line)
############################################################################################################################################################################

Masses            = ['All',900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500]
vstr              = '_vs_'.join(Versions)

##########################################
# Plot 2g efficiency vs mass (Full case)
##########################################
import ROOT,array
# AtlasStyle
ROOT.gROOT.LoadMacro("/home/jbossios/cern/AtlasStyle/AtlasStyle.C")
ROOT.SetAtlasStyle()
ROOT.gROOT.SetBatch(True)
os.system('mkdir -p Plots/{}'.format(vstr))
# TCanvas
Canvas  = ROOT.TCanvas()
outName = "Plots/{}/Compare2gEfficiency_vs_gluino_mass_Full.pdf".format(vstr)
Canvas.Print(outName+"[")
# Legends
Legends = ROOT.TLegend(0.2,0.75,0.45,0.9)
Legends.SetTextFont(42)
# TMultiGraph
MG = ROOT.TMultiGraph()
# SPANet vs mass graphs
Colors = [ROOT.kBlack,ROOT.kRed+1,ROOT.kCyan,ROOT.kOrange,ROOT.kGreen+2,ROOT.kMagenta,ROOT.kAzure]
counter = 0
x = [mass for mass in Masses if mass!='All']
for version in Versions:
  FileName = 'Outputs/{}/2gEfficiency_vs_gluino_mass_Full.root'.format(version)
  File     = ROOT.TFile.Open(FileName)
  if not File:
    print('ERROR: {} not found, exiting'.format(FileName))
    sys.exit(1)
  Graph = File.Get('Graph')
  Graph.SetLineColor(Colors[counter])
  Graph.SetMarkerColor(Colors[counter])
  MG.Add(Graph)
  extra  = ' {}'.format(Versions[version]) if Versions[version] != '' else ''
  #legend = 'SPANet vs mass{}'.format(extra)
  Legends.AddEntry(Graph,extra,'p')
  counter += 1
MG.Draw('apl')
MG.GetXaxis().SetTitle('Gluino Mass [GeV]')
MG.GetYaxis().SetTitle('Reconstruction efficiency (2g)')
Canvas.Update()
Canvas.Modified()
Legends.Draw('same')
Canvas.Print(outName)
Canvas.Print(outName+']')

print('>>> ALL DONE <<<')
