import h5py,os,sys
import pandas as pd
import numpy as np

Versions = {
# SPANet vs CoLaLoLa plot
#  'v11' : '',
#  'v14' : 'w/ energy scaling',
# Compare different SPANet setups
#  'v4'  : 'lr=0.0015,epochs=50',
#  'v10' : 'lr=0.001,epochs=50',
#  'v11' : 'lr=0.0001,epochs=200',
#  'v13' : 'lr=0.0001,epochs=200,PartialEvents',
#  'v15' : 'lr=0.00005,epochs=200',
# See performance of v16 or v11
#  'v16' : 'lr=0.0001,epochs=200,#jets>=8',
#  'v11' : 'lr=0.0001,epochs=200',
#  'v11'              : 'Reference',
  'v11_reproduction' : 'Validation',
#  'v24'              : 'SameStatsPerMass',
#  'v25'              : 'SameStatsPerMass',
# Compare softmin vs min
#  'v11' : '(softmin)',
#  'v17' : '(min)',
# Compare performance when using different learning_rate_cycles values
#  'v11' : 'learning_rate_cycles = 0',
#  'v18' : 'learning_rate_cycles = 1',
#  'v19' : 'learning_rate_cycles = 2',
#  'v20' : 'learning_rate_cycles = 3',
#  'v21' : 'learning_rate_cycles = 4',
#  'v22' : 'learning_rate_cycles = 5',
#  'v23' : 'learning_rate_cycles = 6',
}

Samples = {
  'Pred' : '/home/jbossios/cern/SUSY/RPVMJ/SPANet_outputs/H5predictions/VERSION/signal_testing_VERSIONFLAVOUR_output.h5',
  'True' : '/home/jbossios/cern/SUSY/RPVMJ/CreateH5inputs/Git/Outputs/Signal/v4/FLAVOURSignalData_testing.h5',
#  'True' : '/home/jbossios/cern/SUSY/RPVMJ/CreateH5inputs/Git/Outputs/Signal/v7/FLAVOURSignalData_testing.h5',
#  'True' : '/home/jbossios/cern/SUSY/RPVMJ/CreateH5inputs/Outputs/Signal/v5/AllSignalData_testing.h5',
#  'True' : '/home/jbossios/cern/SUSY/RPVMJ/CreateH5inputs/Outputs/Signal/v6/AllSignalData_testing.h5',
}

MainJetMult = '>=6'

CompareWithCoLaLoLa            = False
ShowInclusivePerformance       = False
ShowPerformanceForExactly6Jets = False
ShowPerformanceForExactly8Jets = False
ShowPerformanceForExactly9Jets = False
CompareFlavourTypes            = False # Used only if len(Versions) == 1

#FlavourTypes      = ['all','UDS','UDB']
FlavourTypes      = ['all']

############################################################################################################################################################################
# DO NOT MODIFY (below this line)
############################################################################################################################################################################

Masses            = ['All',900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500]
JetMultiplicities = ['==6','==7','==8','==9','>=6','>=8','>=9']
vstr              = '_vs_'.join(Versions)

############################################################################################################################################################################
# Type of events to be investigated
EventTypes = ['all','0g','1g','2g','>=1g'] # events with exactly one gluino reconstructable, exactly 2 gluinos reconstructable, or at least one gluino reconstructable
# Explanations:
# When looking at events in which the full event is fully reconstructable, that means that for the original testing dataset, I have mask equal to True on both g1 and g2 (that means jet reco indexes not equal to -1 for all q1/q2/q3). Separatelly look at events in which only one gluino has mask True (the other must have mask False). Then look at events where at least one gluino is reconstructable. Then all events regardless of the number of reconstructable gluinos
############################################################################################################################################################################

############################################################################################################################################################################
# IDEA:
# In those four types of events compare the following quantities depending on the number of reco jets in the events (=6,=7,>=8,All events regardless of the number of jets):
#    Fraction of events (of the requested type and jet multiplicity) w.r.t. all events. Call this the Event Fraction
#    Event Reconstruction Efficiency: Fraction of events (of they chosen type and jet multiplicity) where the predicted indexes match the true (index) values
#        -> taking into account that I don't mind if I jet indexes from q1 are assigned to q2 or q3 (and any other combination)
############################################################################################################################################################################

# Collect numerator/denominator number of events for each event type

# Event Fraction inputs for each event type
EvtFrac_Num = { v : { flav : { mass : { mult : { evtType : 0 for evtType in EventTypes} for mult in JetMultiplicities } for mass in Masses } for flav in FlavourTypes } for v in Versions }
EvtFrac_Den = { v : { flav : { mass : { mult : { evtType : 0 for evtType in EventTypes} for mult in JetMultiplicities } for mass in Masses } for flav in FlavourTypes } for v in Versions }

# Event Efficiency for each event type
EvtEff_Num = { v : { flav : { mass : { mult : { evtType : 0 for evtType in EventTypes} for mult in JetMultiplicities } for mass in Masses } for flav in FlavourTypes } for v in Versions }
EvtEff     = { v : { flav : { mass : { mult : dict() for mult in JetMultiplicities } for mass in Masses } for flav in FlavourTypes } for v in Versions }

# Loop over SPANet versions to compare
for version in Versions:

  # Loop over flavour types
  for flav in FlavourTypes:

    #################################################################################
    # Open input (Pred/True) files
    #################################################################################
    PredFlavour = ''    if flav == 'all' else '_{}'.format(flav)
    TrueFlavour = 'All' if flav == 'all' else flav
    Files   = { sample : h5py.File(name.replace('VERSION',version).replace('FLAVOUR',TrueFlavour if sample == 'True' else PredFlavour),'r')  for sample,name in Samples.items() }
    
    #################################################################################
    # Get data
    #################################################################################
    Data = dict()
    for case in ['source','g1','g2']: # loop over group's keys
      Data[case] = { sample : Files[sample].get(case) for sample in Files }
    
    # Get info about reco jets
    jetMaskInfo = dict()
    for sample in Samples: # loop over True/Pred
      jetMaskInfo[sample] = np.array(Data['source'][sample].get('mask'))
    
    TotalNevents = jetMaskInfo['True'].shape[0]
    
    # Get number of jets on each event
    nJets = np.array([np.sum(x) for x in jetMaskInfo['True']])
    
    # Get gluino mass on each event
    gMass = dict()
    for sample in Samples: # loop over True/Pred
      gMass[sample] = np.array(Data['source'][sample].get('gmass'))
    
    # Get info for each particle
    gluinoInfo = dict()
    for sample in Samples: # loop over True/Pred
      gluinoInfo[sample] = dict()
      for gCase in ['g1','g2']: # loop over gluinos
        gluinoInfo[sample][gCase] = dict()
        for info in ['mask','q1','q2','q3']: # loop over info types
          gluinoInfo[sample][gCase][info] = np.array(Data[gCase][sample].get(info))
    
    # Protections
    if TotalNevents != jetMaskInfo['Pred'].shape[0]:
      print('ERROR: number of events b/w True and Pred do not match, exiting')
      sys.exit(1)
    if TotalNevents != gluinoInfo['True']['g1']['q1'].size:
      print('ERROR: number of events b/w jetMaskInfo and gluinoInfo do not match, exiting')
      sys.exit(1)
    
    print('INFO: Total number of events = {}'.format(TotalNevents))
    
    ############################################
    # Calculate the quantities mentioned above
    ############################################
    
    #####################################################################################################
    # Compare indexes b/w True and Pred on each event type to derive event efficiencies
    # The efficiency is calculated as the fraction of events of a given type
    # that agree on the jet indexes for the appropriate number of gluinos (account for symmetries: q1<->q2,q3)
    # Example:
    # Event Signal Efficiency on 1g events will be the fraction of events in which
    # one gluino can be reconstructed (g1 or g2 with mask==True in True sample)
    # in which all jet (q1,q2,q3) indexes for THAT VERY SAME gluino match to (q1,q2,q3) or to (q2,q1,q3) or (q3,q1,q2) or (q3,q2,q1) or (q1,q3,q2) or (q3,q2,q1)
    # w.r.t. the number of events in which one top can be reconstructed (see above)
    #####################################################################################################
    
    #####################################################################################################
    # Event loop
    #####################################################################################################
    for ievent in range(TotalNevents):
      # Skip events with Njets < 6
      njets = nJets[ievent]
      if njets < 6: continue
      # Categorize event based on the jet multiplicity
      EvtNJetTypes = []
      if njets == 6:
        EvtNJetTypes.append('==6')
      elif njets == 7:
        EvtNJetTypes.append('==7')
      elif njets == 8:
        EvtNJetTypes.append('==8')
      elif njets == 9:
        EvtNJetTypes.append('==9')
      if njets >= 6:
        EvtNJetTypes.append('>=6')
      if njets >= 8:
        EvtNJetTypes.append('>=8')
      if njets >= 9:
        EvtNJetTypes.append('>=9')
      # Find gluino mass
      gmass = gMass['True'][ievent]
      # Identify event types (based on number of reconstructable gluinos)
      EvtTypes            = ['all']
      ReconstructableGluinos = 0 # number of reconstructable gluinos in this event
      for gCase in ['g1','g2']:
        if gluinoInfo['True'][gCase]['mask'][ievent]:
          ReconstructableGluinos += 1
      if ReconstructableGluinos == 0:
        EvtTypes.append('0g')
      elif ReconstructableGluinos == 1:
        EvtTypes.append('1g')
      elif ReconstructableGluinos == 2:
        EvtTypes.append('2g')
      if ReconstructableGluinos >= 1:
        EvtTypes.append('>=1g')
      # Increase number of events for each event type
      for Type in EventTypes: # all event types
        for case in EvtNJetTypes:
          EvtFrac_Den[version][flav]['All'][case][Type] += 1
          EvtFrac_Den[version][flav][gmass][case][Type] += 1
      for Type in EvtTypes: # only for identified event types
        for case in EvtNJetTypes:
          EvtFrac_Num[version][flav]['All'][case][Type] += 1
          EvtFrac_Num[version][flav][gmass][case][Type] += 1
      # For each gluino's decay particle, find out if predicted index matches true index (considering symmetries)
      MatchIndexesByGluino = dict()
      for gCase in ['g1','g2']:
        MatchIndexesByPart_vsTrueCase = dict()
        for altGluinoCase in ['g1','g2']:
          MatchIndexes = dict()
          for partCase in ['q1','q2','q3']:
            # TODO: add protection or check to see if there are duplicated predicted indexes
            if partCase == 'q1':
              altQuarks = ['q2','q3']
            elif partCase == 'q2':
              altQuarks = ['q1','q3']
            elif partCase == 'q3':
              altQuarks = ['q1','q2']
            opts = [ gluinoInfo['Pred'][gCase][partCase][ievent] == gluinoInfo['True'][altGluinoCase][partCase][ievent] ]
            for altQuark in altQuarks:
              opts.append( gluinoInfo['Pred'][gCase][partCase][ievent] == gluinoInfo['True'][altGluinoCase][altQuark][ievent] )
            MatchIndexes[partCase] = True if True in opts else False
          MatchIndexesByPart_vsTrueCase[altGluinoCase] = (MatchIndexes['q1'] + MatchIndexes['q2'] + MatchIndexes['q3']) == 3
        MatchIndexesByGluino[gCase] = MatchIndexesByPart_vsTrueCase['g1'] or MatchIndexesByPart_vsTrueCase['g2']
      # For each gluino, find out if all the predicted indexes for each decay product of the corresponding gluinos (considering constraints from each event type) match the true values
      for mass in ['All',gmass]:
        for case in EvtNJetTypes:
          EvtEff_Num[version][flav][mass][case]['all']    += 1 if (MatchIndexesByGluino['g1']+MatchIndexesByGluino['g2']) >= 1 else False
          if '2g' in EvtTypes:
            EvtEff_Num[version][flav][mass][case]['2g']   += 1 if (MatchIndexesByGluino['g1']+MatchIndexesByGluino['g2']) == 2 else False
          if '1g' in EvtTypes:
            EvtEff_Num[version][flav][mass][case]['1g']   += 1 if (MatchIndexesByGluino['g1']+MatchIndexesByGluino['g2']) == 1 else False
          if '0g' in EvtTypes:
            EvtEff_Num[version][flav][mass][case]['0g']   += 1 if (MatchIndexesByGluino['g1']+MatchIndexesByGluino['g2']) == 0 else False
          if '>=1g' in EvtTypes:
            if '1g' in EvtTypes:
              EvtEff_Num[version][flav][mass][case]['>=1g'] += 1 if (MatchIndexesByGluino['g1']+MatchIndexesByGluino['g2']) == 1 else False
            elif '2g' in EvtTypes:
              EvtEff_Num[version][flav][mass][case]['>=1g'] += 1 if (MatchIndexesByGluino['g1']+MatchIndexesByGluino['g2']) == 2 else False
    
    # Compute efficiencies
    for evtType in EventTypes:
      for mass in Masses:
        for case in JetMultiplicities:
          EvtEff[version][flav][mass][case][evtType] = EvtEff_Num[version][flav][mass][case][evtType] / EvtFrac_Num[version][flav][mass][case][evtType] if EvtFrac_Num[version][flav][mass][case][evtType] != 0 else 0
    
    # Show results
    #print('{} of {} events have at least 6 reconstructed jets'.format(EvtFrac_Den['all'],TotalNevents))
    for mass in Masses:
      for evtType in EventTypes:
        for case in JetMultiplicities:
          if EvtFrac_Den[version][flav][mass][case][evtType] != 0:
            print('###############################################################')
            print('Version: {}'.format(version))
            print('Flavour: {}'.format(flav))
            print('Mass: {}'.format(mass))
            print('Event type: {}'.format(evtType))
            print('Jet multiplicity: {}'.format(case))
            print('Number of events: {}'.format(EvtFrac_Den[version][flav][mass][case][evtType]))
            print('Event Fraction    = {}'.format(EvtFrac_Num[version][flav][mass][case][evtType]/EvtFrac_Den[version][flav][mass][case][evtType]))
            print('SPANet Efficiency = {}'.format(EvtEff[version][flav][mass][case][evtType]))

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
outName = "Plots/{}/2gEfficiency_vs_gluino_mass_Full.pdf".format(vstr)
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
  y = [EvtEff[version]['all'][mass][MainJetMult]['2g'] for mass in Masses if mass!='All']
  Graph = ROOT.TGraph(len(x),array.array('d',x),array.array('d',y))
  Graph.SetLineColor(Colors[counter])
  Graph.SetMarkerColor(Colors[counter])
  # save Graph to a ROOT file
  os.system('mkdir -p Outputs/{}'.format(version))
  outFile = ROOT.TFile('Outputs/{}/2gEfficiency_vs_gluino_mass_Full.root'.format(version),'RECREATE')
  Graph.Write()
  outFile.Close()
  MG.Add(Graph)
  extra  = ' {}'.format(Versions[version]) if Versions[version] != '' else ''
  legend = 'SPANet vs mass{}'.format(extra)
  Legends.AddEntry(Graph,legend,'p')
  counter += 1
  # SPANet average graph
  if ShowInclusivePerformance:
    AvgEff = EvtEff[version]['all']['All'][MainJetMult]['2g']
    yavg = [AvgEff for mass in Masses if mass!='All']
    AvgGraph = ROOT.TGraph(len(x),array.array('d',x),array.array('d',yavg))
    AvgGraph.SetLineColor(Colors[counter])
    AvgGraph.SetMarkerColor(Colors[counter])
    MG.Add(AvgGraph,'l')
    legend = 'SPANet inclusive{}'.format(extra)
    Legends.AddEntry(AvgGraph,legend,'l')
    counter += 1
# CoLaLoLa graphs
if CompareWithCoLaLoLa:
  xCoLaLoLa    = [900,1400,2400]
  yCoLaLoLa    = [0.093,0.114,0.137]
  CoLaLoLaGraph = ROOT.TGraph(len(xCoLaLoLa),array.array('d',xCoLaLoLa),array.array('d',yCoLaLoLa))
  CoLaLoLaGraph.SetLineColor(Colors[counter])
  CoLaLoLaGraph.SetMarkerColor(Colors[counter])
  counter += 1
  yCoLaLoLa_v2 = [0.053,0.175,0.049]
  CoLaLoLaGraph_v2 = ROOT.TGraph(len(xCoLaLoLa),array.array('d',xCoLaLoLa),array.array('d',yCoLaLoLa_v2))
  CoLaLoLaGraph_v2.SetLineColor(Colors[counter])
  CoLaLoLaGraph_v2.SetMarkerColor(Colors[counter])
  Legends.AddEntry(CoLaLoLaGraph_v2,'CoLaLoLa','p')
  Legends.AddEntry(CoLaLoLaGraph,'CoLaLoLa w/ energy scaling','p')
  MG.Add(CoLaLoLaGraph)
  MG.Add(CoLaLoLaGraph_v2)
  counter += 1
#MG.Draw('ap')
MG.Draw('apl')
MG.GetXaxis().SetTitle('Gluino Mass [GeV]')
MG.GetYaxis().SetTitle('Reconstruction efficiency (2g)')
Canvas.Update()
Canvas.Modified()
Legends.Draw('same')
Canvas.Print(outName)
Canvas.Print(outName+']')

##########################################
# Plot >=1g efficiency vs mass (Full case)
##########################################
# TCanvas
Canvas  = ROOT.TCanvas()
outName = "Plots/{}/i1gEfficiency_vs_gluino_mass_Full.pdf".format(vstr)
Canvas.Print(outName+"[")
# Legends
Legends = ROOT.TLegend(0.2,0.75,0.45,0.9)
Legends.SetTextFont(42)
# TMultiGraph
MG = ROOT.TMultiGraph()
# SPANet vs mass graphs
counter = 0
x = [mass for mass in Masses if mass!='All']
for version in Versions:
  y = [EvtEff[version]['all'][mass][MainJetMult]['>=1g'] for mass in Masses if mass!='All']
  Graph = ROOT.TGraph(len(x),array.array('d',x),array.array('d',y))
  Graph.SetLineColor(Colors[counter])
  Graph.SetMarkerColor(Colors[counter])
  MG.Add(Graph)
  extra  = ' {}'.format(Versions[version]) if Versions[version] != '' else ''
  legend = 'SPANet vs mass{}'.format(extra)
  Legends.AddEntry(Graph,legend,'p')
  counter += 1
  # SPANet average graph
  if ShowInclusivePerformance:
    AvgEff = EvtEff[version]['all']['All'][MainJetMult]['>=1g']
    yavg = [AvgEff for mass in Masses if mass!='All']
    AvgGraph = ROOT.TGraph(len(x),array.array('d',x),array.array('d',yavg))
    AvgGraph.SetLineColor(Colors[counter])
    AvgGraph.SetMarkerColor(Colors[counter])
    MG.Add(AvgGraph,'l')
    legend = 'SPANet inclusive{}'.format(extra)
    Legends.AddEntry(AvgGraph,legend,'l')
    counter += 1
MG.Draw('ap')
MG.GetXaxis().SetTitle('Gluino Mass [GeV]')
MG.GetYaxis().SetTitle('Reconstruction efficiency (>=1g)')
Canvas.Update()
Canvas.Modified()
Legends.Draw('same')
Canvas.Print(outName)
Canvas.Print(outName+']')

##########################################
# Plot 2g efficiency vs mass (==6 case)
##########################################
if ShowPerformanceForExactly6Jets:
  # TCanvas
  Canvas  = ROOT.TCanvas()
  outName = "Plots/{}/2gEfficiency_vs_gluino_mass_Exactly6Jets.pdf".format(vstr)
  Canvas.Print(outName+"[")
  # TMultiGraph
  MG = ROOT.TMultiGraph()
  # Legends
  Legends = ROOT.TLegend(0.2,0.75,0.4,0.9)
  Legends.SetTextFont(42)
  counter = 0
  for version in Versions:
    extra = ' {}'.format(Versions[version]) if Versions[version] != '' else ''
    # SPANet vs mass graph
    y = [EvtEff[version]['all'][mass]['==6']['2g'] for mass in Masses if mass!='All']
    Graph = ROOT.TGraph(len(x),array.array('d',x),array.array('d',y))
    Graph.SetLineColor(Colors[counter])
    Graph.SetMarkerColor(Colors[counter])
    Legends.AddEntry(Graph,'SPANet vs mass'+extra,'p')
    MG.Add(Graph,'pl')
    #if counter == 0: Graph.Draw()
    #else: Graph.Draw('same')
    counter += 1
    # SPANet average graph
    AvgEff = EvtEff[version]['all']['All']['==6']['2g']
    yavg = [AvgEff for mass in Masses if mass!='All']
    AvgGraph = ROOT.TGraph(len(x),array.array('d',x),array.array('d',yavg))
    AvgGraph.SetLineColor(Colors[counter])
    AvgGraph.SetMarkerColor(Colors[counter])
    Legends.AddEntry(AvgGraph,'SPANet inclusive'+extra,'l')
    #AvgGraph.Draw('same')
    MG.Add(AvgGraph,'l')
    counter += 1
  MG.Draw('ap')
  MG.GetXaxis().SetTitle('Gluino Mass [GeV]')
  MG.GetYaxis().SetTitle('Reconstruction efficiency (2g)')
  Legends.Draw('same')
  Canvas.Print(outName)
  Canvas.Print(outName+']')

##########################################
# Plot 2g efficiency vs mass (==8 case)
##########################################
if ShowPerformanceForExactly8Jets:
  # TCanvas
  Canvas  = ROOT.TCanvas()
  outName = "Plots/{}/2gEfficiency_vs_gluino_mass_Exactly8Jets.pdf".format(vstr)
  Canvas.Print(outName+"[")
  # TMultiGraph
  MG = ROOT.TMultiGraph()
  # Legends
  Legends = ROOT.TLegend(0.2,0.75,0.4,0.9)
  Legends.SetTextFont(42)
  counter = 0
  for version in Versions:
    extra = ' {}'.format(Versions[version]) if Versions[version] != '' else ''
    # SPANet vs mass graph
    y = [EvtEff[version]['all'][mass]['==8']['2g'] for mass in Masses if mass!='All']
    Graph = ROOT.TGraph(len(x),array.array('d',x),array.array('d',y))
    Graph.SetLineColor(Colors[counter])
    Graph.SetMarkerColor(Colors[counter])
    Legends.AddEntry(Graph,'SPANet vs mass'+extra,'p')
    MG.Add(Graph,'pl')
    #if counter == 0: Graph.Draw()
    #else: Graph.Draw('same')
    counter += 1
    # SPANet average graph
    AvgEff = EvtEff[version]['all']['All']['==8']['2g']
    yavg = [AvgEff for mass in Masses if mass!='All']
    AvgGraph = ROOT.TGraph(len(x),array.array('d',x),array.array('d',yavg))
    AvgGraph.SetLineColor(Colors[counter])
    AvgGraph.SetMarkerColor(Colors[counter])
    Legends.AddEntry(AvgGraph,'SPANet inclusive'+extra,'l')
    #AvgGraph.Draw('same')
    MG.Add(AvgGraph,'l')
    counter += 1
  MG.Draw('ap')
  MG.GetXaxis().SetTitle('Gluino Mass [GeV]')
  MG.GetYaxis().SetTitle('Reconstruction efficiency (2g)')
  Legends.Draw('same')
  Canvas.Print(outName)
  Canvas.Print(outName+']')

##########################################
# Plot 2g efficiency vs mass (==9 case)
##########################################
if ShowPerformanceForExactly9Jets:
  # TCanvas
  Canvas  = ROOT.TCanvas()
  outName = "Plots/{}/2gEfficiency_vs_gluino_mass_Exactly9Jets.pdf".format(vstr)
  Canvas.Print(outName+"[")
  # TMultiGraph
  MG = ROOT.TMultiGraph()
  # Legends
  Legends = ROOT.TLegend(0.2,0.75,0.4,0.9)
  Legends.SetTextFont(42)
  counter = 0
  for version in Versions:
    extra = ' {}'.format(Versions[version]) if Versions[version] != '' else ''
    # SPANet vs mass graph
    y = [EvtEff[version]['all'][mass]['==9']['2g'] for mass in Masses if mass!='All']
    Graph = ROOT.TGraph(len(x),array.array('d',x),array.array('d',y))
    Graph.SetLineColor(Colors[counter])
    Graph.SetMarkerColor(Colors[counter])
    Legends.AddEntry(Graph,'SPANet vs mass'+extra,'p')
    MG.Add(Graph,'pl')
    #if counter == 0: Graph.Draw()
    #else: Graph.Draw('same')
    counter += 1
    # SPANet average graph
    AvgEff = EvtEff[version]['all']['All']['==9']['2g']
    yavg = [AvgEff for mass in Masses if mass!='All']
    AvgGraph = ROOT.TGraph(len(x),array.array('d',x),array.array('d',yavg))
    AvgGraph.SetLineColor(Colors[counter])
    AvgGraph.SetMarkerColor(Colors[counter])
    Legends.AddEntry(AvgGraph,'SPANet inclusive'+extra,'l')
    #AvgGraph.Draw('same')
    MG.Add(AvgGraph,'l')
    counter += 1
  MG.Draw('ap')
  MG.GetXaxis().SetTitle('Gluino Mass [GeV]')
  MG.GetYaxis().SetTitle('Reconstruction efficiency (2g)')
  Legends.Draw('same')
  Canvas.Print(outName)
  Canvas.Print(outName+']')

##########################################
# Nevents vs mass (all + Full)
##########################################
# TCanvas
Canvas  = ROOT.TCanvas()
outName = "Plots/{}/Nevents_vs_gluino_mass_allFull.pdf".format(vstr)
Canvas.Print(outName+"[")
# TMultiGraph
MG = ROOT.TMultiGraph()
# SPANet vs mass graphs
counter = 0
x = [mass for mass in Masses if mass!='All']
for version in Versions:
  y = [EvtFrac_Den[version]['all'][mass][MainJetMult]['2g'] for mass in Masses if mass!='All']
  Graph = ROOT.TGraph(len(x),array.array('d',x),array.array('d',y))
  Graph.SetLineColor(Colors[counter])
  Graph.SetMarkerColor(Colors[counter])
  MG.Add(Graph)
  extra  = ' {}'.format(Versions[version]) if Versions[version] != '' else ''
  counter += 1
MG.Draw('ap')
MG.GetXaxis().SetTitle('Gluino Mass [GeV]')
MG.GetYaxis().SetTitle('Total number of events')
Canvas.Update()
Canvas.Modified()
Canvas.Print(outName)
Canvas.Print(outName+']')


##########################################
# Nevents vs mass (2g + Full)
##########################################
# TCanvas
Canvas  = ROOT.TCanvas()
outName = "Plots/{}/Nevents_vs_gluino_mass_2gFull.pdf".format(vstr)
Canvas.Print(outName+"[")
# TMultiGraph
MG = ROOT.TMultiGraph()
# SPANet vs mass graphs
counter = 0
x = [mass for mass in Masses if mass!='All']
for version in Versions:
  y = [EvtFrac_Num[version]['all'][mass][MainJetMult]['2g'] for mass in Masses if mass!='All']
  Graph = ROOT.TGraph(len(x),array.array('d',x),array.array('d',y))
  Graph.SetLineColor(Colors[counter])
  Graph.SetMarkerColor(Colors[counter])
  MG.Add(Graph)
  extra  = ' {}'.format(Versions[version]) if Versions[version] != '' else ''
  counter += 1
MG.Draw('ap')
MG.GetXaxis().SetTitle('Gluino Mass [GeV]')
MG.GetYaxis().SetTitle('Number of events (2g)')
Canvas.Update()
Canvas.Modified()
Canvas.Print(outName)
Canvas.Print(outName+']')

##########################################
# Nevents vs mass (>=1g + Full)
##########################################
# TCanvas
Canvas  = ROOT.TCanvas()
outName = "Plots/{}/Nevents_vs_gluino_mass_i1gFull.pdf".format(vstr)
Canvas.Print(outName+"[")
# TMultiGraph
MG = ROOT.TMultiGraph()
# SPANet vs mass graphs
counter = 0
x = [mass for mass in Masses if mass!='All']
for version in Versions:
  y = [EvtFrac_Num[version]['all'][mass][MainJetMult]['>=1g'] for mass in Masses if mass!='All']
  Graph = ROOT.TGraph(len(x),array.array('d',x),array.array('d',y))
  Graph.SetLineColor(Colors[counter])
  Graph.SetMarkerColor(Colors[counter])
  MG.Add(Graph)
  extra  = ' {}'.format(Versions[version]) if Versions[version] != '' else ''
  counter += 1
MG.Draw('ap')
MG.GetXaxis().SetTitle('Gluino Mass [GeV]')
MG.GetYaxis().SetTitle('Number of events (>=1g)')
Canvas.Update()
Canvas.Modified()
Canvas.Print(outName)
Canvas.Print(outName+']')

##########################################
# Compare 2g efficiencies b/w flavours
##########################################
if CompareFlavourTypes and len(Versions) == 1:
  # TCanvas
  Canvas  = ROOT.TCanvas()
  outName = "Plots/{}/2gEfficiency_vs_gluino_mass_Full_CompareFlavours.pdf".format(vstr)
  Canvas.Print(outName+"[")
  # Legends
  Legends = ROOT.TLegend(0.2,0.75,0.45,0.9)
  Legends.SetTextFont(42)
  # TMultiGraph
  MG = ROOT.TMultiGraph()
  # SPANet vs mass graphs
  Colors = [ROOT.kBlack,ROOT.kRed+1,ROOT.kCyan,ROOT.kOrange,ROOT.kGreen+2,ROOT.kMagenta]
  counter = 0
  x = [mass for mass in Masses if mass!='All']
  for version in Versions:
    for flav in FlavourTypes:
      y = [EvtEff[version][flav][mass][MainJetMult]['2g'] for mass in Masses if mass!='All']
      Graph = ROOT.TGraph(len(x),array.array('d',x),array.array('d',y))
      Graph.SetLineColor(Colors[counter])
      Graph.SetMarkerColor(Colors[counter])
      MG.Add(Graph)
      legend = 'All(ALL+UDS+UDB)' if flav == 'all' else flav
      Legends.AddEntry(Graph,legend,'p')
      counter += 1
  MG.Draw('ap')
  MG.GetXaxis().SetTitle('Gluino Mass [GeV]')
  MG.GetYaxis().SetTitle('Reconstruction efficiency (2g)')
  Canvas.Update()
  Canvas.Modified()
  Legends.Draw('same')
  Canvas.Print(outName)
  Canvas.Print(outName+']')

##########################################
# Nevents vs mass (2g + Full) vs flavour
##########################################
if CompareFlavourTypes and len(Versions) == 1:
  # TCanvas
  Canvas  = ROOT.TCanvas()
  outName = "Plots/{}/Nevents_vs_gluino_mass_2gFull_CompareFlavours.pdf".format(vstr)
  Canvas.Print(outName+"[")
  # Legends
  Legends = ROOT.TLegend(0.2,0.75,0.45,0.9)
  Legends.SetTextFont(42)
  # TMultiGraph
  MG = ROOT.TMultiGraph()
  # SPANet vs mass graphs
  counter = 0
  x = [mass for mass in Masses if mass!='All']
  for version in Versions:
    for flav in FlavourTypes:
      y = [EvtFrac_Num[version][flav][mass][MainJetMult]['2g'] for mass in Masses if mass!='All']
      Graph = ROOT.TGraph(len(x),array.array('d',x),array.array('d',y))
      Graph.SetLineColor(Colors[counter])
      Graph.SetMarkerColor(Colors[counter])
      MG.Add(Graph)
      legend = 'All(ALL+UDS+UDB)' if flav == 'all' else flav
      Legends.AddEntry(Graph,legend,'p')
      counter += 1
  MG.Draw('ap')
  MG.GetXaxis().SetTitle('Gluino Mass [GeV]')
  MG.GetYaxis().SetTitle('Number of events (2g)')
  Canvas.Update()
  Canvas.Modified()
  Legends.Draw('same')
  Canvas.Print(outName)
  Canvas.Print(outName+']')


# Show number of events for each mass point
Nevents = 0
if 'v11' in Versions:
  x = [mass for mass in Masses if mass!='All']
  y = [EvtFrac_Num['v11']['all'][mass][MainJetMult]['all'] for mass in Masses if mass!='All']
  for i in range(len(x)):
    print('Mass: {}'.format(x[i]))
    print('Nevents: {}'.format(y[i]))
    Nevents += y[i]
  print('Total number of events: {}'.format(Nevents))

print('>>> ALL DONE <<<')

##############################################################
## Glosary:
## source/mask (for counting number of jets with mask==True)
## g1/mask (to know if gluino 1 is reconstructable)
## g2/mask (to know if gluino 2 is reconstructable)
##############################################################
