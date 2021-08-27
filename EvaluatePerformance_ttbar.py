import h5py,os,sys
import pandas as pd
import numpy as np

Samples = {
  # ATLAS data
  #'Pred' : '/home/jbossios/cern/SUSY/RPVMJ/SPANet_outputs/v0/spanet_ttbar_ATLAS_testing_v0_output.h5', # using old version of ATLAS data
  #'True' : '/home/jbossios/cern/SUSY/RPVMJ/CreateH5inputs/Outputs/v1/AllData_testing.h5', # old version of ATLAS data
  #'True' : '/home/jbossios/cern/SUSY/RPVMJ/CreateH5inputs/Outputs/v2/AllData_testing.h5', # new version of ATLAS data
  # SPANet data
  'True' : '/home/jbossios/cern/SUSY/RPVMJ/CheckH5files/ttbar_SPANet/ttbar_testing.h5', # SPANet data
  'Pred' : '/home/jbossios/cern/SUSY/RPVMJ/SPANet_outputs/v1/spanet_ttbar_testing_v1_output.h5', # using SPAnet data
}

############################################################################################################################################################################
# DO NOT MODIFY (below this line)
############################################################################################################################################################################

############################################################################################################################################################################
# Type of events to be investigated
EventTypes = ['all','0t','1t','2t','>=1t'] # events with exactly one top reconstructable, exactly 2 top quarks reconstructable, or at least one top reconstructable
# Explanations:
# When looking at events in which the full event is fully reconstructable, that means that for the original testing dataset, I have mask equal to True on both t1 and t2 (that means jet reco indexes not equal to -1 for all q1/q2/b). Separatelly look at events in which only one top has mask True (the other must have mask False). Then look at events where at least one top is reconstructable. Then all events regardless of the number of reconstructable top quarks
############################################################################################################################################################################

############################################################################################################################################################################
# IDEA:
# In those four types of events compare the following quantities depending on the number of reco jets in the events (=6,=7,>=8,All events regardless of the number of jets):
#    Fraction of events (of the requested type and jet multiplicity) w.r.t. all events. Call this the Event Fraction
#    Event Reconstruction Efficiency: Fraction of events (of they chosen type and jet multiplicity) where the predicted indexes match the true (index) values
#        -> taking into account that I don't mind if I jet indexes from q1 are assigned to q2 or the other way around
############################################################################################################################################################################

#################################################################################
# Open input (Pred/True) files
#################################################################################
Files = { sample : h5py.File(name,'r')  for sample,name in Samples.items() }

#################################################################################
# Get data
#################################################################################
Data = dict()
for case in ['source','t1','t2']: # loop over group's keys
  Data[case] = { sample : Files[sample].get(case) for sample in Files }

# Get info about reco jets
jetMaskInfo = dict()
for sample in Samples: # loop over True/Pred
  jetMaskInfo[sample] = np.array(Data['source'][sample].get('mask'))

TotalNevents = jetMaskInfo['True'].shape[0]

# Get number of jets on each event
nJets = np.array([np.sum(x) for x in jetMaskInfo['True']])

# Get info for each particle
topInfo = dict()
for sample in Samples: # loop over True/Pred
  topInfo[sample] = dict()
  for topCase in ['t1','t2']: # loop over top particles
    topInfo[sample][topCase] = dict()
    for info in ['mask','b','q1','q2']: # loop over info types
      topInfo[sample][topCase][info] = np.array(Data[topCase][sample].get(info))

# Protections
if TotalNevents != jetMaskInfo['Pred'].shape[0]:
  print('ERROR: number of events b/w True and Pred do not match, exiting')
  sys.exit(1)
if TotalNevents != topInfo['True']['t1']['b'].size:
  print('ERROR: number of events b/w jetMaskInfo and topInfo do not match, exiting')
  sys.exit(1)

print('INFO: Total number of events = {}'.format(TotalNevents))

############################################
# Calculate the quantities mentioned above
############################################

#####################################################################################################
# Compare indexes b/w True and Pred on each event type to derive event efficiencies
# The efficiency is calculated as the fraction of events of a given type
# that agree on the jet indexes for the appropriate number of tops (account for symmetries: q1<->q2)
# Example:
# Event Signal Efficiency on 1t events will be the fraction of events in which
# one top can be reconstructed (t1 or t2 with mask==True in True sample)
# in which all jet (q1,q2,b) indexes for THAT VERY SAME top match to (q1,q2,b) or to (q2,q1,b)
# w.r.t. the number of events in which one top can be reconstructed (see above)
#####################################################################################################

# Collect numerator/denominator number of events for each event type

# Event Fraction inputs for each event type
EvtFrac_Num = { evtType : 0 for evtType in EventTypes}
EvtFrac_Den = { evtType : 0 for evtType in EventTypes}

# Event Efficiency for each event type
EvtEff_Num = { evtType : 0 for evtType in EventTypes}
EvtEff     = dict()

#####################################################################################################
# Event loop
#####################################################################################################
for ievent in range(TotalNevents):
  # Skip events with Njets < 6
  if nJets[ievent] < 6: continue
  # Identify event types (based on number of reconstructable top quarks)
  EvtTypes            = ['all']
  ReconstructableTops = 0 # number of reconstructable top quarks in this event
  for topCase in ['t1','t2']:
    if topInfo['True'][topCase]['mask'][ievent]:
      ReconstructableTops += 1
  if ReconstructableTops == 0:
    EvtTypes.append('0t')
  elif ReconstructableTops == 1:
    EvtTypes.append('1t')
  elif ReconstructableTops == 2:
    EvtTypes.append('2t')
  if ReconstructableTops >= 1:
    EvtTypes.append('>=1t')
  # Increase number of events for each event type
  for Type in EventTypes: # all event types
    EvtFrac_Den[Type] += 1
  for Type in EvtTypes: # only for identified event types
    EvtFrac_Num[Type] += 1
  # For each top's decay particle, find out if predicted index matches true index (considering symmetries)
  MatchIndexesByTop = dict()
  for topCase in ['t1','t2']:
    MatchIndexesByPart_vsTrueCase = dict()
    for altTopCase in ['t1','t2']:
      MatchIndexes = dict()
      for partCase in ['b','q1','q2']:
        if partCase == 'b':
          MatchIndexes[partCase] = True if topInfo['Pred'][topCase][partCase][ievent] == topInfo['True'][altTopCase][partCase][ievent] else False
        else: # q1 or q2
          altQuark = 'q2' if partCase == 'q1' else 'q1'
          opt1     = topInfo['Pred'][topCase][partCase][ievent] == topInfo['True'][altTopCase][partCase][ievent]
          opt2     = topInfo['Pred'][topCase][partCase][ievent] == topInfo['True'][altTopCase][altQuark][ievent]
          MatchIndexes[partCase] = True if opt1 or opt2 else False
      MatchIndexesByPart_vsTrueCase[altTopCase] = (MatchIndexes['b'] + MatchIndexes['q1'] + MatchIndexes['q2']) == 3
    MatchIndexesByTop[topCase] = MatchIndexesByPart_vsTrueCase['t1'] or MatchIndexesByPart_vsTrueCase['t2']
  # For each top, find out if all the predicted indexes for each decay product of the correspondings top quarks (considering constraints from each event type) match the true values
  EvtEff_Num['all']    += 1 if (MatchIndexesByTop['t1']+MatchIndexesByTop['t2']) >= 1 else False
  if '2t' in EvtTypes:
    EvtEff_Num['2t']   += 1 if (MatchIndexesByTop['t1']+MatchIndexesByTop['t2']) == 2 else False
  if '1t' in EvtTypes:
    EvtEff_Num['1t']   += 1 if (MatchIndexesByTop['t1']+MatchIndexesByTop['t2']) == 1 else False
  if '0t' in EvtTypes:
    EvtEff_Num['0t']   += 1 if (MatchIndexesByTop['t1']+MatchIndexesByTop['t2']) == 0 else False
  if '>=1t' in EvtTypes:
    if '1t' in EvtTypes:
      EvtEff_Num['>=1t'] += 1 if (MatchIndexesByTop['t1']+MatchIndexesByTop['t2']) == 1 else False
    elif '2t' in EvtTypes:
      EvtEff_Num['>=1t'] += 1 if (MatchIndexesByTop['t1']+MatchIndexesByTop['t2']) == 2 else False

# Compute efficiencies
for evtType in EventTypes:
  EvtEff[evtType] = EvtEff_Num[evtType] / EvtFrac_Num[evtType]

# Show results
print('{} of {} events have at least 6 reconstructed jets'.format(EvtFrac_Den['all'],TotalNevents))
for evtType in EventTypes:
  print('###############################################################')
  print('Event type: {}'.format(evtType))
  print('Event Fraction    = {}'.format(EvtFrac_Num[evtType]/EvtFrac_Den[evtType]))
  print('SPANet Efficiency = {}'.format(EvtEff[evtType]))

print('>>> ALL DONE <<<')

##############################################################
## Glosary:
## source/mask (for counting number of jets with mask==True)
## t1/mask (to know if top 1 is reconstructable)
## t2/mask (to know if top 2 is reconstructable)
##############################################################
