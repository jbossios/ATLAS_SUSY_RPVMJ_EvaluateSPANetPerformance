import h5py
import os
import ROOT
import numpy as np
from multiprocessing import Pool
from functools import partial
from compare_hists import compare_hists

VERSIONS = {
  ## 1.4 TeV + max8jets
  #'spanet': 'v69', # spanet trained with v29 signal (1.4 TeV + max8jets + partial events)
  #'signal': 'v39', # 1.4 TeV + max8jets + normweight
  ## all masses + max8 jets (testing signal data only)
  #'spanet': 'v60', # spanet trained with v24 signal (all masses + max8jets + partial events)
  #'signal': 'v32', # all masses + max8jets + normweight
  ## all masses + max8 jets (full signal data)
  #'spanet': 'v60', # spanet trained with v24 signal (all masses + max8jets + partial events)
  #'signal': 'v38', # all masses + max8jets + normweight (testing+training=full)
  ## 1.4 TeV + max8jets + 50 GeV cut
  #'spanet': 'v71', # spanet trained with v40 signal (1.4 TeV + max8jets + partial events + 50 GeV cut)
  #'signal': 'v40', # 1.4 TeV + max8jets + fixed normweight + 50 GeV cut
  # 1.4 TeV + max8jets + 50 GeV cut
  #'spanet': 'v70', # spanet trained with v40 signal (1.4 TeV + max8jets + 2g events + 50 GeV cut)
  #'signal': 'v40', # 1.4 TeV + max8jets + fixed normweight + 50 GeV cut
  # May 2022
  # 1.4 TeV + max8jets + 50 GeV cut
  #'spanet': 'v93', # spanet trained with all mass points
  #'signal': 'v61', # 1.4 TeV + max8jets + fixed normweight + 50 GeV cut
  # 1.5 TeV + max8jets + 50 GeV cut
  #'spanet': 'v93', # spanet trained with all mass points
  #'signal': 'v65', # 1.5 TeV + max8jets + fixed normweight + 50 GeV cut
  # 1.4 TeV + max8jets + 50 GeV cut
  #'spanet': 'v94', # spanet trained with all mass points except 1.4 TeV
  #'signal': 'v61', # 1.4 TeV + max8jets + fixed normweight + 50 GeV cut
  # 1.5 TeV + max8jets + 50 GeV cut
  #'spanet': 'v95', # spanet trained with all mass points except 1.4 TeV
  #'signal': 'v65', # 1.5 TeV + max8jets + fixed normweight + 50 GeV cut
  # 23 June 2022: new H5 files
  'spanet': 'v96', # spanet trained with all mass points
}

SIGNAL_DSIDS = ['504518', '504539'] # 1.4 TeV

PATHS = {
  'Signal': {
    'spanet_inputs': {
      'path': '/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/ntuples/tag/input/mc16e/signal/HighStats/PROD0/h5/v0/',
      'skip_label': 'GGrpv2x3ALL',
    },
    'spanet_outputs': f'/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/spanet_jona/ML_Pipelines_Signal_Outputs/{VERSIONS["spanet"]}/renamed/',  # Temporary
  },
  'Dijets': {
    'spanet_inputs': {
      'path': '/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/ntuples/tag/input/mc16a/dijets/PROD3/h5/v1/',
      'skip_label': 'jetjet_JZWithSW_SRRPV',  # Temporary
    },
    'spanet_outputs': f'/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/spanet_jona/ML_Pipelines_Dijets_Outputs/{VERSIONS["spanet"]}/',
  }
}

# Path to output npz and root files
OUT_PATH = '/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/spanet_jona/Outputs_spanet_eval'

def get_reco_gluino_masses(case_dict: dict, case: str, use_avg: bool = True) -> tuple[dict, [float]]:
  """ Save reconstructed masses using true matched jets for '2g' events """
  RecoMasses = dict()

  # Open H5DF files and get data
  Files = {level: h5py.File(file_name, 'r') for level, file_name in case_dict.items()}
  groups = {
    'True': ['source', 'g1', 'g2', 'normweight'],
    'Pred': ['source', 'g1', 'g2'],
  }
  Data = {level: {case: Files[level].get(case) for case in groups[level]} for level in case_dict}
  
  # Get gluino info
  gluinoInfo = {level: {gCase: {info: np.array(Data[level][gCase].get(info)) for info in ['mask', 'q1', 'q2', 'q3']} for gCase in ['g1', 'g2']} for level in case_dict}

  # Normalization weight
  normweight_tmp = np.array(Data['True']['normweight'].get('normweight'))
  normweight = {'True': [], 'Pred': []}

  # Get jet info
  jetMaskInfo = {level: np.array(Data[level]['source'].get('mask')) for level in case_dict}
  jetPtInfo   = np.array(Data['True']['source'].get('pt'))
  jetEtaInfo  = np.array(Data['True']['source'].get('eta'))
  jetPhiInfo  = np.array(Data['True']['source'].get('phi'))
  jetMassInfo = np.array(Data['True']['source'].get('mass'))

  nEvents = {'True': 0, 'Pred': 0}

  for level, full_file_name in case_dict.items():
    if case == 'Dijets' and level == 'True': continue
    gcases = {
      True: ['avg'],
      False: ['g1', 'g2'],
    }[use_avg]
    RecoMasses[level] = {gcase: [] for gcase in gcases}
    # Event loop
    for ievent in range(jetMaskInfo[level].shape[0]):
      ReconstructableGluinos = 0 if case == 'Signal' and level == 'True' else 2 # number of reconstructable gluinos in this event
      if case == 'Signal' and level == 'True':
        for gCase in ['g1', 'g2']:
          if gluinoInfo['True'][gCase]['mask'][ievent]:
            ReconstructableGluinos += 1
      if ReconstructableGluinos >= 2: # for signal True, I look at only fully reconstructable events (just for simplicity)
        nEvents[level] += 1
        if use_avg: masses = dict()
        for gcase in ['g1', 'g2']:
          Jets = []
          for qcase in ['q1', 'q2', 'q3']:
            jetIndex = gluinoInfo[level][gcase][qcase][ievent]
            jetPt    = jetPtInfo[ievent][jetIndex]
            jetEta   = jetEtaInfo[ievent][jetIndex]
            jetPhi   = jetPhiInfo[ievent][jetIndex]
            jetM     = jetMassInfo[ievent][jetIndex]
            Jet      = ROOT.TLorentzVector()
            Jet.SetPtEtaPhiM(jetPt,jetEta,jetPhi,jetM)
            Jets.append(Jet)
          if use_avg: masses[gcase] = (Jets[0]+Jets[1]+Jets[2]).M()
          else:
            RecoMasses[level][gcase].append( (Jets[0]+Jets[1]+Jets[2]).M() )
            normweight[level].append(normweight_tmp[ievent])
        if use_avg:
          RecoMasses[level]['avg'].append(0.5 * (masses['g1'] + masses['g2']))
          normweight[level].append(normweight_tmp[ievent])
  for level, ifile in Files.items():
    ifile.close()
  return RecoMasses, normweight

def make_hist(case: str, masses_tuple: tuple) -> dict:
  masses_dict, wgt_dict = masses_tuple
  hists = dict()
  for level, masses in masses_dict.items():
    if 'Dijets' in case and level == 'True': continue
    hist = ROOT.TH1D(f'{case}_{level}', '', 500, 0, 5000)
    for counter, value in enumerate(masses):
      hist.Fill(value, wgt_dict[level][counter].item())
    hists[level] = hist
  return hists

def merge_hists(hists: [dict], name: str) -> dict:
  merged_hists = {key: 0 for key in hists[0].keys()}
  for key in hists[0].keys(): # loop over True/Pred
    for counter, hdict in enumerate(hists): # loop ver histograms
      if not counter:
        merged_hist = hdict[key].Clone(f'{name}_{key}')
      else:
        merged_hist.Add(hdict[key])
    merged_hists[key] = merged_hist
  return merged_hists

if __name__ == '__main__':
  
  # Setup
  use_avg = False # Temporary
  use_dijets = False # Temporary

  # AtlasStyle
  ROOT.gROOT.LoadMacro("/afs/cern.ch/user/j/jbossios/work/public/xAOD/Results/AtlasStyle/AtlasStyle.C")
  ROOT.SetAtlasStyle()
  ROOT.gROOT.SetBatch(True)

  # Get Signal histogram
  print('INFO: Processing signal inputs...')
  output_folder = f'{OUT_PATH}/npz_files/{VERSIONS["spanet"]}/Signal'
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)
  gcases = {
    True: ['avg'],
    False: ['g1', 'g2'],
  }[use_avg]
  signal_hists = {gcase: [] for gcase in gcases}
  signal_dicts = []
  # Loop over spanet input H5 files
  for file_name in os.listdir(PATHS['Signal']['spanet_inputs']['path']):
    if '.h5' not in file_name: continue  # skip other formats
    if PATHS['Signal']['spanet_inputs']['skip_label'] in file_name: continue  # skip undesired file
    dsid = file_name.split('.')[2]
    if dsid not in SIGNAL_DSIDS: continue  # skip undesired mass points
    true_file = f"{PATHS['Signal']['spanet_inputs']['path']}{file_name}"
    pred_file_name = file_name.replace('.h5', f'_spanet_{VERSIONS["spanet"]}.h5')
    pred_file = f"{PATHS['Signal']['spanet_outputs']}{pred_file_name}"
    signal_dicts.append({'True': true_file, 'Pred': pred_file})
  # Divide huge list into small lists
  n_dicts = len(signal_dicts)
  print(f'Number of files = {n_dicts}')
  step_size = 10
  n_lists_int = int(n_dicts/step_size)
  n_extra_files = n_lists_int*step_size - n_dicts
  n_lists = n_lists_int if not n_extra_files else n_lists_int+1
  print(f'{n_lists = }')
  signal_masses = {gcase: {'Pred': []} for gcase in gcases}
  signal_wgts = {'Pred': []}
  for ilist in range(n_lists):
    print(f'        Processing events {ilist+1}/{n_lists}...')
    imin = ilist*step_size
    if ilist != n_lists-1:
      imax = (ilist+1)*step_size
      signal_dicts_small = signal_dicts[imin:imax]
    else:
      signal_dicts_small = signal_dicts[imin:]
    with Pool(4) as p:
      get_reco_gluino_masses_partial = partial(get_reco_gluino_masses, case = 'Signal', use_avg = use_avg)
      result = p.map(get_reco_gluino_masses_partial, signal_dicts_small)
    signal_masses_dict = {gcase: {'Pred': [value for item in result for value in item[0]['Pred'][gcase]]} for gcase in gcases}
    signal_wgts_dict = {'Pred': [value for item in result for value in item[1]['Pred']]}
    for gcase in gcases:
      signal_hists[gcase].append(make_hist(f'Signal_{gcase}_{ilist}', (signal_masses_dict[gcase], signal_wgts_dict)))
    # collect data to save it to a .npz file
    for gcase in gcases:
      signal_masses[gcase]['Pred'] += signal_masses_dict[gcase]['Pred']
    signal_wgts['Pred'] += signal_wgts_dict['Pred']
  # Prepare Signal Pred input for Anthony
  output_file_name = f'{output_folder}/SPANet_{VERSIONS["spanet"]}_Signal_{"and".join(SIGNAL_DSIDS)}{"_avg" if use_avg else ""}.npz'
  print(f'INFO: Creating {output_file_name}')
  if gcases == ['avg']:
    signal_masses_array = signal_masses['avg']['Pred']
  else:
    signal_masses_array = np.column_stack((signal_masses['g1']['Pred'], signal_masses['g2']['Pred']))
  out_dict = {'trees_SRRPV_': {'mass_pred': signal_masses_array}}
  out_dict['trees_SRRPV_']['weights_pred'] = np.array(signal_wgts['Pred'])
  np.savez(output_file_name, **out_dict)
  hists = {'Signal': {gcase: merge_hists(signal_hists[gcase], f'Signal_{gcase}') for gcase in gcases}}

  # Get Dijets histogram
  if use_dijets:
    print('INFO: Processing dijets inputs...')
    output_folder = f'{OUT_PATH}/npz_files/{VERSIONS["spanet"]}/Dijets'
    if not os.path.exists(output_folder):
      os.makedirs(output_folder)
    dijets_hists = {gcase: [] for gcase in gcases}
    dijets_dicts = []
    # Loop over spanet input H5 files
    for file_name in os.listdir(PATHS['Dijets']['spanet_inputs']['path']):
      if '.h5' not in file_name: continue  # skip other formats
      if PATHS['Dijets']['spanet_inputs']['skip_label'] in file_name: continue  # skip undesired file
      true_file = f"{PATHS['Dijets']['spanet_inputs']['path']}{file_name}"
      pred_file_name = file_name.replace('.h5', f'_spanet_{VERSIONS["spanet"]}.h5')
      pred_file = f"{PATHS['Dijets']['spanet_outputs']}{pred_file_name}"
      dijets_dicts.append({'True': true_file, 'Pred': pred_file})
    # Divide huge list into small lists
    n_dicts = len(dijets_dicts)
    print(f'Number of files = {n_dicts}')
    step_size = 10
    n_lists_int = int(n_dicts/step_size)
    n_extra_files = n_lists_int*step_size - n_dicts
    n_lists = n_lists_int if not n_extra_files else n_lists_int+1
    print(f'{n_lists = }')
    dijets_masses = {gcase: {'Pred': []} for gcase in gcases}
    dijets_wgts = {'Pred': []}
    for ilist in range(n_lists):
      print(f'        Processing events {ilist+1}/{n_lists}...')
      imin = ilist*step_size
      if ilist != n_lists-1:
        imax = (ilist+1)*step_size
        dijets_dicts_small = dijets_dicts[imin:imax]
      else:
        dijets_dicts_small = dijets_dicts[imin:]
      with Pool(4) as p:
        get_reco_gluino_masses_partial = partial(get_reco_gluino_masses, case = 'Dijets', use_avg = use_avg)
        result = p.map(get_reco_gluino_masses_partial, dijets_dicts_small)
      dijets_masses_dict = {gcase: {'Pred': [value for item in result for value in item[0]['Pred'][gcase]]} for gcase in gcases}
      dijets_wgts_dict = {'Pred': [value for item in result for value in item[1]['Pred']]}
      for gcase in gcases:
        dijets_hists[gcase].append(make_hist(f'Dijets_{gcase}_{ilist}', (dijets_masses_dict[gcase], dijets_wgts_dict)))
      # collect data to save it to a .npz file
      for gcase in gcases:
        dijets_masses[gcase]['Pred'] += dijets_masses_dict[gcase]['Pred']
      dijets_wgts['Pred'] += dijets_wgts_dict['Pred']
    # Prepare Dijets Pred input for Anthony
    output_file_name = f'{output_folder}/SPANet_{VERSIONS["spanet"]}_Dijets{"_avg" if use_avg else ""}.npz'
    print(f'INFO: Creating {output_file_name}')
    if gcases == ['avg']:
      dijets_masses_array = dijets_masses['avg']['Pred']
    else:
      dijets_masses_array = np.column_stack((dijets_masses['g1']['Pred'], dijets_masses['g2']['Pred']))
    out_dict = {'trees_SRRPV_': {'mass_pred': dijets_masses_array}}
    out_dict['trees_SRRPV_']['weights_pred'] = np.array(dijets_wgts['Pred'])
    np.savez(output_file_name, **out_dict)
    hists['Dijets'] = {gcase: merge_hists(dijets_hists[gcase], f'Dijets_{gcase}') for gcase in gcases}

  # Write histograms
  output_folder = f'{OUT_PATH}/root_files'
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)
  output_file_name = f'root://eosatlas.cern.ch/{output_folder}/Histograms_SPANet_{VERSIONS["spanet"]}_Signal_{"and".join(SIGNAL_DSIDS)}{"_avg" if use_avg else ""}.root'
  print(f'INFO: Creating {output_file_name}')
  out_file = ROOT.TFile(output_file_name, 'RECREATE')
  for case, casedict in hists.items():  # loop over Signal/Dijets
    for gcase, hdict in casedict.items():
      for key, hist in hdict.items():  # loop over True/Pred
        hist.Write()
  out_file.Close()

  # Compare histograms
  compare_hists(hists, VERSIONS, 'g1' if not use_avg else 'avg', f'_Signal_{"and".join(SIGNAL_DSIDS)}')
  print('>>> ALL DONE <<<')
