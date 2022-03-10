import h5py
import os
import ROOT
import numpy as np
from multiprocessing import Pool
from functools import partial
from compare_hists import compare_hists

# TODO
# FIXME
# UPDATEME
# Need to start using the new dijet inputs (now not separated into JZ slices!)

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
  'spanet': 'v70', # spanet trained with v40 signal (1.4 TeV + max8jets + 2g events + 50 GeV cut)
  'signal': 'v40', # 1.4 TeV + max8jets + fixed normweight + 50 GeV cut
}

# Dijets inputs
DJ_in_path = '/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/ntuples/tag/input/mc16e/dijets_expanded_fixed/python/'
DJ_out_path = f'/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/spanet_jona/ML_Pipelines_Dijets_Outputs/{VERSIONS["spanet"]}/'

SAMPLES = {
  'Signal' : { # case : H5 file
    #'True' : f'/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/spanet_jona/SPANET_inputs/signal_UDB_UDS_testing_{VERSIONS["signal"]}.h5', # max8jets including normweight
    #'Pred' : f'/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/spanet_jona/SPANET_Predictions/Signal/{VERSIONS["spanet"]}/signal_testing_{VERSIONS["spanet"]}_output.h5',
    'True' : f'/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/spanet_jona/SPANET_inputs/signal_UDB_UDS_full_{VERSIONS["signal"]}.h5', # max8jets including normweight
    'Pred' : f'/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/spanet_jona/SPANET_Predictions/Signal/{VERSIONS["spanet"]}/signal_full_{VERSIONS["spanet"]}_output.h5',
  },
}

# Path to output npz and root files
OUT_PATH = '/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/spanet_jona/Outputs_spanet_eval'

def get_reco_gluino_masses(case_dict: dict, case: str, use_avg: bool = True) -> tuple[dict, [float]]:
  """ Save reconstructed masses using true matched jets for '2g' events """
  RecoMasses2g = dict()

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
    RecoMasses2g[level] = []
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
            RecoMasses2g[level].append( (Jets[0]+Jets[1]+Jets[2]).M() )
            normweight[level].append(normweight_tmp[ievent])
        if use_avg:
          RecoMasses2g[level].append(0.5 * (masses['g1'] + masses['g2']))
          normweight[level].append(normweight_tmp[ievent])
  for level, ifile in Files.items():
    ifile.close()
  return RecoMasses2g, normweight

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
  use_avg = False
  use_dijets = True

  # AtlasStyle
  ROOT.gROOT.LoadMacro("/afs/cern.ch/user/j/jbossios/work/public/xAOD/Results/AtlasStyle/AtlasStyle.C")
  ROOT.SetAtlasStyle()
  ROOT.gROOT.SetBatch(True)

  # Get Signal histogram
  print('INFO: Processing signal inputs...')
  masses, wgts = get_reco_gluino_masses(SAMPLES['Signal'], 'Signal', use_avg)
  hists = {'Signal': make_hist('Signal', (masses, wgts))}

  # Prepare Signal Pred input for Anthony
  output_folder = f'{OUT_PATH}/npz_files/{VERSIONS["spanet"]}/Signal'
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)
  output_file_name = f'{output_folder}/SPANet_{VERSIONS["spanet"]}_Signal_{VERSIONS["signal"]}{"_avg" if use_avg else ""}.npz'
  print(f'INFO: Creating {output_file_name}')
  np.savez(output_file_name, mass_pred=masses['Pred'], mass_true=masses['True'], weights_pred=wgts['Pred'], weights_true=wgts['True'])

  # Get Dijets histogram
  if use_dijets:
    output_folder = f'{OUT_PATH}/npz_files/{VERSIONS["spanet"]}/Dijets'
    if not os.path.exists(output_folder):
      os.makedirs(output_folder)
    print('INFO: Processing dijet inputs...')
    dijets_hists = []
    for i in range(2, 13): # loop over JZ slices
      print(f'        Processing JZ{i} inputs...')
      dijets_dicts = []
      for h5_file in os.listdir(f'{DJ_in_path}/JZ{i}/'):
        if 'spanet' not in h5_file: continue # skip other formats
        true_file = f'{DJ_in_path}/JZ{i}/{h5_file}'
        jz_slice = f'0{i}' if i < 10 else i
        rtag = [tag for tag in ['r9364', 'r10201', 'r10724'] if tag in h5_file][0]
        file_ext = '.'.join(h5_file.split('.')[4:6])
        pred_file = f'{DJ_out_path}/dijets_{VERSIONS["spanet"]}_output_3647{jz_slice}_{rtag}_{file_ext}.h5'
        dijets_dicts.append({'True': true_file, 'Pred': pred_file})
      # Divide huge list into small lists
      n_dicts = len(dijets_dicts)
      print(f'Number of files = {n_dicts}')
      step_size = 10
      n_lists_int = int(n_dicts/step_size)
      n_extra_files = n_lists_int*step_size - n_dicts
      n_lists = n_lists_int if not n_extra_files else n_lists_int+1
      print(f'{n_lists = }')
      dijet_masses = {'Pred': []}
      dijet_wgts = {'Pred': []}
      for ilist in range(n_lists):
        print(f'        Processing events {ilist+1}/{n_lists}...')
        imin = ilist*step_size
        if ilist != n_lists-1:
          imax = (ilist+1)*step_size
          dijets_dicts_small = dijets_dicts[imin:imax]
        else:
          dijets_dicts_small = dijets_dicts[imin:]
        with Pool(8) as p:
          get_reco_gluino_masses_partial = partial(get_reco_gluino_masses, case = 'Dijets', use_avg = use_avg)
          result = p.map(get_reco_gluino_masses_partial, dijets_dicts_small)
        dijet_masses_dict = {'Pred': [value for item in result for value in item[0]['Pred']]}
        dijet_wgts_dict = {'Pred': [value for item in result for value in item[1]['Pred']]}
        dijets_hists.append(make_hist(f'Dijets_JZ{i}_{ilist}', (dijet_masses_dict, dijet_wgts_dict)))
        # collect data to save it to a .npz file
        dijet_masses['Pred'] += dijet_masses_dict['Pred']
        dijet_wgts['Pred'] += dijet_wgts_dict['Pred']
      # Prepare Dijets Pred input for Anthony
      output_file_name = f'{output_folder}/SPANet_{VERSIONS["spanet"]}_Dijets_JZ{i}{"_avg" if use_avg else ""}.npz'
      print(f'INFO: Creating {output_file_name}')
      np.savez(output_file_name, mass_pred=dijet_masses['Pred'], weights_pred=dijet_wgts['Pred'])
    hists['Dijets'] = merge_hists(dijets_hists, 'Dijets')

  # Write histograms
  output_folder = f'{OUT_PATH}/root_files'
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)
  output_file_name = f'root://eosatlas.cern.ch/{output_folder}/Histograms_SPANet_{VERSIONS["spanet"]}_Signal_{VERSIONS["signal"]}{"_avg" if use_avg else ""}.root'
  print(f'INFO: Creating {output_file_name}')
  out_file = ROOT.TFile(output_file_name, 'RECREATE')
  for case, hdict in hists.items(): # loop over Signal/Dijets
    for key, hist in hdict.items(): # loop over True/Pred
      hist.Write()
  out_file.Close()

  # Compare histograms
  compare_hists(hists, VERSIONS, use_avg)
  print('>>> ALL DONE <<<')
