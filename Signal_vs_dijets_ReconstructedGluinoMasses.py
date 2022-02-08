import h5py
import os
import ROOT
import numpy as np

VERSIONS = {
  # 1.4 TeV + max8jets
  #'spanet': 'v69', # spanet trained with v29 signal (1.4 TeV + max8jets + partial events)
  #'signal': 'v33', # 1.4 TeV + max8jets + normweight
  # all masses + max8 jets
  'spanet': 'v60', # spanet trained with v24 signal (all masses + max8jets + partial events)
  'signal': 'v32', # all masses + max8jets + normweight
}

DJ_in_path = '/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/ntuples/tag/input/mc16e/dijets_expanded/python/'
DJ_out_path = f'/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/spanet_jona/SPANET_Predictions/Dijets/{VERSIONS["spanet"]}/'

SAMPLES = {
  'Signal' : { # case : H5 file
    'True' : f'/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/spanet_jona/SPANET_inputs/signal_UDB_UDS_testing_{VERSIONS["signal"]}.h5', # max8jets including normweight
    'Pred' : f'/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/spanet_jona/SPANET_Predictions/Signal/{VERSIONS["spanet"]}/signal_testing_{VERSIONS["spanet"]}_output.h5',
  },
}

def get_reco_gluino_masses(case: str, case_dict: dict, use_avg: bool = True) -> tuple[dict, [float]]:
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
  return RecoMasses2g, normweight

def make_hist(case: str, masses_tuple: tuple) -> ROOT.TH1D:
  masses_dict, wgt_dict = masses_tuple
  hists = dict()
  for level, masses in masses_dict.items():
    if case == 'Dijets' and level == 'True': continue
    hist = ROOT.TH1D(f'{case}_{level}', '', 500, 0, 5000)
    for counter, value in enumerate(masses):
      hist.Fill(value, wgt_dict[level][counter])
    hists[level] = hist
  return hists

def compare_hists(hists: dict(), use_avg: bool = True):
  if not os.path.exists('Plots'):
    os.makedirs('Plots')

  colors = [ROOT.kBlack, ROOT.kRed, ROOT.kMagenta]
  
  # TCanvas
  Canvas = ROOT.TCanvas()
  comparison = '_vs_'.join(hists.keys())
  versions = f'spanet_{VERSIONS["spanet"]}_signal_{VERSIONS["signal"]}'
  extra = '_avg' if use_avg else ''
  outName = f"Plots/RecoMass_{comparison}_{versions}_2g{extra}.pdf"
  Canvas.Print(outName+"[")
  Canvas.SetLogy()
  Stack = ROOT.THStack()
  Legends = ROOT.TLegend(0.7,0.7,0.92,0.9)
  Legends.SetTextFont(42)
  counter = 0
  for case, case_dict in hists.items():
    for level, hist in case_dict.items():
      hist.SetLineColor(colors[counter])
      hist.SetMarkerColor(colors[counter])
      Stack.Add(hist, 'HIST][')
      Legends.AddEntry(hist, f'{case}_{level}')
      counter += 1
  Stack.Draw('nostack')
  if use_avg:
    Stack.GetXaxis().SetTitle('Averaged reconstructed gluino Mass [GeV]')
  else:
    Stack.GetXaxis().SetTitle('Reconstructed gluino Mass [GeV]')
  Stack.GetYaxis().SetTitle('Number of events')
  Legends.Draw("same")
  Canvas.Update()
  Canvas.Modified()
  Canvas.Print(outName)
  Canvas.Print(outName+']')

if __name__ == '__main__':

  # AtlasStyle
  ROOT.gROOT.LoadMacro("/afs/cern.ch/user/j/jbossios/work/public/xAOD/Results/AtlasStyle/AtlasStyle.C")
  ROOT.SetAtlasStyle()
  ROOT.gROOT.SetBatch(True)

  # Get Signal histogram
  use_avg = False
  print('INFO: Processing signal inputs...')
  hists = {'Signal': make_hist('Signal', get_reco_gluino_masses('Signal', SAMPLES['Signal'], use_avg))}

  # Get Dijets histogram (un-comment and test once I have new dijet inputs)
  dijet_masses = {'Pred': []}
  dijet_wgts = {'Pred': []}
  print('INFO: Processing dijet inputs...')
  for i in range(2, 13): # loop over JZ slices
    for h5_file in os.listdir(f'{DJ_in_path}/JZ{i}/'):
      if 'spanet' not in h5_file: continue # skip other formats
      true_file = f'{DJ_in_path}/JZ{i}/{h5_file}'
      jz_slice = f'0{i}' if i < 10 else i
      rtag = [tag for tag in ['r9364', 'r10201', 'r10724'] if tag in h5_file][0]
      file_ext = '.'.join(h5_file.split('.')[4:6])
      pred_file = f'{DJ_out_path}/dijets_{VERSIONS["spanet"]}_output_3647{jz_slice}_{rtag}_{file_ext}.h5'
      dijets_dict = {'True': true_file, 'Pred': pred_file}
      masses, wgts = get_reco_gluino_masses('Dijets', dijets_dict)
      dijet_masses['Pred'] += masses['Pred']
      dijet_wgts['Pred'] += wgts['Pred']
  hists['Dijets'] = make_hist('Dijets', (dijet_masses, dijet_wgts))
  compare_hists(hists, use_avg)
  print('>>> ALL DONE <<<')
