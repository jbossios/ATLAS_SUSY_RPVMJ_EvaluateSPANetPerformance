from compare_hists import compare_hists
import ROOT
import sys

lumi = 58450.1 # data18

use_avg = False

VERSIONS = {
  # 1.4 TeV + max8jets
  #'spanet': 'v69', # spanet trained with v29 signal (1.4 TeV + max8jets + partial events)
  #'signal': 'v39', # 1.4 TeV + max8jets + normweight
  # all masses + max8 jets
  #'spanet': 'v60', # spanet trained with v24 signal (all masses + max8jets + partial events)
  #'signal': 'v32', # all masses + max8jets + normweight
  # all masses + max8 jets
  #'spanet': 'v60', # spanet trained with v24 signal (all masses + max8jets + partial events)
  #'signal': 'v38', # all masses + max8jets + normweight (testing+training=full)
  # 1.4 TeV + max8jets + 50 GeV cut
  #'spanet': 'v71', # spanet trained with v29 signal (1.4 TeV + max8jets + partial events + 50 GeV cut)
  #'signal': 'v40', # 1.4 TeV + max8jets + normweight
  # 1.4 TeV + max8jets + 50 GeV cut
  'spanet': 'v70', # spanet trained with v29 signal (1.4 TeV + max8jets + 2g events + 50 GeV cut)
  'signal': 'v40', # 1.4 TeV + max8jets + normweight
}

# Open input file
extra = '_avg' if use_avg else ''
input_file_name = f'/eos/atlas/atlascerngroupdisk/phys-susy/RPV_mutlijets_ANA-SUSY-2019-24/spanet_jona/Outputs_spanet_eval/root_files/Histograms_SPANet_{VERSIONS["spanet"]}_Signal_{VERSIONS["signal"]}{extra}.root'
input_file = ROOT.TFile.Open(input_file_name)
if not input_file:
  print(f'ERROR: {input_file_name} not found, exiting')
  sys.exit(1)
hist_names = ['Signal_True', 'Signal_Pred', 'Dijets_Pred']
Hists = {
  'Signal' : {
    'Pred' : 0,
    'True' : 0,
  },
  'Dijets' : {
    'Pred' : 0,
  },
}
for hist_name in hist_names:
  case = 'Signal' if 'Signal' in hist_name else 'Dijets'
  level = 'True' if 'True' in hist_name else 'Pred'
  hist = input_file.Get(hist_name)
  if not hist:
    print(f'ERROR: {hist_name} not found, exiting')
    sys.exit(1)
  hist.Scale(lumi)
  Hists[case][level] = hist

compare_hists(Hists, VERSIONS, use_avg)
