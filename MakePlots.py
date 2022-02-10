from compare_hists import compare_hists
import ROOT

lumi = 58450.1 # data18

VERSIONS = {
  # 1.4 TeV + max8jets
  #'spanet': 'v69', # spanet trained with v29 signal (1.4 TeV + max8jets + partial events)
  #'signal': 'v33', # 1.4 TeV + max8jets + normweight
  # all masses + max8 jets
  #'spanet': 'v60', # spanet trained with v24 signal (all masses + max8jets + partial events)
  #'signal': 'v32', # all masses + max8jets + normweight
  # all masses + max8 jets
  'spanet': 'v60', # spanet trained with v24 signal (all masses + max8jets + partial events)
  'signal': 'v38', # all masses + max8jets + normweight (testing+training=full)
}

# Open input file
input_file_name = f'Outputs/Histograms_SPANet_{VERSIONS["spanet"]}_Signal_{VERSIONS["signal"]}.root'
input_file = ROOT.TFile.Open(input_file_name)
hist_names = ['Signal_True', 'Signal_Pred', 'Pred_Dijets_Pred'] # FIXME
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
  hist.Scale(lumi)
  Hists[case][level] = hist

compare_hists(Hists, VERSIONS)
