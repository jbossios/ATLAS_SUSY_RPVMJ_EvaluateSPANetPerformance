import ROOT
import os

# AtlasStyle
ROOT.gROOT.LoadMacro("/afs/cern.ch/user/j/jbossios/work/public/xAOD/Results/AtlasStyle/AtlasStyle.C")
ROOT.SetAtlasStyle()
ROOT.gROOT.SetBatch(True)

def compare_hists(hists: dict(), VERSIONS, gcase: str = 'avg', extra: str = ''):
  if not os.path.exists('Plots'):
    os.makedirs('Plots')

  colors = [ROOT.kBlack, ROOT.kRed, ROOT.kMagenta]
  
  # TCanvas
  Canvas = ROOT.TCanvas()
  comparison = '_vs_'.join(hists.keys())
  versions = f'spanet_{VERSIONS["spanet"]}'
  outName = f"Plots/RecoMass_{comparison}_{versions}_{gcase}{extra}.pdf"
  Canvas.Print(outName+"[")
  Canvas.SetLogy()
  Stack = ROOT.THStack()
  Legends = ROOT.TLegend(0.7,0.75,0.92,0.92)
  Legends.SetTextFont(42)
  counter = 0
  for case, case_dict in hists.items():
    for level, hist in case_dict[gcase].items():
      hist.Rebin(5)
      hist.SetLineColor(colors[counter])
      hist.SetMarkerColor(colors[counter])
      Stack.Add(hist, 'HIST][')
      Legends.AddEntry(hist, f'{case}_{level}')
      counter += 1
  Stack.Draw('nostack')
  if gcase == 'avg':
    Stack.GetXaxis().SetTitle('Averaged reconstructed gluino Mass [GeV]')
  else:
    Stack.GetXaxis().SetTitle('Reconstructed gluino {} Mass [GeV]'.format(1 if gcase == 'g1' else 2))
  Stack.GetYaxis().SetTitle('Number of events')
  Legends.Draw("same")
  Canvas.Update()
  Canvas.Modified()
  Canvas.Print(outName)
  Canvas.Print(outName+']')


def compare_spanet_versions(hists: dict, postfix: str, use_avg: bool = True):
  if not os.path.exists('Plots'):
    os.makedirs('Plots')

  colors = [ROOT.kBlack, ROOT.kRed, ROOT.kMagenta]

  # TCanvas
  Canvas = ROOT.TCanvas()
  extra = '_avg' if use_avg else ''
  outName = f"Plots/RecoMass_{postfix}_2g{extra}.pdf"
  Canvas.Print(outName+"[")
  Canvas.SetLogy()
  Stack = ROOT.THStack()
  Legends = ROOT.TLegend(0.7,0.75,0.92,0.92)
  Legends.SetTextFont(42)
  counter = 0
  true_plotted = False
  print(hists)
  for version, version_dict in hists.items():
    for level, hist in version_dict.items():
      if level == 'True' and true_plotted: continue
      if level == 'True':
        true_plotted = True
      hist.Rebin(5)
      hist.SetLineColor(colors[counter])
      hist.SetMarkerColor(colors[counter])
      Stack.Add(hist, 'HIST][')
      Legends.AddEntry(hist, f'{version}_{level}' if level != 'True' else 'True')
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

