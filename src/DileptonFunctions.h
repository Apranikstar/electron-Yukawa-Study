#ifndef DILEPTON_H
#define DILEPTON_H

/*
  Computes the combined momentum of the two leading isolated leptons.

  includePaths = ["functions.h", "DileptonFunctions.h"]
  df = df.Define("dilepton",    "FCCAnalyses::Dilepton::DileptonCalculator::calculate(IsolatedElectron_4Vec, IsolatedMuon_4Vec)")
  df = df.Define("Dilepton_P",  "dilepton.P")
  df = df.Define("Dilepton_Pt", "dilepton.Pt")
  df = df.Define("Dilepton_M",  "dilepton.M")
*/

#include <TLorentzVector.h>
#include <ROOT/RVec.hxx>

namespace FCCAnalyses { namespace Dilepton {

struct DileptonCalculator {
    double P  = 0;
    double Pt = 0;
    double M  = 0;

    DileptonCalculator() = default;

    static DileptonCalculator calculate(const ROOT::VecOps::RVec<TLorentzVector>& elVecs,
                                        const ROOT::VecOps::RVec<TLorentzVector>& muVecs) {
        TLorentzVector sum;

        if (elVecs.size() >= 1) sum += elVecs[0];
        if (elVecs.size() >= 2) sum += elVecs[1];
        if (muVecs.size() >= 1) sum += muVecs[0];
        if (muVecs.size() >= 2) sum += muVecs[1];

        DileptonCalculator result;
        result.P  = sum.P();
        result.Pt = sum.Pt();
        result.M  = sum.M();
        return result;
    }
};

}} // namespace

#endif
