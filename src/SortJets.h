#ifndef SORTJETS_H
#define SORTJETS_H

/*
  Sorts jets based on inference scores (highest to lowest).

  Example usage in RDataFrame:

  includePaths = ["functions.h","SortJets.h"]

  df = df.Define("SortedJets", f"FCCAnalyses::JetUtils::JetSorter::sort_jets_by_score({jetClusteringHelper.jets}, recojet_isTAU)")

*/



namespace FCCAnalyses { namespace JetUtils {

struct JetSorter {

  JetSorter() = default;

  // Static sorting function
  static ROOT::VecOps::RVec<fastjet::PseudoJet> sort_jets_by_score(
      const ROOT::VecOps::RVec<fastjet::PseudoJet> &jets,
      const ROOT::VecOps::RVec<float> &scores)
  {
    if (jets.size() != scores.size())
    {
      throw std::runtime_error("JetSorter::sort_jets_by_score: jets and scores must have the same size");
    }

    // Create index vector
    std::vector<size_t> indices(jets.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Sort by descending score
    std::sort(indices.begin(), indices.end(),
              [&](size_t i, size_t j) { return scores[i] > scores[j]; });

    // Build sorted jets
    ROOT::VecOps::RVec<fastjet::PseudoJet> sortedJets;
    sortedJets.reserve(jets.size());

    for (auto i : indices)
      sortedJets.push_back(jets[i]);

    return sortedJets;
  }
};

}} // namespace FCCAnalyses::JetUtils

#endif
