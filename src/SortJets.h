#ifndef SORTJETS_H
#define SORTJETS_H

/*
  Sorts jets based on inference scores (highest to lowest).

  Example usage in RDataFrame:

  includePaths = ["functions.h","SortJets.h"]

  df = df.Define("SortedJets", f"FCCAnalyses::JetUtils::JetSorter::sort_jets_by_score({jetClusteringHelper.jets}, recojet_isTAU)")
  df = df.Define("SortedIndices", f"FCCAnalyses::JetUtils::JetSorter::get_sorted_indices(recojet_isTAU)")
*/

#include <algorithm>
#include <numeric>
#include <vector>
#include <stdexcept>

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

    // Get sorted indices
    auto sortedIndices = get_sorted_indices(scores);
    
    // Build sorted jets using sorted indices
    ROOT::VecOps::RVec<fastjet::PseudoJet> sortedJets;
    sortedJets.reserve(jets.size());

    for (auto i : sortedIndices)
      sortedJets.push_back(jets[i]);

    return sortedJets;
  }

  // Returns indices sorted by scores in descending order (highest to lowest)
  // Example: scores = [0.2, 0.4, 0.3] returns [1, 2, 0]
  static ROOT::VecOps::RVec<int> get_sorted_indices(
      const ROOT::VecOps::RVec<float> &scores)
  {
    // Create index vector
    std::vector<int> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Sort by descending score
    std::sort(indices.begin(), indices.end(),
              [&](int i, int j) { return scores[i] > scores[j]; });

    return ROOT::VecOps::RVec<int>(indices.begin(), indices.end());
  }
};

}} // namespace FCCAnalyses::JetUtils

#endif
