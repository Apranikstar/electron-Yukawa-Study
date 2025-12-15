import json
import os
import copy
import urllib.request

# --------------------------------
# Read config file (environment variable or default)
# --------------------------------

# Parameters with defaults
drmin = 0.01
drmax = 0.2
selection = 0.2

# -----------------------------
# Your existing code
# -----------------------------

processList = {
    # xsecs need to be scaled by 280/989 ...for xsec of ee -> H ...

    # Semileptonic processes
    "wzp6_ee_Henueqq_ecm125": {"fraction": 1},
    "wzp6_ee_Hqqenue_ecm125": {"fraction": 1},
    "wzp6_ee_Hmunumuqq_ecm125": {"fraction": 1},
    "wzp6_ee_Hqqmunumu_ecm125": {"fraction": 1},
    "wzp6_ee_Htaunutauqq_ecm125": {"fraction": 1},
    "wzp6_ee_Hqqtaunutau_ecm125": {"fraction": 1},
    "wzp6_ee_taunutauqq_ecm125": {"fraction": 1},
    "wzp6_ee_tautauqq_ecm125": {"fraction": 1},
    "wzp6_ee_enueqq_ecm125": {"fraction": 1},
    "wzp6_ee_eeqq_ecm125": {"fraction": 1},
    "wzp6_ee_munumuqq_ecm125": {"fraction": 1},
    "wzp6_ee_mumuqq_ecm125": {"fraction": 1},

    # Fully leptonic Processes
    "wzp6_ee_Htautau_ecm125": {"fraction": 1},
    "wzp6_ee_Hllnunu_ecm125": {"fraction": 1},
    "wzp6_ee_eenunu_ecm125": {"fraction": 1},
    "wzp6_ee_mumununu_ecm125": {"fraction": 1},
    "wzp6_ee_tautaununu_ecm125": {"fraction": 1},
    "wzp6_ee_l1l2nunu_ecm125": {"fraction": 1},
    "wzp6_ee_tautau_ecm125": {"fraction": 1},
 
    # Fully hadronic Processes
    "wzp6_ee_Hgg_ecm125": {"fraction": 1},
    "wzp6_ee_Hbb_ecm125": {"fraction": 1},
    "wzp6_ee_qq_ecm125": {"fraction": 1},
    "p8_ee_ZZ_4tau_ecm125": {"fraction": 1},
}

outputDir = "/eos/experiment/fcc/ee/analyses/case-studies/higgs/electron_yukawa/FinalDataGen/on-shell-electron/"
inputDir = "/eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA"
nCPUS = -1
includePaths = ["../../src/functions.h", "../../src/GEOFunctions.h", "../../src/MELAFunctions.h","../../src/SortJets.h" ]

model_name = "fccee_flavtagging_edm4hep_wc"

url_model_dir = "https://fccsw.web.cern.ch/fccsw/testsamples/jet_flavour_tagging/winter2023/wc_pt_13_01_2022/"
url_preproc = "{}/{}.json".format(url_model_dir, model_name)
url_model = "{}/{}.onnx".format(url_model_dir, model_name)

model_dir = (
    "/eos/experiment/fcc/ee/jet_flavour_tagging/winter2023/wc_pt_7classes_12_04_2023/"
)
local_preproc = "{}/{}.json".format(model_dir, model_name)
local_model = "{}/{}.onnx".format(model_dir, model_name)


def get_file_path(url, filename):
    """Return local file path if exists else download from url and return basename."""
    if os.path.exists(filename):
        return os.path.abspath(filename)
    else:
        urllib.request.urlretrieve(url, os.path.basename(url))
        return os.path.basename(url)


weaver_preproc = get_file_path(url_preproc, local_preproc)
weaver_model = get_file_path(url_model, local_model)

from addons.ONNXRuntime.jetFlavourHelper import JetFlavourHelper
from addons.FastJet.jetClusteringHelper import ExclusiveJetClusteringHelper

# helpers to be used inside analysers
jetFlavourHelper = None
jetClusteringHelper = None


class RDFanalysis:
    """RDFanalysis class: defines the transformations applied to the input dataframe."""

    @staticmethod
    def analysers(df):
        """Define aliases, variables, clustering, tagger inference, and filters."""

        # Aliases
        df = df.Alias("Particle0", "Particle#0.index")
        df = df.Alias("Particle1", "Particle#1.index")
        df = df.Alias("MCRecoAssociations0", "MCRecoAssociations#0.index")
        df = df.Alias("MCRecoAssociations1", "MCRecoAssociations#1.index")
        df = df.Alias("Electron0", "Electron#0.index")
        df = df.Alias("Muon0", "Muon#0.index")
        df = df.Alias("Photon0", "Photon#0.index")
        df = df.Alias("Jet2", "Jet#2.index")


        # Missing energy variables
        df = df.Define("MissingEnergy_4Vec", "FCCAnalyses::ReconstructedParticle::get_tlv(MissingET)")
        df = df.Define("MissingEnergy_Pt", "MissingEnergy_4Vec[0].Pt()")
        df = df.Filter("MissingEnergy_Pt > 3")
        df = df.Define("MissingEnergy_P", "MissingEnergy_4Vec[0].P()")
        df = df.Define("MissingEnergy_E", "MissingEnergy_4Vec[0].E()")
        df = df.Define("MissingEnergy_Theta", "MissingEnergy_4Vec[0].Theta()")
        df = df.Define("MissingEnergy_Phi", "MissingEnergy_4Vec[0].Phi()")
        df = df.Define("MissingEnergy_CosTheta", "MissingEnergy_4Vec[0].CosTheta()")
        df = df.Define("MissingEnergy_CosPhi", "TMath::Cos(MissingEnergy_Phi)")

        # Photons and charged hadrons
        df = df.Define("Photons_All", "FCCAnalyses::ReconstructedParticle::get(Photon0, ReconstructedParticles)")
        df = df.Define("Photons_Selected", "FCCAnalyses::ReconstructedParticle::sel_p(20)(Photons_All)")
        df = df.Define(
            "ChargedHadrons",
            "ReconstructedParticle2MC::selRP_ChargedHadrons(MCRecoAssociations0,MCRecoAssociations1,ReconstructedParticles,Particle)",
        )

        # Leptons
        df = df.Define("Electrons_All", "FCCAnalyses::ReconstructedParticle::get(Electron0, ReconstructedParticles)")
        df = df.Define("Muons_All", "FCCAnalyses::ReconstructedParticle::get(Muon0, ReconstructedParticles)")
        df = df.Define("Electrons_PreSelection", "FCCAnalyses::ReconstructedParticle::sel_p(0)(Electrons_All)")
        df = df.Define("Muons_PreSelection", "FCCAnalyses::ReconstructedParticle::sel_p(0)(Muons_All)")

        # Isolation using JSON-configurable parameters
        df = df.Define(
            "Electrons_IsolationValue",
            f"FCCAnalyses::ZHfunctions::coneIsolation({drmin}, {drmax})(Electrons_PreSelection, ChargedHadrons)",
        )
        df = df.Define(
            "Electrons_Isolated",
            f"FCCAnalyses::ZHfunctions::sel_iso({selection})(Electrons_PreSelection, Electrons_IsolationValue)",
        )

        df = df.Define(
            "Muons_IsolationValue",
            f"FCCAnalyses::ZHfunctions::coneIsolation({drmin}, {drmax})(Muons_PreSelection, ChargedHadrons)",
        )
        df = df.Define(
            "Muons_Isolated",
            f"FCCAnalyses::ZHfunctions::sel_iso({selection})(Muons_PreSelection, Muons_IsolationValue)",
        )

        df = df.Define(
            "Photons_IsolationValue", 
            f"FCCAnalyses::ZHfunctions::coneIsolation({drmin}, {drmax})(Photons_Selected, ChargedHadrons)"
        )
        df = df.Define(
            "Photons_Isolated", 
            f"FCCAnalyses::ZHfunctions::sel_iso({selection})(Photons_Selected, Photons_IsolationValue)"
        )

        # Lepton counting
        df = df.Define("N_IsolatedMuons", "Muons_Isolated.size()")
        df = df.Define("N_IsolatedElectrons", "Electrons_Isolated.size()")

        # Require exactly one isolated electron and no isolated muons
        df = df.Filter("N_IsolatedElectrons == 1")
        df = df.Filter("N_IsolatedMuons == 0")

        # Isolated lepton 4-vectors
        df = df.Define("IsolatedElectron_4Vec", "FCCAnalyses::ReconstructedParticle::get_tlv(Electrons_Isolated)")
        df = df.Define("IsolatedMuon_4Vec", "FCCAnalyses::ReconstructedParticle::get_tlv(Muons_Isolated)")

        # Isolated photons info
        df = df.Define("IsolatedPhotons_4Vec", "FCCAnalyses::ReconstructedParticle::get_tlv(Photons_Isolated)")
        df = df.Define("IsolatedPhoton_Phi", "IsolatedPhotons_4Vec[0].Phi()")
        df = df.Define("IsolatedPhoton_Theta", "IsolatedPhotons_4Vec[0].Theta()")
        df = df.Define("IsolatedPhoton_E", "IsolatedPhotons_4Vec[0].E()")
        df = df.Define("IsolatedPhoton_CosTheta", "IsolatedPhotons_4Vec[0].CosTheta()")
        df = df.Define("IsolatedPhoton_CosPhi", "TMath::Cos(IsolatedPhoton_Phi)")
        df = df.Define("N_IsolatedPhotons", "Photons_Isolated.size()")

        # Isolated electron properties
        df = df.Define("IsolatedElectron_3Vec", "IsolatedElectron_4Vec[0].Vect()")
        df = df.Define("IsolatedElectron_P", "IsolatedElectron_4Vec[0].P()")
        df = df.Define("IsolatedElectron_Phi", "IsolatedElectron_4Vec[0].Phi()")
        df = df.Define("IsolatedElectron_Theta", "IsolatedElectron_4Vec[0].Theta()")
        df = df.Define("IsolatedElectron_E", "IsolatedElectron_4Vec[0].E()")
        df = df.Define("IsolatedElectron_CosTheta", "IsolatedElectron_4Vec[0].CosTheta()")
        df = df.Define("IsolatedElectron_CosPhi", "TMath::Cos(IsolatedElectron_Phi)")
        df = df.Define("IsolatedElectron_Charge", "FCCAnalyses::ReconstructedParticle::get_charge(Electrons_Isolated)[0]")

        # Combined masses
        df = df.Define("ElectronNeutrino_InvariantMass", "(MissingEnergy_4Vec[0] + IsolatedElectron_4Vec[0]).M()")

        # Create collections with particles removed
        df = df.Define(
            "RecoParticles_NoPhotons",
            "FCCAnalyses::ReconstructedParticle::remove(ReconstructedParticles, Photons_Selected)",
        )
        df = df.Define(
            "RecoParticles_NoPhotonsNoElectrons",
            "FCCAnalyses::ReconstructedParticle::remove(RecoParticles_NoPhotons, Electrons_PreSelection)",
        )
        df = df.Define(
            "RecoParticles_NoLeptonsNoPhotons",
            "FCCAnalyses::ReconstructedParticle::remove(RecoParticles_NoPhotonsNoElectrons, Muons_PreSelection)",
        )

        # Jet clustering setup
        global jetClusteringHelper
        global jetFlavourHelper

        collections = {
            "GenParticles": "Particle",
            "PFParticles": "ReconstructedParticles",
            "PFTracks": "EFlowTrack",
            "PFPhotons": "EFlowPhoton",
            "PFNeutralHadrons": "EFlowNeutralHadron",
            "TrackState": "EFlowTrack_1",
            "TrackerHits": "TrackerHits",
            "CalorimeterHits": "CalorimeterHits",
            "dNdx": "EFlowTrack_2",
            "PathLength": "EFlowTrack_L",
            "Bz": "magFieldBz",
        }

        collections_noleptons_nophotons = copy.deepcopy(collections)
        collections_noleptons_nophotons["PFParticles"] = "RecoParticles_NoLeptonsNoPhotons"

        jetClusteringHelper = ExclusiveJetClusteringHelper(collections_noleptons_nophotons["PFParticles"], 2)
        df = jetClusteringHelper.define(df)

        # Jet flavour tagger
        jetFlavourHelper = JetFlavourHelper(
            collections_noleptons_nophotons,
            jetClusteringHelper.jets,
            jetClusteringHelper.constituents,
        )
        df = jetFlavourHelper.define(df)

        # Filters and jet definitions
        df = df.Filter("event_njet > 1")
        df = df.Define("Jets_4Vec", "JetConstituentsUtils::compute_tlv_jets({})".format(jetClusteringHelper.jets))
        df = df.Define("DiJet_InvariantMass", "JetConstituentsUtils::InvariantMass(Jets_4Vec[0], Jets_4Vec[1])")
        df = df.Filter("DiJet_InvariantMass < 52.85")
        df = df.Filter("DiJet_InvariantMass > 4")

        # Tagger inference
        df = jetFlavourHelper.inference(weaver_preproc, weaver_model, df)

        jetClusteringHelper_N2 = ExclusiveJetClusteringHelper("RecoParticles_NoLeptonsNoPhotons", 2, "N2")
        df = jetClusteringHelper_N2.define(df)

        df = df.Define("JetClustering_d23", "std::sqrt(JetClusteringUtils::get_exclusive_dmerge(_jet_N2, 2))")
        df = df.Define("JetClustering_d34", "std::sqrt(JetClusteringUtils::get_exclusive_dmerge(_jet_N2, 3))")
        df = df.Define("Jets_Charge", "JetConstituentsUtils::get_charge({})".format(jetClusteringHelper.constituents))
        df = df.Define("Jet1_NConstituents", "jet_nconst[0]")
        df = df.Define("Jet2_NConstituents", "jet_nconst[1]")
        df = df.Filter("Jet1_NConstituents > 2")
        df = df.Filter("Jet2_NConstituents > 2")

        # Jet 1 kinematics
        df = df.Define("Jet1_3Vec", "Jets_4Vec[0].Vect()")
        df = df.Define("Jet1_P", "Jets_4Vec[0].P()")
        df = df.Define("Jet1_Eta", "Jets_4Vec[0].Eta()")
        df = df.Define("Jet1_Phi", "Jets_4Vec[0].Phi()")
        df = df.Define("Jet1_M", "Jets_4Vec[0].M()")
        df = df.Define("Jet1_E", "Jets_4Vec[0].E()")
        df = df.Define("Jet1_Theta", "Jets_4Vec[0].Theta()")
        df = df.Define("Jet1_CosTheta", "Jets_4Vec[0].CosTheta()")
        df = df.Define("Jet1_CosPhi", "TMath::Cos(Jet1_Phi)")

        # Jet 2 kinematics
        df = df.Define("Jet2_3Vec", "Jets_4Vec[1].Vect()")
        df = df.Define("Jet2_P", "Jets_4Vec[1].P()")
        df = df.Define("Jet2_Eta", "Jets_4Vec[1].Eta()")
        df = df.Define("Jet2_Phi", "Jets_4Vec[1].Phi()")
        df = df.Define("Jet2_M", "Jets_4Vec[1].M()")
        df = df.Define("Jet2_E", "Jets_4Vec[1].E()")
        df = df.Define("Jet2_Theta", "Jets_4Vec[1].Theta()")
        df = df.Define("Jet2_CosTheta", "Jets_4Vec[1].CosTheta()")
        df = df.Define("Jet2_CosPhi", "TMath::Cos(Jet2_Phi)")

        # Jet energy and charge
        df = df.Define("Jets_MaxEnergy", "TMath::Max(Jets_4Vec[0].E(), Jets_4Vec[1].E())")
        df = df.Define("Jets_MinEnergy", "TMath::Min(Jets_4Vec[0].E(), Jets_4Vec[1].E())")
        df = df.Define("Jet1_Charge", "ROOT::VecOps::Sum(Jets_Charge[0])")
        df = df.Define("Jet2_Charge", "ROOT::VecOps::Sum(Jets_Charge[1])")

        # Angular separations: Jet1 vs Jet2
        df = df.Define("Jet1Jet2_DeltaR", "Jets_4Vec[0].DeltaR(Jets_4Vec[1])")
        df = df.Define("Jet1Jet2_DeltaPhi", "Jets_4Vec[0].DeltaPhi(Jets_4Vec[1])")
        df = df.Define("Jet1Jet2_DeltaTheta", "Jet1_Theta - Jet2_Theta")
        df = df.Define("Jet1Jet2_Angle", "Jets_4Vec[0].Angle(Jets_4Vec[1].Vect())")
        df = df.Define("Jet1Jet2_CosAngle", "TMath::Cos(Jet1Jet2_Angle)")

        # Angular separations: Electron vs Jet1
        df = df.Define("ElectronJet1_DeltaR", "IsolatedElectron_4Vec[0].DeltaR(Jets_4Vec[0])")
        df = df.Define("ElectronJet1_DeltaPhi", "IsolatedElectron_4Vec[0].DeltaPhi(Jets_4Vec[0])")
        df = df.Define("ElectronJet1_Angle", "IsolatedElectron_4Vec[0].Angle(Jets_4Vec[0].Vect())")
        df = df.Define("ElectronJet1_CosAngle", "TMath::Cos(ElectronJet1_Angle)")

        # Angular separations: Electron vs Jet2
        df = df.Define("ElectronJet2_DeltaR", "IsolatedElectron_4Vec[0].DeltaR(Jets_4Vec[1])")
        df = df.Define("ElectronJet2_DeltaPhi", "IsolatedElectron_4Vec[0].DeltaPhi(Jets_4Vec[1])")
        df = df.Define("ElectronJet2_Angle", "IsolatedElectron_4Vec[0].Angle(Jets_4Vec[1].Vect())")
        df = df.Define("ElectronJet2_CosAngle", "TMath::Cos(ElectronJet2_Angle)")

        # Max/Min electron-jet angular separations
        df = df.Define("ElectronJets_MaxDeltaR", "TMath::Max(ElectronJet1_DeltaR, ElectronJet2_DeltaR)")
        df = df.Define("ElectronJets_MinDeltaR", "TMath::Min(ElectronJet1_DeltaR, ElectronJet2_DeltaR)")
        df = df.Define("ElectronJets_MaxDeltaPhi", "TMath::Max(ElectronJet1_DeltaPhi, ElectronJet2_DeltaPhi)")
        df = df.Define("ElectronJets_MinDeltaPhi", "TMath::Min(ElectronJet1_DeltaPhi, ElectronJet2_DeltaPhi)")
        df = df.Define("ElectronJets_MaxCosAngle", "TMath::Max(ElectronJet1_CosAngle, ElectronJet2_CosAngle)")
        df = df.Define("ElectronJets_MinCosAngle", "TMath::Min(ElectronJet1_CosAngle, ElectronJet2_CosAngle)")

        # Angular separations: MissingEnergy vs Electron
        df = df.Define("MissingEnergyElectron_DeltaR", "MissingEnergy_4Vec[0].DeltaR(IsolatedElectron_4Vec[0])")
        df = df.Define("MissingEnergyElectron_DeltaPhi", "MissingEnergy_4Vec[0].DeltaPhi(IsolatedElectron_4Vec[0])")
        df = df.Define("MissingEnergyElectron_Angle", "MissingEnergy_4Vec[0].Angle(IsolatedElectron_4Vec[0].Vect())")
        df = df.Define("MissingEnergyElectron_CosAngle", "TMath::Cos(MissingEnergyElectron_Angle)")

        # Angular separations: MissingEnergy vs Jet1
        df = df.Define("MissingEnergyJet1_DeltaR", "MissingEnergy_4Vec[0].DeltaR(Jets_4Vec[0])")
        df = df.Define("MissingEnergyJet1_DeltaPhi", "MissingEnergy_4Vec[0].DeltaPhi(Jets_4Vec[0])")
        df = df.Define("MissingEnergyJet1_Angle", "MissingEnergy_4Vec[0].Angle(Jets_4Vec[0].Vect())")
        df = df.Define("MissingEnergyJet1_CosAngle", "TMath::Cos(MissingEnergyJet1_Angle)")

        # Angular separations: MissingEnergy vs Jet2
        df = df.Define("MissingEnergyJet2_DeltaR", "MissingEnergy_4Vec[0].DeltaR(Jets_4Vec[1])")
        df = df.Define("MissingEnergyJet2_DeltaPhi", "MissingEnergy_4Vec[0].DeltaPhi(Jets_4Vec[1])")
        df = df.Define("MissingEnergyJet2_Angle", "MissingEnergy_4Vec[0].Angle(Jets_4Vec[1].Vect())")
        df = df.Define("MissingEnergyJet2_CosAngle", "TMath::Cos(MissingEnergyJet2_Angle)")

        # Max/Min MissingEnergy-jet angular separations
        df = df.Define("MissingEnergyJets_MaxDeltaR", "TMath::Max(MissingEnergyJet1_DeltaR, MissingEnergyJet2_DeltaR)")
        df = df.Define("MissingEnergyJets_MinDeltaR", "TMath::Min(MissingEnergyJet1_DeltaR, MissingEnergyJet2_DeltaR)")
        df = df.Define("MissingEnergyJets_MaxDeltaPhi", "TMath::Max(MissingEnergyJet1_DeltaPhi, MissingEnergyJet2_DeltaPhi)")
        df = df.Define("MissingEnergyJets_MinDeltaPhi", "TMath::Min(MissingEnergyJet1_DeltaPhi, MissingEnergyJet2_DeltaPhi)")
        df = df.Define("MissingEnergyJets_MaxCosAngle", "TMath::Max(MissingEnergyJet1_CosAngle, MissingEnergyJet2_CosAngle)")
        df = df.Define("MissingEnergyJets_MinCosAngle", "TMath::Min(MissingEnergyJet1_CosAngle, MissingEnergyJet2_CosAngle)")

        # Event-level invariant mass
        df = df.Define("Event_InvariantMass", "(Jets_4Vec[0] + Jets_4Vec[1] + MissingEnergy_4Vec[0] + IsolatedElectron_4Vec[0]).M()")

        # Combined system masses
        df = df.Define("ElectronDiJet_InvariantMass", "(Jets_4Vec[0] + Jets_4Vec[1] + IsolatedElectron_4Vec[0]).M()")
        df = df.Define("ElectronJet1_InvariantMass", "(Jets_4Vec[0] + IsolatedElectron_4Vec[0]).M()")
        df = df.Define("ElectronJet2_InvariantMass", "(Jets_4Vec[1] + IsolatedElectron_4Vec[0]).M()")
        df = df.Define("DiJet_Energy", "(Jets_4Vec[0] + Jets_4Vec[1]).E()")

        # System angular properties
        df = df.Define("ElectronDiJet_Phi", "(Jets_4Vec[0] + Jets_4Vec[1] + IsolatedElectron_4Vec[0]).Phi()")
        df = df.Define("DiJet_Phi", "(Jets_4Vec[0] + Jets_4Vec[1]).Phi()")

        # W boson (leptonic) properties
        df = df.Define("WBosonLeptonic_InvariantMass", "(IsolatedElectron_4Vec[0] + MissingEnergy_4Vec[0]).M()")
        df = df.Define("WBosonLeptonic_Theta", "(IsolatedElectron_4Vec[0] + MissingEnergy_4Vec[0]).Theta()")

        # On-shell and off-shell mass variables
        df = df.Define("System_OnShellMass", "TMath::Max((Jets_4Vec[0]+Jets_4Vec[1]).M(), WBosonLeptonic_InvariantMass)")
        df = df.Define("System_OffShellMass", "TMath::Min((Jets_4Vec[0]+Jets_4Vec[1]).M(), WBosonLeptonic_InvariantMass)")
        df = df.Define("System_MaxCosTheta", "TMath::Cos(TMath::Max((Jets_4Vec[0]+Jets_4Vec[1]).Theta(), WBosonLeptonic_Theta))")
        df = df.Define("System_MinCosTheta", "TMath::Cos(TMath::Min((Jets_4Vec[0]+Jets_4Vec[1]).Theta(), WBosonLeptonic_Theta))")
        df = df.Define("Event_EnergyImbalance", "125.0 - MissingEnergy_4Vec[0].E() - IsolatedElectron_4Vec[0].E() - Jets_4Vec[0].E() - Jets_4Vec[1].E()")

        # MELA Variables (production and decay angles)
        df = df.Define(
            "MELA_Angles",
            "FCCAnalyses::MELA::MELACalculator::mela(Jets_4Vec[0], Jets_4Vec[1], MissingEnergy_4Vec[0], IsolatedElectron_4Vec[0], IsolatedElectron_Charge, Jet1_Charge, Jet2_Charge)",
        )
        df = df.Define("MELA_Phi", "MELA_Angles.phi")
        df = df.Define("MELA_CosPhi", "MELA_Angles.cosPhi")
        df = df.Define("MELA_Phi1", "MELA_Angles.phi1")
        df = df.Define("MELA_CosPhi1", "MELA_Angles.cosPhi1")
        df = df.Define("MELA_PhiStar", "MELA_Angles.phiStar")
        df = df.Define("MELA_CosPhiStar", "MELA_Angles.cosPhiStar")
        df = df.Define("MELA_ThetaStar", "MELA_Angles.thetaStar")
        df = df.Define("MELA_CosThetaStar", "MELA_Angles.cosThetaStar")
        df = df.Define("MELA_Theta1", "MELA_Angles.theta1")
        df = df.Define("MELA_CosTheta1", "MELA_Angles.cosTheta1")
        df = df.Define("MELA_Theta2", "MELA_Angles.theta2")
        df = df.Define("MELA_CosTheta2", "MELA_Angles.cosTheta2")

        # Event-shape variables
        df = df.Define("EventShape_Planarity", "FCCAnalyses::GEOFunctions::EventGeoFunctions::calculatePlanarity(Jet1_3Vec, Jet2_3Vec, IsolatedElectron_3Vec)")
        df = df.Define("EventShape_Aplanarity", "FCCAnalyses::GEOFunctions::EventGeoFunctions::calculateAplanarity(Jet1_3Vec, Jet2_3Vec, IsolatedElectron_3Vec)")
        df = df.Define("EventShape_Sphericity", "FCCAnalyses::GEOFunctions::EventGeoFunctions::calculateSphericity(Jet1_3Vec, Jet2_3Vec, IsolatedElectron_3Vec)")
        df = df.Define("EventShape_Asphericity", "FCCAnalyses::GEOFunctions::EventGeoFunctions::calculateAsphericity(Jet1_3Vec, Jet2_3Vec, IsolatedElectron_3Vec)")

        # Jet flavor tagging scores - Products
        df = df.Define("JetFlavor_GluonProduct", "recojet_isG[0] * recojet_isG[1]")
        df = df.Define("JetFlavor_UpQuarkProduct", "recojet_isU[0] * recojet_isU[1]")
        df = df.Define("JetFlavor_DownQuarkProduct", "recojet_isD[0] * recojet_isD[1]")
        df = df.Define("JetFlavor_StrangeQuarkProduct", "recojet_isS[0] * recojet_isS[1]")
        df = df.Define("JetFlavor_CharmQuarkProduct", "recojet_isC[0] * recojet_isC[1]")
        df = df.Define("JetFlavor_BottomQuarkProduct", "recojet_isB[0] * recojet_isB[1]")
        df = df.Define("JetFlavor_TauProduct", "recojet_isTAU[0] * recojet_isTAU[1]")

        return df

    @staticmethod
    def output():
        """Return a list of all branches defined in analysers()."""
        branchList = [
            # Photon variables
            "IsolatedPhoton_Phi",
            "IsolatedPhoton_Theta",
            "IsolatedPhoton_E",
            "IsolatedPhoton_CosTheta",
            "IsolatedPhoton_CosPhi",
            "N_IsolatedPhotons",

            # Electron variables
            "IsolatedElectron_P",
            "IsolatedElectron_Phi",
            "IsolatedElectron_Theta",
            "IsolatedElectron_E",
            "IsolatedElectron_CosTheta",
            "IsolatedElectron_CosPhi",
            "N_IsolatedElectrons",

            # Missing energy variables
            "MissingEnergy_P",
            "MissingEnergy_E",
            "MissingEnergy_Pt",
            "MissingEnergy_Theta",
            "MissingEnergy_Phi",
            "MissingEnergy_CosTheta",
            "MissingEnergy_CosPhi",

            # Combined system masses
            "ElectronNeutrino_InvariantMass",
            "DiJet_InvariantMass",

            # Jet clustering variables
            "JetClustering_d23",
            "JetClustering_d34",

            # Jet constituent counts
            "Jet1_NConstituents",
            "Jet2_NConstituents",
            
            # Jet 1 kinematics
            "Jet1_P",
            "Jet1_Phi",
            "Jet1_M",
            "Jet1_E",
            "Jet1_Theta",
            "Jet1_CosTheta",
            "Jet1_CosPhi",

            # Jet 2 kinematics
            "Jet2_P",
            "Jet2_Phi",
            "Jet2_M",
            "Jet2_E",
            "Jet2_Theta",
            "Jet2_CosTheta",
            "Jet2_CosPhi",

            # Jet energy and charge
            "Jets_MaxEnergy",
            "Jets_MinEnergy",
            "Jet1_Charge",
            "Jet2_Charge",

            # Angular separations: Jet1 vs Jet2
            "Jet1Jet2_DeltaR",
            "Jet1Jet2_DeltaPhi",
            "Jet1Jet2_DeltaTheta",
            "Jet1Jet2_Angle",
            "Jet1Jet2_CosAngle",

            # Angular separations: Electron vs Jets
            "ElectronJet1_DeltaR",
            "ElectronJet1_DeltaPhi",
            "ElectronJet1_Angle",
            "ElectronJet1_CosAngle",
            "ElectronJet2_DeltaR",
            "ElectronJet2_DeltaPhi",
            "ElectronJet2_Angle",
            "ElectronJet2_CosAngle",
            "ElectronJets_MaxDeltaR",
            "ElectronJets_MinDeltaR",
            "ElectronJets_MaxDeltaPhi",
            "ElectronJets_MinDeltaPhi",
            "ElectronJets_MaxCosAngle",
            "ElectronJets_MinCosAngle",

            # Angular separations: MissingEnergy vs Electron
            "MissingEnergyElectron_DeltaR",
            "MissingEnergyElectron_DeltaPhi",
            "MissingEnergyElectron_Angle",
            "MissingEnergyElectron_CosAngle",

            # Angular separations: MissingEnergy vs Jets
            "MissingEnergyJet1_DeltaR",
            "MissingEnergyJet1_DeltaPhi",
            "MissingEnergyJet1_Angle",
            "MissingEnergyJet1_CosAngle",
            "MissingEnergyJet2_DeltaR",
            "MissingEnergyJet2_DeltaPhi",
            "MissingEnergyJet2_Angle",
            "MissingEnergyJet2_CosAngle",
            "MissingEnergyJets_MaxDeltaR",
            "MissingEnergyJets_MinDeltaR",
            "MissingEnergyJets_MaxDeltaPhi",
            "MissingEnergyJets_MinDeltaPhi",
            "MissingEnergyJets_MaxCosAngle",
            "MissingEnergyJets_MinCosAngle",

            # Event-level masses
            "Event_InvariantMass",
            "ElectronDiJet_InvariantMass",
            "ElectronJet1_InvariantMass",
            "ElectronJet2_InvariantMass",
            "DiJet_Energy",

            # System angular properties
            "ElectronDiJet_Phi",
            "DiJet_Phi",

            # W boson properties
            "WBosonLeptonic_InvariantMass",
            "WBosonLeptonic_Theta",
            "System_OnShellMass",
            "System_OffShellMass",
            "System_MaxCosTheta",
            "System_MinCosTheta",
            "Event_EnergyImbalance",

            # MELA angular variables
            "MELA_Phi",
            "MELA_CosPhi",
            "MELA_Phi1",
            "MELA_CosPhi1",
            "MELA_PhiStar",
            "MELA_CosPhiStar",
            "MELA_ThetaStar",
            "MELA_CosThetaStar",
            "MELA_Theta1",
            "MELA_CosTheta1",
            "MELA_Theta2",
            "MELA_CosTheta2",

            # Event shape variables
            "EventShape_Planarity",
            "EventShape_Aplanarity",
            "EventShape_Sphericity",
            "EventShape_Asphericity",

            # Jet flavor tagging scores
            "JetFlavor_GluonProduct",
            "JetFlavor_UpQuarkProduct",
            "JetFlavor_DownQuarkProduct",
            "JetFlavor_StrangeQuarkProduct",
            "JetFlavor_CharmQuarkProduct",
            "JetFlavor_BottomQuarkProduct",
            "JetFlavor_TauProduct",
        ]
        return branchListMELA_Angles.thetaStar")
        df = df.Define("
